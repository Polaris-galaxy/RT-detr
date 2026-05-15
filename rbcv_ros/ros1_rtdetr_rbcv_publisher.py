#!/usr/bin/env python3
"""
ROS1 Noetic：订阅相机图与 CameraInfo，运行 RT-DETR，向 RBCV 下游发布标准 JSON（std_msgs/String）。

发布话题（默认）：/rbcv/semantic_detections
每帧、每个目标一条消息，UTF-8 JSON 对象，字段与 RBCV exploration 的 JSONL 一致::

    {"t_s":123.4,"class_name":"wool","x":1.2,"y":3.4,"confidence":0.88,"track_id":null}

x,y 为 map 系地面交点（见 projection；需 TF 或静态外参）.

依赖：rospy, cv_bridge, sensor_msgs, std_msgs, geometry_msgs, tf2_ros, tf2_geometry_msgs, PyYAML, numpy
以及本仓库 ultralytics（pip install -e .）.

运行（在已 source /opt/ros/noetic/setup.bash 且已安装本包的环境中）::

    rosrun <your_pkg> ros1_rtdetr_rbcv_publisher.py _model_path:=/path/to/best.pt _image_topic:=/camera/color/image_raw
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# 保证可导入本目录 projection
_RBCV_ROS = Path(__file__).resolve().parent
if str(_RBCV_ROS) not in sys.path:
    sys.path.insert(0, str(_RBCV_ROS))

from projection import (  # noqa: E402
    bottom_center_xyxy,
    intersect_ray_ground_in_map,
    ray_dir_optical_from_pixel,
)

try:
    import rospy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    import tf2_ros
    from tf2_ros import Buffer, TransformListener
except ImportError as e:
    raise SystemExit(
        "需要 ROS1 + cv_bridge + tf2_ros。请在 Ubuntu/Noetic 或 docker 中运行。\n"
        f"ImportError: {e}"
    ) from e

from ultralytics import RTDETR  # noqa: E402


def _class_name_from_result(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def _T_from_tf_msg(transform) -> np.ndarray:
    """geometry_msgs Transform -> 4x4，将源坐标系的点变换到目标坐标系（与 ROS tf2 语义一致）。"""
    import tf.transformations as tft

    t = transform.translation
    r = transform.rotation
    q = [r.x, r.y, r.z, r.w]
    T = tft.quaternion_matrix(q)
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


class RTDETRBRBCVNode:
    def __init__(self) -> None:
        rospy.init_node("rtdetr_rbcv_publisher", anonymous=True)

        model_path = rospy.get_param("~model_path", "")
        if not model_path:
            rospy.logfatal("必须设置私有参数 ~model_path:=/绝对或相对路径/best.pt")
            sys.exit(2)
        self._conf = float(rospy.get_param("~conf", 0.25))
        imgsz = int(rospy.get_param("~imgsz", 640))
        self._imgsz = imgsz

        img_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        ci_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        fixed_frame = rospy.get_param("~fixed_frame", "map")
        optical_child = rospy.get_param("~optical_frame", "")  # 若为空用 CameraInfo.header.frame_id

        pub_topic = rospy.get_param("~detections_pub", "/rbcv/semantic_detections")

        z_ground = float(rospy.get_param("~z_map_ground", 0.0))
        static_T_path = rospy.get_param("~static_T_map_cam_yaml", "")  # 可选：无 TF 时使用

        self._fixed_frame = fixed_frame
        self._optical_override = optical_child
        self._z_map_ground = z_ground
        self._T_map_opt_static: np.ndarray | None = None
        self._ci_from_yaml: dict = {}
        if static_T_path:
            import yaml

            with open(static_T_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self._ci_from_yaml = cfg.get("camera", {})
            cam_frame_yaml = cfg.get("optical_frame", "")
            if cam_frame_yaml:
                self._cam_frame_id = cam_frame_yaml
            Tlist = cfg.get("T_map_optical") or cfg.get("T_map_cam")
            if Tlist is None:
                rospy.logfatal("static yaml 缺少 T_map_optical（4x4 行列表）")
                sys.exit(2)
            self._T_map_opt_static = np.array(Tlist, dtype=np.float64)
            rospy.loginfo("使用静态外参 T_map_optical（将不依赖 TF 平移求解）")

        self._bridge = CvBridge()
        self._model = RTDETR(model_path)
        self._K: tuple[float, float, float, float] | None = None
        if self._ci_from_yaml and {"fx", "fy", "cx", "cy"} <= self._ci_from_yaml.keys():
            self._K = (
                float(self._ci_from_yaml["fx"]),
                float(self._ci_from_yaml["fy"]),
                float(self._ci_from_yaml["cx"]),
                float(self._ci_from_yaml["cy"]),
            )
        self._tf_buffer = Buffer(rospy.Duration(10.0))
        self._tf_listener = TransformListener(self._tf_buffer)

        self._pub = rospy.Publisher(pub_topic, String, queue_size=200)
        if self._K is None:
            rospy.Subscriber(ci_topic, CameraInfo, self._on_cam_info, queue_size=1)
        rospy.Subscriber(img_topic, Image, self._on_image, queue_size=1, buff_size=2**24)

        setattr(self, "_cam_frame_id", getattr(self, "_cam_frame_id", ""))

        rospy.loginfo(
            "rtdetr_rbcv: model=%s image=%s cam_info=%s pub=%s fixed_frame=%s",
            model_path,
            img_topic,
            ci_topic,
            pub_topic,
            fixed_frame,
        )

    def _on_cam_info(self, msg: CameraInfo) -> None:
        fx = float(msg.K[0])
        fy = float(msg.K[4])
        cx = float(msg.K[2])
        cy = float(msg.K[5])
        self._K = (fx, fy, cx, cy)
        self._cam_frame_id = msg.header.frame_id

    def _lookup_T_map_optical(self, stamp) -> np.ndarray | None:
        if self._T_map_opt_static is not None:
            return self._T_map_opt_static
        child = self._optical_override or getattr(self, "_cam_frame_id", None)
        if not child or self._K is None:
            return None
        try:
            ts: TransformStamped = self._tf_buffer.lookup_transform(
                self._fixed_frame,
                child,
                rospy.Time(0),
            )
        except Exception as e:
            rospy.logwarn_throttle(5.0, "TF 查询失败（%s -> %s）: %s", self._fixed_frame, child, e)
            return None
        return _T_from_tf_msg(ts.transform)

    def _on_image(self, msg: Image) -> None:
        if self._K is None:
            rospy.logwarn_throttle(5.0, "仍在等待 CameraInfo（或 static yaml 中 camera.fx/fy/cx/cy）")
            return
        fx, fy, cx, cy = self._K
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge: %s", e)
            return

        results = self._model.predict(
            source=bgr,
            conf=self._conf,
            imgsz=self._imgsz,
            verbose=False,
        )
        if not results:
            return
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return

        t_s = _stamp_to_seconds(msg.header.stamp)
        names = getattr(r0, "names", None) or getattr(self._model.model, "names", {})
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)

        T_map_opt = self._lookup_T_map_optical(msg.header.stamp)
        if T_map_opt is None:
            rospy.logwarn_throttle(2.0, "无有效 TF/外参，跳过该帧")
            return

        for i in range(xyxy.shape[0]):
            u, v = bottom_center_xyxy(xyxy[i])
            ray = ray_dir_optical_from_pixel(u, v, fx, fy, cx, cy)
            hit = intersect_ray_ground_in_map(
                T_map_opt,
                np.zeros(3),
                ray,
                z_map_ground=self._z_map_ground,
            )
            if hit is None:
                continue
            x_m, y_m = hit
            cname = _class_name_from_result(names, int(clss[i]))
            payload = {
                "t_s": t_s,
                "class_name": cname,
                "x": x_m,
                "y": y_m,
                "confidence": float(confs[i]),
                "track_id": None,
            }
            s = String()
            s.data = json.dumps(payload, ensure_ascii=False)
            self._pub.publish(s)

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    RTDETRBRBCVNode().spin()


if __name__ == "__main__":
    main()
