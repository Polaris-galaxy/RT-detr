#!/usr/bin/env python3
"""
离线回放 ROS1 bag：解码 ``sensor_msgs/Image``，RT-DETR 推理，输出 RBCV 用 JSONL（无 rospy）。

需要提供 ``--extrinsic-yaml``（camera.fx/fy/cx/cy + ``T_map_optical``）。

用法::

    python rbcv_ros/offline_bag_rtdetr_jsonl.py --bag record.bag --image-topic /camera/color/image_raw ^
      --model best.pt --extrinsic-yaml rbcv_ros/static_extrinsic.example.yaml --out detections.jsonl

依赖：rosbags, numpy, pyyaml, opencv-python, torch, ultralytics（本仓库 editable 安装）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT_SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT_SCR))
from projection import bottom_center_xyxy, intersect_ray_ground_in_map, ray_dir_optical_from_pixel  # noqa: E402


def _deserialize_ros1(raw: bytes, mtype: str):
    from rosbags.typesys import Stores, get_typestore

    for name in ("ROS1_NOETIC", "ROS1_MELODIC", "ROS1_KINETIC"):
        if not hasattr(Stores, name):
            continue
        try:
            ts = get_typestore(getattr(Stores, name))
            return ts.deserialize_ros1(raw, mtype)
        except Exception:
            continue
    raise RuntimeError(f"无法反序列化 {mtype!r}，请升级 rosbags / 检查消息类型")


def _image_msg_to_bgr(msg) -> np.ndarray:
    import cv2

    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    raw = bytes(msg.data) if not isinstance(msg.data, bytes) else msg.data
    arr = np.frombuffer(raw, dtype=np.uint8)
    if enc in ("bgr8", "bgra8", "rgb8", "rgba8"):
        c = arr.size // (h * w)
        im = arr.reshape((h, w, c))
        if enc.startswith("rgb"):
            return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        return im[:, :, :3] if c >= 3 else im
    if enc == "mono8":
        g = arr.reshape((h, w))
        return np.stack([g, g, g], axis=-1)
    raise ValueError(f"暂不支持的 encoding: {msg.encoding}")


def _stamp_to_sec(msg) -> float:
    st = msg.header.stamp
    return float(st.sec) + float(getattr(st, "nanosec", getattr(st, "nsecs", 0))) * 1e-9


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--bag", required=True)
    pa.add_argument("--image-topic", required=True)
    pa.add_argument("--model", required=True)
    pa.add_argument("--extrinsic-yaml", required=True)
    pa.add_argument("--out", default="detections_from_bag.jsonl")
    pa.add_argument("--conf", type=float, default=0.25)
    pa.add_argument("--imgsz", type=int, default=640)
    pa.add_argument("--max-frames", type=int, default=0)
    pa.add_argument("--z-map-ground", type=float, default=0.0)
    args = pa.parse_args()

    import yaml
    from rosbags.rosbag1 import Reader
    from ultralytics import RTDETR

    with open(args.extrinsic_yaml, encoding="utf-8") as f:
        ext = yaml.safe_load(f)
    cam = ext["camera"]
    T = np.array(ext["T_map_optical"], dtype=np.float64)
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])

    model = RTDETR(args.model)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_frame = 0
    n_lines = 0
    with Reader(Path(args.bag)) as reader, outp.open("w", encoding="utf-8") as fout:
        conns = [c for c in reader.connections if c.topic == args.image_topic]
        if not conns:
            raise SystemExit(f"bag 中无 topic {args.image_topic!r}")
        conn = conns[0]
        mt = str(conn.msgtype)

        for _c, _bag_ts, raw in reader.messages(connections=[conn]):
            if args.max_frames and n_frame >= args.max_frames:
                break
            try:
                msg = _deserialize_ros1(raw, mt)
            except Exception:
                n_frame += 1
                continue
            t_s = _stamp_to_sec(msg)
            try:
                bgr = _image_msg_to_bgr(msg)
            except Exception:
                n_frame += 1
                continue

            results = model.predict(
                source=bgr,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )
            n_frame += 1
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            r0 = results[0]
            names = getattr(r0, "names", None) or getattr(model.model, "names", {})
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            for i in range(xyxy.shape[0]):
                u, v = bottom_center_xyxy(xyxy[i])
                ray = ray_dir_optical_from_pixel(u, v, fx, fy, cx, cy)
                hit = intersect_ray_ground_in_map(
                    T, np.zeros(3), ray, z_map_ground=args.z_map_ground
                )
                if hit is None:
                    continue
                x_m, y_m = hit
                cid = int(clss[i])
                if isinstance(names, dict):
                    cname = str(names.get(cid, cid))
                elif isinstance(names, (list, tuple)) and 0 <= cid < len(names):
                    cname = str(names[cid])
                else:
                    cname = str(cid)
                rec = {
                    "t_s": t_s,
                    "class_name": cname,
                    "x": x_m,
                    "y": y_m,
                    "confidence": float(confs[i]),
                    "track_id": None,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_lines += 1

    print(f"处理 {n_frame} 帧，写出 {n_lines} 条检测 -> {outp}")


if __name__ == "__main__":
    main()
