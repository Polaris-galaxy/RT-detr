# RBCV 联用约定（Ultralytics RT-DETR 侧）

本目录与 **[RBCV-Region-Based-Coverage-Navigation-](https://github.com/)** 仓库中 `exploration` 语义栈对接。

## RBCV 需要的「检测一行」字段

与 `exploration/data/detections.example.jsonl` 一致：

| 字段 | 含义 |
|------|------|
| `t_s` | 秒，与 `/odom` 等 bag 时间对齐 |
| `class_name` | 类别名 |
| `x`, `y` | **map** 坐标系地面点（米） |
| `confidence` | 可选 |
| `track_id` | 可选 |

## 话题（实时 ROS1）

- **`/rbcv/semantic_detections`**：`std_msgs/String`，每条消息 **一个 UTF-8 JSON 对象**（不要一次发多行）。

## 脚本

| 文件 | 说明 |
|------|------|
| `projection.py` | 像素射线与地面 z=`z_map_ground` 求交（map 平面） |
| `ros1_rtdetr_rbcv_publisher.py` | **实时**：订阅 `Image` + `CameraInfo`，TF 或静态 YAML，发布上述 String |
| `offline_bag_rtdetr_jsonl.py` | **离线**：rosbags 读 bag + 推理，写 JSONL（无 ROS master） |
| `static_extrinsic.example.yaml` | 内参 + `T_map_optical` 模板，**必须改为真值** |

## 运行实时节点（示例）

```bash
# 已 source /opt/ros/noetic/setup.bash
python rbcv_ros/ros1_rtdetr_rbcv_publisher.py \
  __name:=rtdetr_rbcv \
  _model_path:=/path/to/best.pt \
  _image_topic:=/camera/color/image_raw \
  _camera_info_topic:=/camera/color/camera_info \
  _fixed_frame:=map
```

静态外参（无 TF 时）：

```bash
  _static_T_map_cam_yaml:=$(pwd)/rbcv_ros/static_extrinsic.example.yaml
```

## 离线 bag

```bash
python rbcv_ros/offline_bag_rtdetr_jsonl.py \
  --bag record.bag --image-topic /camera/color/image_raw \
  --model best.pt --extrinsic-yaml rbcv_ros/static_extrinsic.example.yaml \
  --out detections.jsonl
```

然后在 RBCV `src/` 下：

```bash
py exploration/scripts/run_semantic_stack.py \
  --survey exploration/data/survey_zones.example.json \
  --bag record.bag --topic /odom \
  --detections detections.jsonl --out exploration/data/out_semantic_plan.json
```

## 录制 String → JSONL（可选）

使用 RBCV 仓库 `exploration/ros1_bridge/detection_recorder_node.py`，与 `rostopic pub` 调试同样兼容。
