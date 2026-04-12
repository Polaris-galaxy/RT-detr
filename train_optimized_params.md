# `train_optimized.py` 训练参数说明

本文档由 `train_optimized.py` 中的配置提炼而成；实际生效值以脚本内变量及 `model.train(...)` 调用为准。

---

## 1. 训练模式与路径

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `TRAIN_MODE` | `finetune` | `scratch`：按 yaml 建网 + 加载 `PRETRAINED_PT`；`finetune`：仅加载 checkpoint；`resume`：完整续训（含优化器与 epoch） |
| `CHECKPOINT_DIR` | 项目根目录下 `runs/train/wool_small_obj` | 含 `weights/best.pt` 或 `last.pt` 的目录 |
| `WEIGHT_NAME` | `best.pt` | finetune / resume 时使用的权重文件名 |
| `EXPERIMENT_NAME` | `wool_small_obj_v2` | 本次实验输出名（`project/name`） |
| `MODEL_YAML` | `ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml` | scratch 等使用的模型结构配置 |
| `PRETRAINED_PT` | `rtdetr-l.pt` | scratch 时额外加载的预训练权重 |

### 模式与 `lr0`、`resume` 的对应关系

| 模式 | 模型初始化 | `lr0` | `resume` |
|------|------------|-------|----------|
| `scratch` | `RTDETR(MODEL_YAML)` + `load(PRETRAINED_PT)` | `LR0_BASE`（2e-4） | `False` |
| `finetune` | `RTDETR(ckpt)` | `LR0_FINETUNE`（1e-4） | `False` |
| `resume` | `RTDETR(ckpt)` | `LR0_BASE`（2e-4） | checkpoint 路径字符串 |

---

## 2. 学习率与正则

| 参数 | 值 | 说明 |
|------|-----|------|
| `LR0_BASE` | `2e-4` | scratch / resume 时的初始学习率 |
| `LR0_FINETUNE` | `1e-4` | finetune 时的初始学习率 |
| `LRF` | `0.001` | 余弦调度终点约 `lr0 * lrf` |
| `WEIGHT_DECAY` | `1e-3` | AdamW 权重衰减 |
| `LABEL_SMOOTHING` | `0.05` | 标签平滑 |
| `EPOCHS` | `120` | 训练轮数 |
| `PATIENCE` | `18` | 早停耐心值 |
| `WARMUP_STEPS` | `1000` | 传入 `warmup_epochs`；本仓库 fork 中表示**迭代步数**（见 `engine/trainer.py`） |

---

## 3. Mosaic 与混合精度

| 参数 | 值 | 说明 |
|------|-----|------|
| `MOSAIC` | `0.15` | mosaic 概率 |
| `CLOSE_MOSAIC` | `10` | 最后若干 epoch 关闭 mosaic |
| `AMP` | `True` | 自动混合精度；若 loss 不稳定可改为 `False` |

---

## 4. `model.train(...)` 固定参数

以下在脚本中直接写出，未使用文件顶部常量（除已注明外）。

| 参数 | 值 |
|------|-----|
| `data` | `demo.yaml` |
| `cache` | `True` |
| `imgsz` | `800` |
| `epochs` | `EPOCHS`（120） |
| `batch` | `8` |
| `workers` | `0` |
| `device` | `'0'` |
| `project` | `'runs/train'` |
| `name` | `EXPERIMENT_NAME` |
| `optimizer` | `'AdamW'` |
| `lr0` | 见上文模式表 |
| `lrf` | `LRF` |
| `cos_lr` | `True` |
| `weight_decay` | `WEIGHT_DECAY` |
| `label_smoothing` | `LABEL_SMOOTHING` |
| `warmup_epochs` | `WARMUP_STEPS` |
| `scale` | `0.25` |
| `translate` | `0.1` |
| `degrees` | `5.0` |
| `shear` | `1.0` |
| `hsv_h` | `0.015` |
| `hsv_s` | `0.6` |
| `hsv_v` | `0.35` |
| `fliplr` | `0.5` |
| `flipud` | `0.5` |
| `mosaic` | `MOSAIC` |
| `close_mosaic` | `CLOSE_MOSAIC` |
| `mixup` | `0.0` |
| `copy_paste` | `0.0` |
| `amp` | `AMP` |
| `pretrained` | `True` |
| `patience` | `PATIENCE` |

---

## 5. 脚本内注释要点

- 若修改了 `wool_rtdetr.yaml` 的层结构，需用 **scratch** 才会按新结构训练；或自行「yaml 初始化 + `load(ckpt, strict=False)`」做迁移。
- finetune 时网络结构以 `.pt` 内保存为准，与当前 yaml 不一致时以 checkpoint 为准。
