# RT-DETR（Ultralytics 分叉）

本仓库在 [Ultralytics](https://github.com/ultralytics/ultralytics) 基础上集成 **RT-DETR** 实时检测器，并针对 **羊毛等小目标场景** 增加了专用模型配置与推理后处理。根目录提供训练、检测、数据准备与可视化脚本；核心库代码位于 `ultralytics/`。

---

## 环境与安装

1. **Python**：建议 3.8+（与 `setup.py` 一致）。
2. **PyTorch / torchvision**：按本机 CUDA 版本从 [PyTorch 官网](https://pytorch.org/) 安装（`requirements.txt` 中 `torch` 被注释，需自行安装）。
3. **安装本仓库为可编辑包**（推荐在项目根目录执行）：

```bash
pip install -e .
```

4. **依赖**：`pip install -r requirements.txt`  
   其中包含 Ultralytics 基础依赖，以及 `nn/extra_modules` 中部分模块所需的 `efficientnet-pytorch`、`einops`、`timm`、`dill`、`PyWavelets` 等。

5. **按需额外安装**（根目录脚本会用到）：
   - `split_data.py`：`scikit-learn`（`train_test_split`）
   - `heatmap.py`：`pytorch-grad-cam`、`matplotlib`
   - `csv_to_result_png.py`：`pandas`（若未通过其他方式安装）
   - `test_env.py`：可选 CUDA 扩展（`mmcv`、`mamba_ssm`、`DCNv3`、`DCNv4` 等），用于验证环境是否编译成功，**非日常训练必需**

---

## 仓库结构（摘要）

| 路径 | 说明 |
|------|------|
| `ultralytics/` | 核心库：数据、`RTDETR` 模型、训练/验证/推理、大量骨干与 `extra_modules` 扩展 |
| `ultralytics/cfg/models/rt-detr/` | RT-DETR 结构 YAML，含本项目的 `wool_rtdetr.yaml` |
| `ultralytics/models/rtdetr/` | RT-DETR 的 `Model` / `Predictor` / `Trainer` / `Validator` |
| `ultralytics/nn/AddModules/SPDConv.py` | **SPDConv**：Space-to-Depth + 卷积，用于 PAN 下采样（见下文） |
| `demo.yaml` | 数据集配置示例（`train`/`val` 路径、`nc`、`names`） |
| 根目录 `*.py` | 见下文「根目录脚本说明」 |

`ultralytics/nn/backbone/`、`ultralytics/nn/extra_modules/` 等目录体量较大，包含多种可选骨干与算子（含部分需本地编译的 CUDA 扩展）；仅在对应 YAML 引用到相关模块时才需要编译或安装对应依赖。

---

## 配套数据增强：Polaris Data Albumentations

训练与推理在本仓库完成；离线扩增与标注处理使用独立仓库 **[Polaris_Data_Albumentations](https://github.com/Polaris-galaxy/Polaris_Data_Albumentations.git)**。数据流为：在增强仓库生成或整理 **YOLO / COCO** 数据 → 将本仓库 `demo.yaml` 中 `train` / `val` 指向对应目录 → 在本仓库执行训练。

**Polaris_Data_Albumentations** 使用 **Albumentations** 等依赖，功能概览如下。

| 类别 | 内容 |
|------|------|
| 通用图片增强 | 两阶段流水线：第一阶段以颜色、几何拉伸、噪声为主；第二阶段以环境、光照为主；可按强度、目标倍数等参数批量生成图像。 |
| YOLO 数据集增强 | 对 **YOLO 格式** 做单图增强（几何 / 颜色 / 噪声）、**Mosaic**、**MixUp**，并同步改写标注；含分步入口脚本与 matplotlib 抽查脚本，用于核对框与类别。 |
| 标注与格式 | 训练前图与标签检查、按目标张数扩增、目录诊断、标注可视化；**YOLO → COCO** 转换；**COCO JSON 内图片路径** 在 Linux 服务器上的修正脚本。 |

脚本文件名含 `_中文说明` 后缀时，终端中须使用 **完整文件名**。参数与入口以对方仓库 [README](https://github.com/Polaris-galaxy/Polaris_Data_Albumentations/blob/main/README.md) 为准。

**仓库地址**：<https://github.com/Polaris-galaxy/Polaris_Data_Albumentations.git>

---

## SPDConv 模块（Space-to-Depth 下采样）

羊毛专用结构中，**PAN 下采样** 使用 **SPDConv** 替代常见的 **stride=2 卷积**：下采样时先保留更多空间信息再融合，相对 stride=2 卷积直接抽像素，更利于 **细长 / 小目标**（如羊毛条）特征传递。

### 代码位置与注册

- **实现**：`ultralytics/nn/AddModules/SPDConv.py`，类名 `SPDConv`。
- **导出**：`ultralytics/nn/AddModules/__init__.py` 中导出；`ultralytics/nn/tasks.py` 通过 `from .AddModules import *` 将 `SPDConv` 注册进解析表，YAML 里可直接写 `SPDConv`。

### 前向计算步骤

1. **`_space_to_depth`**：将特征图按 2×2 邻域拆成四张子图（`::2` / `1::2` 步长组合），在 **通道维拼接**，等价于 **空间高宽各减半、通道×4**（Space-to-Depth）。高或宽为奇数时先对右下 **pad 1**，保证四张子图尺寸一致。
2. **卷积**：`__init__` 将输入通道按 `c1 * 4` 接入 `Conv2d`，后接 `BatchNorm2d` 与默认 **SiLU**；**`stride` 固定为 1**，空间尺度减半由 Space-to-Depth 完成，不由 stride=2 卷积完成。

与 stride=2 卷积下采样相比：本模块先把四邻域信息压入通道再卷积融合，属于 SPD / Space-to-Depth 类常见做法，利于保留细粒度结构。

### `wool_rtdetr.yaml` 中的配置

PAN 中有 **两处** `SPDConv`，注释写明用于下采样并与上一路特征拼接：

```yaml
# PAN 下采样：用 SPDConv 代替 stride=2 的 Conv（空间减半由 space-to-depth 完成，这里 stride 用 1）
- [-1, 1, SPDConv, [256, 3, 1]]
```

YAML 中参数 **`[256, 3, 1]`** 对应构造函数 `(c1, c2, k, s)`：**输出通道 256、卷积核 3、stride=1**（空间步进已由 Space-to-Depth 完成）。`c1` 由上一层输出通道自动推断。

### 部署与融合

类中提供 `forward_fuse`：仅 `Conv+激活`（无 BN），便于与 Ultralytics 常规的 `fuse()` 流程一起做推理融合。

---

## 相对上游 Ultralytics 的改动

### 1. 羊毛场景结构 `wool_rtdetr.yaml`

- 路径：`ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml`
- 内容：DRBC 风格骨干与混合编码器上的 RT-DETR 头（多层 `AIFI`、`DRBC3`、**PAN 侧 SPDConv**、`RTDETRDecoder` 等），`detr_loss_gain`（如 bbox/GIoU 权重）已按 **单类羊毛（`nc: 1`）** 与 YAML 内注释调整。SPDConv 行为见上一节。
- 权重：与官方 `rtdetr-l` 等结构不一致，整网 **strict** 加载常失败；需按本 YAML 训练或做部分加载。

### 2. 推理阶段「羊毛先验」框过滤

- 实现：`ultralytics/models/rtdetr/predict.py` 中 `RTDETRPredictor.postprocess` 在置信度过滤后调用 `_filter_wool_priors`。
- 行为：在 **归一化 xyxy（0–1）** 下按框 **面积占图比例** 与 **长宽比**（`max(w,h)/min(w,h)`）滤除检测结果，用于压制与先验尺度、长宽比不符的框。
- 配置项：`ultralytics/cfg/default.yaml` 中：
  - `wool_prior_filter`：总开关（默认 `False`）
  - `wool_min_area_ratio` / `wool_max_area_ratio`
  - `wool_min_aspect_ratio` / `wool_max_aspect_ratio`  
  也可在 `model.predict(...)` 中传同名参数覆盖（见 `detect.py` 示例）。

### 3. 数据集与训练入口

- `demo.yaml`：`train` / `val` 图片路径、`nc`、`names`；使用前改为本地路径。
- `train_optimized.py`：从 `wool_rtdetr.yaml` 与 `rtdetr-l.pt` 启动训练，或在 `RESUME = True` 时从 `runs/.../last.pt` 续训；数据配置默认指向 `demo.yaml`，可在脚本内常量中修改。

### 4. 训练曲线 `csv_to_result_png.py`

- Ultralytics 默认 `plot_results` 按 YOLO 列名解析 `results.csv`，与 RT-DETR 导出的列名不一致。
- 本脚本直接读取 RT-DETR 的 `results.csv`，绘制 GIoU/cls/L1、precision、recall、mAP 等曲线，输出 `result.png`。

---

## 根目录脚本说明

| 脚本 | 作用 |
|------|------|
| `train_optimized.py` | 使用 `MODEL_YAML`、`PRETRAINED`、`demo.yaml` 训练；支持 `RESUME` 与 `LAST_PT` 指向含 `last.pt` 的目录或文件 |
| `detect.py` | 加载权重对 `testset` 推理，`save=True`，并开启 `wool_prior_filter` 等参数；首张结果另存为项目根目录 `result.png` |
| `main_profile.py` | 对指定 YAML 构建模型，`info`、`profile`、`fuse`，用于参数量与速度分析 |
| `get_FPS.py` | 加载 `.pt` 或 YAML，warmup 后统计批量推理延迟与 FPS（支持 `--half`） |
| `split_data.py` | 将平铺的 YOLO 格式 `images`/`labels` 按比例划分为 `train/val/test`（需 `scikit-learn`）；路径在 `__main__` 中硬编码，使用前请修改 |
| `xml_to_txt.py` | VOC XML 转 YOLO txt；需修改 `classes` 与 `xml_folder`/`output_folder`；含个别类名纠错（如 `buning`→`burning`） |
| `csv_to_result_png.py` | `results.csv` → `result.png`（RT-DETR 列） |
| `heatmap.py` | 基于 Grad-CAM 类方法生成热力图（需 `pytorch-grad-cam` 等） |
| `track.py` | 使用 `RTDETR` 对视频做跟踪（`model.track`） |
| `test_env.py` | 探测可选第三方 CUDA 扩展是否可导入，用于环境自检 |

---

## 使用方法速查

### 训练

先修改 `demo.yaml` 中的数据路径与类别，再执行：

```bash
python train_optimized.py
```

或在 Python 中：

```python
from ultralytics import RTDETR
model = RTDETR("ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml")
model.load("rtdetr-l.pt")  # 或官方预训练权重路径
model.train(data="demo.yaml", epochs=100, imgsz=640, batch=4)
```

### 推理 / 验证

```bash
python detect.py
```

或使用 CLI（安装包后）：

```bash
yolo detect model=路径/best.pt source=图片或文件夹 conf=0.25
```

`detect.py` 中使用的先验过滤参数示例如下（数值需按数据分布调整）：

```python
model.predict(
    source="testset",
    wool_prior_filter=True,
    wool_min_area_ratio=0.0001,
    wool_max_area_ratio=0.92,
    wool_min_aspect_ratio=1.0,
    wool_max_aspect_ratio=15.0,
)
```

### 训练曲线图

```bash
python csv_to_result_png.py path/to/results.csv -o path/to/result.png
```

### 官方 RT-DETR 论文

[RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/pdf/2304.08069.pdf)

---

## 许可

`ultralytics` 包默认遵循 **AGPL-3.0**（见 `setup.py`）。使用与分发时请遵守原项目及本仓库各子模块（如 CUTLASS、Mamba 等）的许可证要求。
