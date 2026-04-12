# 羊毛场景 RT-DETR 专项改动说明

本文档记录为「羊毛检测微调」在仓库内做过的修改，便于复查与迁移。

---

## 1. 新增文件

### `ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml`

- 在 DRBC 风格骨干与颈部基础上，于 P5 分支堆叠 **4×`AIFI`**（混合编码侧加深），并在其后插入 **`SimAM`**（见下文第 5 节）。
- 检测头：`RTDETRDecoder`，参数为 **查询数 300、可变形注意力采样点 `ndp=4`、注意力头数 8、解码器层数 3**。
- 通过 **`detr_loss_gain`** 配置 DETR 损失权重（与 `DETRLoss` 键名一致）：
  - `class: 1`
  - `bbox: 0.5`（L1）
  - `giou: 2.0`
  - 其余键与默认对齐（`no_object`、`mask`、`dice`）。
- 文件内注释说明了与 Paddle 版 PP-HGNetV2 / MobileNetV4 命名不对齐、可选 `r50.yaml` 等说明。

---

## 2. 修改的核心代码

### `ultralytics/nn/tasks.py`

- **`RTDETRDetectionModel.init_criterion()`**：读取模型 YAML 中的可选字段 **`detr_loss_gain`**，与默认字典合并后传入 **`RTDETRDetectionLoss(..., loss_gain=...)`**，从而在羊毛 yaml 中覆盖 GIoU / L1 权重。

### `ultralytics/cfg/default.yaml`

- 新增推理用可选参数（默认关闭先验过滤）：
  - `wool_prior_filter`
  - `wool_min_area_ratio` / `wool_max_area_ratio`
  - `wool_min_aspect_ratio` / `wool_max_aspect_ratio`  
  面积比在 **归一化 xyxy（0～1）** 下计算，与 `RTDETRPredictor` 中在缩放到原图前过滤一致。

### `ultralytics/models/rtdetr/predict.py`

- **`RTDETRPredictor.postprocess`**：在置信度与类别过滤之后，增加 **`_filter_wool_priors`** 调用。
- 新增 **`_aspect_ratio_xyxy`**（静态方法）、**`_filter_wool_priors`**：当 `wool_prior_filter=True` 时按面积占比与长宽比（`max(w,h)/min(w,h)`）滤框。

---

## 3. 项目根脚本

### `train.py`

- 默认模型改为 **`ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml`**。
- 训练超参对齐推荐配置要点：**AdamW**、`cos_lr`、`lr0`/`lrf`/`weight_decay`、`warmup_epochs`（实为迭代步数，与 Ultralytics 约定一致）。
- **多尺度**：`scale=0.4`，对应 RandomPerspective 中约 **0.6～1.4** 倍缩放。
- **光度**：减弱 `hsv_h` / `hsv_s` / `hsv_v`。
- **`copy_paste=0.3`**、`mosaic=0`、`mixup=0`、`amp=False`，并注释 COCO 预训练与自定义结构兼容性、TensorRT 导出注意点。

### `detect.py`

- 增加 **`imgsz`** 与 **`wool_prior_*`** 传参示例，演示开启羊毛框先验过滤。

---

## 4. 使用与注意

1. **类别数**：训练前将 `wool_rtdetr.yaml` 中的 **`nc`** 与数据 `demo.yaml`（或你的 data yaml）中的 **`nc`** 保持一致。
2. **COCO 预训练**：官方 `rtdetr-l.pt` 与当前自定义 YAML **结构可能不一致**，不可盲目 `load`；需结构匹配的 checkpoint 或从 yaml 训练。
3. **Copy-Paste**：Ultralytics 实现依赖 **分割多边形**；纯检测标注时该增强常 **不生效**，参数仍保留以符合策略文档。
4. **TensorRT**：FP16 与动态 batch 在部分导出路径上 **可能互斥**，可按注释尝试 `engine` / `onnx` 或 ONNX→TRT 流水线。

---

## 5. 将 SimAM 加入训练（过程说明）

### 5.1 仓库里 SimAM 从哪来

- **`ultralytics/nn/AddModules/SimAM.py`**：本仓库用于 YAML 解析的 **`SimAM`** 类（`tasks.py` 中在 `extra_modules` 之后又 `from ultralytics.nn.AddModules import *`，**后者覆盖同名符号**，因此配置里写的 `SimAM` 对应该文件）。
- 构造函数为 `SimAM(channels=None, out_channels=None, e_lambda=1e-4)`，前向 **不显式依赖通道参数**，与通道数无关的注意力图乘到特征上。
- 另有 **`ultralytics/nn/extra_modules/attention.py`** 中也有同名 `SimAM`（仅 `e_lambda`），一般 **不要与 YAML 混用**；若你改 import 顺序，需自行确认解析到的是哪一个类。

### 5.2 能否直接加入训练

**可以。** 无需再改 Python 训练逻辑：只要在模型 YAML 的 `backbone` 或 `head` 里增加一行模块定义，`parse_model` 会注册该层（`tasks.py` 中 `SimAM` 与 `SpatialGroupEnhance` 同属 `c2 = ch[f]` 分支，**不改变通道数**）。

### 5.3 已在 `wool_rtdetr.yaml` 中的改法（推荐位置）

在 **P5 颈部**：`1×1 Conv → 4×AIFI` 之后、**Y5 的 `Conv 256`** 之前插入一层，使高层自注意力输出再经 SimAM 精炼：

```yaml
  - [-1, 1, AIFI, [1024, 8]]
  - [-1, 1, AIFI, [1024, 8]]
  - [-1, 1, AIFI, [1024, 8]]
  - [-1, 1, AIFI, [1024, 8]]
  - [-1, 1, SimAM, []]
  - [-1, 1, Conv, [256, 1, 1]]
```

- **`SimAM` 的参数写 `[]`** 即可，等价于 `SimAM()`，使用默认 `e_lambda`。
- **插入任意新层会改变后续层的全局下标**，必须与 `Concat`、`RTDETRDecoder` 的 **绝对层号** 对齐。当前羊毛配置在插入 SimAM 后已改为：
  - 与 Y4、Y5 拼接：`[[-1, 19], ...]`、`[[-1, 14], ...]`
  - 解码器三路输入：`[[23, 26, 29], 1, RTDETRDecoder, ...]`

若你自行移动 SimAM 位置或增删层，请从 YAML 顶向下 **重新数一遍每层在「backbone+head」合并后的全局索引**，再改上述数字。

### 5.4 其他可选接法（需自行改索引）

| 位置 | 作用简述 |
|------|----------|
| 骨干 **P5 最后一个 `Blocks` 之后** | 在 512 通道特征上做 SimAM，再进颈部（参考 `rtdetr-1-SimAM.yaml` 在 HG 骨干上的写法，该文件使用 `SimAM, [1024]` 仅作占位，AddModules 版会忽略多余通道参数）。 |
| 使用 **`HGBlock_SimAM`** | 见 `ultralytics/cfg/models/rt-detr/rtdetr-I-HGBlock_SimAM.yaml`，需 **HGStem/HGBlock** 骨干，与当前 DRBC 羊毛结构不同，整网需一起换。 |

### 5.5 训练与权重

- **从头训练**：直接运行现有 `train.py` 即可（已指向 `wool_rtdetr.yaml`）。
- **加载旧 checkpoint**：插入 SimAM 后 **结构已变**，旧 `best.pt` **无法整网加载**；需重新训练，或仅用 `strict=False` 等方式做部分加载（需自己写脚本，本仓库未内置）。

---

## 6. 文件一览（本次相关）

| 路径 | 变更类型 |
|------|----------|
| `ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml` | 新增（含 SimAM 与层号修正） |
| `ultralytics/nn/tasks.py` | 修改 |
| `ultralytics/cfg/default.yaml` | 修改 |
| `ultralytics/models/rtdetr/predict.py` | 修改 |
| `train.py` | 修改 |
| `detect.py` | 修改 |
| `WOOL_RT-DETR_改动说明.md` | 新增 / 更新（本说明，含 SimAM 一节） |

---

## 7. 后续更新（见主文档第 9 节）

**`RT-DETR训练与结构改动说明.md` → 第 9 节「2026-04 优化批次」** 汇总了：

- `train_optimized.py` 三模式（scratch / finetune / resume）、新默认超参与新实验名；
- `wool_rtdetr.yaml`：`nc=1`、AIFI 与 RTDETRDecoder 的 dropout 等；
- `track.py` 默认使用 `best.pt`。

网络结构有改动时，请优先阅读该节关于 **checkpoint 与 yaml 一致性** 的说明。

---

*文档生成目的：集中记录羊毛专项配置与代码改动，非产品对外文档。*
