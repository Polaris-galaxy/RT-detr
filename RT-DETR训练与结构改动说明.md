# RT-DETR 羊毛场景：训练脚本与结构改动说明

本文档汇总本仓库内针对**小目标羊毛检测**所做的配置、`train_optimized.py` 参数、`wool_rtdetr.yaml` 结构变更，以及 **SPDConv** 的接入与影响。路径与实验名请按本机情况修改。

---

## 1. 涉及文件一览

| 文件 | 作用 |
|------|------|
| `train_optimized.py` | 统一入口：预训练加载 / 断点续训 / 训练超参 |
| `demo.yaml` | 数据集路径、`nc`、类别名 |
| `ultralytics/cfg/models/rt-detr/wool_rtdetr.yaml` | 模型结构、DETR 损失权重、SPDConv 插入位置 |
| `ultralytics/nn/AddModules/SPDConv.py` | SPDConv 实现（含奇数 H/W 填充） |
| `ultralytics/nn/AddModules/__init__.py` | 导出 `SPDConv` |
| `ultralytics/nn/tasks.py` | `parse_model` 中注册 `SPDConv`（与 `Conv` 同一分支） |

---

## 2. `demo.yaml` 注意事项

- **禁止使用 Python 写法 `r"D:\..."`**：在 YAML 中会被当成普通字符串，路径前会多出字符 `r` 和 `"`，导致 `WinError 123`。
- 推荐：**正斜杠** `D:/path/to/...`，或合法的双引号 YAML 字符串。
- `train` / `val`（及可选 `test`）需指向**真实存在**的 `images` 目录；无 `test` 时可删除或注释 `test` 行。

---

## 3. `train_optimized.py` 说明

### 3.1 断点续训（文件顶部）

- **`RESUME`**：`True` 从检查点接着训，`False` 从 yaml 构建网络并 `load('rtdetr-l.pt')`。
- **`LAST_CKPT`**：指向**同结构**实验的 `weights/last.pt`（或你要续训的 pt）。
- **`resume=RESUME`**：传入 `model.train`，与 Ultralytics 内部恢复逻辑一致。

### 3.2 从零训练时的预训练

- `RTDETR('wool_rtdetr.yaml')` 后执行 **`model.load('rtdetr-l.pt')`**：自动下载/加载官方权重；与自定义结构不一致的层会 **strict=False 跳过**，骨干中与 DRBC 一致的部分可对齐（日志里会显示 `Transferred x/y items`）。

### 3.3 `model.train(...)` 各参数含义

| 参数 | 含义 |
|------|------|
| `data` | 数据配置 yaml，如 `demo.yaml` |
| `resume` | 是否续训（与 `RESUME` 同步） |
| `cache` | 缓存数据以加速重复 epoch（占内存） |
| `imgsz` | 输入边长；800 有利于小目标，显存占用更高 |
| `epochs` | 总轮数 |
| `batch` | 每 GPU batch；OOM 时可减小 |
| `workers` | DataLoader 进程数；Windows 常用 `0` |
| `device` | 如 `'0'` 使用第一块 GPU |
| `project` / `name` | 结果目录 `runs/train/<name>` |
| `optimizer` | 如 `AdamW` |
| `lr0` | 初始学习率（相对默认 `1e-4` 略大时需配合预训练与数据量权衡） |
| `lrf` | 余弦结束时 `lr ≈ lr0 * lrf` |
| `cos_lr` | 余弦学习率调度 |
| `weight_decay` | 权重衰减（如 `5e-4`） |
| `warmup_epochs` | **本仓库 fork**：表示 **warmup 迭代步数**（见 `ultralytics/engine/trainer.py` 中 `nw = self.args.warmup_epochs`），不是 epoch 数 |
| `scale` | 随机缩放强度；`0.25` 表示约 0.75～1.25，比默认 `0.5` 温和，减轻小目标被缩没 |
| `translate` | 平移增强；`0.1` 比 `0.2` 保守 |
| `degrees` / `shear` | 旋转、剪切强度 |
| `hsv_h` / `hsv_s` / `hsv_v` | HSV 颜色增强 |
| `fliplr` / `flipud` | 左右、上下翻转概率；羊毛场景可保留 `flipud` |
| `mosaic` / `mixup` / `copy_paste` | RT-DETR 侧常为 0 |
| `amp` | 混合精度；RT-DETR 易出现数值问题，此处关闭 |
| `pretrained` | 语义上与预训练一致；显式 `load` 时仍以实际加载为准 |
| `patience` | 早停耐心轮数 |

---

## 4. `wool_rtdetr.yaml` 结构改动

### 4.1 `detr_loss_gain`

- 将 **`bbox`** 从过小值恢复为与 RT-DETR 默认量级一致（如 **5**），避免框回归梯度过弱、小目标更难定位。
- **`giou`** 可适当加大（如 **2.5**）以强调几何一致性；可按验证集微调。

### 4.2 去掉 SimAM

- 已从 head 中移除 **`SimAM`** 层。
- 全局层号少一层后，**Concat 的绝对层号**与 **`RTDETRDecoder` 输入**已重算：
  - Concat：`19→18`，`14→13`（相对带 SimAM 时的编号）。
  - Decoder：`[[22, 25, 28], ...]`（P3 / P4 / P5 三路特征）。

若再改 head（增删层），必须重新核对 **Concat 下标** 与 **Decoder 三个层号**。

### 4.3 插入 SPDConv

- 在 PAN 两处，将原 **`Conv [256, 3, 2]`**（stride=2 下采样）改为 **`SPDConv [256, 3, 1]`**。
- **含义**：空间减半由 **space-to-depth**（四向子采样拼通道）完成，卷积使用 **stride=1**，输出特征图空间尺寸与原先 stride-2 Conv 对齐，后续 `Concat` / `DRBC3` / Decoder **层号不变**。
- 这两层与官方 `rtdetr-l.pt` 中对应 Conv **参数形状不一致**，预训练通常**无法直接对齐**，需依赖训练重新学习。

---

## 5. SPDConv 代码接入方式

1. **`ultralytics/nn/AddModules/SPDConv.py`**：实现 `SPDConv`；对 **H 或 W 为奇数** 时在右下 **pad 1**，保证四个子图空间尺寸一致再 `cat`。
2. **`ultralytics/nn/AddModules/__init__.py`**：`from .SPDConv import SPDConv`（避免 `import *` 把 `autopad` 等带入全局）。
3. **`ultralytics/nn/tasks.py`**：文件头部已有 `from .AddModules import *`；在 **`parse_model`** 里与 **`Conv` 同类的超长元组**中加入 **`SPDConv`**，以便 YAML 写 `- [-1, 1, SPDConv, [c2, k, s, ...]]` 时走 `c1,c2` 自动注入逻辑。

**YAML 中不写 `SPDConv` 则网络中不会出现该模块**；是否使用完全由 `wool_rtdetr.yaml`（或其它模型 yaml）决定。

---

## 6. SPDConv 对训练效果的预期

| 方面 | 说明 |
|------|------|
| 机制 | 相对 stride-2 卷积下采样，SPD 更倾向保留细粒度空间信息，理论上利于小目标/细边界 |
| 预训练 | 颈部两处与 `rtdetr-l.pt` 不匹配，这两层多从零或随机初始化学习 |
| 验证 | 是否涨点需 **对照实验**（仅去掉 SPDConv、其余超参与数据相同） |
| 开销 | 通道先 ×4 再卷积，局部算量/参数略增 |

---

## 7. 续训与检查点

- 续训使用 **`last.pt`**（含优化器状态）；**`best.pt`** 多用于推理或手动微调。
- `RESUME=True` 且 `LAST_CKPT` 指向正确路径；或 `RTDETR('.../last.pt').train(resume=True)`。
- **`LAST_CKPT` 结构须与训练时 yaml 一致**；改结构后旧权重只能部分加载或需重新训练。

---

## 8. 版本与提示

- 控制台若提示升级上游 `ultralytics`，本仓库为 **魔改 fork**，升级前请确认兼容性。
- `scales` 未在命令行指定时可能出现 `WARNING no model scale passed. Assuming scale='l'.`，一般可忽略（yaml 内已定义 `l`）。

---

## 9. 2026-04 优化批次（超参 / Transformer / 网络）

本节记录一次集中改动：**训练脚本默认策略**、**`wool_rtdetr.yaml` 中编码器与解码器正则**，以及 **`track.py` 默认权重**。便于复现实验与对照旧 `runs/train/wool_small_obj`。

### 9.1 `train_optimized.py`

| 项目 | 说明 |
|------|------|
| `TRAIN_MODE` | `scratch`：yaml + `rtdetr-l.pt`；`finetune`：加载 `weights/best.pt` 或 `last.pt` 且 `resume=False`，新优化器与新实验目录；`resume`：完整续训（含优化器） |
| `EXPERIMENT_NAME` | 默认 `wool_small_obj_v2`，避免与旧实验**共用同一份 `results.csv` 追加混淆** |
| `CHECKPOINT_DIR` / `WEIGHT_NAME` | 指定从哪次实验、哪种权重启动 |
| `LR0_FINETUNE` | `1e-4`，在已有好权重附近**降低步长**，减轻后期指标崩坏 |
| `LRF` | `0.001`（原为 `0.01`），余弦末端更细 |
| `WEIGHT_DECAY` | `1e-3`（原为 `5e-4`） |
| `LABEL_SMOOTHING` | `0.05` |
| `EPOCHS` / `PATIENCE` | `120` / `18`，缩短长训、收紧早停 |
| `MOSAIC` / `CLOSE_MOSAIC` | `0.15` / `10`，弱 mosaic，末段关闭 |
| `AMP` | 默认 `True`；若 loss 不稳定可改 `False` |

### 9.2 `wool_rtdetr.yaml`（网络 / Transformer）

| 项目 | 说明 |
|------|------|
| `nc` | 改为 **1**，与 `demo.yaml` 单类羊毛一致（避免与数据配置混淆） |
| `AIFI`（4 层） | 参数由 `[1024, 8]` 改为 **`[1024, 8, 0.05]`**，为编码侧 **dropout=0.05** |
| `RTDETRDecoder` | 在原有 `hd,nq,ndp,nh,ndl` 后增加 **`d_ffn=1024`、`eval_idx=-1`、`dropout=0.05`**，解码器正则略增 |

**重要**：从已有 `.pt` **finetune/resume** 时，**网络结构以 checkpoint 为准**，不会自动套用当前 yaml 的新 dropout；要使新结构生效，请使用 **`TRAIN_MODE='scratch'`**，或自行实现 yaml 构建 + `load(ckpt, strict=False)` 迁移。

### 9.3 `track.py`

- 默认权重由 `last.pt` 改为 **`best.pt`**，并加注释说明二者差异。

### 9.4 与旧权重的兼容性

- 本次 **yaml 中 AIFI / RTDETRDecoder 参数表变更** 后，**旧 `best.pt` / `last.pt` 无法保证整网 `state_dict` 键完全一致**；若加载报错，请用 `scratch` 重训或部分加载迁移。
- 仅 **`train_optimized.py` 超参** 变更、不改 yaml 时，旧 checkpoint 仍可 **`finetune` / `resume`**（结构未变的前提下）。

---

*文档随仓库当前状态整理；若你修改了 yaml 层号或脚本默认值，请同步更新本节对应描述。*
