# QCResUNet

基于独立 PyTorch 训练框架的 **QCResUNet 分割质量控制复现**。给定 MRI 和待评估的分割结果，模型同时预测病例级质量分数与组织级分割错误图。

对应论文：*QCResUNet: Joint subject-level and voxel-level segmentation quality prediction*, Medical Image Analysis 107 (2026), 103718。
[论文 DOI](https://doi.org/10.1016/j.media.2025.103718) · [详细复现说明](docs/qcresunet.md) · [插件扩展指南](docs/extending.md)

**当前状态：**已完成主方法的代码实现，跑通合成数据训练、最佳权重重载和推理；真实数据上的论文性能指标尚未验证。最近一次开发检查通过 59 项测试，以及 Ruff、格式检查、mypy 和包构建。

## 模型做什么

QCResUNet 评估已有分割的质量。输入中的分割可以来自 nnUNet、nnFormer 或其他方法。

| 输入 / 输出 | 内容 |
| --- | --- |
| MRI 输入 | BraTS 的 T1、T1c、T2、FLAIR；ACDC 使用单模态 |
| 查询分割输入 | 与 MRI 同一坐标网格的待评估整数标签图 |
| 病例级输出 | Dice similarity coefficient（DSC）和 normalized surface Dice（NSD）的预测值 |
| 体素级输出 | 每个组织区域的 segmentation error map（SEM） |

BraTS 的 SEM 输出区域为全肿瘤 **WT**、肿瘤核心 **TC** 和增强肿瘤 **ET**；ACDC 对应左心室 **LV**、心肌 **Myo** 和右心室 **RV**。训练使用专家标注生成质量分数和错误图监督；单例推理只需 MRI 与查询分割。

模型采用 3D ResNet-34 编码器，从 Block 3 接出解码器，通过 ECA 聚合组织级错误图，联合优化病例级 MAE 和体素级 Dice + BCE。实现还包括 SegGen 查询生成、DSC 质量区间平衡采样、受试者级三折划分和三模型集成。

## 快速开始

### 1. 安装环境

需要 Python **3.9–3.13** 和 `uv`。在仓库根目录执行：

```bash
uv sync --extra dev --extra brats
```

环境安装在本项目的 `.venv/`。`brats` 扩展提供 NIfTI 读取、几何变换和 NSD 所需的可选依赖；只运行通用框架与 QC 合成数据示例时，也可使用 `uv sync --extra dev`。

### 2. 跑通 CPU 示例

```bash
uv run training-framework fit \
  --configs configs/experiment.qcresunet.smoke.toml \
  --model configs/model.qcresunet.smoke.toml
```

示例使用合成体积、缩小网络和 2 个 epoch，依次完成训练、验证、保存最佳权重及测试。它用于验证接口和运行流程，输出指标不代表正式模型性能。

### 3. 查看结果

运行结束后会打印实验目录，默认位于 `runs/`。查看训练日志：

```bash
uv run training-tensorboard --logdir runs
```

## 使用真实数据

### 数据要求

QC 训练需要三类数据：MRI、专家分割标注，以及不同方法或训练阶段生成的查询分割。BraTS MRI 应预先完成配准、1 mm 等体素重采样和去颅骨。

每个受试者的输入 NPZ 包含以下字段，查询分割另存为 `.npy`：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `image` | `[M, D, H, W]` | MRI，BraTS 的 `M=4`，模态顺序为 T1/T1c/T2/FLAIR |
| `reference` | `[D, H, W]` | 专家整数标签图 |
| `spacing` | `[3]` | 按 D/H/W 轴排列的毫米间距 |
| 查询 NPY | `[D, H, W]` | 与 MRI、专家标注使用相同网格 |

通过 `source.json` 指定受试者 ID、数据路径、查询来源和 train/val/test 划分。完整示例见[数据准备说明](docs/qcresunet.md#准备自己的数据)。同一受试者的所有查询和时间点应保留在同一个划分中。

### 准备 BraTS QC 数据

```bash
uv run python -m training_framework.plugins.qcresunet.prepare \
  --source data/qc-source/source.json \
  --output data/qc/fold0 \
  --tolerance-mm 1.0
```

准备流程进行非零体素 z-score、公共前景裁剪和补零，导入已有查询，并默认为训练/验证受试者各生成三份 SegGen 查询。输出包括共享 MRI/标注文件、查询标签、`manifest.json` 和 DSC 分布报告。

**NSD 容差需要明确记录。**所提供的论文 PDF 未明确给出这一参数；示例的 `1.0 mm` 是当前实现默认值。准备命令、实验配置和评估必须使用相同容差。

### 训练正式模型

先确认 [BraTS 实验配置](configs/experiment.qcresunet.brats.toml) 中的 manifest 路径和设备设置，再运行：

```bash
uv run training-framework fit \
  --configs configs/experiment.qcresunet.brats.toml \
  --model configs/model.qcresunet.toml
```

正式配置使用 Adam、batch size 4、100 epochs、初始学习率 `2.1e-4`、weight decay `1e-4`，学习率每 epoch 乘 `0.9`，最低为 `1e-6`。默认使用 CUDA 与 AMP。

质量平衡采样将 DSC 分成 10 个区间，并把各区间下采样到相同数量；训练每 epoch 重采样，验证和测试保持固定。每个区间都需有样本，小规模接口试验可关闭 `balance_train` 和 `balance_eval`。

| 场景 | 实验配置 | 模型配置 |
| --- | --- | --- |
| CPU 合成数据 | [QC smoke](configs/experiment.qcresunet.smoke.toml) | [缩小网络](configs/model.qcresunet.smoke.toml) |
| BraTS QC | [BraTS](configs/experiment.qcresunet.brats.toml) | [QCResUNet](configs/model.qcresunet.toml) |
| ACDC QC | [ACDC](configs/experiment.qcresunet.acdc.toml) | [ACDC 网络](configs/model.qcresunet.acdc.toml) |

三折训练、外部测试集评估及 ACDC 数据约定见[详细复现说明](docs/qcresunet.md)。

## 评估与推理

### 评估已有权重

```bash
uv run training-framework test \
  --configs configs/experiment.qcresunet.brats.toml \
  --model configs/model.qcresunet.toml \
  --checkpoint runs/YOUR_RUN/checkpoints/best.pt
```

通用命令的行为如下：

| 命令 | 行为 |
| --- | --- |
| `train` | 训练并进行每轮验证 |
| `fit` | 训练后加载最佳权重，在测试集评估 |
| `validate` | 在验证集评估指定权重 |
| `test` | 在测试集评估指定权重 |

`validate` / `test` 省略 `--checkpoint` 时使用初始化权重。通用入口会组装 train/val/test 三个划分；外部纯测试集使用 QC 插件的 `predict --evaluate-manifest` 入口。

QC 指标包括 DSC/NSD 的 MAE、标准差与 Pearson 相关系数，以及病例平均和各组织区域的 SEM Dice。

### 单例推理

输入 NPZ 包含预处理后的 `image` 和 `query`，可附 `spacing`，无需专家标注：

```bash
uv run python -m training_framework.plugins.qcresunet.predict \
  --configs configs/experiment.qcresunet.brats.toml \
  --model configs/model.qcresunet.toml \
  --checkpoint runs/YOUR_RUN/checkpoints/best.pt \
  --input data/qc-inference/case.npz \
  --output runs/case-qc.npz \
  --device cuda
```

输出 NPZ 包含 `quality=[DSC, NSD]`、组织级 `sem_probability`、二值 `sem` 和 `spacing`。结果位于预处理网格中，恢复原始 NIfTI 坐标需要使用对应的逆变换。

`--checkpoint` 支持传入三折模型权重：病例分数取算术平均，SEM 使用二值图多数投票。完整集成评估命令见[三折训练和集成](docs/qcresunet.md#三折训练和集成)。

## 实验产物

```text
runs/EXPERIMENT_MODEL_TIMESTAMP/
├── experiment.snapshot.toml       # 合并后的实验配置
├── splits/                        # train/val/test 样本 ID
├── checkpoints/
│   ├── best.pt                    # 按验证指标选择的最佳权重
│   └── last.pt                    # 最后一轮权重
├── logs/                          # TensorBoard
├── metrics.jsonl                  # 每轮训练与验证指标
└── plots/
```

Checkpoint 保存模型权重、epoch 和验证指标，可用于重载评估。目前不保存完整 optimizer、scheduler、AMP scaler 和随机数状态，因此不支持完整断点续训。生成数据、运行目录和模型权重均由 Git 忽略。

## 框架与项目结构

所有运行模块位于 `src/training_framework/`。框架提供注册表、TOML 配置、训练与评估、早停、checkpoint、CPU/CUDA/MPS 设备选择、AMP、DDP、TensorBoard 和 JSONL 日志；QCResUNet 作为显式加载的应用插件接入。

```text
src/training_framework/
├── plugins/qcresunet/             # 模型、QC 任务、数据准备、指标与推理
├── contracts/                     # 模型、数据、任务和指标接口
├── registry/                      # 组件注册与配置工厂
├── core/                          # Trainer、Evaluator、设备与分布式运行
├── data_adapters/                 # 合成数据与可选 BraTS 分割数据适配器
├── tasks/                         # 通用分类与分割任务
├── losses/                        # 通用训练损失
├── metrics/                       # 通用分类、分割与 BraTS 指标
├── models/                        # SimpleNet、TinySegNet 接口示例
├── logging/                       # TensorBoard 与日志协议
└── cli/                           # 命令行入口
configs/                           # 实验与模型 TOML
examples/custom_project/           # 可安装插件示例
tests/                            # 合成数据与临时目录测试
docs/                             # 复现、扩展与验证说明
```

框架内置多标签分类与 2D/3D 多类分割任务。两套通用 CPU 示例也可独立运行：

```bash
uv run training-framework fit --configs configs/experiment.classification.toml --model configs/model.simple_net.toml
uv run training-framework fit --configs configs/experiment.segmentation.toml --model configs/model.tiny_seg_net.toml
```

通用 BraTS 分割模板为 `configs/experiment.brats.example.toml`。它与 QC 插件使用不同的数据约定，模板中的 TinySegNet 用于验证分割接口。`brats_dice` 计算 WT/TC/ET Dice；`brats_hd` 计算最大对称 Hausdorff 距离，不是 HD95。

### 扩展自己的组件

模型、Task、DataAdapter 和 Metric 通过 `register(registries)` 注册，并在 TOML 中显式加载：

```toml
[plugins]
modules = ["training_framework.plugins.qcresunet.components"]
```

批次采用 `Batch(signal, label, id, meta)`；通用分类和分割模型返回 logits，由 Task 计算损失并后处理。QC 插件在 Tensor 合同内打包两个预测头，具体格式见[复现说明](docs/qcresunet.md)。新算法应放在应用插件中，保留默认模型作为小型接口示例。组件实现方法见[扩展指南](docs/extending.md)。

### 多卡训练

配置 `[trainer].device="cuda"` 后，可以通过 `torchrun` 启动：

```bash
uv run torchrun --standalone --nproc_per_node=2 -m training_framework train \
  --configs configs/experiment.qcresunet.brats.toml \
  --model configs/model.qcresunet.toml
```

训练集使用 DistributedSampler 分片，验证/测试集在各进程完整评估，仅主进程写日志和权重。训练指标按各进程标量平均，不等同于合并所有预测后计算的全局 Dice/AUC。框架的双进程 CPU DDP 已由测试验证，QCResUNet 的完整 CUDA AMP/DDP 训练仍需在目标硬件验证。

## 开发与验证

```bash
uv sync --extra dev --extra brats
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy src
uv build
```

测试只使用合成数据、临时路径和小型模型。QC 测试覆盖双头梯度、Block 3 解码路径、输入隔离、区域 XOR 标签、NSD 物理间距、损失公式、指标、受试者划分、平衡采样、数据准备和 checkpoint 重载。

## 复现范围

当前实现覆盖 QCResUNet 主方法、论文最佳训练设置及相关数据和评估流程。仓库不包含私有数据、作者的实际划分 ID 或预训练权重，也不负责训练生成查询的 nnUNet/nnFormer/DeepMedic。

论文未完整指定的 NSD 容差、损失归一化、ECA 细节及 ACDC 体积尺寸已在[复现边界与待核实约定](docs/qcresunet.md#复现边界与待核实约定)中记录。RayTune 搜索、其他 QC 基线和解释性分析尚未重跑；合成数据验证不能替代真实数据上的论文性能复现。

| 文档 | 内容 |
| --- | --- |
| [QCResUNet 复现](docs/qcresunet.md) | 数据 schema、三折训练、集成评估与实现约定 |
| [扩展指南](docs/extending.md) | 自定义模型、数据适配器、任务和指标 |
| [验证说明](docs/validation.md) | 框架验证记录 |
| [迁移说明](docs/extraction.md) | 独立框架的保留功能与边界 |
