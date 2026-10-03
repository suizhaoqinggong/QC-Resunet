# Training Framework

独立的 PyTorch 训练框架。核心代码使用 `training_framework` 命名空间，项目拥有自己的源码、配置、依赖锁和测试。

提供注册表、TOML 配置、训练/验证/测试、早停、模型 checkpoint、CPU/CUDA/MPS 设备选择、AMP、DDP、TensorBoard 和 JSONL 日志。

内置任务为**多标签分类**和**多类语义分割（2D/3D）**。BraTS 数据适配器及 WT/TC/ET 指标作为可选扩展保留。模型目录只有两个用于检查接口的微型示例：`SimpleNet` 和 `TinySegNet`；新算法通过插件接入。

## 安装与启动

在本目录执行，使用 uv 创建独立 `.venv`：

```bash
uv sync --extra dev --extra brats
uv run training-framework fit --configs configs/experiment.classification.toml --model configs/model.simple_net.toml
uv run training-framework fit --configs configs/experiment.segmentation.toml --model configs/model.tiny_seg_net.toml
```

不使用 BraTS 时执行 `uv sync --extra dev` 即可，核心训练无需 nibabel、SciPy。示例使用 CPU 和合成数据，2 个 epoch；示例指标不代表真实模型性能。

也可以用 `uv run python -m training_framework ...`。`train` 只训练，`fit` 训练后加载最佳权重测试。

验证已有权重：

```bash
uv run training-framework validate --configs configs/experiment.classification.toml --model configs/model.simple_net.toml --checkpoint runs/YOUR_RUN/checkpoints/best.pt
uv run training-framework test --configs configs/experiment.classification.toml --model configs/model.simple_net.toml --checkpoint runs/YOUR_RUN/checkpoints/best.pt
```

`validate`/`test` 不提供 `--checkpoint` 时评估当前初始化的模型，评估训练结果时务必指定权重。所有命令目前都会组装三个数据划分。

## 项目布局

```text
src/training_framework/
  contracts/       模型、数据、任务、指标、runner 的接口
  registry/        注册表与配置组件工厂
  core/            Trainer、Evaluator、checkpoint、DDP、设备和实验目录
  logging/         TensorBoard 与 logger 协议
  losses/          BCE/Focal、CE、Soft Dice、Dice+CE
  metrics/         Accuracy、PR/F1、AUC、Dice、IoU、可选 BraTS 指标
  tasks/           分类与分割的损失、标签和后处理
  data_adapters/   合成数据与可选 BraTS 数据适配器
  models/          小型接口示例
  cli/             独立命令入口
configs/           合成数据示例及 BraTS 模板
examples/          新模型注册插件示例
tests/            组件与训练流程测试
docs/             扩展方法和迁移说明
```

## 新模型接入

不需要修改 Trainer。推荐在新项目自己的 Python 包中实现模型和 `register(registries)`，通过实验配置加载：

```toml
[plugins]
modules = ["your_package.components"]
```

该 Python 包需要安装到当前环境。仓库提供 `examples/custom_project/` 的独立可安装示例，详见 [扩展指南](docs/extending.md)。注册表按实验创建，不共享全局注册状态。

## 数据与任务约定

批次为 `Batch(signal, label, id, meta)`：

- 多标签分类：signal `[B, input_dim]`；label `[B, C]`，0/1 浮点；模型返回 logits `[B, C]`。
- 多类分割：signal `[B, M, ...]`；label `[B, ...]`，整数类别；模型返回 logits `[B, C, ...]`。
- 模型不需要内置 softmax/sigmoid；Task 负责后处理。
- 模型构造器接收 `num_classes`；数据适配器接收 `seed` 和 `num_classes`，可以忽略不需要的值。

`[experiment].class_names` 决定类别数。`[data]` 中的 DataLoader 参数由工厂处理，其余参数传给数据适配器。`[model]` 中除 `name` 外的参数传给模型。`[task]` 中除 `name` 外的参数传给 Task；内置 Task 对未知参数会报错。

分割损失支持 `ce`、`dice`、`dice_ce`（别名 `dc_and_ce`）、`focal`；`dice` 确实计算 Soft Dice。`include_background=false` 可让 Dice 损失排除背景。

## BraTS

```bash
uv sync --extra brats --extra dev
uv run training-framework fit --configs configs/experiment.brats.example.toml --model configs/model.tiny_seg_net.toml
```

先在模板中设置**自己的数据路径和划分**。本项目不包含数据、真实患者 ID、历史实验配置和模型权重。模板模型仅用于接通流程。

支持以下结构：

```text
data/brats/CASE_ID/{t1,t1ce,t2,flair}.npy  # 每个模态 [D, H, W]，float32
data/brats/CASE_ID/seg.npy      # 整数类别，[D, H, W]
```

NPZ 文件为 `preprocessed_dir/CASE_ID.npz`，键为 `t1`、`t1ce`、`t2`、`flair`、`seg`；root_dir 中的病例目录用于发现和划分病例；原始 NIfTI 支持 `CASE_ID-t1n.nii.gz`、`t1c`、`t2w`、`t2f` 及 `seg`，可在适配器代码中配置自己的模态约定。预处理数据应已按需要归一化，标签统一为 `0=background, 1=NCR/NET, 2=ED, 3=ET`。原始读取会将标签 4 映射为 3。

可用 `train_ids_file`、`val_ids_file`、`test_ids_file` 指定逐行 case ID，或使用随机划分。随机划分将相关 `-000/-001` 变体归为同一受试者；固定列表检查 ID 重复和划分交叉，使用者还需保证受试者级别独立。

`auc` 输出宏平均 AUROC 和阶梯积分的 AUPRC（Average Precision）；不含正样本或不含负样本的类别不参与平均。

`brats_dice` 计算病例平均 WT/TC/ET Dice。`brats_hd` 为最大对称 Hausdorff 距离，**不是 HD95**。当前 Evaluator 直接评估加载器输出的体积/裁剪，不包含滑窗整图推理、MC Dropout 或论文特定后处理。

## 实验输出

```text
runs/EXPERIMENT_MODEL_TIMESTAMP/
  experiment.snapshot.toml
  splits/{train,val,test}_ids.txt
  checkpoints/{best,last}.pt
  logs/                 # TensorBoard
  metrics.jsonl
  plots/
```

可用 `uv run training-tensorboard --logdir runs` 查看日志。Checkpoint 包含 epoch、模型权重和验证指标，可加载评估；目前不恢复 optimizer、scheduler、AMP scaler 或随机数状态，**不提供完整断点续训**。

## 多卡训练

将 `[trainer].device` 设为 `cuda`，使用你自己的模型和数据配置：

```bash
uv run torchrun --standalone --nproc_per_node=2 -m training_framework train --configs YOUR_EXPERIMENT.toml --model YOUR_MODEL.toml
```

训练数据用 DistributedSampler 分片；验证/测试集在各进程完整评估；只有主进程写日志和权重，所有进程更新最佳指标及早停状态。`fit` 要求共享可访问的 checkpoint 路径。训练指标沿用各进程标量平均，不能将其理解为所有预测合并后的全局 Dice/AUC。

## 开发检查

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy src
uv build
```

[迁移说明](docs/extraction.md) 记录保留的功能、独立边界和修正。Python 保留 3.9 兼容性，也可使用 3.10–3.13；CPU 分类/分割和双进程 CPU DDP 使用自动测试验证，实际 CUDA/MPS 与真实数据训练需要在目标硬件验证。
