# QCResUNet 论文复现

对应用户提供的期刊版论文：Qiu et al., *QCResUNet: Joint subject-level and voxel-level segmentation quality prediction*, Medical Image Analysis 107 (2026), 103718，DOI [10.1016/j.media.2025.103718](https://doi.org/10.1016/j.media.2025.103718)。实现依据正文第 4–6 页、Fig. 2、Table 2/3 和 §3.5。

这是**分割质量控制**应用：输入 MRI 和其他方法生成的待检查分割，预测病例级 DSC、NSD 和组织级分割错误图 SEM。它不会训练 nnUNet/nnFormer，也不直接输出肿瘤分割。已完成独立代码实现和合成数据流程验证；尚未在真实数据上验证论文数值。

## 运行现有合成数据验证

在仓库根目录执行：

```bash
uv sync --extra dev --extra brats
uv run training-framework fit --configs configs/experiment.qcresunet.smoke.toml --model configs/model.qcresunet.smoke.toml
```

这是 2 epoch、CPU、缩小网络的接口检查，只有完美/全空查询两种质量，不构成实验结果。正式模型使用 `configs/model.qcresunet.toml`。所有运行代码位于 `src/training_framework/plugins/qcresunet/`，只在 `[plugins]` 显式加载；默认模型注册表仍只有框架微型示例。无需安装或导入其他训练项目。

## 实现与论文的对应

| 论文要求 | 当前实现 |
| --- | --- |
| 四个 MRI 模态 + 一个查询分割 | `signal=[B,5,D,H,W]`，前四通道为 T1/T1c/T2/FLAIR，最后为整数标签图 |
| 3D ResNet-34 | 64/128/256/512 通道，3/4/6/3 residual blocks，7³ stride-2 stem、2³ max pool |
| InstanceNorm、spatial dropout | 所有残差分支使用 affine InstanceNorm3d、ReLU、Dropout3d(0.3) |
| Block 4 回归、Block 3 解码 | GAP + Linear(512,2)；解码器从 256 通道 Block 3 出发，跳接 Block 2、Block 1、stem |
| 逐级 nearest upsample + 1³ conv | 解码通道 128→64→64→32，每级两次 3³ Conv/IN/LeakyReLU |
| ECA SEM aggregation | 32→3 的 1³ 投影，与查询的三个区域通道连接；GAP→Conv1d(k=3)→sigmoid 加权→1³ 输出 |
| WT/TC/ET 错误图 | WT={1,2,3}、TC={1,3}、ET={3}；每个区域的 query XOR reference |
| 病例 DSC/NSD | 各区域分数的宏平均；NSD 使用物理间距与 surfel 面积权重 |
| 多任务损失 | 两个病例 MAE 相加 + λ(负的 batch soft Dice + BCEWithLogits)，λ=1 |
| Adam / batch 4 / 100 epochs | lr=2.1e-4、weight decay=1e-4；每 epoch ×0.9，最低 1e-6 |
| 查询质量平衡 | [0,1] 均分 10 bins，每 bin 下采样到最少数量；训练每 epoch 重采样，验证/测试固定 |
| 三模型集成 | DSC/NSD 算术均值；SEM 使用二值图多数投票 |

完整默认模型有 **65,652,797** 个可训练参数。GPU 全体积训练所需显存取决于 AMP、batch 和体积大小；这里未实测 CUDA。保持全体积输入，不能把随机 patch 的质量标签当作整例标签。输入太小时 InstanceNorm 在单体素瓶颈上无法计算；CPU 示例使用 32×32×64。

框架的模型、Task 和 Metric 合同均传递单个 Tensor，因此插件将输出打包为 `[B,2+C*D*H*W]`：前两项为 DSC/NSD，随后是通道优先展开的 SEM logits。监督采用同样格式。`forward_heads(signal)`、`representation.unpack` 提供原始两个头。查询标签同时转成重叠区域掩膜供 ECA 使用；专家标注不会进入模型输入。

## 准备自己的数据

先完成配准、1 mm 等体素重采样和去颅骨。准备命令会对每模态非零体素 z-score、裁剪公共前景、补零到 160×192×160。裁剪包含查询和专家标注，避免裁掉错误区域；超过目标尺寸时明确报错，不截断病灶。代码不替代 SRI24 配准或 nnUNet 的原始影像预处理流程。

每个受试者的输入 NPZ 包含：

```text
image       float32 [4,D,H,W]      # T1, T1c, T2, FLAIR
reference   integer [D,H,W]        # 专家分割
spacing     float [3]              # 按 D,H,W 轴对应的毫米间距
```

查询分割分别为 `.npy`，与原始 NPZ 的 reference 同一坐标网格。BraTS 标签 `4=ET` 在准备时统一映射为 `3=ET`；其余为 `0=背景, 1=NCR/NET, 2=ED`。多个时间点/心脏相位必须使用同一 subject_id，以保持受试者级隔离。目前准备入口要求一个 subject 一个 volume；多相位可自行生成下述样本 manifest，沿用同一 subject_id。

创建 `data/qc-source/source.json`，路径相对于此文件：

```json
{
  "subjects": [
    {"subject_id": "case-a", "split": "train", "volume": "case-a.npz",
     "queries": [{"path": "case-a-nnunet-early.npy", "method": "nnunet"},
                 {"path": "case-a-nnformer-final.npy", "method": "nnformer"}]},
    {"subject_id": "case-b", "split": "val", "volume": "case-b.npz",
     "queries": [{"path": "case-b-nnunet.npy", "method": "nnunet"}]},
    {"subject_id": "case-c", "split": "test", "volume": "case-c.npz",
     "queries": [{"path": "case-c-deepmedic.npy", "method": "deepmedic"}]}
  ]
}
```

准备单个已有划分：

```bash
uv run python -m training_framework.plugins.qcresunet.prepare --source data/qc-source/source.json --output data/qc/fold0 --tolerance-mm 1.0
```

命令生成每例三份 SegGen 查询（只用于训练/验证）：rotation ±15°、scale 0.85–1.25、translation ±20 voxels、elastic displacement 最大 20 voxels，每种操作各有 0.5 概率。标签使用 nearest interpolation。它还接收预先生成的 nnUNet/nnFormer 查询；`--seggen 0` 可仅导入已有查询。DeepMedic 查询在训练/验证中被过滤，数据适配器也拒绝将其放入这两个集合。

输出 `manifest.json` 中每条样本包含 `id, subject_id, split, path, volume, method, dsc, nsd`，顶层包含 `regions, tolerance_mm`。`volume` 指向共享的 subject NPZ，`path` 指向单个 query NPY；也可直接提供包含 image/query/reference/spacing 的单样本 NPZ 并省略 volume。所有路径必须在 manifest 目录内，以相对路径保存。MRI 只保存一次，避免数万查询重复存储。未增强样本读取准备阶段的 DSC/NSD；修改任何标签或 NSD 约定后应重新准备数据。训练的联合几何增强后重新计算监督质量。

`preparation-report.json` 给出样本数、受试者数、各划分的 DSC 直方图。默认质量平衡要求十个 bin 均有数据；缺失时会报错。上面的三个病例只是格式示例，无法满足此要求；对小规模接口试验设置 `balance_train=false, balance_eval=false`。正式比较不能把关闭重采样的结果直接与论文表格比较。

## 三折训练和集成

论文 BraTS 协议：1251 subjects 中固定 251 个内部测试 subjects；其余 1000 subjects 轮流使用约 667/333 个训练/验证。三个 fold 必须使用相同 seed、同一 source 文件和固定测试集：

```bash
uv run python -m training_framework.plugins.qcresunet.prepare --source data/qc-source/source.json --output data/qc/fold0 --tolerance-mm 1.0 --fold 0 --test-subjects 251
uv run python -m training_framework.plugins.qcresunet.prepare --source data/qc-source/source.json --output data/qc/fold1 --tolerance-mm 1.0 --fold 1 --test-subjects 251
uv run python -m training_framework.plugins.qcresunet.prepare --source data/qc-source/source.json --output data/qc/fold2 --tolerance-mm 1.0 --fold 2 --test-subjects 251
```

`--fold` 根据 subject_id 决定划分，忽略 source 中原来的 split。1000 不能被 3 整除，所以实际三组为 334/333/333；这是明确的实现选择，不是作者的实际 ID 列表。要保持作者固定 ID 列表，请提前写好 split 并省略 `--fold`。小样本可降低 `--test-subjects`，不会重现论文规模。

训练 fold0：

```bash
uv run training-framework fit --configs configs/experiment.qcresunet.brats.toml --model configs/model.qcresunet.toml
```

复制实验 TOML 为 fold1/fold2，分别修改 `experiment.name` 和 `data.manifest` 后运行。Trainer 会保存 best/last checkpoint、配置快照、epoch JSONL 指标及划分 ID。训练平衡样本每 epoch 更新；初始抽样的 ID 快照与候选池 manifest 一起保留。验证/测试抽样固定，由 seed 可复原。

对同一个测试 manifest 进行三折集成评估：

```bash
uv run python -m training_framework.plugins.qcresunet.predict --configs configs/experiment.qcresunet.brats.toml --model configs/model.qcresunet.toml --checkpoint runs/FOLD0/checkpoints/best.pt runs/FOLD1/checkpoints/best.pt runs/FOLD2/checkpoints/best.pt --evaluate-manifest data/qc/fold0/manifest.json --output runs/qc-ensemble-metrics.json --device cuda
```

指标包括 `qc_dsc_mae, qc_nsd_mae` 及标准差、两个 Pearson r、`qc_sem_dice` 与 WT/TC/ET SEM Dice。SEM Dice 按病例分别计算再平均；两个错误图都空时记为 1。Pearson 在分数不变化或仅一个病例时未定义，返回 NaN。最佳权重按两个病例 MAE 的平均选择。集成评估的 SEM loss 使用多数投票编码的 logits，正式比较应使用 MAE/SEM Dice，而非该 loss。

实际测试需同时覆盖内部 nnUNet/nnFormer、未见过的 DeepMedic 方法、外部 BraTS-SSA 和 WUSM。外部数据使用新的 manifest，保持同样 regions/tolerance，不能将外部病例加入训练。外部纯测试数据的 source 全部设为 `split="test"`，准备命令加 `--test-only --seggen 0`，随后用上述 `predict --evaluate-manifest` 命令评估该 manifest。训练用的通用 `training-framework fit` 仍要求完整 train/val/test。

## 单例推理，不需要专家标注

输入必须已按训练约定预处理，NPZ 包含 `image [M,D,H,W], query [D,H,W]`，可附 spacing。单模型也可运行：

```bash
uv run python -m training_framework.plugins.qcresunet.predict --configs configs/experiment.qcresunet.brats.toml --model configs/model.qcresunet.toml --checkpoint runs/YOUR_RUN/checkpoints/best.pt --input data/qc-inference/case.npz --output runs/case-qc.npz --device cuda
```

输出 `quality=[DSC,NSD]`、`sem_probability=[C,D,H,W]`、`sem=[C,D,H,W]`、spacing。三折时 sem 是多数投票，不能用平均概率图阈值代替；坐标仍是预处理后网格，写回原始 NIfTI 时需要应用自己的逆变换和 affine。回归头遵照图示使用线性层，预测可能越过 [0,1]；评估不裁剪，避免改变 MAE。

## ACDC

提供 `configs/model.qcresunet.acdc.toml` 和 `configs/experiment.qcresunet.acdc.toml`。配置按常见 ACDC 标注 `1=RV, 2=Myo, 3=LV`，输出顺序 LV/Myo/RV。准备时使用 `--modalities 1 --regions '[[3],[2],[1]]' --shape 16 160 160`。一个受试者的 ED/ES 相位必须保留同一 subject_id。正文的 ACDC 尺寸写成 16×16×160，但紧接着说首轴比其他轴小十倍；本实现选用 16×160×160，并显式记录这一歧义。

## 复现边界与待核实约定

1. 提供的 PDF 没有给出 NSD tolerance，也未包含实际附录定义。配置中的 **1.0 mm 是实现默认值，不是已核实的论文参数**。准备命令强制传入 tolerance，manifest 与配置必须匹配。区域宏平均、both-empty=1 和 surfel 面积权重也是当前明确约定。
2. 正文 Dice 式的归一化系数写成 1/V，但后面按 C 个 tissue 求和；CE 式只列正项。当前使用组织平均的负 soft Dice 与完整二元 BCE，适配三个可重叠错误图，避免把 SEM 当作互斥多类分割。没有按 1/V 使 Dice 项随体积缩小。
3. ECA 的拼接投影位置、InstanceNorm affine、LeakyReLU 斜率 0.01、Dice smooth=1e-5、弹性场的平滑与训练增强强度没有完整指定，已在代码中明确选定。训练 rotation/scale 使用 Table 2 的范围，外加 mirroring、Gaussian noise 和 gamma correction。
4. [作者公开代码](https://github.com/sotiraslab/QCResUNet/tree/e9ddc2418c72cc1c20b69bd3074b7f96b8b5f408)还含 input ECA、多尺度 deep supervision、trilinear upsampling 等选项，和提供 PDF 的图示不同。这里优先独立实现这份 PDF，未复制或依赖作者工程，权重格式也不兼容作者 checkpoint。
5. nnUNet/nnFormer 的七组输入为 T1、T1c、T2、FLAIR、T1+FLAIR、T1c+T2、四模态；论文使用 lr=1e-6、沿训练阶段保存查询。生成这些查询的模型应有独立训练与数据隔离记录。这里只提供导入与 SegGen；没有这些查询、真实 BraTS/ACDC/外部数据和目标 GPU，就不能宣称复现论文指标。
6. 论文还有 RayTune 超参搜索、基线方法及解释性分析；本次实现 QCResUNet 主方法及表 3 最佳设置，没有重跑这些附加实验。真实性能、完整 CUDA AMP/DDP 与官方数据划分尚待验证。

## 验证

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy src
uv build
```

新增测试使用合成体积和临时目录，覆盖双头梯度、Block 3 解码路径、无专家标注输入泄漏、区域 XOR、area-weighted NSD 间距、损失公式、病例指标、受试者划分、bin 重采样、预处理、checkpoint 重载与指数衰减最低学习率。生成数据、权重与 runs 均不纳入 Git。
