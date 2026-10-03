# 新项目接入指南

## 先运行独立插件示例

框架和算法项目分别拥有 Python 包，算法项目注册自己的组件。示例安装到当前框架的独立环境：

```bash
uv pip install -e examples/custom_project
uv run --no-sync training-framework fit --configs configs/experiment.classification.toml --model configs/model.plugin_example.toml
```

`--no-sync` 保留额外安装的示例插件；普通 `uv sync` 会移除未声明在依赖中的额外包。在正式项目中，使用自己的 pyproject.toml 声明框架依赖，或用 `uv add --editable ./examples/custom_project` 把应用包登记为本项目的本地依赖。

## 自己的模型

```python
from torch import nn
from training_framework.contracts.types import Batch

class MyModel(nn.Module):
    def __init__(self, num_classes: int, input_dim: int):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, batch: Batch):
        return self.head(batch["signal"])

def register(registries):
    registries.models.register("my_model", MyModel)
```

在模型配置中指定：

```toml
[plugins]
modules = ["my_project.components"]
[model]
name = "my_model"
input_dim = 128
```

模型返回 logits，损失和后处理交给 Task。可以注册自己的 task、data_adapter、metric，无需修改通用 Trainer。

## 自己的数据

实现 `prepare()`、`get_splits()`、`collate_fn(batch)`。`get_splits()` 返回 train/val/test 三个 PyTorch Dataset，样本包含 signal、label、id、meta。构造器接收 seed、num_classes 及 `[data]` 中自定义的配置参数。

```python
def register(registries):
    registries.data_adapters.register("my_data", MyDataAdapter)
```

## 自己的任务和指标

Task 实现 `compute_loss(outputs, batch)`、`extract_targets(batch)`、`postprocess_outputs(outputs)`、`infer_problem_type()`。工厂传入 num_classes 和 threshold。Task 可持有自己的 loss_fn，损失对象可以提供 `get_last_components()` 返回分项标量，训练器自动记录 `loss_` 开头的分项。

Metric 实现 update、compute、reset，定义 name 和 higher_is_better。构造器需接受 num_classes、threshold、class_names、problem_type（不用的参数可通过 **kwargs 接收）。compute 返回标量字典，日志键为 `METRIC_NAME_RESULT_KEY`；checkpoint monitor 可加 `val_` 前缀。

## 嵌入自己的 Python 入口

```python
from training_framework.registry.defaults import create_default_registries
from training_framework.registry.factories import load_merged_experiment_config
from training_framework.cli.main import build_trainer

registries = create_default_registries(["my_project.components"])
config = load_merged_experiment_config("configs/experiment.toml", "configs/model.toml")
trainer, layout = build_trainer(config, registries)
try:
    trainer.fit()
finally:
    trainer.logger.end_run()
```

如果不需要示例组件，可以自己创建 RegistryBundle，并只注册应用需要的组件。不要导入外部工程的 Trainer 或把外部源码目录加入 sys.path。
