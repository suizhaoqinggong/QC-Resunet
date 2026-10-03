# 本地验证记录

验证环境：macOS，独立 `.venv`，Python 3.9.6，PyTorch 2.8.0，CPU。

- 常规测试：43 项通过。
- 双进程 CPU Gloo DDP：1 项通过，两个 rank 在 epoch 2 早停，并加载 epoch 1 的同一份最佳权重。
- Ruff lint 和格式检查通过。
- mypy：53 个源码模块无问题。
- uv build：wheel 和 source distribution 均成功。
- wheel 中 53 个 Python 模块全部位于 training_framework 命名空间；source distribution 包含配置、扩展文档和 DDP worker。
- 使用 `python -I` 从临时目录导入已安装包并前向计算；阻止旧通用包名及 nibabel/scipy 后，核心示例仍能运行。
- 分类、分割、NPY/NPZ/raw NIfTI BraTS 均通过合成数据 train/fit 测试。
- 独立可安装的 custom_project 示例插件通过 CLI fit，正确写入本项目的 runs。

执行命令：

```bash
uv sync --extra dev --extra brats
.venv/bin/pytest -q --ignore=tests/test_ddp_fit.py --disable-warnings --tb=short
.venv/bin/pytest tests/test_ddp_fit.py -q --tb=short
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
.venv/bin/mypy src
uv build
uv pip install -e examples/custom_project
uv run --no-sync training-framework fit --configs configs/experiment.classification.toml --model configs/model.plugin_example.toml
```

DDP 测试需要本机 TCP 通信，本次在获准的沙箱外执行。真实数据和 CUDA/MPS 训练未进行。Checkpoint 的完整续训及滑窗推理不属于当前能力范围，详见 README。
