import ast
import subprocess
import sys
from pathlib import Path

from training_framework.registry.defaults import create_default_registries

PROJECT = Path(__file__).resolve().parents[1]


def test_only_example_models_are_registered():
    registries = create_default_registries()
    assert registries.models.names() == ("simple_net", "tiny_seg_net")
    assert "brats" in registries.data_adapters
    assert "segmentation" in registries.tasks
    assert registries.models is not create_default_registries().models


def test_no_external_project_imports_or_source_links():
    for source in (PROJECT / "src").rglob("*.py"):
        assert not source.is_symlink(), source
        for node in ast.walk(ast.parse(source.read_text())):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                assert name.split(".")[0] not in {"framework", "models", "tasks", "data_adapters"}, source


def test_installed_package_imports_outside_checkout_without_optional_dependencies(tmp_path):
    code = """
import importlib.abc
import sys
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'framework', 'models', 'tasks', 'data_adapters', 'nibabel', 'scipy'}:
            raise ImportError('Blocked external import: ' + fullname)
sys.meta_path.insert(0, BlockExternal())
from training_framework.registry.defaults import create_default_registries
from training_framework.registry.factories import build_component_bundle
config = {
    'experiment': {'class_names': ['a', 'b']},
    'data': {'adapter': 'dummy', 'input_dim': 4, 'num_samples': 12},
    'model': {'name': 'simple_net', 'input_dim': 4, 'hidden_dim': 4},
    'task': {'name': 'classification'},
}
registries = create_default_registries()
bundle = build_component_bundle(config, registries)
assert bundle.model(next(iter(bundle.train_loader))).shape[1] == 2
print('independent import and forward passed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
