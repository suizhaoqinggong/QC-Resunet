"""Register only when this application module is explicitly loaded."""

from training_framework.plugins.qcresunet.data import QCDataAdapter, SyntheticQCDataAdapter
from training_framework.plugins.qcresunet.metrics import QCMetrics
from training_framework.plugins.qcresunet.model import QCResUNet
from training_framework.plugins.qcresunet.task import QCTask
from training_framework.registry.defaults import RegistryBundle


def register(registries: RegistryBundle) -> None:
    registries.models.register("qcresunet", QCResUNet)
    registries.tasks.register("segmentation_qc", QCTask)
    registries.data_adapters.register("qc_manifest", QCDataAdapter)
    registries.data_adapters.register("qc_synthetic", SyntheticQCDataAdapter)
    registries.metrics.register("qc", QCMetrics)
