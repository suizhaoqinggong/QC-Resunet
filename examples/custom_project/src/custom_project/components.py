"""A project-owned model and its registration entry point."""

from torch import nn

from training_framework.contracts.types import Batch, ModelOutput
from training_framework.registry.defaults import RegistryBundle


class ProjectClassifier(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 16) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, batch: Batch) -> ModelOutput:
        return self.classifier(batch["signal"])


def register(registries: RegistryBundle) -> None:
    registries.models.register("project_classifier", ProjectClassifier)
