"""Minimal 3D segmentation example; replace with your own architecture."""

from torch import nn

from training_framework.contracts.types import Batch, ModelOutput


class TinySegNet(nn.Module):
    def __init__(self, num_classes: int, num_modalities: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(num_modalities, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, batch: Batch) -> ModelOutput:
        return self.net(batch["signal"])
