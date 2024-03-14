import torch
from torch import nn


class PatchEmbedding(nn.Module):

    def __init__(self, in_features: int, out_features: int, pooling: bool = True):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2) if pooling else nn.Identity()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.avg_pool(x))
