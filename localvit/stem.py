from typing import Tuple

import torch
from torch import nn


class Stem(nn.Module):
    """
    Stem
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, out_features: int = 64, stem_channels: Tuple[int] = (64, 32, 64)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=stem_channels[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_channels[0], out_channels=stem_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_channels[1], out_channels=stem_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_channels[2], out_channels=out_features, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)
