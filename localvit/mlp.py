import torch
from torch import nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, out_features: int, expansion_ratio: int = 1):
        super().__init__()
        hidden_features = in_features * expansion_ratio
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(num_features=in_features),
            nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
