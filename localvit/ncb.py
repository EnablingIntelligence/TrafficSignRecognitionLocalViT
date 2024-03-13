import torch
from timm.models.layers import DropPath
from torch import nn

from localvit.attention import MHCA
from localvit.mlp import MLP


class NCB(nn.Module):
    """
    Next Convolution Block
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, out_features: int, dropout_path: float, expansion_ratio: int = 3):
        super().__init__()
        self.mhca = MHCA(in_features)
        self.mlp = MLP(in_features, out_features, expansion_ratio)
        self.drop_path = DropPath(dropout_path) if dropout_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.drop_path(self.mhca(x)) + x
        return self.drop_path(self.mlp(features)) + features
