from enum import Enum

import torch
from torch import nn

from localvit.block import NextViTBlock
from localvit.embedding import PatchEmbedding
from localvit.stem import Stem


class NextViTType(Enum):
    SMALL = 2
    BASE = 4
    LARGE = 6


class LocalViT(nn.Module):
    """
    Next Vision Transformer
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, out_features: int, num_classes: int, dropout_path: float = 0.2,
                 vit_type: NextViTType = NextViTType.BASE):
        super().__init__()
        self.stem = Stem(in_features=in_features, out_features=64)

        self.blocks = nn.Sequential(
            # stage 1
            PatchEmbedding(in_features=64, out_features=96, pooling=False),
            NextViTBlock(in_features=96, out_features=96, num_ncb_layers=1, num_ntb_layers=0, sr_ratio=8,
                         dropout_path=dropout_path),
            # stage 2
            PatchEmbedding(in_features=96, out_features=192),
            NextViTBlock(in_features=192, out_features=256, num_ncb_layers=3, num_ntb_layers=1, sr_ratio=4,
                         dropout_path=dropout_path),
            # stage 3
            PatchEmbedding(in_features=256, out_features=384),
            NextViTBlock(in_features=384, out_features=512, num_ncb_layers=4, num_ntb_layers=1, sr_ratio=2,
                         depth=vit_type.value, dropout_path=dropout_path),
            # stage 4
            PatchEmbedding(in_features=512, out_features=768),
            NextViTBlock(in_features=768, out_features=1024, num_ncb_layers=2, num_ntb_layers=1, sr_ratio=1,
                         dropout_path=dropout_path),
        )

        self.batch_norm = nn.BatchNorm2d(num_features=out_features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_proj = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_out = self.stem(x)
        blocks_out = self.blocks(stem_out)
        norm_out = self.batch_norm(blocks_out)
        pool_out = self.avg_pool(norm_out)
        return self.out_proj(torch.flatten(pool_out, 1))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
