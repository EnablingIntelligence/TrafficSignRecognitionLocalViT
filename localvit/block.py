import torch
from torch import nn

from localvit.ncb import NCB
from localvit.ntb import NTB


class NextViTBlock(nn.Module):
    """
    Next Vision Transformer Block
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, out_features: int, num_ncb_layers: int, num_ntb_layers: int, depth: int = 1,
                 sr_ratio: int = 1, dropout_path: float = 0., local: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_ncb_layers = num_ncb_layers
        self.num_ntb_layers = num_ntb_layers
        self.dropout_path = dropout_path
        self.sr_ratio = sr_ratio
        self.depth = depth
        self.local = local

        self.block = self.__make_block_layers()

    def __make_block_layers(self) -> nn.Sequential:
        block = []

        for depth_idx in range(self.depth):
            local = depth_idx == 0 and self.local
            is_last_layer = depth_idx == self.depth - 1
            out_features = self.out_features if is_last_layer else self.in_features
            block.append(self.__make_layer(out_features, local))

        return nn.Sequential(*block)

    def __make_layer(self, out_features: int, local: bool) -> nn.Sequential:
        return nn.Sequential(
            *[
                *[NCB(self.in_features, self.in_features, self.dropout_path) for _ in range(self.num_ncb_layers)],
                *[
                    NTB(
                        in_features=self.in_features,
                        out_features=out_features,
                        sr_ratio=self.sr_ratio,
                        dropout_path=self.dropout_path,
                        local=local
                    ) for _ in range(self.num_ntb_layers)
                ]
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
