import torch
from torch import nn


class MultiHeadChannelAttention(nn.Module):

    def __init__(self, out_channels: int, n_heads: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.att_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=n_heads, bias=False)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.att_conv(x)
        x = self.batch_norm(x)
        x = self.act(x)
        return self.out_conv(x)


class EfficientMultiHeadSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
