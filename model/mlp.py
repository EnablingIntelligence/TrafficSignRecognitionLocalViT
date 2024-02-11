import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, dropout: float = 0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.dropout(x)
