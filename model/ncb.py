from torch import nn

from model.attention import MultiHeadChannelAttention
from model.embedding import PatchEmbedding
from model.mlp import MLP


class NextConvolutionBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_heads: int, dropout: float = 0.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, out_channels)
        self.attention = MultiHeadChannelAttention(out_channels, n_heads)
        self.mlp = MLP(out_channels, out_channels, hidden_channels, dropout)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x
