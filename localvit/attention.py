import torch
from einops import rearrange
from torch import nn


class MHCA(nn.Module):
    """
    Multi-Head Channel Attention
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, head_dim: int = 32):
        super().__init__()
        n_heads = in_features // head_dim
        self.mhca = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, padding=1, groups=n_heads),
            nn.BatchNorm2d(num_features=in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mhca(x)


class EMHSA(nn.Module):
    """
    Efficient Multi-Head Self-Attention
    https://arxiv.org/pdf/2207.05501.pdf
    """

    def __init__(self, in_features: int, sr_ratio: int = 2, n_heads: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = in_features
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.scaling_factor = hidden_dim ** (-0.5)

        self.norm = nn.BatchNorm2d(num_features=in_features)
        self.avg_pool = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
        self.q_proj = nn.Linear(in_features=in_features, out_features=hidden_dim * n_heads)
        self.k_proj = nn.Linear(in_features=in_features, out_features=hidden_dim * n_heads)
        self.v_proj = nn.Linear(in_features=in_features, out_features=hidden_dim * n_heads)
        self.out_proj = nn.Linear(in_features=hidden_dim * n_heads, out_features=in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.norm(x)
        x_reshaped = x.view(b, c, h * w).permute(0, 2, 1)

        # query
        q = (self.q_proj(x_reshaped)
             .view(b, h * w, self.n_heads, self.hidden_dim)
             .permute(0, 2, 1, 3).contiguous())

        # key
        k = (self.k_proj(x_reshaped)
             .view(b, self.n_heads * self.hidden_dim, h, w))
        k = self.avg_pool(k)
        k = rearrange(k, "b (head n) h w -> b head (h w) n", head=self.n_heads)

        # value
        v = (self.v_proj(x_reshaped)
             .view(b, self.n_heads * self.hidden_dim, h, w))
        v = self.avg_pool(v)
        v = rearrange(v, "b (head n) h w -> b head (h w) n", head=self.n_heads)

        # attention
        scores = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaling_factor
        weights = torch.softmax(scores, dim=-1)

        attention = (torch.matmul(weights, v)
                     .permute(0, 2, 1, 3)
                     .contiguous()
                     .view(b, h * w, self.n_heads * self.hidden_dim))

        out = self.out_proj(attention).view(b, self.in_dim, h, w)
        return out + x
