from torch import nn


class LocalityModule(nn.Module):
    """
    Locality Module
    https://arxiv.org/abs/2311.06651
    """

    def __init__(self, in_features: int, out_features: int, expansion_ratio: int = 6, stride: int = 1):
        super().__init__()
        hidden_features = int(in_features * expansion_ratio)
        self.locality = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_features),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1),
        )

    def forward(self, x):
        return self.locality(x)
