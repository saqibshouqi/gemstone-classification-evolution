import torch
import torch.nn as nn


class CBAM(nn.Module):
    """Channel attention followed by spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_out = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        x = x * torch.sigmoid(avg_out + max_out)

        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        x = x * torch.sigmoid(self.spatial(torch.cat([avg_sp, max_sp], dim=1)))
        return x