from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("This module requires PyTorch. Install torch before neural training.")


if nn is not None:

    class ConvBNAct(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int = 7,
            stride: int = 1,
            dilation: int = 1,
            groups: int = 1,
            dropout: float = 0.0,
        ):
            super().__init__()
            padding = ((kernel_size - 1) // 2) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)


    class SEBlock(nn.Module):
        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            hidden = max(4, channels // reduction)
            self.net = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.net(x)


    class ResidualConvBlock(nn.Module):
        def __init__(
            self,
            channels: int,
            *,
            kernel_size: int = 7,
            dilation: int = 1,
            dropout: float = 0.0,
            use_se: bool = True,
        ):
            super().__init__()
            self.block = nn.Sequential(
                ConvBNAct(channels, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout),
                ConvBNAct(channels, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout),
                SEBlock(channels) if use_se else nn.Identity(),
            )

        def forward(self, x):
            return x + self.block(x)

else:

    class ConvBNAct:
        def __init__(self, *args, **kwargs):
            require_torch()

    class SEBlock:
        def __init__(self, *args, **kwargs):
            require_torch()

    class ResidualConvBlock:
        def __init__(self, *args, **kwargs):
            require_torch()
