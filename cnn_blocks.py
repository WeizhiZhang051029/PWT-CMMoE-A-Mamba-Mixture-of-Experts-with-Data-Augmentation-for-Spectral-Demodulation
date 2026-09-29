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


else:

    class ConvBNAct:
        def __init__(self, *args, **kwargs):
            require_torch()

    pass
