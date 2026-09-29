from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None

from cnn_blocks import require_torch


def smoothness_loss(reconstructed):
    if torch is None:
        require_torch()
    second = reconstructed[..., 2:] - 2 * reconstructed[..., 1:-1] + reconstructed[..., :-2]
    return torch.mean(second**2)
