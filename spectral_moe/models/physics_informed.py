from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None

from spectral_moe.models.cnn_blocks import require_torch


def smoothness_loss(reconstructed):
    if torch is None:
        require_torch()
    second = reconstructed[..., 2:] - 2 * reconstructed[..., 1:-1] + reconstructed[..., :-2]
    return torch.mean(second**2)


def masked_reconstruction_loss(reconstructed, target, mask=None):
    if torch is None:
        require_torch()
    if mask is None:
        return torch.mean((reconstructed - target) ** 2)
    mask = mask.to(dtype=reconstructed.dtype)
    denom = mask.sum().clamp_min(1.0)
    return torch.sum(((reconstructed - target) ** 2) * mask) / denom


def physics_auxiliary_loss(predicted_features, target_features):
    if torch is None:
        require_torch()
    if predicted_features is None or target_features is None:
        if target_features is not None:
            return target_features.new_tensor(0.0)
        if predicted_features is not None:
            return predicted_features.new_tensor(0.0)
        return torch.tensor(0.0)
    return torch.mean((predicted_features - target_features) ** 2)
