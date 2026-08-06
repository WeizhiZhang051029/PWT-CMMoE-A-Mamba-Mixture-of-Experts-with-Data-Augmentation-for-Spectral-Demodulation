from __future__ import annotations

from typing import Sequence

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from spectral_moe.models.cnn_blocks import require_torch


def reconstruction_loss(pred, target):

    if torch is None or nn is None:
        require_torch()
    return nn.functional.mse_loss(pred, target)


def regression_loss(pred, target):

    if torch is None or nn is None:
        require_torch()
    return nn.functional.mse_loss(pred, target)


def weighted_regression_loss(
    pred,
    target,
    target_weights=None,
    *,
    loss_name: str = "mse",
    smooth_l1_beta: float = 1.0,
):

    if torch is None or nn is None:
        require_torch()
    loss_name = str(loss_name).lower()
    if loss_name == "mse":
        err = (pred - target) ** 2
    elif loss_name in {"mae", "l1"}:
        err = torch.abs(pred - target)
    elif loss_name in {"smooth_l1", "huber"}:
        err = nn.functional.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=float(smooth_l1_beta),
        )
    else:
        raise ValueError(f"Unsupported regression loss: {loss_name}")
    if target_weights is not None:
        weights = target_weights.to(device=pred.device, dtype=pred.dtype)
        while weights.ndim < err.ndim:
            weights = weights.unsqueeze(0)
        err = err * weights
    return torch.mean(err)


def load_balance_loss(gate_weights_list: Sequence) -> object:


    if torch is None:
        require_torch()
    if hasattr(gate_weights_list, "shape"):
        gate_weights_list = [gate_weights_list]
    if len(gate_weights_list) == 0:
        raise ValueError("gate_weights_list must not be empty")

    total = torch.tensor(0.0, device=gate_weights_list[0].device)
    for weights in gate_weights_list:
        if weights.ndim != 2:
            raise ValueError(f"gate weights must have shape [batch, experts], got {tuple(weights.shape)}")
        num_experts = weights.shape[1]
        uniform = 1.0 / num_experts

        importance = weights.sum(dim=0)
        importance = importance / (importance.sum() + 1e-8)
        importance_loss = ((importance - uniform) ** 2).mean()

        load = weights.mean(dim=0)
        load_loss = ((load - uniform) ** 2).mean()

        total = total + importance_loss + load_loss
    return total


def total_moe_regression_loss(
    pred,
    target,
    gate_weights_list: Sequence,
    *,
    aux_weight: float = 0.01,
    target_weights=None,
    loss_name: str = "mse",
    smooth_l1_beta: float = 1.0,
    mae_blend_weight: float = 0.0,
):

    regression_val = weighted_regression_loss(
        pred,
        target,
        target_weights=target_weights,
        loss_name=loss_name,
        smooth_l1_beta=smooth_l1_beta,
    )
    if mae_blend_weight > 0:
        regression_val = regression_val + float(mae_blend_weight) * weighted_regression_loss(
            pred,
            target,
            target_weights=target_weights,
            loss_name="mae",
        )
    aux_val = load_balance_loss(gate_weights_list)
    return regression_val + aux_weight * aux_val, regression_val, aux_val
