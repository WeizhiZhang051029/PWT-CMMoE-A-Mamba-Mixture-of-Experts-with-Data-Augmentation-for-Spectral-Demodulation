"""Adapter fine-tuning for a pretrained WGAN-GP MoE model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_moe.data.dataset import load_spectrum_bundle
from spectral_moe.data.pca_reducer import SpectralPCAReducer
from spectral_moe.data.physical_features import apply_feature_standardizer, extract_physics_features
from spectral_moe.evaluate.metrics import regression_metrics
from spectral_moe.train.physics_consistency import fit_forward_feature_calibrator, fit_forward_trough_calibrator
from spectral_moe.models.heterogeneous_moe import (
    HeterogeneousMoE,
    QuadraticPhysicsTemperatureHead,
)
from spectral_moe.models.adapter import (
    apply_adapter_to_model,
    count_parameters,
    freeze_non_adapter,
)
from spectral_moe.train.pareto import balanced_mae_score, pareto_frontier
from spectral_moe.utils.config import load_config
from spectral_moe.utils.io import ensure_dir, write_json
from spectral_moe.utils.seed import set_seed
from spectral_moe.utils.splits import resolve_split_seed, split_from_config, subsample_train_indices


class PCASpectralDataset:
    """Dataset used during supervised adapter fine-tuning."""

    def __init__(
        self,
        z_pca: np.ndarray,
        physics: np.ndarray,
        y: np.ndarray | None = None,
        raw_spectrum: np.ndarray | None = None,
        forward_physics: np.ndarray | None = None,
    ) -> None:
        import torch
        self.z = torch.from_numpy(z_pca.astype(np.float32))
        self.phy = torch.from_numpy(physics.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
        self.raw = None if raw_spectrum is None else torch.from_numpy(raw_spectrum.astype(np.float32))
        self.forward_physics = None if forward_physics is None else torch.from_numpy(forward_physics.astype(np.float32))

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int) -> dict:
        item = {"z": self.z[idx], "physics": self.phy[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        if self.raw is not None:
            item["raw"] = self.raw[idx]
        if self.forward_physics is not None:
            item["forward_physics"] = self.forward_physics[idx]
        return item


def _load_pretrained_moe(
    pretrain_dir: Path,
    device: "torch.device",
) -> "tuple[HeterogeneousMoE, dict, list[int]]":
    import json

    import torch

    ckpt = torch.load(
        pretrain_dir / "pretrained_moe_best.pt", map_location=device, weights_only=False
    )
    moe_cfg = ckpt["moe_cfg"]
    norm = np.load(pretrain_dir / "normalization.npz")
    with (pretrain_dir / "pretrain_metrics.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    trough_indices = meta.get("trough_indices", [])

    reducer = SpectralPCAReducer.load(pretrain_dir / "pca_reducer.npz")
    k = reducer.n_components_selected_

    state = ckpt["model"]
    shared_proj_weight = state["shared_proj.1.weight"]
    in_dim = shared_proj_weight.shape[1]
    phys_dim = in_dim - k

    temp_context_out_dim = int(moe_cfg.get("temp_context_out_dim", 0))
    model = HeterogeneousMoE(
        pca_dim=k,
        phys_dim=phys_dim,
        expert_out_dim=int(moe_cfg.get("expert_out_dim", 64)),
        hidden_dim=int(moe_cfg.get("hidden_dim", 128)),
        top_k=int(moe_cfg.get("top_k", 2)),
        trough_indices=trough_indices,
        dropout=float(moe_cfg.get("dropout", 0.1)),
        head_hidden_dim=int(moe_cfg.get("head_hidden_dim", 64)),
        decouple_temperature=bool(moe_cfg.get("decouple_temperature", True)),
        temp_context_out_dim=temp_context_out_dim,
        condition_film_cfg=moe_cfg.get("condition_film", None),
        physics_heads_cfg=moe_cfg.get("physics_heads", None),
        expert_types=moe_cfg.get("expert_types", None),
        use_moe=bool(moe_cfg.get("use_moe", True)),
        hsg_cfg=moe_cfg.get("hsg", None),
        mamba_cfg=moe_cfg.get("mamba", None),
    ).to(device)
    current_sd = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in current_sd and current_sd[k].shape == v.shape}
    skipped = [k for k in state if k not in compatible]
    if skipped:
        print(f"[checkpoint] 跳过形状不匹配/多余参数 ({len(skipped)} 项)，如: {skipped[:3]}")
    load_result = model.load_state_dict(compatible, strict=False)
    if load_result.missing_keys:
        print(f"[checkpoint] 随机初始化参数: {len(load_result.missing_keys)} 项")

    extra = {
        "reducer": reducer,
        "z_mean": norm["z_mean"], "z_std": norm["z_std"],
        "y_mean": norm["y_mean"], "y_std": norm["y_std"],
        "phys_mean": norm["phys_mean"], "phys_std": norm["phys_std"],
        "trough_indices": trough_indices,
        "temp_context_out_dim": temp_context_out_dim,
    }
    return model, extra, trough_indices


def _apply_hsg_schedule(model, moe_cfg, epoch):

    hsg_cfg = moe_cfg.get("hsg", {}) or {}
    configured_mode = str(hsg_cfg.get("mode", "sparse")).lower()
    warmup_epochs = int(hsg_cfg.get("warmup_epochs", 0))
    if not hasattr(model, "router") or not hasattr(model.router, "set_mode"):
        return
    active_mode = "dense" if configured_mode == "sparse" and epoch <= warmup_epochs else configured_mode
    model.router.set_mode(active_mode)


class AdaptiveMTLBalancer:


    def __init__(self, alpha=1.0, beta=0.0, gamma=0.5, ema_span=10,
                 min_weight=0.55, prior_lambda_T=0.714, prior_strength=0.7,
                 min_weight_S=None):
        self.alpha = alpha

        self.beta = beta
        self.gamma = gamma
        self.ema_alpha = 2.0 / (ema_span + 1)


        self.min_weight_T = float(min_weight)
        self.min_weight_S = float(min_weight_S) if min_weight_S is not None else 0.15
        self.prior_lambda_T = float(prior_lambda_T)
        self.prior_strength = float(prior_strength)
        self.L_ema = [None, None]
        self.L_prev_norm = [None, None]


        self.lambda_T = self.prior_lambda_T
        self.lambda_S = 1.0 - self.prior_lambda_T
        self.history = []

    def update(self, L_T, L_S, C_epoch, epoch=0):
        for i, L in enumerate([L_T, L_S]):
            if self.L_ema[i] is None:
                self.L_ema[i] = L
            else:
                self.L_ema[i] = self.ema_alpha * L + (1 - self.ema_alpha) * self.L_ema[i]
        L_T_norm = L_T / max(self.L_ema[0], 1e-8)
        L_S_norm = L_S / max(self.L_ema[1], 1e-8)

        dL_T = dL_S = 0.0
        if self.beta > 0:
            dL_T = abs(L_T_norm - (self.L_prev_norm[0] if self.L_prev_norm[0] is not None else L_T_norm))
            dL_S = abs(L_S_norm - (self.L_prev_norm[1] if self.L_prev_norm[1] is not None else L_S_norm))
        self.L_prev_norm = [L_T_norm, L_S_norm]

        S_T = self.alpha * L_T_norm + self.beta * dL_T
        S_S = self.alpha * L_S_norm + self.beta * dL_S
        denom = S_T + S_S + 1e-8
        lambda_T_adapt = S_T / denom


        lambda_T_raw = (self.prior_strength * self.prior_lambda_T
                        + (1.0 - self.prior_strength) * lambda_T_adapt)


        C = float(max(0.0, C_epoch))
        lambda_T_ = (1 - self.gamma * C) * lambda_T_raw + self.gamma * C * self.prior_lambda_T


        upper = 1.0 - self.min_weight_S
        lambda_T_ = max(self.min_weight_T, min(upper, lambda_T_))
        lambda_S_ = 1.0 - lambda_T_

        self.lambda_T = lambda_T_
        self.lambda_S = lambda_S_
        self.history.append({
            "epoch": epoch,
            "lambda_T": lambda_T_, "lambda_S": lambda_S_,
            "C": C, "L_T": L_T, "L_S": L_S,
            "L_T_norm": L_T_norm, "L_S_norm": L_S_norm,
            "lambda_T_adapt": lambda_T_adapt,
            "lambda_T_prior": self.prior_lambda_T,
        })
        return lambda_T_, lambda_S_

def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("finetune_adapter requires PyTorch.") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--pretrain-dir", default="runs/diffusion_pretrain")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)


    adapter_cfg = config.get("adapter", {})
    bottleneck_dim = int(adapter_cfg.get("bottleneck_dim", 16))
    adapter_dropout = float(adapter_cfg.get("dropout", 0.0))
    adapter_scale = float(adapter_cfg.get("scale", 1.0))
    exclude_modules = list(adapter_cfg.get("exclude_modules", ["temperature_head", "salinity_head"]))


    ft_cfg = config.get("adapter_finetune", {})
    output_dir_str = args.output_dir or ft_cfg.get("output_dir", f"runs/adapter_b{bottleneck_dim}")
    output_dir = ensure_dir(output_dir_str)

    epochs = args.epochs if args.epochs is not None else int(ft_cfg.get("epochs", 500))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_dir = Path(args.pretrain_dir)
    print(f"[设备] {device} | Adapter bottleneck={bottleneck_dim}, scale={adapter_scale}")


    bundle = load_spectrum_bundle(config)
    split_cfg = config.get("data", {}).get("split", {})
    split_seed = resolve_split_seed(split_cfg, seed)
    train_idx, val_idx, test_idx, split_meta = split_from_config(
        len(bundle.y), seed=split_seed, split_cfg=split_cfg,
        labels=bundle.labels, x_raw_dbm=bundle.x_raw_dbm,
    )
    train_fraction = float(split_cfg.get("train_fraction", 1.0))
    train_idx = subsample_train_indices(train_idx, fraction=train_fraction, seed=split_seed + 20000)
    print(f"[数据] 微调训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_idx)}")


    model, artifacts, trough_indices = _load_pretrained_moe(pretrain_dir, device)

    moe_cfg = config.get("heterogeneous_moe", {})
    if ft_cfg.get("reinit_temp_encoder", False) and model.temp_encoder is not None:
        from spectral_moe.models.heterogeneous_moe import SpectralTempEncoder as _STE
        _out_dim = model.temp_context_out_dim
        _drop = float(moe_cfg.get("dropout", 0.1))
        model.temp_encoder = _STE(out_dim=_out_dim, dropout=_drop).to(device)
        print(f"[温度编码器] 已重新初始化为增强版 SE-CNN（out_dim={_out_dim}）")

    reducer = artifacts["reducer"]
    z_mean, z_std = artifacts["z_mean"], artifacts["z_std"]
    y_mean, y_std = artifacts["y_mean"], artifacts["y_std"]
    phys_mean, phys_std = artifacts["phys_mean"], artifacts["phys_std"]
    temp_context_out_dim = artifacts.get("temp_context_out_dim", 0)


    feat_cfg = config.get("features", {})
    physics, feature_names = extract_physics_features(
        bundle.x_raw_dbm, bundle.wavelength_nm,
        num_dips=int(feat_cfg.get("num_dips", 6)),
        num_bands=int(feat_cfg.get("num_bands", 8)),
        tracked_centers_nm=feat_cfg.get("tracked_centers_nm", [1599.0]),
        tracked_half_window_nm=float(feat_cfg.get("tracked_half_window_nm", 35.0)),
    )
    forward_cfg = ft_cfg.get("forward_physics", {})
    forward_mode = str(forward_cfg.get("feature_mode", "main_tracked"))
    forward_model, forward_values = None, None
    if bool(forward_cfg.get("enabled", False)) and forward_mode == "multifeature":
        forward_raw, forward_names = extract_physics_features(
            bundle.x_raw_dbm, bundle.wavelength_nm,
            num_dips=int(feat_cfg.get("num_dips", 6)), num_bands=int(feat_cfg.get("num_bands", 8)),
            tracked_centers_nm=forward_cfg.get("tracked_centers_nm", [1477.0, 1517.0, 1599.0]),
            tracked_half_window_nm=float(forward_cfg.get("tracked_half_window_nm", 12.0)),
        )
        suffixes = tuple(str(v) for v in forward_cfg.get("feature_suffixes", ["wavelength_nm", "intensity"]))
        selected = [i for i, name in enumerate(forward_names) if name.startswith("tracked_dip_") and any(name.endswith("_" + suffix) for suffix in suffixes)]
        if not selected: raise ValueError("multifeature forward physics selected no tracked features")
        forward_values = forward_raw[:, selected]
        forward_model = fit_forward_feature_calibrator(bundle.y[train_idx], forward_values[train_idx], ridge_alpha=float(forward_cfg.get("ridge_alpha", 1e-3)))
        print(f"[ForwardPhysics] features={len(selected)}, r2={np.round(forward_model['r2'], 3).tolist()}")
    elif bool(forward_cfg.get("enabled", False)):
        from spectral_moe.train.physics_consistency import trough_reliability
        forward_indices, forward_centers = [], []
        for i, name in enumerate(feature_names):
            if name.startswith("tracked_dip_") and name.endswith("_wavelength_nm"):
                forward_indices.append(i); forward_centers.append(float(name.split("_")[2]))
        reliability = trough_reliability(physics[train_idx][:, forward_indices], np.asarray(forward_centers), half_window_nm=float(feat_cfg.get("tracked_half_window_nm", 35.0)), edge_margin_nm=float(forward_cfg.get("edge_margin_nm", 12.0)), minimum_weight=float(forward_cfg.get("minimum_reliability", 0.05)))
        forward_model = fit_forward_trough_calibrator(bundle.y[train_idx], physics[train_idx], feature_names, ridge_alpha=float(forward_cfg.get("ridge_alpha", 1e-3)), sample_weight=reliability)

    phys_train = apply_feature_standardizer(physics[train_idx], phys_mean, phys_std)
    phys_val = apply_feature_standardizer(physics[val_idx], phys_mean, phys_std)
    phys_test = apply_feature_standardizer(physics[test_idx], phys_mean, phys_std)

    z_train = (reducer.transform(bundle.x_raw_dbm[train_idx]) - z_mean) / z_std
    z_val = (reducer.transform(bundle.x_raw_dbm[val_idx]) - z_mean) / z_std
    z_test = (reducer.transform(bundle.x_raw_dbm[test_idx]) - z_mean) / z_std

    y_train_s = apply_feature_standardizer(bundle.y[train_idx], y_mean, y_std)
    y_val_s = apply_feature_standardizer(bundle.y[val_idx], y_mean, y_std)
    y_test_s = apply_feature_standardizer(bundle.y[test_idx], y_mean, y_std)


    adapter_enabled = bool(adapter_cfg.get("enabled", True))
    if adapter_enabled:
        print(f"\n[Adapter] 插入瓶颈适配层（r={bottleneck_dim}, scale={adapter_scale}）")
        apply_adapter_to_model(
            model,
            bottleneck_dim=bottleneck_dim,
            dropout=adapter_dropout,
            scale=adapter_scale,
            exclude_modules=exclude_modules,
            verbose=True,
        )
        freeze_non_adapter(model)
        trainable_name_filters = list(ft_cfg.get("trainable_name_filters", []))
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in trainable_name_filters):
                param.requires_grad_(True)
    else:
        print("\nAdapter disabled: fine-tuning all model parameters.")
        for param in model.parameters():
            param.requires_grad_(True)
        trainable_name_filters = []

    stats = count_parameters(model)
    print(
        f"[参数] 总计={stats['total']:,}, "
        f"可训练={stats['trainable']:,} ({100*stats['trainable']/stats['total']:.2f}%)"
    )


    use_raw = (temp_context_out_dim > 0) or ("mamba" in getattr(model, "active_expert_types", []))
    train_ds = PCASpectralDataset(z_train, phys_train, y_train_s, raw_spectrum=bundle.x_raw_dbm[train_idx] if use_raw else None, forward_physics=forward_values[train_idx] if forward_values is not None else None)
    val_ds = PCASpectralDataset(z_val, phys_val, y_val_s, raw_spectrum=bundle.x_raw_dbm[val_idx] if use_raw else None, forward_physics=forward_values[val_idx] if forward_values is not None else None)
    test_ds = PCASpectralDataset(z_test, phys_test, y_test_s, raw_spectrum=bundle.x_raw_dbm[test_idx] if use_raw else None, forward_physics=forward_values[test_idx] if forward_values is not None else None)
    bs = int(ft_cfg.get("batch_size", 16))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)


    temp_w = float(ft_cfg.get("temperature_weight", 2.5))
    sal_w = float(ft_cfg.get("salinity_weight", 1.0))
    bal_w = float(ft_cfg.get("load_balance_weight", 0.005))

    use_uw_loss = bool(ft_cfg.get("use_uw_loss", False))
    uw_log_sigma_T = None
    uw_log_sigma_S = None
    if use_uw_loss:
        uw_log_sigma_T = torch.nn.Parameter(torch.zeros(1, device=device))
        uw_log_sigma_S = torch.nn.Parameter(torch.zeros(1, device=device))
        print(f"[UW Loss] Kendall Uncertainty Weighting enabled (log_sigma_T,S 可学习)")


    use_adaptive_mtl = bool(ft_cfg.get("use_adaptive_mtl", False))
    use_pcgrad       = bool(ft_cfg.get("use_pcgrad", True))
    mtl_alpha        = float(ft_cfg.get("mtl_alpha", 1.0))

    mtl_beta         = float(ft_cfg.get("mtl_beta", 0.0))
    mtl_gamma        = float(ft_cfg.get("mtl_gamma", 0.5))
    mtl_ema_span     = int(ft_cfg.get("mtl_ema_span", 10))


    mtl_min_weight   = float(ft_cfg.get("mtl_min_weight", 0.55))
    mtl_min_weight_S = float(ft_cfg.get("mtl_min_weight_S", 0.15))


    default_prior_lambda_T = temp_w / max(temp_w + sal_w, 1e-8)
    mtl_prior_lambda_T = float(ft_cfg.get("mtl_prior_lambda_T", default_prior_lambda_T))
    mtl_prior_strength = float(ft_cfg.get("mtl_prior_strength", 0.7))
    if use_adaptive_mtl:
        mtl_balancer = AdaptiveMTLBalancer(
            alpha=mtl_alpha, beta=mtl_beta, gamma=mtl_gamma,
            ema_span=mtl_ema_span, min_weight=mtl_min_weight,
            prior_lambda_T=mtl_prior_lambda_T,
            prior_strength=mtl_prior_strength,
            min_weight_S=mtl_min_weight_S,
        )
        print(f"[Adaptive MTL] prior_lambda_T={mtl_prior_lambda_T:.3f} "
              f"prior_strength={mtl_prior_strength}, "
              f"clip=[{mtl_min_weight:.2f}, {1-mtl_min_weight_S:.2f}], "
              f"alpha={mtl_alpha}, beta={mtl_beta}, gamma={mtl_gamma}, "
              f"PCGrad={'on' if use_pcgrad else 'off'}")
    else:
        mtl_balancer = None

    selection_temp_weight = float(ft_cfg.get("selection_temperature_weight", temp_w))
    selection_sal_weight = float(ft_cfg.get("selection_salinity_weight", sal_w))
    selection_strategy = str(ft_cfg.get("selection_strategy", "weighted_mse"))
    pareto_history: list[dict] = []


    ssl_weight = float(ft_cfg.get("ssl_consistency_weight", 0.0))
    ssl_noise_std = float(ft_cfg.get("ssl_noise_std", 0.05))
    ssl_warmup = int(ft_cfg.get("ssl_warmup_epochs", 0))
    if ssl_weight > 0:
        print(f"[SSL] 自监督一致性正则 weight={ssl_weight}, noise_std={ssl_noise_std}, "
              f"warmup={ssl_warmup} epochs")

    forward_weight = float(forward_cfg.get("weight", 0.0))
    forward_tensors = None
    if forward_model is not None and forward_weight > 0:
        if forward_mode == "multifeature":
            quality = np.ones_like(forward_model["r2"])
            if bool(forward_cfg.get("quality_weighting", False)):
                quality = np.clip(forward_model["r2"], float(forward_cfg.get("quality_floor", 0.25)), 1.0)
                quality = quality / quality.mean()
            forward_tensors = {"mode": "multifeature", "coef": torch.tensor(forward_model["coefficients"], device=device), "scale": torch.tensor(forward_model["scale"], device=device), "quality": torch.tensor(quality, device=device), "y_mean": torch.tensor(y_mean, device=device), "y_std": torch.tensor(y_std, device=device)}
        else:
            forward_tensors = {"mode": "main_tracked", "indices": forward_model["indices"], "coef": torch.tensor(forward_model["coefficients"], device=device), "scale": torch.tensor(forward_model["scale_nm"], device=device), "centers": torch.tensor(forward_model["centers_nm"], device=device), "half_window_nm": float(feat_cfg.get("tracked_half_window_nm", 35.0)), "edge_margin_nm": float(forward_cfg.get("edge_margin_nm", 12.0)), "minimum_reliability": float(forward_cfg.get("minimum_reliability", 0.05)), "y_mean": torch.tensor(y_mean, device=device), "y_std": torch.tensor(y_std, device=device), "phys_mean": torch.tensor(phys_mean, device=device), "phys_std": torch.tensor(phys_std, device=device)}

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for adapter fine-tuning.")

    base_lr = float(ft_cfg.get("learning_rate", 3e-4))
    encoder_lr_scale = float(ft_cfg.get("encoder_lr_scale", 1.0))
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "temp_encoder" in name:
            encoder_params.append(param)
        else:
            head_params.append(param)
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": base_lr * encoder_lr_scale})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    if use_uw_loss and uw_log_sigma_T is not None:
        param_groups.append({
            "params": [uw_log_sigma_T, uw_log_sigma_S],
            "lr": float(ft_cfg.get("uw_sigma_lr", base_lr)),
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=float(ft_cfg.get("weight_decay", 1e-3)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(ft_cfg.get("scheduler_T0", 150)),
        T_mult=int(ft_cfg.get("scheduler_T_mult", 2)),
        eta_min=base_lr * 0.001,
    )

    lr_warmup_epochs = int(ft_cfg.get("lr_warmup_epochs", 0))
    warmup_start_factor = float(ft_cfg.get("lr_warmup_start_factor", 0.1))
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("lr_warmup_start_factor must be in (0, 1]")
    _base_lrs = [g["lr"] for g in optimizer.param_groups]
    if lr_warmup_epochs > 0:
        for group, base_lr in zip(optimizer.param_groups, _base_lrs):
            group["lr"] = base_lr * warmup_start_factor
        print(f"[LR Warmup] {lr_warmup_epochs} epochs, start_factor={warmup_start_factor:.3f}")

    patience = int(ft_cfg.get("early_stop_patience", 200))

    early_stop_warmup = int(ft_cfg.get("early_stop_warmup_epochs", 30))

    val_ema_alpha = float(ft_cfg.get("val_ema_alpha", 0.3))


    selection_metric_kind = str(ft_cfg.get("selection_metric_kind", "weighted_mse"))
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    val_ema = None


    use_model_ema = bool(ft_cfg.get("use_model_ema", False))
    ema_decay = float(ft_cfg.get("ema_decay", 0.999))
    ema_state = None


    eval_use_ema = bool(ft_cfg.get("eval_use_ema", use_model_ema))
    if eval_use_ema and not use_model_ema:
        raise ValueError("eval_use_ema=true requires use_model_ema=true")
    if use_model_ema:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[Model EMA] enabled, decay={ema_decay}, select_with_ema={eval_use_ema}")

    def _update_ema():
        if ema_state is None:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                else:
                    ema_state[k].copy_(v.detach())

    def _swap_to_ema():

        if ema_state is None:
            return None
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state, strict=False)
        return backup

    def _restore_from_backup(backup):
        if backup is not None:
            model.load_state_dict(backup, strict=False)

    def _ssl_consistency(pred_clean, z, phy, raw):


        z2 = z + torch.randn_like(z) * ssl_noise_std
        phy2 = phy + torch.randn_like(phy) * ssl_noise_std
        raw2 = (raw + torch.randn_like(raw) * ssl_noise_std) if raw is not None else None
        out_aug = model(z2, phy2, raw_spectrum=raw2)
        return torch.mean((pred_clean - out_aug["prediction"]) ** 2)

    print(f"\n[Adapter 微调] epochs={epochs}, patience={patience}")
    for epoch in range(1, epochs + 1):
        _apply_hsg_schedule(model, moe_cfg, epoch)
        if lr_warmup_epochs > 0 and epoch <= lr_warmup_epochs:
            progress = float(epoch - 1) / float(max(1, lr_warmup_epochs - 1))
            factor = warmup_start_factor + (1.0 - warmup_start_factor) * progress
            for group, base_lr in zip(optimizer.param_groups, _base_lrs):
                group["lr"] = base_lr * factor
        model.train()
        train_losses = []
        _mtl_lT_buf = []
        _mtl_lS_buf = []
        _mtl_conflict_buf = []
        ssl_active = ssl_weight > 0 and epoch > ssl_warmup
        for batch in train_loader:
            z = batch["z"].to(device)
            phy = batch["physics"].to(device)
            y = batch["y"].to(device)
            raw = batch["raw"].to(device) if "raw" in batch else None

            out = model(z, phy, raw_spectrum=raw)
            pred = out["prediction"]

            loss_t = torch.mean((pred[:, 0:1] - y[:, 0:1]) ** 2)
            loss_s = torch.mean((pred[:, 1:2] - y[:, 1:2]) ** 2)
            bal = out["route_weights"].mean(dim=0)
            bal_loss = torch.sum(bal * torch.softmax(bal, dim=0)) * 4

            if use_adaptive_mtl and mtl_balancer is not None:
                ssl_term = (ssl_weight * _ssl_consistency(pred, z, phy, raw)
                            if ssl_active else None)
                lT = mtl_balancer.lambda_T
                lS = mtl_balancer.lambda_S
                if use_pcgrad:


                    _zero_scalar = torch.zeros((), device=device)
                    _ssl_half = (0.5 * ssl_term) if ssl_term is not None else _zero_scalar

                    optimizer.zero_grad()
                    lT_task = loss_t + _ssl_half
                    lT_task.backward(retain_graph=True)
                    gT = [p.grad.detach().clone() if p.grad is not None
                          else torch.zeros_like(p) for p in trainable_params]
                    optimizer.zero_grad()
                    lS_task = loss_s + _ssl_half
                    lS_task.backward(retain_graph=True)
                    gS = [p.grad.detach().clone() if p.grad is not None
                          else torch.zeros_like(p) for p in trainable_params]
                    optimizer.zero_grad()

                    _bal_only = bal_w * bal_loss
                    _bal_only.backward()
                    gB = [p.grad.detach().clone() if p.grad is not None
                          else torch.zeros_like(p) for p in trainable_params]
                    optimizer.zero_grad()

                    gT_flat = torch.cat([g.reshape(-1) for g in gT])
                    gS_flat = torch.cat([g.reshape(-1) for g in gS])
                    gB_flat = torch.cat([g.reshape(-1) for g in gB])
                    cos_TS = (torch.dot(gT_flat, gS_flat)
                              / (gT_flat.norm() * gS_flat.norm() + 1e-8))
                    C_batch = float(max(0.0, -cos_TS.item()))
                    _mtl_conflict_buf.append(C_batch)
                    if cos_TS.item() < 0:


                        gT_orig = gT_flat.clone()
                        gS_orig = gS_flat.clone()
                        proj_T = (torch.dot(gT_orig, gS_orig)
                                  / (gS_orig.norm() ** 2 + 1e-8)) * gS_orig
                        gT_flat = gT_orig - proj_T
                        proj_S = (torch.dot(gS_orig, gT_orig)
                                  / (gT_orig.norm() ** 2 + 1e-8)) * gT_orig
                        gS_flat = gS_orig - proj_S

                    gc = lT * gT_flat + lS * gS_flat + gB_flat
                    _off = 0
                    for _p in trainable_params:
                        _sz = _p.numel()
                        _p.grad = gc[_off:_off + _sz].reshape(_p.shape).clone()
                        _off += _sz
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    loss_log = (lT * float(loss_t.item()) + lS * float(loss_s.item())
                                + bal_w * float(bal_loss.item()))
                else:
                    loss = lT * loss_t + lS * loss_s + bal_w * bal_loss
                    if ssl_term is not None:
                        loss = loss + ssl_term
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    loss_log = float(loss.item())
                    _mtl_conflict_buf.append(0.0)
                _mtl_lT_buf.append(float(loss_t.item()))
                _mtl_lS_buf.append(float(loss_s.item()))
                train_losses.append(loss_log)
                if use_model_ema:
                    _update_ema()
                continue


            if use_uw_loss and uw_log_sigma_T is not None:
                precision_T = torch.exp(-2.0 * uw_log_sigma_T)
                precision_S = torch.exp(-2.0 * uw_log_sigma_S)
                uw_loss = 0.5 * (precision_T * loss_t + precision_S * loss_s)\
                          + uw_log_sigma_T + uw_log_sigma_S
                loss = uw_loss.squeeze() + bal_w * bal_loss
            else:
                loss = temp_w * loss_t + sal_w * loss_s + bal_w * bal_loss

            if ssl_active:
                loss = loss + ssl_weight * _ssl_consistency(pred, z, phy, raw)
            if forward_tensors is not None:
                raw_pred = pred * forward_tensors["y_std"] + forward_tensors["y_mean"]
                t, q = raw_pred[:, :1], raw_pred[:, 1:]
                d = torch.cat([torch.ones_like(t), t, t**2, q, t*q, q**2], dim=1)
                recon = d @ forward_tensors["coef"].T
                if forward_tensors["mode"] == "multifeature":
                    obs = batch["forward_physics"].to(device)
                    feature_quality = forward_tensors["quality"]
                    if str(forward_cfg.get("branch_mode", "joint")) == "conditional":
                        raw_true = y * forward_tensors["y_std"] + forward_tensors["y_mean"]
                        t_true, s_true = raw_true[:, :1], raw_true[:, 1:]
                        d_t = torch.cat([torch.ones_like(t), t, t**2, s_true, t*s_true, s_true**2], dim=1)
                        d_s = torch.cat([torch.ones_like(t_true), t_true, t_true**2, q, t_true*q, q**2], dim=1)
                        error_t = ((d_t @ forward_tensors["coef"].T - obs) / forward_tensors["scale"]) ** 2
                        error_s = ((d_s @ forward_tensors["coef"].T - obs) / forward_tensors["scale"]) ** 2
                        branch_t = float(forward_cfg.get("temperature_branch_weight", 1.0))
                        branch_s = float(forward_cfg.get("salinity_branch_weight", 1.0))
                        physics_loss = (branch_t * (error_t * feature_quality).mean() + branch_s * (error_s * feature_quality).mean()) / (branch_t + branch_s)
                    else:
                        forward_error = ((recon - obs) / forward_tensors["scale"]) ** 2
                        physics_loss = (forward_error * feature_quality).mean()
                    loss = loss + forward_weight * physics_loss
                else:
                    idx = forward_tensors["indices"]
                    obs = phy[:, idx] * forward_tensors["phys_std"][:, idx] + forward_tensors["phys_mean"][:, idx]
                    distance_to_edge = forward_tensors["half_window_nm"] - torch.abs(obs - forward_tensors["centers"])
                    reliability = torch.clamp(distance_to_edge / forward_tensors["edge_margin_nm"], 0.0, 1.0)
                    reliability = forward_tensors["minimum_reliability"] + (1.0 - forward_tensors["minimum_reliability"]) * reliability
                    forward_error = ((recon - obs) / forward_tensors["scale"]) ** 2
                    loss = loss + forward_weight * (forward_error * reliability).sum() / reliability.sum().clamp_min(1e-8)


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

            if use_model_ema:
                _update_ema()


        if use_adaptive_mtl and mtl_balancer is not None and _mtl_lT_buf:
            _ep_C  = float(sum(_mtl_conflict_buf) / max(len(_mtl_conflict_buf), 1))
            _ep_LT = float(sum(_mtl_lT_buf) / max(len(_mtl_lT_buf), 1))
            _ep_LS = float(sum(_mtl_lS_buf) / max(len(_mtl_lS_buf), 1))
            mtl_balancer.update(_ep_LT, _ep_LS, _ep_C, epoch=epoch)


        if lr_warmup_epochs <= 0 or epoch > lr_warmup_epochs:
            scheduler.step()


        _validation_backup = _swap_to_ema() if eval_use_ema else None
        model.eval()
        val_losses = []
        val_abs_errors = []
        with torch.no_grad():
            for batch in val_loader:
                z = batch["z"].to(device)
                phy = batch["physics"].to(device)
                y = batch["y"].to(device)
                raw = batch["raw"].to(device) if "raw" in batch else None
                out = model(z, phy, raw_spectrum=raw)
                pred = out["prediction"]
                vloss = selection_temp_weight * torch.mean((pred[:, 0:1] - y[:, 0:1]) ** 2)\
                      + selection_sal_weight * torch.mean((pred[:, 1:2] - y[:, 1:2]) ** 2)
                val_losses.append(float(vloss.item()))
                val_abs_errors.append(torch.abs(pred - y).cpu().numpy())
        _restore_from_backup(_validation_backup)

        train_avg = float(np.mean(train_losses))
        val_mse_avg = float(np.mean(val_losses)) if val_losses else train_avg
        val_mae_avg = None
        val_temp_mae = None
        val_sal_mae = None
        if val_abs_errors:
            _mae = np.concatenate(val_abs_errors, axis=0).mean(axis=0)
            val_mae_avg = float(selection_temp_weight * _mae[0] + selection_sal_weight * _mae[1])
            val_temp_mae = float(_mae[0])
            val_sal_mae = float(_mae[1])

        if selection_metric_kind == "rank_ensemble" and val_temp_mae is not None:
            val_avg = val_temp_mae + val_sal_mae
        elif selection_metric_kind == "weighted_mae" and val_mae_avg is not None:
            val_avg = val_mae_avg
        else:
            val_avg = val_mse_avg

        val_ema = val_avg if val_ema is None else val_ema_alpha * val_avg + (1.0 - val_ema_alpha) * val_ema
        if selection_strategy == "pareto_normalized_mae" and val_abs_errors:
            target_mae = np.concatenate(val_abs_errors, axis=0).mean(axis=0)
            candidate = {"epoch": epoch, "temperature_mae": float(target_mae[0]), "salinity_mae": float(target_mae[1])}
            candidate["balanced_score"] = balanced_mae_score(candidate["temperature_mae"], candidate["salinity_mae"])
            pareto_history.append(candidate)
            val_avg = candidate["balanced_score"]


        cmp_val = val_ema
        _past_warmup = (lr_warmup_epochs == 0) or (epoch > lr_warmup_epochs)
        if _past_warmup and cmp_val < best_val:
            best_val = cmp_val
            best_epoch = epoch
            stale = 0
            _save_payload = {
                "model": model.state_dict(),
                "bottleneck_dim": bottleneck_dim,
                "trough_indices": trough_indices,
                "moe_cfg": moe_cfg,
            }

            if ema_state is not None:
                _save_payload["ema"] = {k: v.detach().clone() for k, v in ema_state.items()}
            torch.save(_save_payload, Path(output_dir) / "best_adapter.pt")
        else:

            if _past_warmup:
                stale += 1

            if epoch >= early_stop_warmup and stale >= patience:
                print(f"  早停于 epoch={epoch}（patience={patience}, warmup={early_stop_warmup}）")
                break

        if epoch % 30 == 0 or epoch == epochs:
            _mae_str = f" mae={val_mae_avg:.4f}" if val_mae_avg is not None else ""
            _mtl_str = ""
            if use_adaptive_mtl and mtl_balancer and mtl_balancer.history:
                _h = mtl_balancer.history[-1]
                _mtl_str = (f" lT={_h['lambda_T']:.3f} lS={_h['lambda_S']:.3f}"
                            f" C={_h['C']:.3f}")
            print(
                f"  epoch={epoch:>4}  train={train_avg:.5f}  val={val_avg:.5f}"
                f"{_mae_str}  ema={val_ema:.5f}  best={best_val:.5f}@ep{best_epoch}{_mtl_str}"
            )


    ckpt = torch.load(Path(output_dir) / "best_adapter.pt", map_location=device, weights_only=False)

    if eval_use_ema and "ema" in ckpt:
        print("[评估] 使用 EMA 权重（v14）")
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    def _eval_split(loader, split_name):
        y_preds, y_trues = [], []
        with torch.no_grad():
            for batch in loader:
                raw = batch["raw"].to(device) if "raw" in batch else None
                out = model(batch["z"].to(device), batch["physics"].to(device), raw_spectrum=raw)
                y_preds.append(out["prediction"].cpu().numpy())
                y_trues.append(batch["y"].numpy())
        if not y_preds:
            print(f"[评估 - {split_name}集] 跳过（无样本）")
            n_targets = len(bundle.target_names)
            return {}, np.empty((0, n_targets)), np.empty((0, n_targets))
        pred = np.concatenate(y_preds) * y_std + y_mean
        true = np.concatenate(y_trues) * y_std + y_mean

        salinity_discretize = bool(ft_cfg.get("salinity_discretize", False))
        if salinity_discretize:
            SALINITY_LEVELS = np.array([0.0, 5.0, 20.0, 30.0, 35.0, 40.0])

            sal_col_idx = 1
            pred_sal = pred[:, sal_col_idx]

            pred_sal_quantized = np.array([SALINITY_LEVELS[np.argmin(np.abs(s - SALINITY_LEVELS))] for s in pred_sal])
            pred[:, sal_col_idx] = pred_sal_quantized
        m = regression_metrics(true, pred, bundle.target_names)
        print(f"[评估 - {split_name}集] {m}")
        return m, pred, true

    base_info = {
        "adapter_bottleneck_dim": bottleneck_dim,
        "adapter_scale": adapter_scale,
        "adapter_exclude_modules": exclude_modules,
        "param_stats": stats,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "selection_temperature_weight": selection_temp_weight,
        "selection_salinity_weight": selection_sal_weight,
        "selection_strategy": selection_strategy,
        "selection_metric_kind": selection_metric_kind,
        "lr_warmup_epochs": lr_warmup_epochs,
        "lr_warmup_start_factor": warmup_start_factor,
        "use_model_ema": use_model_ema,
        "ema_decay": ema_decay if use_model_ema else None,
        "eval_use_ema": bool(ft_cfg.get("eval_use_ema", use_model_ema)),
        "use_adaptive_mtl": use_adaptive_mtl,
        "use_pcgrad": (use_pcgrad if use_adaptive_mtl else None),
        "mtl_gamma": (mtl_gamma if use_adaptive_mtl else None),
        "mtl_final_lambda_T": (mtl_balancer.lambda_T if mtl_balancer else None),
        "mtl_final_lambda_S": (mtl_balancer.lambda_S if mtl_balancer else None),
        "mtl_final_C": (mtl_balancer.history[-1]["C"] if mtl_balancer and mtl_balancer.history else None),
        "ssl_consistency_weight": ssl_weight,
        "forward_physics": forward_cfg,
        "trainable_name_filters": trainable_name_filters,
        "split": split_meta,

        "use_uw_loss": use_uw_loss,
        "uw_final_log_sigma_T": (float(uw_log_sigma_T.detach().cpu().item()) if use_uw_loss and uw_log_sigma_T is not None else None),
        "uw_final_log_sigma_S": (float(uw_log_sigma_S.detach().cpu().item()) if use_uw_loss and uw_log_sigma_S is not None else None),
    }

    if pareto_history:
        write_json(Path(output_dir) / "pareto_frontier.json", {
            "selection_strategy": selection_strategy, "points": pareto_frontier(pareto_history),
        })

    metrics_val, y_pred_val, y_true_val = _eval_split(val_loader, "val")
    write_json(Path(output_dir) / "metrics_val.json", {
        **base_info, "metrics": metrics_val, "evaluation_split": "val",
    })

    metrics_test, y_pred_test, y_true_test = _eval_split(test_loader, "test")
    write_json(Path(output_dir) / "metrics_test.json", {
        **base_info, "metrics": metrics_test, "evaluation_split": "test",
    })
    write_json(Path(output_dir) / "metrics.json", {
        **base_info, "metrics": metrics_test, "evaluation_split": "test",
    })

    if use_adaptive_mtl and mtl_balancer and mtl_balancer.history:
        write_json(Path(output_dir) / "mtl_conflict_history.json",
                   {"config": {"alpha": mtl_alpha, "beta": mtl_beta,
                               "gamma": mtl_gamma, "pcgrad": use_pcgrad,
                               "min_weight": mtl_min_weight},
                    "history": mtl_balancer.history})
        print(f"[adaptive-mtl] conflict history saved ({len(mtl_balancer.history)} epochs)")

    pred_df = bundle.labels.iloc[test_idx].copy()
    pred_df["temperature_true"] = y_true_test[:, 0]
    pred_df["salinity_true"] = y_true_test[:, 1]
    pred_df["temperature_pred"] = y_pred_test[:, 0]
    pred_df["salinity_pred"] = y_pred_test[:, 1]
    pred_df.to_csv(Path(output_dir) / "predictions.csv", index=False, encoding="utf-8-sig")

    pred_val_df = bundle.labels.iloc[val_idx].copy()
    pred_val_df["temperature_true"] = y_true_val[:, 0]
    pred_val_df["salinity_true"] = y_true_val[:, 1]
    pred_val_df["temperature_pred"] = y_pred_val[:, 0]
    pred_val_df["salinity_pred"] = y_pred_val[:, 1]
    pred_val_df.to_csv(Path(output_dir) / "predictions_val.csv", index=False, encoding="utf-8-sig")

    print(f"[完成] Adapter 微调结果保存至：{output_dir}")


if __name__ == "__main__":
    main()
