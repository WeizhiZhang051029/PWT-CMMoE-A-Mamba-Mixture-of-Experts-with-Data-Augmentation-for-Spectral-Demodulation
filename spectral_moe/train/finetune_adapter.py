"""Adapter fine-tuning for a pretrained WGAN-GP MoE model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from spectral_moe.data.dataset import TorchRegressionDataset, load_spectrum_bundle
from spectral_moe.data.physical_features import apply_feature_standardizer, extract_physics_features
from spectral_moe.models.heterogeneous_moe import (
    HeterogeneousMoE,
)
from spectral_moe.models.adapter import (
    apply_adapter_to_model,
    count_parameters,
    freeze_non_adapter,
)
from spectral_moe.utils.config import load_config
from spectral_moe.utils.io import ensure_dir, write_json
from spectral_moe.utils.seed import set_seed
from spectral_moe.utils.splits import resolve_split_seed, split_from_config, subsample_train_indices


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
    with (pretrain_dir / "pretrain_summary.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    trough_indices = meta.get("trough_indices", [])

    state = ckpt["model"]
    shared_proj_weight = state["shared_proj.1.weight"]
    in_dim = shared_proj_weight.shape[1]
    spectrum_dim = int(ckpt["spectrum_dim"])
    phys_dim = in_dim - spectrum_dim

    model = HeterogeneousMoE(
        spectrum_dim=spectrum_dim,
        phys_dim=phys_dim,
        expert_out_dim=int(moe_cfg.get("expert_out_dim", 64)),
        hidden_dim=int(moe_cfg.get("hidden_dim", 128)),
        top_k=int(moe_cfg.get("top_k", 2)),
        trough_indices=trough_indices,
        dropout=float(moe_cfg.get("dropout", 0.1)),
        head_hidden_dim=int(moe_cfg.get("head_hidden_dim", 64)),
        condition_film_cfg=moe_cfg.get("condition_film", None),
        physics_heads_cfg=moe_cfg.get("physics_heads", None),
        expert_types=moe_cfg.get("expert_types", None),
        mamba_cfg=moe_cfg.get("mamba", None),
    ).to(device)
    current_sd = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in current_sd and current_sd[k].shape == v.shape}
    skipped = [k for k in state if k not in compatible]
    if skipped:
        print(f"[checkpoint] skipped {len(skipped)} incompatible parameters, e.g. {skipped[:3]}")
    load_result = model.load_state_dict(compatible, strict=False)
    if load_result.missing_keys:
        print(f"[checkpoint] initialized missing parameters: {len(load_result.missing_keys)}")

    extra = {
        "spectrum_mean": norm["spectrum_mean"], "spectrum_std": norm["spectrum_std"],
        "y_mean": norm["y_mean"], "y_std": norm["y_std"],
        "phys_mean": norm["phys_mean"], "phys_std": norm["phys_std"],
        "trough_indices": trough_indices,
    }
    return model, extra, trough_indices


class AdaptiveMTLBalancer:
    """CATB task priorities with the manuscript's equal-weight calibration."""

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, ema_span=10):
        self.alpha = alpha

        self.beta = beta
        self.gamma = gamma
        self.ema_alpha = 2.0 / (ema_span + 1)

        self.L_ema = [None, None]
        self.L_prev_norm = [None, None]
        # Algorithm 2: epoch 1 is always an equal-weight calibration epoch.
        self.lambda_T = 0.5
        self.lambda_S = 0.5
        self.calibration_losses = None
        self.history = []

    def initialize_from_calibration(self, L_T, L_S, epoch=1):
        """Initialize EMA baselines and priorities from calibration means."""
        denom = max(L_T + L_S, 1e-8)
        self.L_ema = [float(L_T), float(L_S)]
        self.L_prev_norm = [1.0, 1.0]
        self.lambda_T = float(L_T) / denom
        self.lambda_S = float(L_S) / denom
        self.calibration_losses = {"temperature": float(L_T), "salinity": float(L_S)}
        self.history.append({
            "epoch": epoch,
            "phase": "equal_weight_calibration",
            "lambda_T": self.lambda_T,
            "lambda_S": self.lambda_S,
            "C": None,
            "L_T": float(L_T),
            "L_S": float(L_S),
            "L_T_norm": 1.0,
            "L_S_norm": 1.0,
            "lambda_T_adapt": self.lambda_T,
        })

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


        C = float(max(0.0, C_epoch))
        # Eq. (12): increasingly conflicting gradients drive the two task
        # weights toward the balanced allocation (0.5, 0.5), not the initial
        # fine-tuning allocation.
        lambda_T_ = (1 - self.gamma * C) * lambda_T_adapt + self.gamma * C * 0.5


        lambda_S_ = 1.0 - lambda_T_

        self.lambda_T = lambda_T_
        self.lambda_S = lambda_S_
        self.history.append({
            "epoch": epoch,
            "phase": "adaptive",
            "lambda_T": lambda_T_, "lambda_S": lambda_S_,
            "C": C, "L_T": L_T, "L_S": L_S,
            "L_T_norm": L_T_norm, "L_S_norm": L_S_norm,
            "lambda_T_adapt": lambda_T_adapt,
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
    parser.add_argument("--pretrain-dir", default="outputs/pretrain")
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
    print(f"[device] {device} | adapter bottleneck={bottleneck_dim}, scale={adapter_scale}")


    bundle = load_spectrum_bundle(config)
    split_cfg = config.get("data", {}).get("split", {})
    split_seed = resolve_split_seed(split_cfg, seed)
    train_idx, val_idx, split_meta = split_from_config(
        len(bundle.y), seed=split_seed, split_cfg=split_cfg,
        labels=bundle.labels, x_raw_dbm=bundle.x_raw_dbm,
    )
    train_fraction = float(split_cfg.get("train_fraction", 1.0))
    train_idx = subsample_train_indices(train_idx, fraction=train_fraction, seed=split_seed + 20000)
    print(f"[data] fine-tune={len(train_idx)}, validation={len(val_idx)}")


    model, artifacts, trough_indices = _load_pretrained_moe(pretrain_dir, device)

    moe_cfg = config.get("heterogeneous_moe", {})
    spectrum_mean, spectrum_std = artifacts["spectrum_mean"], artifacts["spectrum_std"]
    y_mean, y_std = artifacts["y_mean"], artifacts["y_std"]
    phys_mean, phys_std = artifacts["phys_mean"], artifacts["phys_std"]

    feat_cfg = config.get("features", {})
    physics, feature_names = extract_physics_features(
        bundle.x_raw_dbm, bundle.wavelength_nm,
        num_dips=int(feat_cfg.get("num_dips", 6)),
        num_bands=int(feat_cfg.get("num_bands", 8)),
        tracked_centers_nm=feat_cfg.get("tracked_centers_nm", [1516.0, 1599.0]),
        tracked_half_window_nm=float(feat_cfg.get("tracked_half_window_nm", 35.0)),
    )
    phys_train = apply_feature_standardizer(physics[train_idx], phys_mean, phys_std)
    phys_val = apply_feature_standardizer(physics[val_idx], phys_mean, phys_std)

    z_train = (bundle.x[train_idx] - spectrum_mean) / spectrum_std
    z_val = (bundle.x[val_idx] - spectrum_mean) / spectrum_std

    y_train_s = apply_feature_standardizer(bundle.y[train_idx], y_mean, y_std)
    y_val_s = apply_feature_standardizer(bundle.y[val_idx], y_mean, y_std)


    adapter_enabled = bool(adapter_cfg.get("enabled", True))
    if adapter_enabled:
        print(f"\n[Adapter] inserting bottleneck adapter (r={bottleneck_dim}, scale={adapter_scale})")
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
        f"[parameters] total={stats['total']:,}, "
        f"trainable={stats['trainable']:,} ({100*stats['trainable']/stats['total']:.2f}%)"
    )


    use_raw = "mamba" in getattr(model, "active_expert_types", [])
    train_ds = TorchRegressionDataset(z_train, phys_train, y_train_s, raw_spectrum=bundle.x[train_idx] if use_raw else None)
    val_ds = TorchRegressionDataset(z_val, phys_val, y_val_s, raw_spectrum=bundle.x[val_idx] if use_raw else None)
    bs = int(ft_cfg.get("batch_size", 16))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)


    temp_w = float(ft_cfg.get("temperature_weight", 2.5))
    sal_w = float(ft_cfg.get("salinity_weight", 1.0))
    bal_w = float(ft_cfg.get("load_balance_weight", 0.005))

    use_adaptive_mtl = bool(ft_cfg.get("use_adaptive_mtl", False))
    use_pcgrad       = bool(ft_cfg.get("use_pcgrad", True))
    mtl_alpha        = float(ft_cfg.get("mtl_alpha", 1.0))

    mtl_beta         = float(ft_cfg.get("mtl_beta", 1.0))
    mtl_gamma        = float(ft_cfg.get("mtl_gamma", 0.5))
    mtl_ema_span     = int(ft_cfg.get("mtl_ema_span", 10))


    if use_adaptive_mtl:
        mtl_balancer = AdaptiveMTLBalancer(
            alpha=mtl_alpha, beta=mtl_beta, gamma=mtl_gamma,
            ema_span=mtl_ema_span,
        )
        print(f"[CATB] equal-weight calibration: lambda_T=0.500 lambda_S=0.500; "
              f"alpha={mtl_alpha}, beta={mtl_beta}, gamma={mtl_gamma}, "
              f"PCGrad={'on' if use_pcgrad else 'off'}")
    else:
        mtl_balancer = None

    selection_temp_weight = float(ft_cfg.get("selection_temperature_weight", temp_w))
    selection_sal_weight = float(ft_cfg.get("selection_salinity_weight", sal_w))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for adapter fine-tuning.")
    # CATB/PCGrad applies only to shared adaptation parameters; the two
    # task-specific prediction heads retain their independent gradients.
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    shared_param_mask = [
        "temperature_head" not in name
        and "salinity_head" not in name
        for name in trainable_names
    ]

    base_lr = float(ft_cfg.get("learning_rate", 3e-4))
    head_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        head_params.append(param)
    param_groups = [{"params": head_params, "lr": base_lr}]

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


    print(f"\n[Adapter fine-tuning] epochs={epochs}, patience={patience}")
    for epoch in range(1, epochs + 1):
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
                lT = mtl_balancer.lambda_T
                lS = mtl_balancer.lambda_S
                if use_pcgrad:


                    optimizer.zero_grad()
                    lT_task = loss_t
                    lT_task.backward(retain_graph=True)
                    gT = [p.grad.detach().clone() if p.grad is not None
                          else torch.zeros_like(p) for p in trainable_params]
                    optimizer.zero_grad()
                    lS_task = loss_s
                    lS_task.backward(retain_graph=True)
                    gS = [p.grad.detach().clone() if p.grad is not None
                          else torch.zeros_like(p) for p in trainable_params]
                    optimizer.zero_grad()

                    _bal_only = bal_w * bal_loss
                    _bal_only.backward()
                    gB = [p.grad.detach().clone() if p.grad is not None
                          else torch.zeros_like(p) for p in trainable_params]
                    optimizer.zero_grad()

                    gT_flat = torch.cat([g.reshape(-1) for g, shared in zip(gT, shared_param_mask) if shared])
                    gS_flat = torch.cat([g.reshape(-1) for g, shared in zip(gS, shared_param_mask) if shared])
                    gB_flat = torch.cat([g.reshape(-1) for g, shared in zip(gB, shared_param_mask) if shared])
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

                    gc_shared = lT * gT_flat + lS * gS_flat + gB_flat
                    _off = 0
                    for _p, _gT, _gS, _gB, _is_shared in zip(trainable_params, gT, gS, gB, shared_param_mask):
                        if _is_shared:
                            _sz = _p.numel()
                            _p.grad = gc_shared[_off:_off + _sz].reshape(_p.shape).clone()
                            _off += _sz
                        else:
                            _p.grad = (lT * _gT + lS * _gS + _gB).clone()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    loss_log = (lT * float(loss_t.item()) + lS * float(loss_s.item())
                                + bal_w * float(bal_loss.item()))
                else:
                    loss = lT * loss_t + lS * loss_s + bal_w * bal_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    loss_log = float(loss.item())
                    _mtl_conflict_buf.append(0.0)
                _mtl_lT_buf.append(float(loss_t.item()))
                _mtl_lS_buf.append(float(loss_s.item()))
                train_losses.append(loss_log)
                continue


            loss = temp_w * loss_t + sal_w * loss_s + bal_w * bal_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))



        if use_adaptive_mtl and mtl_balancer is not None and _mtl_lT_buf:
            _ep_C  = float(sum(_mtl_conflict_buf) / max(len(_mtl_conflict_buf), 1))
            _ep_LT = float(sum(_mtl_lT_buf) / max(len(_mtl_lT_buf), 1))
            _ep_LS = float(sum(_mtl_lS_buf) / max(len(_mtl_lS_buf), 1))
            if epoch == 1:
                mtl_balancer.initialize_from_calibration(_ep_LT, _ep_LS, epoch=epoch)
                print("[CATB] calibration complete: "
                      f"mean_LT={_ep_LT:.6f} mean_LS={_ep_LS:.6f} "
                      f"-> lambda_T={mtl_balancer.lambda_T:.3f} "
                      f"lambda_S={mtl_balancer.lambda_S:.3f}")
            else:
                mtl_balancer.update(_ep_LT, _ep_LS, _ep_C, epoch=epoch)


        if lr_warmup_epochs <= 0 or epoch > lr_warmup_epochs:
            scheduler.step()


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

            torch.save(_save_payload, Path(output_dir) / "best_adapter.pt")
        else:

            if _past_warmup:
                stale += 1

            if epoch >= early_stop_warmup and stale >= patience:
                print(f"  early stopping at epoch={epoch} (patience={patience}, warmup={early_stop_warmup})")
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


    base_info = {
        "adapter_bottleneck_dim": bottleneck_dim,
        "adapter_scale": adapter_scale,
        "adapter_exclude_modules": exclude_modules,
        "param_stats": stats,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "selection_temperature_weight": selection_temp_weight,
        "selection_salinity_weight": selection_sal_weight,
        "selection_metric_kind": selection_metric_kind,
        "lr_warmup_epochs": lr_warmup_epochs,
        "lr_warmup_start_factor": warmup_start_factor,
        "use_adaptive_mtl": use_adaptive_mtl,
        "use_pcgrad": (use_pcgrad if use_adaptive_mtl else None),
        "mtl_gamma": (mtl_gamma if use_adaptive_mtl else None),
        "mtl_final_lambda_T": (mtl_balancer.lambda_T if mtl_balancer else None),
        "mtl_final_lambda_S": (mtl_balancer.lambda_S if mtl_balancer else None),
        "mtl_final_C": (mtl_balancer.history[-1]["C"] if mtl_balancer and mtl_balancer.history else None),
        "trainable_name_filters": trainable_name_filters,
        "split": split_meta,

    }

    if use_adaptive_mtl and mtl_balancer and mtl_balancer.history:
        write_json(Path(output_dir) / "mtl_conflict_history.json",
                   {"config": {"alpha": mtl_alpha, "beta": mtl_beta,
                               "gamma": mtl_gamma, "pcgrad": use_pcgrad,
                               "calibration": "epoch 1, lambda_T=lambda_S=0.5"},
                    "history": mtl_balancer.history})
        print(f"[adaptive-mtl] conflict history saved ({len(mtl_balancer.history)} epochs)")

    write_json(Path(output_dir) / "training_summary.json", base_info)

    print(f"[complete] adapter fine-tuning results saved to {output_dir}")


if __name__ == "__main__":
    main()
