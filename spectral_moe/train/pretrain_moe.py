from __future__ import annotations


import argparse

import json

from pathlib import Path


import numpy as np

import pandas as pd


from spectral_moe.data.dataset import load_spectrum_bundle

from spectral_moe.data.physical_features import (

    apply_feature_standardizer,

    extract_physics_features,

    fit_feature_standardizer,

)

from spectral_moe.evaluate.metrics import regression_metrics

from spectral_moe.evaluate.synthetic_quality import synthetic_acceptance_mask, synthetic_quality_report

from spectral_moe.train.physics_consistency import fit_forward_trough_calibrator, predict_forward_troughs

from spectral_moe.models.heterogeneous_moe import HeterogeneousMoE

from spectral_moe.utils.config import load_config

from spectral_moe.utils.io import ensure_dir, write_json

from spectral_moe.utils.seed import set_seed

from spectral_moe.utils.splits import (

    resolve_split_seed,

    split_from_config,

    subsample_train_indices,

)


def standardize_labels(

    y_train: np.ndarray,

    *others: np.ndarray,

) -> tuple[np.ndarray, ...]:

    mean = y_train.mean(axis=0, keepdims=True)

    std = y_train.std(axis=0, keepdims=True)

    std[std < 1e-8] = 1.0

    scaled = tuple(((a - mean) / std).astype(np.float32) for a in (y_train, *others))

    return (*scaled, mean.astype(np.float32), std.astype(np.float32))


def standardize_spectrum(

    spectrum_train: np.ndarray,

    *others: np.ndarray,

) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:

    mean = spectrum_train.mean(axis=0, keepdims=True)

    std = spectrum_train.std(axis=0, keepdims=True)

    std[std < 1e-8] = 1.0

    scaled = tuple(((a - mean) / std).astype(np.float32) for a in (spectrum_train, *others))

    return scaled, mean.astype(np.float32), std.astype(np.float32)


def load_balance_loss_fn(route_weights: "torch.Tensor") -> "torch.Tensor":

    import torch

    n_experts = route_weights.shape[-1]

    expert_frac = route_weights.mean(dim=0)

    return torch.sum(expert_frac * torch.softmax(expert_frac, dim=0)) * n_experts


class SpectralDataset:


    def __init__(

        self,

        spectrum_features: np.ndarray,

        physics: np.ndarray,

        y: np.ndarray | None = None,

        sample_weight: np.ndarray | None = None,

        raw_spectrum: np.ndarray | None = None,

    ) -> None:

        import torch

        self.z = torch.from_numpy(spectrum_features.astype(np.float32))

        self.phy = torch.from_numpy(physics.astype(np.float32))

        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))

        self.sample_weight = (

            None if sample_weight is None else torch.from_numpy(sample_weight.astype(np.float32))

        )

        self.raw = (

            None if raw_spectrum is None else torch.from_numpy(raw_spectrum.astype(np.float32))

        )


    def __len__(self) -> int:

        return int(self.z.shape[0])


    def __getitem__(self, idx: int) -> dict:

        item = {"z": self.z[idx], "physics": self.phy[idx]}

        if self.y is not None:

            item["y"] = self.y[idx]

        if self.sample_weight is not None:

            item["sample_weight"] = self.sample_weight[idx]

        if self.raw is not None:

            item["raw"] = self.raw[idx]

        return item


def _apply_hsg_schedule(model, moe_cfg, epoch):

    hsg_cfg = moe_cfg.get("hsg", {}) or {}

    configured_mode = str(hsg_cfg.get("mode", "sparse")).lower()

    warmup_epochs = int(hsg_cfg.get("warmup_epochs", 0))

    if not hasattr(model, "router") or not hasattr(model.router, "set_mode"):

        return

    active_mode = "dense" if configured_mode == "sparse" and epoch <= warmup_epochs else configured_mode

    model.router.set_mode(active_mode)


def pretrain_moe(

    spectrum_all_norm: np.ndarray,

    phys_all: np.ndarray,

    y_all_norm: np.ndarray,

    sample_weight: np.ndarray | None,

    trough_indices: list[int],

    moe_cfg: dict,

    pretrain_cfg: dict,

    output_dir: Path,

    device: "torch.device",

    raw_spectrum_all: np.ndarray | None = None,

) -> "HeterogeneousMoE":

    import torch

    from torch.utils.data import DataLoader


    dataset = SpectralDataset(

        spectrum_all_norm, phys_all, y_all_norm, sample_weight,

        raw_spectrum=raw_spectrum_all,

    )

    loader = DataLoader(

        dataset,

        batch_size=int(pretrain_cfg.get("batch_size", 32)),

        shuffle=True,

        drop_last=True,

    )


    temp_context_out_dim = int(moe_cfg.get("temp_context_out_dim", 0))

    model = HeterogeneousMoE(

        spectrum_dim=spectrum_all_norm.shape[1],

        phys_dim=phys_all.shape[1],

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


    masked_encoder_path = pretrain_cfg.get("masked_encoder_path", None)

    if masked_encoder_path and model.temp_encoder is not None:

        import torch as _torch

        enc_path = Path(masked_encoder_path)

        if enc_path.exists():

            state = _torch.load(enc_path, map_location=device, weights_only=True)

            result = model.temp_encoder.load_state_dict(state, strict=False)

            print(f"[Phase 3] loaded SpectralTempEncoder weights from {enc_path}")

            if result.missing_keys:

                print(f"  missing_keys={result.missing_keys}")

            if result.unexpected_keys:

                print(f"  unexpected_keys={result.unexpected_keys}")

        else:

            print(f"[warning] masked encoder checkpoint not found: {enc_path}")

    elif masked_encoder_path and model.temp_encoder is None:

        print("[warning] temperature encoder is disabled because temp_context_out_dim=0")


    temp_w = float(pretrain_cfg.get("temperature_weight", 1.5))

    sal_w = float(pretrain_cfg.get("salinity_weight", 1.3))

    bal_w = float(pretrain_cfg.get("load_balance_weight", 0.01))


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=float(pretrain_cfg.get("learning_rate", 3e-4)),

        weight_decay=float(pretrain_cfg.get("weight_decay", 1e-4)),

    )

    epochs = int(pretrain_cfg.get("epochs", 150))

    best_loss = float("inf")


    for epoch in range(1, epochs + 1):

        _apply_hsg_schedule(model, moe_cfg, epoch)

        model.train()

        epoch_losses = []

        for batch in loader:

            z = batch["z"].to(device)

            phy = batch["physics"].to(device)

            y = batch["y"].to(device)

            weight = batch.get("sample_weight")

            if weight is not None:

                weight = weight.to(device).view(-1, 1)

            raw = batch.get("raw")

            if raw is not None:

                raw = raw.to(device)


            out = model(z, phy, raw_spectrum=raw)

            pred = out["prediction"]


            err_t = (pred[:, 0:1] - y[:, 0:1]) ** 2

            err_s = (pred[:, 1:2] - y[:, 1:2]) ** 2

            if weight is not None:

                err_t = err_t * weight

                err_s = err_s * weight

            loss_t = torch.mean(err_t)

            loss_s = torch.mean(err_s)

            balance = load_balance_loss_fn(out["route_weights"])

            loss = temp_w * loss_t + sal_w * loss_s + bal_w * balance


            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_losses.append(float(loss.item()))


        avg_loss = float(np.mean(epoch_losses))

        if avg_loss < best_loss:

            best_loss = avg_loss


            saved_cfg = dict(moe_cfg)

            saved_cfg["temp_context_out_dim"] = temp_context_out_dim

            torch.save({
                "model": model.state_dict(),
                "moe_cfg": saved_cfg,
                "spectrum_dim": int(spectrum_all_norm.shape[1]),
            },

                       output_dir / "pretrained_moe_best.pt")

        if epoch % 30 == 0 or epoch == epochs:

            print(f"  [MoE Pretrain] epoch={epoch}/{epochs}  loss={avg_loss:.6f}  best={best_loss:.6f}")


    ckpt = torch.load(output_dir / "pretrained_moe_best.pt", map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])

    print(f"[Phase 3] MoE pretraining complete, best loss={best_loss:.6f}")

    return model


def main() -> None:

    try:

        import torch

    except ImportError as exc:

        raise ImportError("pretrain_moe requires PyTorch.") from exc


    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/config.yaml")

    parser.add_argument(

        "--output-dir", default=None,

        help="Override pretrain.output_dir from the configuration.",

    )

    args = parser.parse_args()


    config = load_config(args.config)

    seed = int(config.get("seed", 42))

    set_seed(seed)


    output_dir_str = args.output_dir or config.get("pretrain", {}).get("output_dir", "runs/diffusion_pretrain")

    output_dir = ensure_dir(output_dir_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[device] {device}")


    bundle = load_spectrum_bundle(config)

    split_cfg = config.get("data", {}).get("split", {})

    split_seed = resolve_split_seed(split_cfg, seed)

    train_idx, val_idx, test_idx, split_meta = split_from_config(

        len(bundle.y), seed=split_seed, split_cfg=split_cfg,

        labels=bundle.labels, x_raw_dbm=bundle.x_raw_dbm,

    )


    data_fraction = float(

        config.get("pretrain", {}).get("data_fraction",

            split_cfg.get("train_fraction", 1.0))

    )

    if data_fraction < 1.0:

        train_idx = subsample_train_indices(train_idx, fraction=data_fraction, seed=split_seed + 20000)

        print(f"[data] fraction={data_fraction:.2f}, training samples={len(train_idx)}")

    print(f"[data] train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")


    feat_cfg = config.get("features", {})

    physics, feature_names = extract_physics_features(

        bundle.x_raw_dbm,

        bundle.wavelength_nm,

        num_dips=int(feat_cfg.get("num_dips", 6)),

        num_bands=int(feat_cfg.get("num_bands", 8)),

        tracked_centers_nm=feat_cfg.get("tracked_centers_nm", [1599.0]),

        tracked_half_window_nm=float(feat_cfg.get("tracked_half_window_nm", 35.0)),

    )

    phys_mean, phys_std = fit_feature_standardizer(physics[train_idx])

    phys_train = apply_feature_standardizer(physics[train_idx], phys_mean, phys_std)

    phys_val = apply_feature_standardizer(physics[val_idx], phys_mean, phys_std)

    phys_test = apply_feature_standardizer(physics[test_idx], phys_mean, phys_std)


    trough_indices = [

        i for i, n in enumerate(feature_names)

        if n.startswith("tracked_dip_") and n.endswith("_wavelength_nm")

    ]

    print(f"[physics features] dimension={physics.shape[1]}, trough indices={trough_indices}")


    spectrum_train = bundle.x[train_idx].astype(np.float32)
    spectrum_val = bundle.x[val_idx].astype(np.float32)
    spectrum_test = bundle.x[test_idx].astype(np.float32)
    (spectrum_train_norm, spectrum_val_norm, spectrum_test_norm), spectrum_mean, spectrum_std = standardize_spectrum(
        spectrum_train, spectrum_val, spectrum_test
    )


    y_results = standardize_labels(bundle.y[train_idx], bundle.y[val_idx], bundle.y[test_idx])

    y_train_norm, y_val_norm, y_test_norm = y_results[:3]

    y_mean, y_std = y_results[3], y_results[4]


    np.savez(

        Path(output_dir) / "normalization.npz",

        spectrum_mean=spectrum_mean, spectrum_std=spectrum_std,

        y_mean=y_mean, y_std=y_std,

        phys_mean=phys_mean, phys_std=phys_std,

    )



    pretrain_cfg = config.get("pretrain", {})



    gan_synthetic_path = pretrain_cfg.get("gan_synthetic_path")


    if gan_synthetic_path:


        payload = np.load(gan_synthetic_path, allow_pickle=False)

        if "x_spectrum" not in payload or "y" not in payload:

            raise ValueError("GAN augmentation file must contain x_spectrum and y")

        x_synth_spectrum = np.asarray(payload["x_spectrum"], dtype=np.float32)

        y_synth_raw = np.asarray(payload["y"], dtype=np.float32)

        if x_synth_spectrum.ndim != 2 or x_synth_spectrum.shape[1] != bundle.x.shape[1]:
            raise ValueError("GAN synthetic spectra must use the configured resampled wavelength grid")

        if y_synth_raw.shape != (len(x_synth_spectrum), 2):

            raise ValueError("GAN synthetic labels must have shape [n, 2]")

        n_synthetic = len(x_synth_spectrum)

        spectrum_synth_norm = ((x_synth_spectrum - spectrum_mean) / spectrum_std).astype(np.float32)

        phys_synth_raw, _ = extract_physics_features(

            x_synth_spectrum, bundle.input_wavelength_nm,

            num_dips=int(feat_cfg.get("num_dips", 6)),

            num_bands=int(feat_cfg.get("num_bands", 8)),

            tracked_centers_nm=feat_cfg.get("tracked_centers_nm", [1599.0]),

            tracked_half_window_nm=float(feat_cfg.get("tracked_half_window_nm", 35.0)),

        )

        phys_synth = apply_feature_standardizer(phys_synth_raw, phys_mean, phys_std)

        if bool(pretrain_cfg.get("audit_gan_synthetic_quality", True)):

            tracked_wavelength_idx = [i for i, name in enumerate(feature_names)

                                      if name.startswith("tracked_dip_") and name.endswith("_wavelength_nm")]

            tracked_distribution_idx = [i for i, name in enumerate(feature_names)

                                        if name.startswith("tracked_dip_") and

                                        (name.endswith("_wavelength_nm") or name.endswith("_intensity"))]

            forward_calibrator = fit_forward_trough_calibrator(bundle.y[train_idx], physics[train_idx], feature_names)

            observed_wavelengths = phys_synth_raw[:, tracked_wavelength_idx]

            expected_wavelengths = predict_forward_troughs(y_synth_raw, forward_calibrator["coefficients"])

            audit_points = min(128, spectrum_train_norm.shape[1])
            audit_indices = np.linspace(
                0, spectrum_train_norm.shape[1] - 1, audit_points
            ).round().astype(int)
            audit_real = spectrum_train_norm[:, audit_indices]
            audit_synthetic = spectrum_synth_norm[:, audit_indices]


            quality = synthetic_quality_report(

                audit_real, audit_synthetic,

                physics[train_idx][:, tracked_distribution_idx], phys_synth_raw[:, tracked_distribution_idx],

                observed_synthetic_wavelengths=observed_wavelengths,

                expected_synthetic_wavelengths=expected_wavelengths,

            )

            quality["condition_in_train_range_fraction"] = float(np.logical_and(

                y_synth_raw >= bundle.y[train_idx].min(axis=0),

                y_synth_raw <= bundle.y[train_idx].max(axis=0),

            ).all(axis=1).mean())

            gate_cfg = pretrain_cfg.get("synthetic_quality_gate", {}) or {}

            if bool(gate_cfg.get("enabled", False)):

                accepted, gate_audit = synthetic_acceptance_mask(

                    audit_real, audit_synthetic, observed_wavelengths, expected_wavelengths,

                    max_conditional_trough_mae_nm=float(

                        gate_cfg.get("max_conditional_trough_mae_nm", 8.0)

                    ),

                    manifold_percentile=float(gate_cfg.get("manifold_percentile", 95.0)),

                )

                quality["quality_gate"] = gate_audit

                synthetic_weight_cfg = float(pretrain_cfg.get("synthetic_weight", 0.05))

                if synthetic_weight_cfg <= 0:

                    raise ValueError("synthetic_weight must be positive when a GAN quality gate is enabled")


                has_explicit_target = "target_accepted_samples" in gate_cfg

                target_accepted = int(gate_cfg.get(

                    "target_accepted_samples",

                    round(len(spectrum_train_norm) / synthetic_weight_cfg),

                ))

                if target_accepted <= 0:

                    raise ValueError("target_accepted_samples must be positive")

                min_accepted = (

                    target_accepted if has_explicit_target

                    else max(int(gate_cfg.get("min_accepted_samples", 0)), target_accepted)

                )

                quality["quality_gate"]["target_accepted_count_for_one_to_one_weight"] = target_accepted

                if int(accepted.sum()) >= min_accepted:

                    accepted_idx = np.flatnonzero(accepted)

                    if len(accepted_idx) > target_accepted:

                        rng = np.random.default_rng(seed + 42017)

                        accepted_idx = np.sort(rng.choice(accepted_idx, size=target_accepted, replace=False))

                    x_synth_spectrum = x_synth_spectrum[accepted_idx]

                    y_synth_raw = y_synth_raw[accepted_idx]

                    spectrum_synth_norm = spectrum_synth_norm[accepted_idx]

                    phys_synth_raw = phys_synth_raw[accepted_idx]

                    phys_synth = phys_synth[accepted_idx]

                    n_synthetic = len(x_synth_spectrum)

                    quality["quality_gate"]["fallback_to_real_only"] = False

                    print(f"[GAN quality gate] accepted {n_synthetic}/{len(accepted)} spectra")

                else:


                    n_synthetic = 0

                    quality["quality_gate"]["fallback_to_real_only"] = True

                    quality["quality_gate"]["minimum_accepted_samples"] = min_accepted

                    print(

                        "[GAN quality gate] insufficient accepted spectra; "

                        "falling back to real-only pretraining"

                    )

            write_json(Path(output_dir) / "synthetic_quality.json", quality)

        print(f"[GAN augmentation] loaded {n_synthetic} spectra from {gan_synthetic_path}")


    print("\n" + "=" * 60)

    print("Phase 3: Physics-Guided HeterogeneousMoE pretraining")

    print("=" * 60)


    if n_synthetic > 0:

        y_synth_norm = ((y_synth_raw - y_mean) / y_std).astype(np.float32)

        spectrum_all_norm = np.concatenate([spectrum_train_norm, spectrum_synth_norm], axis=0)

        phys_all = np.concatenate([phys_train, phys_synth], axis=0)

        y_all_norm = np.concatenate([y_train_norm, y_synth_norm], axis=0)

    else:

        spectrum_all_norm = spectrum_train_norm

        phys_all = phys_train

        y_all_norm = y_train_norm


    moe_cfg = config.get("heterogeneous_moe", {})


    decouple_temp_pretrain = pretrain_cfg.get("decouple_temperature_pretrain", False)

    moe_cfg_for_pretrain = dict(moe_cfg)

    moe_cfg_for_pretrain["decouple_temperature"] = bool(decouple_temp_pretrain)

    if not decouple_temp_pretrain:

        print("[Phase 3] decouple_temperature=False (temperature loss updates the backbone)")


    _uses_raw_spectrum = int(moe_cfg.get("temp_context_out_dim", 0)) > 0 or (

        any(str(t).lower() in {"mamba", "transformer"} for t in moe_cfg.get("expert_types", []))

    )

    if _uses_raw_spectrum:

        if n_synthetic > 0:

            raw_spectrum_all = np.concatenate(

                [bundle.x[train_idx].astype(np.float32), x_synth_spectrum.astype(np.float32)],

                axis=0,

            )

        else:

            raw_spectrum_all = bundle.x[train_idx].astype(np.float32)

    else:

        raw_spectrum_all = None


    if n_synthetic > 0:

        print(f"  training samples: real={len(spectrum_train_norm)}, synthetic={len(spectrum_synth_norm)}, total={len(spectrum_all_norm)}")

        synthetic_weight = float(pretrain_cfg.get("synthetic_weight", 0.25))

        sample_weight = np.concatenate([

            np.ones(len(spectrum_train_norm), dtype=np.float32),

            np.full(len(spectrum_synth_norm), synthetic_weight, dtype=np.float32),

        ])

    else:

        print(f"  training samples: real={len(spectrum_train_norm)} (no synthetic data)")

        synthetic_weight = 0.0

        sample_weight = np.ones(len(spectrum_train_norm), dtype=np.float32)


    if n_synthetic > 0:

        real_total_weight = float(len(spectrum_train_norm))

        synthetic_total_weight = float(len(spectrum_synth_norm)) * synthetic_weight

        synthetic_to_real_weight_ratio = (

            synthetic_total_weight / real_total_weight if real_total_weight > 0 else float("nan")

        )

        print(

            f"  synthetic_weight={synthetic_weight}  "

            f"real_total_weight={real_total_weight:.1f}  "

            f"synthetic_total_weight={synthetic_total_weight:.1f}  "

            f"synthetic/real ratio={synthetic_to_real_weight_ratio:.2f}"

        )

    else:

        real_total_weight = float(len(spectrum_train_norm))

        synthetic_total_weight = 0.0

        synthetic_to_real_weight_ratio = float("nan")


    moe_model = pretrain_moe(

        spectrum_all_norm, phys_all, y_all_norm, sample_weight,

        trough_indices, moe_cfg_for_pretrain, pretrain_cfg,

        Path(output_dir), device,

        raw_spectrum_all=raw_spectrum_all,

    )


    import torch

    from torch.utils.data import DataLoader as DL

    moe_model.eval()

    val_ds = SpectralDataset(

        spectrum_val_norm, phys_val, y_val_norm,

        raw_spectrum=bundle.x[val_idx] if _uses_raw_spectrum else None,

    )

    val_loader = DL(val_ds, batch_size=32, shuffle=False)

    y_preds, y_trues = [], []

    with torch.no_grad():

        for batch in val_loader:

            raw = batch.get("raw")

            if raw is not None:

                raw = raw.to(device)

            out = moe_model(batch["z"].to(device), batch["physics"].to(device), raw_spectrum=raw)

            y_preds.append(out["prediction"].cpu().numpy())

            y_trues.append(batch["y"].numpy())

    if y_preds:

        y_pred_s = np.concatenate(y_preds) * y_std + y_mean

        y_true_s = np.concatenate(y_trues) * y_std + y_mean

        metrics = regression_metrics(y_true_s, y_pred_s, bundle.target_names)

        print(f"[validation metrics] {metrics}")

    else:

        metrics = {}

        print("[validation] skipped because the validation split is empty")


    write_json(Path(output_dir) / "pretrain_metrics.json", {

        "val_metrics": metrics,

        "spectrum_length": int(bundle.x.shape[1]),

        "n_synthetic": n_synthetic,

        "synthetic_weight": synthetic_weight,

        "real_total_weight": real_total_weight,

        "synthetic_total_weight": synthetic_total_weight,

        "synthetic_to_real_weight_ratio": synthetic_to_real_weight_ratio,

        "trough_indices": trough_indices,

        "split": split_meta,

    })

    print(f"\n[complete] pretraining results saved to {output_dir}")


if __name__ == "__main__":

    main()
