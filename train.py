from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np

from data import (
    TorchRegressionDataset,
    TorchSpectrumDataset,
    ensure_dir,
    load_config,
    load_spectrum_bundle,
    resolve_split_seed,
    set_seed,
    split_from_config,
    subsample_train_indices,
    write_json,
)
from models.gan import (
    AntiResonanceConfig,
    AntiResonancePINN,
    ConditionalCritic,
    ConditionalGenerator,
    antiresonance_trough_loss,
    calibrate_antiresonance_prior,
    gradient_penalty,
    smoothness_loss,
    soft_trough_locations,
)
from models.moe import (
    HeterogeneousMoE,
    apply_adapter_to_model,
    count_parameters,
    freeze_non_adapter,
)
from physics import (
    apply_feature_standardizer,
    design_matrix,
    extract_physics_features,
    fit_feature_standardizer,
    fit_forward_trough_calibrator,
    predict_forward_troughs,
    real_manifold_distances,
    select_cmi_pcqd,
    synthetic_acceptance_mask,
    synthetic_quality_report,
)

ROOT = Path(__file__).resolve().parent


def _project_path(value: str | Path) -> Path:
    """Resolve repository-relative paths independently of the caller's cwd."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def train_gan_stage(args) -> None:

    try:

        import torch

        from torch.utils.data import DataLoader

    except ImportError as exc:

        raise ImportError("train_gan requires PyTorch. Install torch first.") from exc



    if args.generate_only:

        if not args.checkpoint or not args.output:

            raise ValueError("--generate-only requires --checkpoint and --output")

        generate_synthetic(args.config, args.checkpoint, args.output, args.n_synthetic)

        return


    config = load_config(args.config)

    seed = int(config.get("seed", 42))

    set_seed(seed)

    bundle = load_spectrum_bundle(config)

    gan_cfg = config.get("gan", {})

    if not bool(gan_cfg.get("enabled", False)) and not args.force:

        print("GAN training skipped because gan.enabled is false. Pass --force to override.")

        return

    output_dir = ensure_dir(_project_path(gan_cfg.get("output_dir", "runs/gan")))


    split_cfg = config.get("data", {}).get("split", {})

    split_seed = resolve_split_seed(split_cfg, seed)

    train_idx, _, split_meta = split_from_config(

        len(bundle.y),

        seed=split_seed,

        split_cfg=split_cfg,

        labels=bundle.labels,

        x_raw_dbm=bundle.x_raw_dbm,

    )

    split_meta["seed"] = split_seed
    train_fraction = float(split_cfg.get("train_fraction", 1.0))
    train_idx = subsample_train_indices(
        train_idx, fraction=train_fraction, seed=split_seed + 20000
    )
    split_meta["train_fraction"] = train_fraction


    condition = bundle.y[train_idx].astype(np.float32)

    condition_mean = condition.mean(axis=0, keepdims=True)

    condition_std = condition.std(axis=0, keepdims=True)

    condition_std[condition_std == 0] = 1.0

    condition = (condition - condition_mean) / condition_std


    train_spectra = bundle.x[train_idx].astype(np.float32)

    representation = "resampled_spectrum"
    representation_train = train_spectra

    spectrum_mean = np.asarray(representation_train.mean(axis=0, keepdims=True), dtype=np.float32)

    spectrum_std = np.asarray(representation_train.std(axis=0, keepdims=True), dtype=np.float32)

    spectrum_std[spectrum_std < 1e-6] = 1.0

    normalized_train_spectra = (representation_train - spectrum_mean) / spectrum_std

    dataset = TorchSpectrumDataset(normalized_train_spectra, condition)

    loader = DataLoader(dataset, batch_size=int(gan_cfg.get("batch_size", 16)), shuffle=True, drop_last=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spectrum_mean_t = torch.as_tensor(spectrum_mean, device=device, dtype=torch.float32)

    spectrum_std_t = torch.as_tensor(spectrum_std, device=device, dtype=torch.float32)

    latent_dim = int(gan_cfg.get("latent_dim", 64))
    generator = ConditionalGenerator(latent_dim, condition_dim=2, output_length=bundle.x.shape[1]).to(device)
    critic = ConditionalCritic(condition_dim=2).to(device)

    learning_rate = float(gan_cfg.get("learning_rate", 1e-4))

    g_opt = torch.optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

    c_opt = torch.optim.AdamW(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))


    critic_steps = int(gan_cfg.get("critic_steps", 5))

    gp_weight = float(gan_cfg.get("gradient_penalty_weight", 10.0))

    smooth_weight = float(gan_cfg.get("smoothness_weight", 0.001))

    moment_weight = float(gan_cfg.get("moment_weight", 0.0))


    pinn_cfg = gan_cfg.get("pinn", {})

    pinn_enabled = bool(pinn_cfg.get("enabled", False))

    physics = None

    wavelength_grid = None

    soft_coefficients = None

    pinn_calibration = None

    strict_weight = 0.0

    if pinn_enabled:

        centers = list(pinn_cfg.get("tracked_centers_nm", []))

        if len(centers) != 2:

            raise ValueError(
                "gan.pinn requires exactly two ordered tracked_centers_nm values "
                "for the joint dip-1/dip-2 physics constraint"
            )

        nominal_thickness = float(pinn_cfg.get("wall_thickness_um", 26.5))

        if bool(pinn_cfg.get("auto_calibrate", True)):

            pinn_calibration = calibrate_antiresonance_prior(

                centers,

                reference_temperature_c=float(

                    pinn_cfg.get("reference_temperature_c", np.median(bundle.y[train_idx, 0]))

                ),

                reference_salinity_ppt=float(

                    pinn_cfg.get("reference_salinity_ppt", np.median(bundle.y[train_idx, 1]))

                ),

                nominal_wall_thickness_um=nominal_thickness,

                candidate_orders=range(

                    int(pinn_cfg.get("candidate_order_min", 10)),

                    int(pinn_cfg.get("candidate_order_max", 31)) + 1,

                ),

                fixed_point_steps=int(pinn_cfg.get("fixed_point_steps", 12)),

            )

            orders = tuple(pinn_calibration["orders"])

            calibrated_thickness = float(pinn_calibration["wall_thickness_um"])

            max_error = float(pinn_cfg.get("max_calibration_error_nm", 5.0))

            max_deviation = float(pinn_cfg.get("max_nominal_thickness_deviation_um", 2.0))

            strict_active = (

                pinn_calibration["max_center_error_nm"] <= max_error

                and abs(pinn_calibration["nominal_thickness_deviation_um"]) <= max_deviation

            )

            pinn_calibration.update({

                "strict_active": bool(strict_active),

                "max_calibration_error_nm": max_error,

                "max_nominal_thickness_deviation_um": max_deviation,

            })

        else:

            orders = tuple(int(v) for v in pinn_cfg.get("resonance_orders", []))

            if len(centers) != len(orders):

                raise ValueError("gan.pinn requires matching tracked_centers_nm and resonance_orders")

            calibrated_thickness = nominal_thickness

            strict_active = bool(pinn_cfg.get("allow_uncalibrated_strict", False))

            pinn_calibration = {

                "orders": list(orders), "wall_thickness_um": calibrated_thickness,

                "strict_active": strict_active, "auto_calibrate": False,

            }

        strict_weight = float(pinn_cfg.get("strict_weight", 0.1)) if strict_active else 0.0

        pinn_calibration["configured_strict_weight"] = float(pinn_cfg.get("strict_weight", 0.1))

        pinn_calibration["effective_strict_weight"] = strict_weight

        write_json(Path(output_dir) / "pinn_calibration.json", pinn_calibration)

        print(

            "[PINN calibration] "

            f"orders={list(orders)} wall={calibrated_thickness:.4f} um "

            f"strict_active={strict_active}"

        )

        physics = AntiResonancePINN(

            AntiResonanceConfig(

                wall_thickness_um=calibrated_thickness,

                resonance_orders=orders,

                salinity_to_percent=float(pinn_cfg.get("salinity_to_percent", 0.1)),

                max_thickness_correction_um=float(pinn_cfg.get("max_thickness_correction_um", 0.5)),

                max_cladding_index_correction=float(pinn_cfg.get("max_cladding_index_correction", 0.01)),

                fixed_point_steps=int(pinn_cfg.get("fixed_point_steps", 12)),

            )

        ).to(device)


        soft_coefficients = fit_empirical_soft_guide(

            bundle.x[train_idx], bundle.y[train_idx], bundle.input_wavelength_nm, centers,

            half_window_nm=float(pinn_cfg.get("half_window_nm", 35.0)),

        )

        soft_coefficients = torch.from_numpy(soft_coefficients).to(device=device, dtype=torch.float32)

        wavelength_grid = torch.as_tensor(bundle.input_wavelength_nm, device=device, dtype=torch.float32)

        g_opt.add_param_group({"params": physics.parameters()})


    for epoch in range(1, int(gan_cfg.get("epochs", 300)) + 1):

        c_losses = []

        g_losses = []

        for batch in loader:

            real = batch["x"].to(device)

            cond = batch["y"].to(device)

            for _ in range(critic_steps):

                z = torch.randn(real.shape[0], latent_dim, device=device)

                fake = generator(z, cond).detach()

                c_loss = critic(fake, cond).mean() - critic(real, cond).mean()

                c_loss = c_loss + gp_weight * gradient_penalty(critic, real, fake, cond)

                c_opt.zero_grad()

                c_loss.backward()

                c_opt.step()

            z = torch.randn(real.shape[0], latent_dim, device=device)

            fake = generator(z, cond)

            g_loss = -critic(fake, cond).mean()

            fake_raw_for_regularizers = fake * spectrum_std_t + spectrum_mean_t

            g_loss = g_loss + smooth_weight * smoothness_loss(fake_raw_for_regularizers)

            if moment_weight > 0:

                g_loss = g_loss + moment_weight * moment_matching_loss(fake, real)

            if physics is not None:

                raw_cond = cond * torch.as_tensor(condition_std, device=device) + torch.as_tensor(condition_mean, device=device)

                fake_raw = fake_raw_for_regularizers

                # [batch, 2]: differentiable locations of dip 1 and dip 2.
                observed = soft_trough_locations(

                    fake_raw, wavelength_grid, pinn_cfg["tracked_centers_nm"],

                    half_window_nm=float(pinn_cfg.get("half_window_nm", 35.0)),

                    temperature=float(pinn_cfg.get("softargmin_temperature", 0.08)),

                )

                strict_target = physics.predicted_troughs_nm(raw_cond[:, 0], raw_cond[:, 1])

                strict_loss = antiresonance_trough_loss(

                    observed, strict_target, scale_nm=float(pinn_cfg.get("strict_scale_nm", 1.0))

                )

                soft_target = empirical_soft_targets(raw_cond, soft_coefficients)

                soft_loss = antiresonance_trough_loss(

                    observed, soft_target, scale_nm=float(pinn_cfg.get("soft_scale_nm", 1.0))

                )

                g_loss = g_loss + strict_weight * strict_loss

                g_loss = g_loss + float(pinn_cfg.get("empirical_soft_weight", 0.01)) * soft_loss

            g_opt.zero_grad()

            g_loss.backward()

            g_opt.step()

            c_losses.append(float(c_loss.detach().cpu()))

            g_losses.append(float(g_loss.detach().cpu()))

        print(f"epoch={epoch} critic={np.mean(c_losses):.6f} generator={np.mean(g_losses):.6f}")

        if epoch % 50 == 0:

            torch.save(

                _checkpoint_payload(generator, critic, physics, soft_coefficients, config,

                                    split_meta, condition_mean, condition_std, spectrum_mean, spectrum_std,

                                    representation),

                Path(output_dir) / f"gan_epoch_{epoch}.pt",

            )

    torch.save(

        _checkpoint_payload(generator, critic, physics, soft_coefficients, config,

                            split_meta, condition_mean, condition_std, spectrum_mean, spectrum_std,

                            representation),

        Path(output_dir) / "gan_final.pt",

    )


def generate_synthetic(config_path: str, checkpoint_path: str, output_path: str, n_synthetic: int | None = None) -> None:

    import torch

    config = load_config(config_path)

    if bool(config.get("data", {}).get("use_zscore", True)):

        raise ValueError("GAN-to-MoE chain requires data.use_zscore=false so generated spectra are in raw dBm")

    seed = int(config.get("seed", 42))

    set_seed(seed)

    bundle = load_spectrum_bundle(config)

    gan_cfg = config.get("gan", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    latent_dim = int(gan_cfg.get("latent_dim", 64))

    if str(checkpoint.get("representation", "")).lower() != "resampled_spectrum":

        raise ValueError("checkpoint was not trained on the configured resampled spectrum")

    generator = ConditionalGenerator(latent_dim, 2, bundle.x.shape[1]).to(device)

    generator.load_state_dict(checkpoint["generator"], strict=True)

    generator.eval()

    n = n_synthetic or int(gan_cfg.get("n_synthetic", 2000))

    batch_size = int(gan_cfg.get("sample_batch_size", 64))

    rng = np.random.default_rng(seed + 30000)

    split_cfg = config.get("data", {}).get("split", {})

    split_seed = resolve_split_seed(split_cfg, seed)

    train_idx, _, _ = split_from_config(

        len(bundle.y), seed=split_seed, split_cfg=split_cfg,

        labels=bundle.labels, x_raw_dbm=bundle.x_raw_dbm,

    )

    train_idx = subsample_train_indices(

        train_idx,

        fraction=float(split_cfg.get("train_fraction", 1.0)),

        seed=split_seed + 20000,

    )

    y = rng.uniform(bundle.y[train_idx].min(axis=0), bundle.y[train_idx].max(axis=0), size=(n, 2)).astype(np.float32)

    cond_mean = np.asarray(checkpoint["condition_mean"], dtype=np.float32)

    cond_std = np.asarray(checkpoint["condition_std"], dtype=np.float32)

    if "spectrum_mean" not in checkpoint or "spectrum_std" not in checkpoint:

        raise ValueError("checkpoint lacks train-split spectrum normalization statistics")

    spectrum_mean = torch.as_tensor(checkpoint["spectrum_mean"], device=device, dtype=torch.float32)

    spectrum_std = torch.as_tensor(checkpoint["spectrum_std"], device=device, dtype=torch.float32)

    generated = []

    with torch.no_grad():

        for start in range(0, n, batch_size):

            raw = y[start:start + batch_size]

            cond = torch.from_numpy((raw - cond_mean) / cond_std).to(device)

            z = torch.randn(len(raw), latent_dim, device=device)

            generated_normalized = generator(z, cond)

            generated_representation = generated_normalized * spectrum_std + spectrum_mean

            generated.append(generated_representation.squeeze(1).cpu().numpy().astype(np.float32))

    output = Path(output_path)

    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output, x_spectrum=np.concatenate(generated), y=y, seed=seed)

    print(f"saved {n} generated resampled spectra to {output}")


def moment_matching_loss(fake, real):

    fake_flat = fake.flatten(1)

    real_flat = real.flatten(1)

    fake_mean = fake_flat.mean(dim=1)

    real_mean = real_flat.mean(dim=1)

    fake_std = fake_flat.std(dim=1)

    real_std = real_flat.std(dim=1)

    return ((fake_mean - real_mean) ** 2).mean() + ((fake_std - real_std) ** 2).mean()


def _checkpoint_payload(generator, critic, physics, soft_coefficients, config, split_meta,

                        condition_mean, condition_std, spectrum_mean, spectrum_std,

                        representation):

    return {

        "generator": generator.state_dict(),

        "critic": critic.state_dict(),

        "physics_pinn": None if physics is None else physics.state_dict(),

        "empirical_soft_coefficients": None if soft_coefficients is None else soft_coefficients.detach().cpu(),

        "config": config,

        "split": split_meta,

        "condition_mean": condition_mean,

        "condition_std": condition_std,

        "spectrum_mean": spectrum_mean,

        "spectrum_std": spectrum_std,

        "representation": representation,


    }


def fit_empirical_soft_guide(spectra, labels, wavelengths_nm, centers_nm, half_window_nm=35.0):

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)

    troughs = []

    for center in centers_nm:

        mask = np.abs(wavelengths_nm - float(center)) <= half_window_nm

        if int(mask.sum()) < 3:

            raise ValueError("empirical trough window is empty")

        local = np.asarray(spectra)[:, mask]

        troughs.append(wavelengths_nm[mask][np.argmin(local, axis=1)])

    design = design_matrix(labels)

    return np.linalg.lstsq(design, np.stack(troughs, axis=1), rcond=None)[0].T.astype(np.float32)


def empirical_soft_targets(temperature_salinity, coefficients):

    import torch

    t, sal = temperature_salinity[:, 0], temperature_salinity[:, 1]

    design = torch.stack([torch.ones_like(t), t, t.square(), sal, t * sal], dim=-1)

    return design @ coefficients.T

def standardize_labels(
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    mean = y_train.mean(axis=0, keepdims=True)

    std = y_train.std(axis=0, keepdims=True)

    std[std < 1e-8] = 1.0

    scaled = ((y_train - mean) / std).astype(np.float32)

    return scaled, mean.astype(np.float32), std.astype(np.float32)


def standardize_spectrum(
    spectrum_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    mean = spectrum_train.mean(axis=0, keepdims=True)

    std = spectrum_train.std(axis=0, keepdims=True)

    std[std < 1e-8] = 1.0

    scaled = ((spectrum_train - mean) / std).astype(np.float32)

    return scaled, mean.astype(np.float32), std.astype(np.float32)


def load_balance_loss_fn(route_weights: "torch.Tensor") -> "torch.Tensor":

    import torch

    n_experts = route_weights.shape[-1]

    expert_frac = route_weights.mean(dim=0)

    return torch.sum(expert_frac * torch.softmax(expert_frac, dim=0)) * n_experts


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


    dataset = TorchRegressionDataset(

        spectrum_all_norm, phys_all, y_all_norm, sample_weight,

        raw_spectrum=raw_spectrum_all,

    )

    loader = DataLoader(

        dataset,

        batch_size=int(pretrain_cfg.get("batch_size", 32)),

        shuffle=True,

        drop_last=True,

    )


    model = HeterogeneousMoE(

        spectrum_dim=spectrum_all_norm.shape[1],

        phys_dim=phys_all.shape[1],

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


def pretrain_stage(args) -> None:

    try:

        import torch

    except ImportError as exc:

        raise ImportError("pretrain_moe requires PyTorch.") from exc




    config = load_config(args.config)

    seed = int(config.get("seed", 42))

    set_seed(seed)


    output_dir_str = args.output_dir or _project_path(
        config.get("pretrain", {}).get("output_dir", "runs/diffusion_pretrain")
    )

    output_dir = ensure_dir(output_dir_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[device] {device}")


    bundle = load_spectrum_bundle(config)

    split_cfg = config.get("data", {}).get("split", {})

    split_seed = resolve_split_seed(split_cfg, seed)

    train_idx, val_idx, split_meta = split_from_config(

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

    print(f"[data] train={len(train_idx)}, validation={len(val_idx)}")


    feat_cfg = config.get("features", {})

    pinn_cfg = config.get("gan", {}).get("pinn", {})
    pinn_centers = list(pinn_cfg.get("tracked_centers_nm", []))
    feature_centers = list(feat_cfg.get("tracked_centers_nm", []))
    if bool(pinn_cfg.get("enabled", False)) and (
        len(feature_centers) != 2 or feature_centers != pinn_centers
    ):
        raise ValueError(
            "features.tracked_centers_nm and gan.pinn.tracked_centers_nm must be the "
            "same ordered two-dip vector for dual-dip PCST screening"
        )

    physics, feature_names = extract_physics_features(

        bundle.x_raw_dbm,

        bundle.wavelength_nm,

        num_dips=int(feat_cfg.get("num_dips", 6)),

        num_bands=int(feat_cfg.get("num_bands", 8)),

        tracked_centers_nm=feat_cfg.get("tracked_centers_nm", [1516.0, 1599.0]),

        tracked_half_window_nm=float(feat_cfg.get("tracked_half_window_nm", 35.0)),

    )

    phys_mean, phys_std = fit_feature_standardizer(physics[train_idx])

    phys_train = apply_feature_standardizer(physics[train_idx], phys_mean, phys_std)


    trough_indices = [

        i for i, n in enumerate(feature_names)

        if n.startswith("tracked_dip_") and n.endswith("_wavelength_nm")

    ]

    print(f"[physics features] dimension={physics.shape[1]}, trough indices={trough_indices}")


    spectrum_train = bundle.x[train_idx].astype(np.float32)
    spectrum_train_norm, spectrum_mean, spectrum_std = standardize_spectrum(spectrum_train)

    y_train_norm, y_mean, y_std = standardize_labels(bundle.y[train_idx])


    np.savez(

        Path(output_dir) / "normalization.npz",

        spectrum_mean=spectrum_mean, spectrum_std=spectrum_std,

        y_mean=y_mean, y_std=y_std,

        phys_mean=phys_mean, phys_std=phys_std,

    )



    pretrain_cfg = config.get("pretrain", {})



    gan_synthetic_path = pretrain_cfg.get("gan_synthetic_path")
    if gan_synthetic_path:
        gan_synthetic_path = _project_path(gan_synthetic_path)

    n_synthetic = 0
    synthetic_confidence = None


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

        synthetic_confidence = np.ones(n_synthetic, dtype=np.float32)

        spectrum_synth_norm = ((x_synth_spectrum - spectrum_mean) / spectrum_std).astype(np.float32)

        phys_synth_raw, _ = extract_physics_features(

            x_synth_spectrum, bundle.input_wavelength_nm,

            num_dips=int(feat_cfg.get("num_dips", 6)),

            num_bands=int(feat_cfg.get("num_bands", 8)),

            tracked_centers_nm=feat_cfg.get("tracked_centers_nm", [1516.0, 1599.0]),

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

                nearest_manifold_dist_syn, manifold_radius = real_manifold_distances(
                    audit_real,
                    audit_synthetic,
                    percentile=float(gate_cfg.get("manifold_percentile", 95.0)),
                )
                accepted, gate_audit = synthetic_acceptance_mask(

                    audit_real, audit_synthetic, observed_wavelengths, expected_wavelengths,

                    max_conditional_trough_mae_nm=float(

                        gate_cfg.get("max_conditional_trough_mae_nm", 8.0)

                    ),

                    manifold_percentile=float(gate_cfg.get("manifold_percentile", 95.0)),
                    precomputed_nearest=nearest_manifold_dist_syn,
                    precomputed_radius=manifold_radius,

                )

                quality["quality_gate"] = gate_audit

                synthetic_weight_cfg = float(pretrain_cfg.get("synthetic_weight", 1.0))

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

                synthetic_confidence = None
                if int(accepted.sum()) >= min_accepted:

                    accepted_idx = np.flatnonzero(accepted)

                    selector_cfg = gate_cfg.get("cmi_pcqd_selector", {}) or {}
                    if bool(selector_cfg.get("enabled", False)):
                        trough_mae = np.mean(
                            np.abs(observed_wavelengths - expected_wavelengths), axis=1
                        )
                        selection = select_cmi_pcqd(
                            F_real=audit_real,
                            y_real=bundle.y[train_idx],
                            z_real_norm=spectrum_train_norm,
                            F_syn=audit_synthetic,
                            y_syn=y_synth_raw,
                            z_syn_norm=spectrum_synth_norm,
                            trough_mae_syn_nm=trough_mae,
                            nearest_manifold_dist_syn=nearest_manifold_dist_syn,
                            manifold_radius=manifold_radius,
                            max_conditional_trough_mae_nm=float(
                                selector_cfg.get("soft_trough_mae_nm", 8.0)
                            ),
                            hard_reject_trough_mae_nm=float(
                                selector_cfg.get("hard_reject_trough_mae_nm", 16.0)
                            ),
                            target_count=target_accepted,
                            min_accepted_samples=min_accepted,
                            condition_bins=tuple(selector_cfg.get("condition_bins", [4, 4])),
                            diversity_weight=float(selector_cfg.get("diversity_weight", 0.30)),
                            min_per_nonempty_bin=int(selector_cfg.get("min_per_nonempty_bin", 50)),
                            knn_k=int(selector_cfg.get("knn_k", 7)),
                            tau_quantile=float(selector_cfg.get("tau_quantile", 0.90)),
                            w_phys=float(selector_cfg.get("w_phys", 0.20)),
                            w_manifold=float(selector_cfg.get("w_manifold", 0.15)),
                            w_temp=float(selector_cfg.get("w_temp", 0.20)),
                            w_sal=float(selector_cfg.get("w_sal", 0.45)),
                            confidence_clip=tuple(selector_cfg.get("confidence_clip", [0.25, 2.0])),
                            seed=seed,
                        )
                        accepted_idx = selection.selected_idx
                        synthetic_confidence = selection.confidence
                        quality["cmi_pcqd_selector"] = selection.audit
                    elif len(accepted_idx) > target_accepted:

                        rng = np.random.default_rng(seed + 42017)

                        accepted_idx = np.sort(rng.choice(accepted_idx, size=target_accepted, replace=False))

                    x_synth_spectrum = x_synth_spectrum[accepted_idx]

                    y_synth_raw = y_synth_raw[accepted_idx]

                    spectrum_synth_norm = spectrum_synth_norm[accepted_idx]

                    phys_synth_raw = phys_synth_raw[accepted_idx]

                    phys_synth = phys_synth[accepted_idx]

                    if synthetic_confidence is None:
                        synthetic_confidence = np.ones(len(accepted_idx), dtype=np.float32)
                    else:
                        synthetic_confidence = synthetic_confidence.astype(np.float32)

                    n_synthetic = len(x_synth_spectrum)

                    quality["quality_gate"]["fallback_to_real_only"] = False

                    print(f"[GAN quality gate] accepted {n_synthetic}/{len(accepted)} spectra")

                else:


                    n_synthetic = 0

                    synthetic_confidence = None

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


    synthetic_only = bool(pretrain_cfg.get("synthetic_only", False))
    if synthetic_only and n_synthetic <= 0:
        raise RuntimeError(
            "pretrain.synthetic_only=true requires PCST-selected synthetic spectra; "
            "the quality gate retained none"
        )

    if n_synthetic > 0 and synthetic_only:

        y_synth_norm = ((y_synth_raw - y_mean) / y_std).astype(np.float32)

        spectrum_all_norm = spectrum_synth_norm

        phys_all = phys_synth

        y_all_norm = y_synth_norm

    elif n_synthetic > 0:

        y_synth_norm = ((y_synth_raw - y_mean) / y_std).astype(np.float32)

        spectrum_all_norm = np.concatenate([spectrum_train_norm, spectrum_synth_norm], axis=0)

        phys_all = np.concatenate([phys_train, phys_synth], axis=0)

        y_all_norm = np.concatenate([y_train_norm, y_synth_norm], axis=0)

    else:

        spectrum_all_norm = spectrum_train_norm

        phys_all = phys_train

        y_all_norm = y_train_norm


    moe_cfg = config.get("heterogeneous_moe", {})


    moe_cfg_for_pretrain = dict(moe_cfg)
    print("[Phase 3] joint temperature/salinity optimization updates the shared backbone")


    _uses_raw_spectrum = any(
        str(t).lower() == "mamba" for t in moe_cfg.get("expert_types", [])
    )

    if _uses_raw_spectrum:

        if n_synthetic > 0 and synthetic_only:

            raw_spectrum_all = x_synth_spectrum.astype(np.float32)

        elif n_synthetic > 0:

            raw_spectrum_all = np.concatenate(

                [bundle.x[train_idx].astype(np.float32), x_synth_spectrum.astype(np.float32)],

                axis=0,

            )

        else:

            raw_spectrum_all = bundle.x[train_idx].astype(np.float32)

    else:

        raw_spectrum_all = None


    if n_synthetic > 0 and synthetic_only:

        print(f"  training samples: synthetic={len(spectrum_synth_norm)} (PCST-selected only)")

        synthetic_weight = 1.0

        sample_weight = synthetic_confidence

    elif n_synthetic > 0:

        print(f"  training samples: real={len(spectrum_train_norm)}, synthetic={len(spectrum_synth_norm)}, total={len(spectrum_all_norm)}")

        synthetic_weight = float(pretrain_cfg.get("synthetic_weight", 1.0))

        sample_weight = np.concatenate([

            np.ones(len(spectrum_train_norm), dtype=np.float32),

            synthetic_weight * synthetic_confidence,

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


    write_json(Path(output_dir) / "pretrain_summary.json", {
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

def finetune_stage(args) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("finetune_adapter requires PyTorch.") from exc


    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)


    adapter_cfg = config.get("adapter", {})
    bottleneck_dim = int(adapter_cfg.get("bottleneck_dim", 16))
    adapter_dropout = float(adapter_cfg.get("dropout", 0.0))
    adapter_scale = float(adapter_cfg.get("scale", 1.0))
    exclude_modules = list(adapter_cfg.get("exclude_modules", ["temperature_head", "salinity_head"]))


    ft_cfg = config.get("adapter_finetune", {})
    output_dir_str = args.output_dir or _project_path(
        ft_cfg.get("output_dir", f"runs/adapter_b{bottleneck_dim}")
    )
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
    consistency_weight = float(ft_cfg.get("consistency_weight", 0.0))
    consistency_noise_std = float(ft_cfg.get("consistency_noise_std", 0.05))
    consistency_warmup_epochs = int(ft_cfg.get("consistency_warmup_epochs", 0))
    if consistency_weight < 0 or consistency_noise_std < 0:
        raise ValueError("consistency_weight and consistency_noise_std must be non-negative")
    if consistency_weight > 0:
        print(
            f"[Consistency] weight={consistency_weight}, "
            f"noise_std={consistency_noise_std}, "
            f"warmup={consistency_warmup_epochs} epochs"
        )
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    val_ema = None

    def _consistency_loss(pred_clean, z, phy, raw):
        z_perturbed = z + torch.randn_like(z) * consistency_noise_std
        phy_perturbed = phy + torch.randn_like(phy) * consistency_noise_std
        raw_perturbed = (
            raw + torch.randn_like(raw) * consistency_noise_std
            if raw is not None else None
        )
        out_perturbed = model(z_perturbed, phy_perturbed, raw_spectrum=raw_perturbed)
        return torch.mean((pred_clean - out_perturbed["prediction"]) ** 2)


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
        consistency_active = (
            consistency_weight > 0 and epoch > consistency_warmup_epochs
        )
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
            consistency_term = (
                consistency_weight * _consistency_loss(pred, z, phy, raw)
                if consistency_active else None
            )

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

                    if consistency_term is not None:
                        consistency_term.backward(retain_graph=True)
                    gC = [p.grad.detach().clone() if p.grad is not None
                          else torch.zeros_like(p) for p in trainable_params]
                    optimizer.zero_grad()

                    gT_flat = torch.cat([g.reshape(-1) for g, shared in zip(gT, shared_param_mask) if shared])
                    gS_flat = torch.cat([g.reshape(-1) for g, shared in zip(gS, shared_param_mask) if shared])
                    gB_flat = torch.cat([g.reshape(-1) for g, shared in zip(gB, shared_param_mask) if shared])
                    gC_flat = torch.cat([g.reshape(-1) for g, shared in zip(gC, shared_param_mask) if shared])
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

                    gc_shared = lT * gT_flat + lS * gS_flat + gB_flat + gC_flat
                    _off = 0
                    for _p, _gT, _gS, _gB, _gC, _is_shared in zip(
                        trainable_params, gT, gS, gB, gC, shared_param_mask
                    ):
                        if _is_shared:
                            _sz = _p.numel()
                            _p.grad = gc_shared[_off:_off + _sz].reshape(_p.shape).clone()
                            _off += _sz
                        else:
                            _p.grad = (lT * _gT + lS * _gS + _gB + _gC).clone()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    loss_log = (lT * float(loss_t.item()) + lS * float(loss_s.item())
                                + bal_w * float(bal_loss.item())
                                + (float(consistency_term.item())
                                   if consistency_term is not None else 0.0))
                else:
                    loss = lT * loss_t + lS * loss_s + bal_w * bal_loss
                    if consistency_term is not None:
                        loss = loss + consistency_term
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
            if consistency_term is not None:
                loss = loss + consistency_term

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
                _conflict_text = (
                    "n/a" if _h["C"] is None else f"{_h['C']:.3f}"
                )
                _mtl_str = (f" lT={_h['lambda_T']:.3f} lS={_h['lambda_S']:.3f}"
                            f" C={_conflict_text}")
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
        "consistency_weight": consistency_weight,
        "consistency_noise_std": consistency_noise_std,
        "consistency_warmup_epochs": consistency_warmup_epochs,
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

def run_all(args: Namespace) -> None:
    config_path = str(_project_path(args.config))
    loaded_config = load_config(config_path)

    gan_dir = _project_path(loaded_config.get("gan", {}).get("output_dir", "outputs/gan"))
    pretrain_dir = _project_path(args.pretrain_dir)
    adapter_dir = _project_path(args.adapter_dir)

    if not args.skip_gan:
        train_gan_stage(Namespace(
            config=config_path,
            force=args.force,
            generate_only=False,
            checkpoint=None,
            output=None,
            n_synthetic=None,
        ))
        train_gan_stage(Namespace(
            config=config_path,
            force=False,
            generate_only=True,
            checkpoint=str(gan_dir / "gan_final.pt"),
            output=str(gan_dir / "gan_synthetic.npz"),
            n_synthetic=None,
        ))

    if not args.skip_pretrain:
        pretrain_stage(Namespace(
            config=config_path,
            output_dir=str(pretrain_dir),
        ))

    finetune_stage(Namespace(
        config=config_path,
        pretrain_dir=str(pretrain_dir),
        output_dir=str(adapter_dir),
        epochs=None,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PWT-CMMoE pipeline.")
    parser.add_argument(
        "--stage",
        choices=("all", "gan", "generate", "pretrain", "finetune"),
        default="all",
        help="Training stage to run; the default runs the complete pipeline.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--pretrain-dir", default="outputs/pretrain")
    parser.add_argument("--adapter-dir", default="outputs/adapter")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-synthetic", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-gan", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    args = parser.parse_args()

    if args.stage == "all":
        run_all(args)
    elif args.stage == "gan":
        train_gan_stage(Namespace(
            config=str(_project_path(args.config)),
            force=args.force,
            generate_only=False,
            checkpoint=None,
            output=None,
            n_synthetic=None,
        ))
    elif args.stage == "generate":
        if not args.checkpoint or not args.output:
            parser.error("--stage generate requires --checkpoint and --output")
        train_gan_stage(Namespace(
            config=str(_project_path(args.config)),
            force=False,
            generate_only=True,
            checkpoint=str(_project_path(args.checkpoint)),
            output=str(_project_path(args.output)),
            n_synthetic=args.n_synthetic,
        ))
    elif args.stage == "pretrain":
        pretrain_stage(Namespace(
            config=str(_project_path(args.config)),
            output_dir=str(_project_path(args.output_dir or args.pretrain_dir)),
        ))
    elif args.stage == "finetune":
        finetune_stage(Namespace(
            config=str(_project_path(args.config)),
            pretrain_dir=str(_project_path(args.pretrain_dir)),
            output_dir=str(_project_path(args.output_dir or args.adapter_dir)),
            epochs=args.epochs,
        ))


if __name__ == "__main__":
    main()
