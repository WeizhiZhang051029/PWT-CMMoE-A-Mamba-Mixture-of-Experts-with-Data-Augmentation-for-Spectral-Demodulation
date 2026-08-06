"""Train the physics-guided WGAN-GP generator on a training split."""

from __future__ import annotations


import argparse

from pathlib import Path


import numpy as np


from spectral_moe.data.dataset import TorchSpectrumDataset, load_spectrum_bundle

from spectral_moe.models.antiresonance_pinn import (

    AntiResonanceConfig,

    AntiResonancePINN,

    antiresonance_trough_loss,

    calibrate_antiresonance_prior,

    soft_trough_locations,

)

from spectral_moe.models.gan import (

    ConditionalCritic,

    ConditionalGenerator,

    gradient_penalty,

    r3gan_critic_loss,

    r3gan_generator_loss,

)

from spectral_moe.models.physics_informed import smoothness_loss

from spectral_moe.train.physics_consistency import design_matrix

from spectral_moe.utils.config import load_config

from spectral_moe.utils.io import ensure_dir, write_json

from spectral_moe.utils.seed import set_seed

from spectral_moe.utils.splits import resolve_split_seed, split_audit, split_from_config


def main() -> None:

    try:

        import torch

        from torch.utils.data import DataLoader

    except ImportError as exc:

        raise ImportError("train_gan requires PyTorch. Install torch first.") from exc


    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/default.yaml")

    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()


    config = load_config(args.config)

    seed = int(config.get("seed", 42))

    set_seed(seed)

    bundle = load_spectrum_bundle(config)

    gan_cfg = config.get("gan", {})

    if not bool(gan_cfg.get("enabled", False)) and not args.force:

        print("GAN training skipped because gan.enabled is false. Pass --force to override.")

        return

    output_dir = ensure_dir(gan_cfg.get("output_dir", "runs/gan"))


    split_cfg = config.get("data", {}).get("split", {})

    split_seed = resolve_split_seed(split_cfg, seed)

    train_idx, val_idx, test_idx, split_meta = split_from_config(

        len(bundle.y),

        seed=split_seed,

        split_cfg=split_cfg,

        labels=bundle.labels,

        x_raw_dbm=bundle.x_raw_dbm,

    )

    split_meta["seed"] = split_seed

    audit = split_audit(bundle.labels, train_idx, val_idx, test_idx, x_raw_dbm=bundle.x_raw_dbm)

    write_json(Path(output_dir) / "split_audit.json", {"split": split_meta, "audit": audit})


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
    critic = ConditionalCritic(condition_dim=2, input_length=bundle.x.shape[1]).to(device)

    learning_rate = float(gan_cfg.get("learning_rate", 1e-4))

    g_opt = torch.optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

    c_opt = torch.optim.AdamW(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))


    critic_steps = int(gan_cfg.get("critic_steps", 5))

    gp_weight = float(gan_cfg.get("gradient_penalty_weight", 10.0))

    smooth_weight = float(gan_cfg.get("smoothness_weight", 0.001))

    moment_weight = float(gan_cfg.get("moment_weight", 0.0))

    variant = str(gan_cfg.get("variant", "wgan_gp")).lower()

    if variant not in {"wgan_gp", "r3gan"}:

        raise ValueError("gan.variant must be 'wgan_gp' or 'r3gan'")


    pinn_cfg = gan_cfg.get("pinn", {})

    pinn_enabled = bool(pinn_cfg.get("enabled", False))

    physics = None

    wavelength_grid = None

    soft_coefficients = None

    pinn_calibration = None

    strict_weight = 0.0

    if pinn_enabled:

        centers = list(pinn_cfg.get("tracked_centers_nm", []))

        if not centers:

            raise ValueError("gan.pinn requires at least one tracked_centers_nm value")

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

                if variant == "wgan_gp":

                    c_loss = critic(fake, cond).mean() - critic(real, cond).mean()

                    c_loss = c_loss + gp_weight * gradient_penalty(critic, real, fake, cond)

                else:

                    c_loss = r3gan_critic_loss(

                        critic, real, fake, cond,

                        r1_weight=float(gan_cfg.get("r1_weight", 1.0)),

                        r2_weight=float(gan_cfg.get("r2_weight", 1.0)),

                    )

                c_opt.zero_grad()

                c_loss.backward()

                c_opt.step()

            z = torch.randn(real.shape[0], latent_dim, device=device)

            fake = generator(z, cond)

            if variant == "wgan_gp":

                g_loss = -critic(fake, cond).mean()

            else:

                g_loss = r3gan_generator_loss(critic, real, fake, cond)

            fake_raw_for_regularizers = fake * spectrum_std_t + spectrum_mean_t

            g_loss = g_loss + smooth_weight * smoothness_loss(fake_raw_for_regularizers)

            if moment_weight > 0:

                g_loss = g_loss + moment_weight * moment_matching_loss(fake, real)

            if physics is not None:

                raw_cond = cond * torch.as_tensor(condition_std, device=device) + torch.as_tensor(condition_mean, device=device)

                fake_raw = fake_raw_for_regularizers

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

        print(f"epoch={epoch} variant={variant} critic={np.mean(c_losses):.6f} generator={np.mean(g_losses):.6f}")

        if epoch % 50 == 0:

            torch.save(

                _checkpoint_payload(generator, critic, physics, soft_coefficients, config,

                                    split_meta, condition_mean, condition_std, spectrum_mean, spectrum_std,

                                    representation, variant),

                Path(output_dir) / f"gan_epoch_{epoch}.pt",

            )

    torch.save(

        _checkpoint_payload(generator, critic, physics, soft_coefficients, config,

                            split_meta, condition_mean, condition_std, spectrum_mean, spectrum_std,

                            representation, variant),

        Path(output_dir) / "gan_final.pt",

    )


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

                        representation, variant):

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

        "gan_variant": variant,

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


if __name__ == "__main__":

    main()
