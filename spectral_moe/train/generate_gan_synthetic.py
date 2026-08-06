"""Generate condition-labelled spectra from a trained WGAN-GP model."""

from __future__ import annotations


import argparse

from pathlib import Path


import numpy as np


from spectral_moe.data.dataset import load_spectrum_bundle

from spectral_moe.models.gan import ConditionalGenerator, ConditionalVectorGenerator

from spectral_moe.utils.config import load_config

from spectral_moe.utils.seed import set_seed

from spectral_moe.utils.splits import resolve_split_seed, split_from_config


def main() -> None:

    import torch


    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--output", required=True)

    parser.add_argument("--n-synthetic", type=int, default=None)

    args = parser.parse_args()


    config = load_config(args.config)

    if bool(config.get("data", {}).get("use_zscore", True)):

        raise ValueError("GAN-to-MoE chain requires data.use_zscore=false so generated spectra are in raw dBm")

    seed = int(config.get("seed", 42))

    set_seed(seed)

    bundle = load_spectrum_bundle(config)

    gan_cfg = config.get("gan", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    latent_dim = int(gan_cfg.get("latent_dim", 64))

    representation = str(checkpoint.get("representation", "raw")).lower()

    if representation == "pca":

        pca_components = torch.as_tensor(checkpoint["pca_components"], device=device, dtype=torch.float32)

        pca_mean = torch.as_tensor(checkpoint["pca_mean"], device=device, dtype=torch.float32)

        generator = ConditionalVectorGenerator(latent_dim, 2, int(pca_components.shape[0])).to(device)

    elif representation == "raw":

        pca_components = pca_mean = None

        generator = ConditionalGenerator(latent_dim, 2, bundle.x_raw_dbm.shape[1]).to(device)

    else:

        raise ValueError(f"unsupported GAN checkpoint representation: {representation}")

    generator.load_state_dict(checkpoint["generator"], strict=True)

    generator.eval()


    n = args.n_synthetic or int(gan_cfg.get("n_synthetic", 2000))

    batch_size = int(gan_cfg.get("sample_batch_size", 64))

    rng = np.random.default_rng(seed + 30000)


    split_cfg = config.get("data", {}).get("split", {})

    train_idx, _, _, _ = split_from_config(

        len(bundle.y), seed=resolve_split_seed(split_cfg, seed), split_cfg=split_cfg,

        labels=bundle.labels, x_raw_dbm=bundle.x_raw_dbm,

    )

    train_min = bundle.y[train_idx].min(axis=0)

    train_max = bundle.y[train_idx].max(axis=0)

    y = rng.uniform(train_min, train_max, size=(n, 2)).astype(np.float32)

    cond_mean = np.asarray(checkpoint["condition_mean"], dtype=np.float32)

    cond_std = np.asarray(checkpoint["condition_std"], dtype=np.float32)

    if "spectrum_mean" not in checkpoint or "spectrum_std" not in checkpoint:

        raise ValueError(

            "checkpoint lacks train-split spectrum normalization statistics; "

            "regenerate it with the normalized GAN training protocol"

        )

    spectrum_mean = torch.as_tensor(checkpoint["spectrum_mean"], device=device, dtype=torch.float32)

    spectrum_std = torch.as_tensor(checkpoint["spectrum_std"], device=device, dtype=torch.float32)

    x_parts = []

    with torch.no_grad():

        for start in range(0, n, batch_size):

            raw = y[start:start + batch_size]

            cond = torch.from_numpy((raw - cond_mean) / cond_std).to(device)

            z = torch.randn(len(raw), latent_dim, device=device)

            generated_normalized = generator(z, cond)

            generated_representation = generated_normalized * spectrum_std + spectrum_mean

            if representation == "pca":

                raw = generated_representation @ pca_components + pca_mean

            else:

                raw = generated_representation.squeeze(1)

            x_parts.append(raw.cpu().numpy().astype(np.float32))

    out = Path(args.output)

    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out, x_raw=np.concatenate(x_parts), y=y,

                        gan_variant=checkpoint.get("gan_variant", "unknown"), seed=seed)

    print(f"saved {n} generated raw-dBm spectra to {out}")


if __name__ == "__main__":

    main()
