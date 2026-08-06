# WGAN-GP MoE for Joint Temperature and Salinity Regression

This repository contains a training implementation for joint prediction of seawater temperature and salinity from transmission spectra. The pipeline uses a physics-guided WGAN-GP to generate condition-labelled spectra, followed by a heterogeneous mixture-of-experts regressor with Top-2 sparse routing and prior-anchored adaptive multi-task learning with PCGrad.

The repository intentionally excludes datasets, trained checkpoints, experiment logs, ablation studies, baseline models, performance-analysis scripts, and figure-generation code.

## Repository layout

```text
configs/config.yaml              training configuration
scripts/train.py                 single-run training entry point
spectral_moe/data/               loading, PCA, and physics features
spectral_moe/models/             WGAN-GP, PINN prior, four-expert MoE, and adapters
spectral_moe/train/              WGAN-GP, MoE pretraining, and adapter fine-tuning
spectral_moe/evaluate/           regression and synthetic-sample quality metrics
```

## Installation

Use Python 3.10 or later. Create an environment, install PyTorch appropriate for your CUDA version, then install the remaining dependencies.

```bash
pip install -r requirements.txt
pip install mamba-ssm --no-build-isolation
```

`mamba-ssm` is required by the default Mamba expert. If it is unavailable on your platform, change `heterogeneous_moe.mamba.backend` in the configuration to a supported fallback implemented in your environment.

## Data format

Place user-supplied data under `data/`. No data files are distributed with this repository.

`data/spectra.npz` must contain these arrays:

| Key | Shape | Description |
| --- | --- | --- |
| `X_raw_dbm` | `[N, L]` | Raw transmission spectra in dB |
| `X_zscore` | `[N, L]` | Optional z-score-normalized spectra |
| `wavelength_nm` | `[L]` | Wavelength axis in nm |
| `sample_id` | `[N]` | Sample identifiers matching the CSV |

`data/labels.csv` must contain `sample_id`, `temperature_c`, and `salinity_ppt`. Its row order must match `sample_id` in the NPZ file. An optional `experiment_type` column may be used for filtering.

The default configuration trains WGAN-GP on the training partition, generates condition-labelled spectra, applies the spectral quality gate, then pretrains and fine-tunes the MoE.

## Training

Run the complete single-seed pipeline:

```bash
python scripts/train.py
```

Or run the stages separately:

```bash
python -m spectral_moe.train.train_gan \
  --config configs/config.yaml

python -m spectral_moe.train.generate_gan_synthetic \
  --config configs/config.yaml \
  --checkpoint outputs/gan/gan_final.pt \
  --output outputs/gan/gan_synthetic.npz

python -m spectral_moe.train.pretrain_moe \
  --config configs/config.yaml \
  --output-dir outputs/pretrain

python -m spectral_moe.train.finetune_adapter \
  --config configs/config.yaml \
  --pretrain-dir outputs/pretrain \
  --output-dir outputs/adapter
```

Use `--skip-gan` or `--skip-pretrain` only when the corresponding compatible artifacts already exist under `outputs/`.

## Outputs

Training writes generated artifacts under `outputs/`:

```text
outputs/pretrain/                pretrained MoE and preprocessing artifacts
outputs/adapter/                 adapter checkpoint, metrics, and predictions
outputs/gan/                     WGAN-GP checkpoint and generated spectra
```

These outputs are ignored by Git and are not part of this source release.
