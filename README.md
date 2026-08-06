# PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity

### [Project Page](https://github.com/WeizhiZhang051029/PWT-CMMoE-A-Mamba-Mixture-of-Experts-with-Data-Augmentation-for-Spectral-Demodulation) | [Paper](#citation)

The official implementation of [**PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity**](#citation).

PWT-CMMoE is designed for joint temperature and salinity demodulation from full transmission spectra when labelled calibration data are limited. The framework combines Physics-guided WGAN-GP with Teacher (PWT) data augmentation and a conflict-aware Mamba mixture-of-experts (CMMoE) regressor. The generator incorporates anti-resonance constraints, while the Physics-Consistent Sample Teacher (PCST) screens and weights synthetic spectra before MoE pretraining. The regressor uses sparse Top-2 routing across MLP, CNN, physics, and Mamba experts, then applies adaptive multi-task learning with PCGrad during fine-tuning.

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n pwt_cmmoe python=3.10
conda activate pwt_cmmoe
pip install -r requirements.txt
pip install mamba-ssm --no-build-isolation
```

Install a PyTorch build compatible with your CUDA environment before running the project. `mamba-ssm` is required by the default Mamba expert.

## Dataset

The experimental spectra and trained checkpoints are not distributed with this repository. Create a `data` directory in the project root and place your data in the following format:

```text
.
|-- data/
|   |-- spectra.npz
|   `-- labels.csv
|-- configs/
|   `-- config.yaml
|-- scripts/
|   `-- train.py
`-- spectral_moe/
```

`spectra.npz` must contain the following arrays:

| Key | Shape | Description |
| --- | --- | --- |
| `X_raw_dbm` | `[N, L]` | Raw transmission spectra in dB |
| `X_zscore` | `[N, L]` | Optional z-score-normalized spectra |
| `wavelength_nm` | `[L]` | Wavelength axis in nm |
| `sample_id` | `[N]` | Sample identifiers |

`labels.csv` must contain `sample_id`, `temperature_c`, and `salinity_ppt`. The `sample_id` values must align with the NPZ file. An optional `experiment_type` column can be used to exclude specified records through `configs/config.yaml`.

Before training, each spectrum is linearly interpolated onto a uniform 2048-point wavelength grid. The resampled spectrum is used by WGAN-GP and CMMoE, while physics features are extracted from the original high-resolution spectrum.

## Running

Run the complete training pipeline:

```bash
python scripts/train.py
```

The pipeline trains the physics-guided WGAN-GP, generates condition-labelled spectra, applies PCST quality screening, pretrains the CMMoE regressor, and performs adapter fine-tuning.

The stages can also be run separately:

```bash
python -m spectral_moe.train.train_gan --config configs/config.yaml

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

Generated checkpoints, synthetic spectra, metrics, and predictions are written to `outputs/` and ignored by Git.

## Repository Structure

```text
configs/                         Training configuration
scripts/train.py                 Complete single-run pipeline
spectral_moe/data/               Data loading, linear resampling, and physics features
spectral_moe/models/             WGAN-GP, PINN prior, MoE, and adapters
spectral_moe/train/              Generation, pretraining, and fine-tuning
spectral_moe/evaluate/           Regression and synthetic-sample quality metrics
```

## Acknowledgements

This project uses PyTorch, scikit-learn, and Mamba-based sequence modelling tools. We thank the open-source community for these resources.

## Citation

Citation information will be added after the manuscript is finalized. If you use this code before then, please cite the project page above.
