# PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity

### [Project Page](https://github.com/WeizhiZhang051029/PWT-CMMoE-A-Mamba-Mixture-of-Experts-with-Data-Augmentation-for-Spectral-Demodulation) | [Paper](#citation)

The official implementation of [**PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity**](#citation).

![PWT-CMMoE Overall Architecture](images/fig01_pwt_cmmoe_overall_architecture.png)

PWT-CMMoE is a physics-guided spectral demodulation framework designed for joint temperature and salinity estimation under limited labelled calibration data.

The framework combines **Physics-guided WGAN-GP with Teacher (PWT)** for conditional spectral data augmentation and a **Conflict-aware Mamba Mixture-of-Experts (CMMoE)** network for multi-parameter spectral demodulation.

To improve the physical reliability of generated spectra, anti-resonance response constraints are incorporated into the conditional generation process. A **Physics-Consistent Sample Teacher (PCST)** then evaluates, screens, and weights the generated spectra before model pretraining.

For downstream demodulation, CMMoE employs sparse Top-2 routing across heterogeneous MLP, CNN, physics-aware, and Mamba experts. The pretrained model is subsequently adapted using a small number of measured spectra, while conflict-aware task balancing and PCGrad are introduced to alleviate gradient conflicts between temperature and salinity estimation.

## 🔥 Highlights

* **Physics-guided spectral generation:** introduces anti-resonance constraints into WGAN-GP to improve the physical consistency of synthetic transmission spectra.
* **Teacher-based sample selection:** uses PCST to screen and assign confidence weights to generated spectra.
* **Heterogeneous expert collaboration:** integrates MLP, CNN, physics-aware, and Mamba experts for complementary spectral feature extraction.
* **Sparse expert routing:** activates only the most relevant experts through input-dependent Top-2 routing.
* **Conflict-aware multi-task learning:** mitigates optimization conflicts between temperature and salinity demodulation.
* **Data-scarce adaptation:** supports synthetic-data pretraining followed by lightweight fine-tuning on limited measured spectra.

## 🧩 Framework

The complete PWT-CMMoE pipeline consists of the following stages:

```text
Measured spectra
       |
       v
Physics-guided WGAN-GP training
       |
       v
Condition-labelled spectral generation
       |
       v
Physics-Consistent Sample Teacher
       |
       v
Synthetic-sample screening and weighting
       |
       v
CMMoE synthetic-data pretraining
       |
       v
Adapter fine-tuning on measured spectra
       |
       v
Joint temperature and salinity demodulation
```

## 🛠️ Installation

Create a new Conda environment and install the required packages:

```bash
conda create -n pwt_cmmoe python=3.10
conda activate pwt_cmmoe
pip install -r requirements.txt
```

Install a PyTorch version compatible with your CUDA environment before installing `mamba-ssm`.

Then install the Mamba dependency:

```bash
pip install mamba-ssm --no-build-isolation
```

The default Mamba expert requires `mamba-ssm`. Please verify that the installed PyTorch, CUDA toolkit, and CUDA driver versions are compatible.

## 📊 Dataset

The experimental transmission spectra and trained checkpoints are not distributed with this repository.

Create a folder named `data` in the root directory and organize the dataset as follows:

```text
.
├── data/
│   ├── spectra.npz
│   └── labels.csv
├── configs/
│   └── config.yaml
├── scripts/
│   └── train.py
├── spectral_moe/
├── requirements.txt
└── README.md
```

### Spectral Data

The `spectra.npz` file should contain the following arrays:

| Key             | Shape    | Description                         |
| --------------- | -------- | ----------------------------------- |
| `X_raw_dbm`     | `[N, L]` | Raw transmission spectra in dB      |
| `X_zscore`      | `[N, L]` | Optional z-score-normalized spectra |
| `wavelength_nm` | `[L]`    | Wavelength axis in nanometres       |
| `sample_id`     | `[N]`    | Unique sample identifiers           |

Here, `N` denotes the number of measured spectra and `L` denotes the number of wavelength sampling points.

### Label Data

The `labels.csv` file should contain the following columns:

| Column            | Description                                |
| ----------------- | ------------------------------------------ |
| `sample_id`       | Unique sample identifier                   |
| `temperature_c`   | Temperature label in degrees Celsius       |
| `salinity_ppt`    | Salinity label in parts per thousand       |
| `experiment_type` | Optional experimental-condition identifier |

The values in the `sample_id` column must be consistent with those stored in `spectra.npz`.

The optional `experiment_type` column can be used to exclude specified records or define experimental subsets through `configs/config.yaml`.

## ⚙️ Configuration

The main training parameters are defined in:

```text
configs/config.yaml
```

The configuration file controls:

* dataset paths and data partitions
* spectral preprocessing
* physics-guided WGAN-GP training
* synthetic-spectrum generation
* PCST screening and sample weighting
* CMMoE architecture
* sparse routing parameters
* synthetic-data pretraining
* adapter fine-tuning
* multi-task optimization
* checkpoint and output paths

Please update the dataset paths and training parameters before running the code.

## 🚀 Running

### Complete Training Pipeline

Run the complete PWT-CMMoE training pipeline using:

```bash
python scripts/train.py
```

The complete pipeline performs the following operations:

1. trains the physics-guided WGAN-GP;
2. generates condition-labelled synthetic spectra;
3. evaluates synthetic spectra using PCST;
4. screens and weights physically consistent samples;
5. pretrains the CMMoE regressor using synthetic spectra;
6. fine-tunes the adapters and prediction heads using measured spectra;
7. evaluates temperature and salinity demodulation performance.

### Stage-by-Stage Execution

The individual stages can also be executed separately.

#### 1. Train the Physics-Guided WGAN-GP

```bash
python -m spectral_moe.train.train_gan \
  --config configs/config.yaml
```

#### 2. Generate Synthetic Spectra

```bash
python -m spectral_moe.train.generate_gan_synthetic \
  --config configs/config.yaml \
  --checkpoint outputs/gan/gan_final.pt \
  --output outputs/gan/gan_synthetic.npz
```

#### 3. Pretrain the CMMoE Regressor

```bash
python -m spectral_moe.train.pretrain_moe \
  --config configs/config.yaml \
  --output-dir outputs/pretrain
```

#### 4. Fine-Tune the Adapter

```bash
python -m spectral_moe.train.finetune_adapter \
  --config configs/config.yaml \
  --pretrain-dir outputs/pretrain \
  --output-dir outputs/adapter
```

## 📁 Repository Structure

```text
PWT-CMMoE/
├── configs/
│   └── config.yaml
├── data/
│   ├── spectra.npz
│   └── labels.csv
├── scripts/
│   └── train.py
├── spectral_moe/
│   ├── data/
│   │   ├── data loading
│   │   ├── spectral preprocessing
│   │   ├── PCA processing
│   │   └── physics-feature extraction
│   ├── models/
│   │   ├── physics-guided WGAN-GP
│   │   ├── PINN-based physical prior
│   │   ├── heterogeneous experts
│   │   ├── sparse MoE routing
│   │   └── lightweight adapters
│   ├── train/
│   │   ├── GAN training
│   │   ├── synthetic-spectrum generation
│   │   ├── PCST sample screening
│   │   ├── MoE pretraining
│   │   └── adapter fine-tuning
│   └── evaluate/
│       ├── regression metrics
│       ├── synthetic-sample quality metrics
│       └── prediction analysis
├── outputs/
├── requirements.txt
└── README.md
```

## 📈 Outputs

The generated files are saved in the `outputs/` directory.

```text
outputs/
├── gan/
│   ├── gan_final.pt
│   └── gan_synthetic.npz
├── pretrain/
│   ├── checkpoints
│   ├── training logs
│   └── validation metrics
├── adapter/
│   ├── fine-tuned checkpoints
│   ├── predictions
│   └── evaluation metrics
└── figures/
```

The output files may include:

* trained model checkpoints
* generated synthetic spectra
* PCST confidence scores
* sample-selection results
* expert-routing statistics
* temperature and salinity predictions
* regression evaluation metrics
* training and validation logs

Generated checkpoints, synthetic spectra, predictions, and intermediate outputs are ignored by Git by default.

## 📏 Evaluation Metrics

The framework evaluates temperature and salinity demodulation using common regression metrics, including:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination ((R^2))
* Relative prediction error
* Synthetic-spectrum quality metrics
* Expert-routing and load-balancing statistics

## 📰 News

* **May 2026** — PWT-CMMoE framework completed
* **June 2026** — Manuscript prepared
* **2026** — Source code released

## 🙏 Acknowledgements

This project is implemented using PyTorch, scikit-learn, NumPy, SciPy, and Mamba-based sequence modelling tools.

We sincerely thank the open-source community for providing valuable implementations of generative modelling, state-space sequence modelling, mixture-of-experts learning, and multi-task optimization methods.

## 📖 Citation

If you find this repository useful in your research or project, please consider citing our paper:

```bibtex
@article{zhang2026pwtcmmoe,
  title   = {PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

The citation information will be updated after the paper is officially published.

## 📬 Contact

For questions regarding the implementation, dataset format, or experimental configuration, please open an issue in this repository.
