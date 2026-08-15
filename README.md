# 🌊 PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity

<p align="center">
  <b>Physics-Guided Data Augmentation · Mamba-MoE · Full-Spectrum Demodulation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PWT-Data%20Augmentation-green">
  <img src="https://img.shields.io/badge/CMMoE-Joint%20Spectral%20Demodulation-purple">
  <img src="https://img.shields.io/badge/Application-Temperature%20%26%20Salinity-blue">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C">
</p>

<p align="center">
  <a href="https://github.com/WeizhiZhang051029/PWT-CMMoE-A-Mamba-Mixture-of-Experts-with-Data-Augmentation-for-Spectral-Demodulation">
    <b>Project Page</b>
  </a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#citation">
    <b>Paper</b>
  </a>
</p>

## 📌 Overview

The official implementation of **PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity**.

<p align="center">
  <img src="images/framework_overview.jpg" width="100%">
</p>

<p align="center">
  <em>Overall framework of the proposed PWT-CMMoE method for data augmentation and joint temperature–salinity demodulation.</em>
</p>

We propose PWT-CMMoE for joint temperature and salinity demodulation from full transmission spectra under data scarcity. The framework combines physics-guided data augmentation with a sparse Mamba mixture-of-experts (MoE) for adaptive spectral modeling, while CATB mitigates task imbalance and gradient conflicts during joint optimization. Experimental results on Dataset1 achieve RMSEs of 1.461 °C and 1.247‰ for temperature and salinity, with corresponding R² values of 0.9737 and 0.9927. Cross-dataset experiments on Dataset2 further demonstrate the adaptability of PWT-CMMoE to spectral distribution shifts across independently fabricated sensors.

## 🔬 Experimental Platform

<p align="center">
  <img src="images/platform.jpg" width="100%">
</p>

<p align="center">
  <em>Experimental platform for transmission-spectrum acquisition and online temperature–salinity demodulation.</em>
</p>

The experimental platform consists of a supercontinuum light source (SLS), the proposed fiber-optic sensor, a constant-temperature oil bath, a precision thermometer, an optical spectrum analyzer (OSA), and a data-processing system. Standard seawater samples with different salinity levels are injected into the sensor using a syringe. The oil bath provides controlled temperature conditions, while the OSA records the corresponding transmission spectra for subsequent joint temperature and salinity demodulation based on PWT-CMMoE.

## 🔥 Highlights

* **Physics-guided augmentation:** embeds AR physics into WGAN-GP for spectrum generation.
* **Physics-Consistent Sample Teacher:** screens and weights high-confidence spectra.
* **Heterogeneous expert routing:** activates complementary experts with Top-2 routing.
* **Bidirectional Mamba:** captures long-range and cross-band spectral dependencies.
* **Conflict-aware task balancing:** mitigates task imbalance and gradient conflicts.

## 🧩 Framework

The offline training procedure of PWT-CMMoE consists of the following stages:

```text
Measured training spectra and labels
       |
       v
Physics-guided WGAN-GP training
       |
       v
Condition-labelled candidate spectrum generation
       |
       v
PCST screening and confidence weighting
       |
       v
High-confidence weighted synthetic spectra
       |
       v
CMMoE pretraining on synthetic spectra
       |
       v
CATB-guided optimization on measured spectra
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

Install a PyTorch build compatible with your CUDA environment before installing the Mamba dependency:

```bash
pip install mamba-ssm --no-build-isolation
```

The default Mamba expert requires `mamba-ssm`. Please ensure that the installed PyTorch, CUDA toolkit, and GPU driver versions are compatible.

## ⚙️ Configuration

All experiment settings are specified in:

```text
configs/config.yaml
```

The configuration file controls:

* dataset paths and data partitioning
* wavelength-grid alignment, linear interpolation, and spectral normalization
* physics-guided WGAN-GP training
* candidate-spectrum generation
* PCST-based sample screening and confidence weighting
* CMMoE architecture and sparse Top-2 routing
* synthetic-data pretraining
* CATB-guided joint optimization
* checkpoint, prediction, and output paths

Please update the dataset paths and relevant hyperparameters before running the experiments.


## 🚀 Running

### Complete Training Pipeline

Run the complete PWT-CMMoE pipeline with:

```bash
python scripts/train.py
```

The pipeline sequentially:

1. trains the physics-guided WGAN-GP;
2. generates condition-labelled candidate spectra;
3. screens and confidence-weights synthetic spectra using PCST;
4. pretrains CMMoE on the selected synthetic spectra;
5. optimizes CMMoE on measured spectra under CATB;
6. evaluates joint temperature and salinity demodulation performance.

### Ten-seed Repetition Reported in the Manuscript

To reproduce the reported mean and standard deviation over independent random seeds,
run the repetition driver (the default configuration defines seeds 1--10):

```bash
python scripts/repeat_experiments.py
```

It writes each isolated run under `outputs/repeated_runs/seed_XX/` and writes the
aggregated test metrics to `outputs/repeated_runs/test_metrics_mean_std.json`.

### Stage-by-Stage Execution

Each training stage can also be executed independently.

#### 1. Train the Physics-Guided WGAN-GP

```bash
python -m spectral_moe.train.train_gan \
  --config configs/config.yaml
```

#### 2. Generate and Screen Synthetic Spectra

Generate condition-labelled candidate spectra:

```bash
python -m spectral_moe.train.generate_gan_synthetic \
  --config configs/config.yaml \
  --checkpoint outputs/gan/gan_final.pt \
  --output outputs/gan/gan_synthetic.npz
```

PCST screening and confidence weighting are performed according to the settings specified in `configs/config.yaml`.

#### 3. Pretrain CMMoE

```bash
python -m spectral_moe.train.pretrain_moe \
  --config configs/config.yaml \
  --output-dir outputs/pretrain
```

#### 4. Perform CATB-Guided Optimization

```bash
python -m spectral_moe.train.finetune_adapter \
  --config configs/config.yaml \
  --pretrain-dir outputs/pretrain \
  --output-dir outputs/adapter
```

During this stage, the pretrained CMMoE is adapted to measured spectra, while CATB coordinates the temperature and salinity tasks through dynamic task prioritization, conflict-aware gating, and PCGrad-based gradient correction.

## 📁 Repository Structure

```text
PWT-CMMoE/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── labels.csv
├── scripts/
│   └── train.py
├── spectral_moe/
│   ├── data/
│   ├── models/
│   ├── train/
│   └── evaluate/
├── outputs/
├── requirements.txt
└── README.md
```

The main components are organized as follows:

* `spectral_moe/data/`: data loading, wavelength-grid alignment, linear interpolation, normalization, and physics-feature extraction
* `spectral_moe/models/`: physics-guided WGAN-GP, heterogeneous experts, sparse routing, Mamba modules, and adapters
* `spectral_moe/train/`: spectrum generation, PCST screening, CMMoE pretraining, and CATB-guided optimization
* `spectral_moe/evaluate/`: regression metrics, predictions, and model-analysis utilities

## 📈 Outputs

All generated artifacts are saved in the `outputs/` directory:

```text
outputs/
├── gan/
│   ├── gan_final.pt
│   └── gan_synthetic.npz
├── pretrain/
│   ├── checkpoints/
│   ├── training_logs/
│   └── validation_metrics/
├── adapter/
│   ├── fine_tuned_checkpoints/
│   ├── predictions/
│   └── evaluation_metrics/
└── figures/
```

The outputs include:

* trained model checkpoints
* generated candidate spectra
* PCST confidence scores and sample-selection results
* CMMoE pretraining and optimization logs
* expert-routing statistics
* temperature and salinity predictions
* regression metrics and visualization results

Generated checkpoints, synthetic spectra, predictions, and intermediate files are excluded from version control by default.

## 📏 Evaluation Metrics

Temperature and salinity demodulation performance is evaluated using three standard regression metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)

Lower MAE and RMSE values indicate smaller demodulation errors, while a higher R² indicates better agreement between the predicted and measured values.

## 📰 News

* **August 2026** — PWT-CMMoE framework completed
* **August 2026** — Manuscript prepared
* **August 2026** — Source code released

## 🙏 Acknowledgements

This project is built upon PyTorch, scikit-learn, NumPy, SciPy, and open-source Mamba implementations.

We gratefully acknowledge the open-source community for providing valuable resources in generative modeling, state-space sequence modeling, mixture-of-experts architectures, and multi-task optimization.

## 📖 Citation

If you find this repository useful in your research or project, please consider citing our paper:

```bibtex
@article{zhang2026pwtcmmoe,
  title   = {PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity},
  author  = {TODO},
  year    = {2026}
}
```

The citation information will be updated after the paper is officially published.

## 📬 Contact

For questions regarding the implementation, dataset format, or experimental configuration, please open an issue in this repository.
