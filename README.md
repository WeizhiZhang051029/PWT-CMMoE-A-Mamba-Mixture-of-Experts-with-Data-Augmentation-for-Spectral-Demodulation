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

* PWT-CMMoE framework is proposed for data-scarce multi-parameter demodulation.
* PWT combines physics-constrained WGAN-GP and PCST for reliable spectrum augmentation.
* CMMoE combines sparse heterogeneous experts with bidirectional Mamba modeling.
* Achieves state-of-the-art performance on the real-world spectral dataset.
* Demonstrates robustness and cross-dataset transferability across different sensors.

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

Create a new Conda environment and install a PyTorch build compatible with your CUDA environment:

```bash
conda create -n pwt_cmmoe python=3.10
conda activate pwt_cmmoe
# Install the PyTorch build matching your CUDA toolkit and GPU driver.
pip install -r requirements.txt
```

The default Mamba expert requires `mamba-ssm`, which is installed separately:

```bash
pip install mamba-ssm --no-build-isolation
```

Before running the pipeline, place the user-provided dataset at the paths specified in `configs/config.yaml` (by default, `data/spectra.npz` and `data/labels.csv`).

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

Run the complete training pipeline with:

```bash
python train.py --stage all --config configs/config.yaml
```

The pipeline sequentially:

1. trains the physics-guided WGAN-GP;
2. generates condition-labelled candidate spectra;
3. screens and confidence-weights synthetic spectra using PCST;
4. pretrains CMMoE on the selected synthetic spectra;
5. fine-tunes CMMoE on measured spectra using Adapters and CATB.

### Stage-by-Stage Execution

Each training stage can also be executed independently.

#### 1. Train the Physics-Guided WGAN-GP

```bash
python train.py --stage gan --config configs/config.yaml
```

#### 2. Generate Candidate Spectra

```bash
python train.py --stage generate \
  --config configs/config.yaml \
  --checkpoint outputs/gan/gan_final.pt \
  --output outputs/gan/gan_synthetic.npz
```

PCST screening and confidence weighting are performed during the pretraining stage.

#### 3. Pretrain CMMoE

```bash
python train.py --stage pretrain \
  --config configs/config.yaml \
  --pretrain-dir outputs/pretrain
```

#### 4. Perform CATB-Guided Fine-Tuning

```bash
python train.py --stage finetune \
  --config configs/config.yaml \
  --pretrain-dir outputs/pretrain \
  --adapter-dir outputs/adapter
```

During this stage, the pretrained CMMoE is adapted to measured spectra, while CATB coordinates the temperature and salinity tasks through dynamic task prioritization, conflict-aware gating, and PCGrad-based gradient correction.

## 📁 Repository Structure

```text
PWT-CMMoE/
├── configs/
│   └── config.yaml
├── data/                         # user-provided; not included
│   ├── spectra.npz
│   └── labels.csv
├── models/
│   ├── __init__.py
│   ├── gan.py                    # WGAN-GP and anti-resonance physics modules
│   └── moe.py                    # heterogeneous MoE, Mamba, and adapters
├── data.py                       # data loading and dataset utilities
├── physics.py                    # physical features and PCST utilities
├── train.py                      # unified training entrypoint
├── images/
├── requirements.txt
└── README.md
```

The repository contains the final training implementation only. Test, inference, and evaluation scripts are not included.

## 📈 Outputs

All generated training artifacts are saved in the `outputs/` directory:

```text
outputs/
├── gan/
│   ├── gan_final.pt
│   ├── gan_synthetic.npz
│   └── pinn_calibration.json
├── pretrain/
│   ├── pretrained_moe_best.pt
│   ├── normalization.npz
│   ├── synthetic_quality.json
│   └── pretrain_summary.json
└── adapter/
    ├── best_adapter.pt
    ├── training_summary.json
    └── mtl_conflict_history.json
```

The outputs include:

* trained model checkpoints;
* generated candidate spectra;
* PCST quality reports, confidence weights, and selection records;
* normalization statistics and training summaries;
* CATB task-weight and gradient-conflict history.

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
  author  = {Zhang, Weizhi and Li, Yiteng and Liu, Yanze and Xie, Yuhan and Zhao, Jian and Zhang, Yanan and Zhao, Yong},
  year    = {2026}
}
```

The citation information will be updated after the paper is officially published.

## 📬 Contact

For questions regarding the implementation, dataset format, or experimental configuration, please open an issue in this repository.
