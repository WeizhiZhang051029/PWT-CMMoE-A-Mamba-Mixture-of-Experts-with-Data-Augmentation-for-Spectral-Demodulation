# PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity

### [Project Page](https://github.com/WeizhiZhang051029/PWT-CMMoE-A-Mamba-Mixture-of-Experts-with-Data-Augmentation-for-Spectral-Demodulation) | [Paper](#citation)

The official implementation of [**PWT-CMMoE: A Mamba Mixture-of-Experts with Data Augmentation for Spectral Demodulation under Data Scarcity**](#citation).

We propose PWT-CMMoE, a physics-guided Mamba mixture-of-experts framework for joint temperature and salinity demodulation from full transmission spectra under limited calibration data. PWT generates candidate spectra under anti-resonance constraints and employs a Physics-Consistent Sample Teacher (PCST) to screen and confidence-weight reliable synthetic samples. CMMoE employs Top-2 routing to select complementary heterogeneous experts, including a bidirectional Mamba expert. Conflict-aware task balancing (CATB), together with PCGrad, mitigates task imbalance and gradient conflicts between temperature and salinity demodulation.

## 🔥 Highlights

* **Physics-guided data augmentation:** incorporates anti-resonance constraints into WGAN-GP to enhance the physical consistency of generated transmission spectra.
* **Teacher-guided sample selection:** employs PCST to screen generated spectra and assign confidence weights to physically reliable samples.
* **Heterogeneous sparse expert routing:** adopts Top-2 routing to activate complementary experts for input-adaptive spectral representation learning.
* **Bidirectional Mamba modeling:** captures long-range dependencies and cross-band correlations within full transmission spectra.
* **Conflict-aware task balancing:** alleviates task imbalance and gradient conflicts in joint temperature and salinity demodulation.

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
