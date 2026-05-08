# 🧬 FoldNet: Protein Structure & Contact Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Lightning-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white)](https://www.pytorchlightning.ai/)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/)

**FoldNet** is a state-of-the-art multi-task deep learning framework designed to predict **Secondary Structure** and **Residue-Level Contact Maps** directly from protein sequences. By leveraging pre-computed **ESM-2 Embeddings**, FoldNet achieves high accuracy while comparing multiple architectural backbones.

---

## ✨ Key Features

*   🚀 **Multi-Task Learning:** Simultaneous prediction of 3-class secondary structure (Helix, Sheet, Coil) and binary contact maps.
*   🧠 **Multiple Encoders:** Comparative support for **1D CNN (Residual)**, **BiLSTM**, and **Transformer** architectures.
*   📊 **ESM-2 Integration:** Uses high-dimensional (1280-dim) protein language model embeddings for superior feature representation.
*   📈 **Rich Visualisation:** Automatic generation of colored secondary structure bar plots and contact map heatmaps.
*   🧪 **Experiment Tracking:** Full integration with **Weights & Biases** for live monitoring of Q3, MCC, and Precision@L.
*   🛠️ **Robust Pipeline:** End-to-end support from data preprocessing to benchmarking and visualization.

---

## 🏗️ Project Architecture

```
FoldNet/
├── foldnet/
│   ├── data/                 # Data Pipeline (Shubham)
│   │   ├── dataset.py        # PyTorch Dataset & Collate logic
│   │   ├── preprocess_*.py   # Feature & Label extraction scripts
│   ├── models/               # Model Architectures (Shivam)
│   │   ├── encoders.py       # CNN, BiLSTM, Transformer implementations
│   │   ├── heads.py          # SS & Contact prediction heads
│   │   └── foldnet.py        # Multi-task LightningModule
│   ├── evaluation/           # Metrics & Plotting (Tanishka & Vaibhav)
│   │   ├── metrics_ss.py     # Q3, MCC, Confusion Matrix
│   │   ├── metrics_contacts.py# Precision@L, Long-range precision
│   │   └── visualisation.py  # Static PNG generation
├── configs/                  # Experiment Configurations (YAML)
├── results/                  # Checkpoints, Logs, and Visualisations
├── scripts/                  # Master Pipeline & Benchmarking
└── tests/                    # Unit Tests (data & model verification)
```

---

## 🚀 Getting Started

### 1. Installation
```powershell
git clone https://github.com/Yuyutsu01/FoldNet.git
cd FoldNet
pip install -r requirements.txt
```

### 2. Preprocessing
Ensure your ESM-2 embeddings and contact maps are in `data/processed/`.
```powershell
python foldnet/data/preprocess_contacts.py
```

### 3. Training
Run an experiment using a YAML config:
```powershell
python run.py --config configs/baseline_cnn.yaml
```

### 4. Benchmarking & Evaluation
Compare all models and generate a report:
```powershell
python scripts/benchmark.py --cnn results/checkpoints/cnn_best.ckpt
```

---

## 📊 Visualisation Examples

FoldNet generates detailed visual reports for every protein in the validation set:

*   **Secondary Structure:** Predicted vs. True labels (Red=Helix, Yellow=Sheet, Green=Coil).
*   **Contact Maps:** Side-by-side comparison of Predicted probabilities vs. Ground truth binary maps.

---

## 🏆 Performance Benchmarks (CB513)

| Model | Q3 Accuracy (%) | MCC (Macro) | Precision@L |
| :--- | :--- | :--- | :--- |
| **BiLSTM** | **83.66%** | **0.7448** | **0.0258** |
| **CNN** | 82.93% | 0.7326 | 0.0226 |
| **Transformer** | Training... | Training... | Training... |

---

## 👥 Contributors

*   **Shubham** — Data Engineering & Feature Extraction
*   **Shivam** — Model Architecture & Multi-task Loss
*   **Tanishka** — Evaluation Metrics & Quantitative Analysis
*   **Vaibhav** — Master Pipeline & Experiment Integration

---
*Developed for the Advanced Bioinformatics Modeling Project.*
