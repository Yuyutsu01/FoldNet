<p align="center">
  <img src="assets/foldnet_banner.png" alt="FoldNet Banner" width="750"/>
</p>

<h1 align="center">FoldNet</h1>

<p align="center">
  <b>Deep Learning Framework for Multi-Task Protein Structural Analysis</b><br>
  <i>Transforming amino-acid sequences into secondary structure predictions & 2D contact maps.</i>
</p>

---

## 🛠️ Technologies Used

| Domain | Technologies / Libraries |
| :--- | :--- |
| **Deep Learning & ML** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![Lightning](https://img.shields.io/badge/PyTorch_Lightning-792EE5?style=flat-square&logo=pytorchlightning&logoColor=white) **ESM-2 Transformer** (`fair-esm`), **BiLSTM**, **Residual 1D/2D CNNs** |
| **Bioinformatics & Science** | ![Biopython](https://img.shields.io/badge/Biopython-3776AB?style=flat-square&logo=python&logoColor=white) **Biopython**, **NumPy**, **Pandas**, **SciPy**, **Scikit-learn**, **h5py** |
| **Web Dashboard & API** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-4998F5?style=flat-square&logo=python&logoColor=white) **FastAPI**, **Uvicorn**, **Pydantic** |
| **3D & 2D Visualization** | **Plotly.js**, **3Dmol.js**, **HTML5**, **Vanilla CSS**, **JavaScript**, **Bootstrap 5** |
| **DevOps & Monitoring** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) **Docker**, **Docker Compose**, **TensorBoard**, **Weights & Biases (W&B)** |

---

## Overview
FoldNet is a deep learning framework for protein structural analysis that transforms amino-acid sequences into biologically meaningful structural representations.

The system combines Evolutionary Scale Modeling (ESM-2) with task-specific neural prediction heads to perform two complementary structural prediction tasks:
* **Secondary Structure Prediction**: Three-class classification:
  * **H** — $\alpha$-helix
  * **E** — $\beta$-sheet
  * **C** — coil
* **Contact Map Prediction**: Predicts residue-residue contact probabilities. Contacts are defined using an 8 Å distance threshold.

The architecture combines global sequence representations from a pretrained protein language model with specialized 1D and 2D neural components for local structural modeling and long-range residue interactions.

* **Input**: Amino-acid sequence
* **Output**: Secondary structure + residue-residue contact probability map

---

## Why FoldNet?
Protein structure contains information that cannot be recovered from sequence alone using conventional sequence models.

FoldNet addresses this by combining:
* Evolutionary representations from ESM-2
* Bidirectional sequence modeling using BiLSTM layers
* Residual convolutional processing for local structural patterns
* 2D pairwise representations for residue-residue interactions
* Multi-task prediction of complementary structural properties
* Interactive structural visualization
* Reproducible evaluation and benchmarking

The goal is not simply to classify residues. The system attempts to preserve structural relationships across the entire sequence.

---

## Key Capabilities

| Capability | Description |
| :--- | :--- |
| **ESM-2 Embeddings** | Extracts 1280-dimensional contextual representations from a pretrained 650M-parameter protein language model |
| **Secondary Structure** | Predicts H/E/C residue-level structural classes |
| **Contact Maps** | Predicts pairwise residue contact probabilities |
| **Multi-Task Architecture** | Shares sequence representations across complementary structural tasks |
| **Benchmarking** | Evaluates multiple model variants using structural metrics |
| **Interactive Dashboard** | FastAPI-powered interface for inference and visualization |
| **Contact Visualization** | Interactive probability heatmaps and structural maps |
| **Dataset Evaluation** | Supports evaluation against curated structural datasets |
| **Docker Support** | Containerized deployment for reproducible environments |

---

## Architecture
FoldNet follows a multi-stage representation and prediction pipeline:

```text
┌──────────────────────┐
│ Amino Acid Sequence  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ESM-2 Transformer   │
│     650M Params      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   1280-dim Residue   │
│ Contextual Embedding │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Hybrid Encoder    │
│  Stacked BiLSTM      │
│  Dropout             │
│  LayerNorm           │
└──────────┬───────────┘
           │
 ┌─────────┴───────────┐
 │                     │
 ▼                     ▼
┌──────────────────┐ ┌────────────────────┐
│ 1D Residual CNN  │ │ Pairwise / 2D Head │
│  SS Prediction   │ │ Contact Prediction │
└────────┬─────────┘ └──────────┬─────────┘
         │                      │
         ▼                      ▼
┌──────────────────┐ ┌────────────────────┐
│    Secondary     │ │ Contact Probability│
│ Structure H/E/C  │ │        Map         │
└──────────────────┘ └────────────────────┘
```

### Representation Flow
For a protein sequence of length $L$:

```text
Amino Acid Sequence
       │
       ▼
     ESM-2
       │
       ▼
L × 1280 Residue Embeddings
       │
       ▼
Hybrid Sequence Encoder
       ├──────────────► 1D Structural Features
       └──────────────► Pairwise Structural Features
                              │
                              ▼
                      L × L Contact Map
```

The model therefore operates at both:
* **1D level**: residue-wise structural classification
* **2D level**: pairwise residue interaction modeling

---

## Model Components

### 1. ESM-2 Representation Layer
FoldNet uses a pretrained ESM-2 650M protein language model to generate contextual residue representations.
For a sequence of length $L$, the model produces:
$$L \times 1280$$
representations.
These embeddings encode sequence context learned from large-scale protein sequence data.

### 2. Hybrid Sequence Encoder
The contextual representations are passed through a sequence-processing stack containing:
* Bidirectional LSTM layers
* Dropout
* Layer normalization

The BiLSTM allows the model to incorporate both upstream and downstream sequence context when constructing structural representations.

### 3. Secondary Structure Head
The secondary structure branch performs residue-level classification:

```text
Embedding
    │
    ▼
1D Residual CNN
    │
    ▼
Classification Head
    ├── H → Helix
    ├── E → Sheet
    └── C → Coil
```

The output is a sequence of structural labels corresponding to the input residues.

### 4. Contact Map Head
The contact branch converts residue-level representations into pairwise features.

```text
Residue Features
    │
    ▼
Pairwise Feature Construction
    │
    ▼
2D Convolutional Processing
    │
    ▼
Contact Prediction
    │
    ▼
L × L Probability Matrix
```

Each matrix element represents the predicted probability that two residues form a structural contact under the selected distance criterion.

---

## Results
FoldNet has been evaluated on the CB513 test set using multiple architectural variants.

### Benchmark Comparison

| Architecture | Q3 Accuracy | Macro MCC | Precision@L | Long-Range Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Residual CNN** | **83.66%** | **0.7448** | 0.0258 | 0.0120 |
| **BiLSTM Fusion** | 82.55% | 0.7270 | **0.1022** | **0.0845** |
| **Transformer-Hybrid** | 82.93% | 0.7326 | 0.0226 | 0.0105 |

### Interpretation
The benchmark highlights a useful architectural trade-off:
* **Residual CNN** achieves the strongest Q3 secondary-structure accuracy in this comparison.
* **BiLSTM Fusion** produces substantially stronger contact-map precision, particularly for long-range interactions.
* **Transformer-Hybrid** provides competitive structural classification while combining pretrained transformer representations with downstream structural modeling.

*These results should be interpreted as FoldNet's reported experimental benchmark results, rather than as a universal state-of-the-art claim across all published protein structure prediction systems.*

---

## Dashboard
FoldNet includes an interactive inference and visualization interface powered by FastAPI.

### Dashboard Capabilities
* Real-time sequence inference
* Secondary structure visualization
* Contact probability heatmaps
* Structural comparison
* Dataset-based evaluation
* Prediction inspection
* Benchmark analysis
* Interactive Plotly visualizations

### Inference Pipeline

```text
User Sequence
     │
     ▼
  FastAPI
     │
     ▼
ESM-2 Embedding
     │
     ▼
FoldNet Model
     ├──────────────► Secondary Structure
     └──────────────► Contact Map
            │
            ▼
   Interactive Viewer
```

---

## Quick Start

### Requirements

#### Software
* Python 3.9 – 3.11
* PyTorch 2.x
* Git
* Docker (optional)

#### Hardware
For practical ESM-2 inference:
* NVIDIA GPU recommended
* 4 GB+ VRAM recommended
* CPU inference is possible but substantially slower

#### Storage
Approximately 5 GB or more may be required for model weights, datasets, and caches depending on the selected configuration.

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/Yuyutsu01/FoldNet.git
cd FoldNet
```

#### 2. Create a virtual environment
**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Start the API server
```bash
python -m uvicorn viewer.app:app --reload --port 8000
```

Open: [http://localhost:8000](http://localhost:8000)

### Docker Deployment
Docker provides a reproducible runtime environment and simplifies deployment across development machines and servers.

**Build and start**
```bash
docker compose up --build -d
```

**View logs**
```bash
docker compose logs -f
```

**Stop services**
```bash
docker compose down
```

The dashboard is exposed on: [http://localhost:8000](http://localhost:8000)

> **Note:** The Docker configuration is designed to minimize container size by separating runtime dependencies from host-side model caching where applicable.

---

## Repository Structure

```text
FoldNet/
│
├── foldnet/
│   ├── data/
│   │   ├── datasets/
│   │   └── preprocessing/
│   │
│   ├── models/
│   │   ├── encoders/
│   │   ├── heads/
│   │   ├── losses/
│   │   └── training/
│   │
│   └── evaluation/
│       ├── metrics/
│       └── plots/
│
├── configs/
│   └── experiment configurations
│
├── docs/
│   └── technical documentation
│
├── results/
│   └── checkpoints and experiment outputs
│
├── viewer/
│   ├── app.py
│   └── static/
│       ├── templates/
│       └── assets/
│
├── Dockerfile
├── docker-compose.yml
├── run.py
├── requirements.txt
└── README.md
```

---

## Reproducibility
FoldNet is designed around reproducible experimentation.

Experiments should record:
* Model configuration
* Dataset split
* Random seed
* Training parameters
* Checkpoint
* Evaluation metrics
* Hardware configuration
* Software environment

Recommended experiment configuration:
```yaml
experiment:
  name: cb513_baseline
  seed: 42
  model:
    backbone: esm2
    variant: 650m
    tasks:
      secondary_structure: true
      contact_map: true
  contact:
    threshold_angstrom: 8.0
```

For scientific comparisons, keep dataset splits and evaluation procedures fixed across model variants.

---

## Evaluation
FoldNet supports evaluation across complementary structural metrics.

### Secondary Structure
* **Q3 Accuracy**: Measures the percentage of residues correctly classified into **H** / **E** / **C**.
* **Matthews Correlation Coefficient (MCC)**: MCC provides a more informative measure when class distributions are imbalanced.

### Contact Prediction
FoldNet evaluates contact prediction using:
* Precision@L
* Long-range precision
* Contact probability matrices
* Pairwise structural comparisons

Long-range contacts are particularly important because they test whether the model can capture relationships between residues that are distant in sequence but close in three-dimensional structure.

---

## Data
FoldNet's structural evaluation pipeline is designed around curated protein datasets, including CB513 for secondary-structure benchmarking.

The project separates:
```text
Raw Dataset
     │
     ▼
Preprocessing
     │
     ▼
Sequence / Structure Alignment
     │
     ▼
Model Input
     │
     ▼
Prediction
     │
     ▼
Metric Evaluation
```

Dataset files should be obtained and stored according to the project's dataset configuration and licensing requirements.

---

## API
The FastAPI backend exposes the inference layer used by the dashboard.

Typical workflow:
```text
POST Sequence
     │
     ▼
Validate Input
     │
     ▼
Generate ESM-2 Embeddings
     │
     ▼
Run FoldNet
     ├──────────────► Secondary Structure
     └──────────────► Contact Map
```

The interactive API documentation is available when the FastAPI server is running: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Development

**Run locally**
```bash
python -m uvicorn viewer.app:app --reload --port 8000
```

**Run the project entry point**
```bash
python run.py
```

### Recommended Development Workflow
```text
Feature / Experiment
     │
     ▼
Configuration
     │
     ▼
Implementation
     │
     ▼
Unit / Integration Testing
     │
     ▼
Experiment
     │
     ▼
Evaluation
     │
     ▼
Results
     │
     ▼
Documentation
```

---

## Research & Engineering Principles
FoldNet follows several principles intended to keep the project suitable for continued research and engineering development:

* **Reproducibility**: Experiments should be deterministic where practical and report their configuration.
* **Modularity**: Encoders, prediction heads, losses, datasets, and evaluation modules should remain independently replaceable.
* **Benchmark-Driven Development**: Architectural changes should be evaluated against consistent datasets and metrics.
* **Separation of Concerns**: The inference API, machine-learning pipeline, visualization layer, and experiment infrastructure are maintained as separate components.
* **Extensibility**: The architecture is designed to support additional structural prediction tasks without replacing the complete pipeline.

---

## Limitations
FoldNet currently focuses on secondary structure and residue-residue contact prediction.
It does not directly predict complete atomic 3D coordinates.

Consequently:
* Contact maps are not equivalent to full 3D structures.
* Secondary structure predictions do not fully describe protein geometry.
* ESM-2 inference can be computationally expensive.
* Prediction quality depends on sequence characteristics and training data.
* Benchmark results should not be interpreted as universal performance across every protein-structure prediction method.

> *Scientific honesty: the annoying part of research, and also the part that prevents README files from becoming fiction.*

---

## Roadmap

### Structural Modeling
* 3D Cartesian prediction head
* Coordinate-based structural reconstruction
* PDB generation pipeline
* Improved long-range contact modeling

### Multi-Chain Proteins
* Multi-chain sequence support
* Protein-protein interaction modeling
* Complex-level structural prediction

### Efficient Inference
* Quantized ESM-2 inference
* Memory-efficient embedding generation
* Batch inference
* Browser-compatible inference experiments

### Research Infrastructure
* Automated experiment tracking
* Expanded benchmark datasets
* Ablation studies
* Cross-dataset evaluation
* Model checkpoint registry

---

## Project Status
**Status:** Active Research & Development

FoldNet is an evolving research project. Architecture, benchmarks, APIs, and visualization components may change as experiments progress.

---

## Contributing
Contributions are welcome.

Before submitting a pull request:
1. Create a focused branch.
2. Keep changes modular.
3. Add or update tests where appropriate.
4. Document architectural changes.
5. Include benchmark results for model-related changes.
6. Ensure existing functionality remains intact.

**Example workflow:**
```bash
git checkout -b feature/contact-attention
git add .
git commit -m "feat: improve contact prediction head"
git push origin feature/contact-attention
```

Then open a pull request describing:
* What changed
* Why it changed
* Experimental results
* Known limitations

---

## Citation
If FoldNet contributes to your research, please cite the project:

```bibtex
@software{foldnet,
  title  = {FoldNet: Protein Structure Prediction Framework},
  author = {Shubham and Shivam and Tanishka and Vaibhav},
  year   = {2026},
  url    = {https://github.com/Yuyutsu01/FoldNet}
}
```

---

## License
FoldNet is released under the MIT License.
See [LICENSE](LICENSE) for the complete license text.

---

## Team
* **Shubham**
* **Shivam**
* **Tanishka**
* **Vaibhav**
