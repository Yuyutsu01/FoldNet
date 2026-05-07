# FoldNet: Protein Structure & Secondary Structure Prediction

FoldNet is a deep learning project for predicting protein secondary structure and contacts from sequence data using the CullPDB dataset.

## Data Pipeline

The project uses the CullPDB dataset. The full processing pipeline consists of the following steps:

### 1. Extract PDB IDs
```bash
python -m foldnet.data.extract_cull_pdbids \
  --csv foldnet/data/processed/cullpdb_ss_labels.csv \
  --out foldnet/data/processed/cullpdb_pdbids.csv
```
- Recovers PDB IDs and chain IDs for CullPDB entries.
- Uses the RCSB Sequence Search API to match sequences back to their original PDB entries (since raw CullPDB files often lack PDB IDs).
- Output: `foldnet/data/processed/cullpdb_pdbids.csv`

### 2. Preprocess Secondary Structure Labels
```bash
python -m foldnet.data.preprocess_ss
```
- Filters sequences and generates clean secondary structure labels.

### 3. Preprocess Contact Maps
```bash
python -m foldnet.data.preprocess_contacts
```
- Downloads PDB structures for each protein in `cullpdb_pdbids.csv`.
- Computes residue-level contact maps from 3D coordinates.
- Output: `foldnet/data/processed/pdb_contact_maps/` — one `.npy` file per protein chain.

### 4. Extract Features
```bash
python -m foldnet.data.extract_features
```
- Generates features for model training from the processed data.

### 5. Dataset Splits
- `foldnet/data/splits.py` — Handles train/val/test splits.
- `foldnet/data/dataset.py` — PyTorch `Dataset` class for loading processed data.

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Yuyutsu01/FoldNet.git
   cd FoldNet
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
FoldNet/
├── foldnet/                  # Core Python package
│   ├── data/
│   │   ├── download.py           # Data download utilities
│   │   ├── extract_cull_pdbids.py# PDB ID recovery via RCSB API
│   │   ├── preprocess_ss.py      # Secondary structure label preprocessing
│   │   ├── preprocess_contacts.py# Contact map generation from PDB structures
│   │   ├── extract_features.py   # Feature extraction for training
│   │   ├── splits.py             # Train/val/test splitting
│   │   ├── dataset.py            # PyTorch Dataset class
│   │   └── utils.py              # Shared utilities
│   ├── models/                   # Model architecture definitions
│   └── utils/                    # Shared utilities (metrics, etc.)
├── configs/                  # Training configurations (YAML)
└── README.md
```

---

## Branch Structure

| Branch | Owner | Purpose |
|--------|-------|---------|
| `main` | — | Stable releases |
| `master` | — | Main development branch |
| `shubham` | Shubham | Data pipeline (PDB ID extraction, contact maps) |

---

## Contributors

- **Yuyutsu01** — Project lead
- **Shubham** — Data pipeline: PDB ID extraction, contact map preprocessing
- **Tanishka** — Evaluation & Visualization
- **Shivam** — Machine Learning Models (Encoders, Heads, Loss)
