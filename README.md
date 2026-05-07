# FoldNet: Protein Structure & Secondary Structure Prediction

FoldNet is a deep learning project for predicting protein secondary structure and contacts.

## Data Pipeline

The project uses the CullPDB dataset. The processing pipeline consists of several steps:

### 1. Data Preprocessing
- **Secondary Structure Labels**: `python -m foldnet.data.preprocess_ss`
  - Filters sequences and generates clean secondary structure labels.
- **PDB ID Extraction**: `python -m foldnet.data.extract_cull_pdbids`
  - Recovers PDB IDs and chain IDs for CullPDB entries.
  - Note: Since raw CullPDB files often lack PDB IDs, this script uses the RCSB Sequence Search API to match sequences back to their original PDB entries.

### 2. Feature Extraction
- `python -m foldnet.data.extract_features`
  - Generates features for model training.

## Installation & Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Extracting PDB IDs
To recover PDB IDs from the processed labels:
```bash
python -m foldnet.data.extract_cull_pdbids \
  --csv foldnet/data/processed/cullpdb_ss_labels.csv \
  --out foldnet/data/processed/cullpdb_pdbids.csv
```

## Project Structure
- `foldnet/`: Core package
  - `data/`: Data loading, preprocessing scripts, and processed data
  - `models/`: Model architecture definitions
- `configs/`: Training configurations (YAML)
- `data/`: Raw and processed data files (ignored by git)
