"""
extract_features.py  —  Shubham's Module (Step 3)
==================================================
Generates per-residue features:
  1. One-hot encoding        → (L, 20)
  2. Physicochemical props   → (L, 5)
  3. ESM-2 loader            → loads pre-saved .npy files (L, 1280)

Usage (one-hot + physchem only, no GPU needed):
  python -m foldnet.data.extract_features \
      --csv     data/processed/cb513_ss_labels.csv \
      --out_dir data/processed/basic_features
"""

import os
import argparse
import numpy as np
import pandas as pd

# ── Amino acid definitions ────────────────────────────────────────────────────
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX   = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# 5 physicochemical properties per AA
# [hydrophobicity, charge, polarity, volume, aromaticity]
PHYSCHEM = {
    'A': [ 1.800,  0.00, 0.00,  88.6, 0.0],
    'C': [ 2.500,  0.00, 1.48, 108.5, 0.0],
    'D': [-3.500, -1.00, 1.00, 111.1, 0.0],
    'E': [-3.500, -1.00, 1.00, 138.4, 0.0],
    'F': [ 2.800,  0.00, 0.00, 189.9, 1.0],
    'G': [-0.400,  0.00, 0.00,  60.1, 0.0],
    'H': [-3.200,  0.10, 0.58, 153.2, 1.0],
    'I': [ 4.500,  0.00, 0.00, 166.7, 0.0],
    'K': [-3.900,  1.00, 0.33, 168.6, 0.0],
    'L': [ 3.800,  0.00, 0.00, 166.7, 0.0],
    'M': [ 1.900,  0.00, 0.00, 162.9, 0.0],
    'N': [-3.500,  0.00, 1.00, 114.1, 0.0],
    'P': [-1.600,  0.00, 0.00, 112.7, 0.0],
    'Q': [-3.500,  0.00, 1.00, 143.8, 0.0],
    'R': [-4.500,  1.00, 1.00, 173.4, 0.0],
    'S': [-0.800,  0.00, 1.00,  89.0, 0.0],
    'T': [-0.700,  0.00, 1.00, 116.1, 0.0],
    'V': [ 4.200,  0.00, 0.00, 140.0, 0.0],
    'W': [-0.900,  0.00, 0.00, 227.8, 1.0],
    'Y': [-1.300,  0.00, 0.76, 193.6, 1.0],
}

# Pre-compute min/max for normalisation
_all_vals = np.array(list(PHYSCHEM.values()), dtype=np.float32)
_PHYS_MIN = _all_vals.min(axis=0)
_PHYS_MAX = _all_vals.max(axis=0)
_PHYS_RANGE = np.where(_PHYS_MAX - _PHYS_MIN == 0, 1.0, _PHYS_MAX - _PHYS_MIN)


# ── Feature functions ─────────────────────────────────────────────────────────

def one_hot_encode(seq: str) -> np.ndarray:
    """Returns (L, 20) float32. Unknown AA → all zeros."""
    enc = np.zeros((len(seq), 20), dtype=np.float32)
    for i, aa in enumerate(seq.upper()):
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            enc[i, idx] = 1.0
    return enc


def physchem_encode(seq: str) -> np.ndarray:
    """Returns (L, 5) float32, normalised to [0,1]."""
    enc = np.array(
        [PHYSCHEM.get(aa.upper(), [0.0]*5) for aa in seq],
        dtype=np.float32
    )
    enc = (enc - _PHYS_MIN) / _PHYS_RANGE
    return enc


def load_esm2(protein_id: str, esm_dir: str) -> np.ndarray:
    """Load pre-saved ESM-2 embedding. Returns (L, 1280) float32."""
    path = os.path.join(esm_dir, f"{protein_id}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ESM-2 embedding not found: {path}")
    return np.load(path).astype(np.float32)


def combine_features(seq: str, protein_id: str,
                     esm_dir: str = None,
                     use_esm: bool = True) -> np.ndarray:
    """
    Build full feature matrix for one protein.

    If use_esm=True and esm_dir provided → returns (L, 1305)  [onehot+physchem+esm]
    If use_esm=False                      → returns (L, 25)   [onehot+physchem only]
    """
    oh   = one_hot_encode(seq)       # (L, 20)
    phys = physchem_encode(seq)      # (L, 5)

    if use_esm and esm_dir:
        esm = load_esm2(protein_id, esm_dir)   # (L, 1280)
        # trim/pad ESM to match sequence length (safety)
        L = len(seq)
        if esm.shape[0] > L:
            esm = esm[:L]
        elif esm.shape[0] < L:
            pad = np.zeros((L - esm.shape[0], 1280), dtype=np.float32)
            esm = np.vstack([esm, pad])
        return np.concatenate([oh, phys, esm], axis=-1)   # (L, 1305)
    else:
        return np.concatenate([oh, phys], axis=-1)         # (L, 25)


# ── Script: save basic features (no GPU needed) ───────────────────────────────

def save_basic_features(csv_path: str, out_dir: str):
    """
    Save one-hot + physchem features for every protein as .npy files.
    Runs on CPU, no dependencies except numpy.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(f"[extract_features] Processing {len(df)} proteins -> {out_dir}")

    for i, row in df.iterrows():
        pid = row['protein_id']
        seq = row['sequence']
        feat = combine_features(seq, pid, esm_dir=None, use_esm=False)  # (L, 25)
        np.save(os.path.join(out_dir, f"{pid}.npy"), feat)
        if i % 100 == 0:
            print(f"  [{i+1}/{len(df)}] {pid} -> shape {feat.shape}")

    print(f"[extract_features] Done. {len(df)} files saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     default='foldnet/data/processed/cb513_ss_labels.csv')
    parser.add_argument('--out_dir', default='foldnet/data/processed/basic_features')
    args = parser.parse_args()
    save_basic_features(args.csv, args.out_dir)