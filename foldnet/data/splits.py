"""
splits.py 
--------------------------------------------------------
Creates reproducible 5-fold cross-validation splits on CB513
and writes them to JSON for use by the DataLoader.

Usage:
  python -m foldnet.data.splits \\
      --csv     data/processed/cb513_ss_labels.csv \\
      --out_dir data/processed/cb513_splits_5fold
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def create_splits(csv_path: str, out_dir: str, n_folds: int = 5, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    n  = len(df)
    print(f"[splits] Loaded {n} proteins from {csv_path}")

    # Hold out a fixed 10% test set BEFORE any CV (sorted for reproducibility)
    np.random.seed(seed)
    all_idx   = np.arange(n)
    test_size = max(1, int(0.1 * n))
    test_idx  = np.sort(np.random.choice(all_idx, size=test_size, replace=False))
    trainval_idx = np.array([i for i in all_idx if i not in set(test_idx.tolist())])

    print(f"[splits] Hold-out test set: {len(test_idx)} proteins")
    print(f"[splits] Train+Val pool:    {len(trainval_idx)} proteins")

    # 5-fold CV on the train+val pool
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = {
        'test': test_idx.tolist(),
        'folds': {}
    }

    for fold, (train_rel, val_rel) in enumerate(kf.split(trainval_idx)):
        train_abs = trainval_idx[train_rel].tolist()
        val_abs   = trainval_idx[val_rel].tolist()
        splits['folds'][str(fold)] = {
            'train': train_abs,
            'val':   val_abs,
        }
        print(f"[splits]   Fold {fold}: train={len(train_abs)}  val={len(val_abs)}")

    out_path = os.path.join(out_dir, 'splits.json')
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"[splits]  Saved → {out_path}")
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      default='foldnet/data/processed/cb513_ss_labels.csv')
    parser.add_argument('--out_dir',  default='foldnet/data/processed/cb513_splits_5fold')
    parser.add_argument('--n_folds',  type=int, default=5)
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()
    create_splits(args.csv, args.out_dir, args.n_folds, args.seed)