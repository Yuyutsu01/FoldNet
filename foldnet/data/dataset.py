"""
dataset.py 
=========================================
ProteinDataset + collate_fn + get_dataloaders()

Integration with Shivam's train.py:
    train_loader, val_loader = get_dataloaders(fold=0, config=config)
    trainer, model = train_foldnet(train_loader, val_loader, config)
"""

import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class ProteinDataset(Dataset):

    def __init__(self, df, esm_dir, contact_dir=None):
        self.df          = df.reset_index(drop=True)
        self.esm_dir     = esm_dir
        self.contact_dir = contact_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['protein_id']
        L   = int(row['length'])

        # ── features (L, 1280) ──
        emb_path = os.path.join(self.esm_dir, f"{pid}.npy")
        feat     = np.load(emb_path).astype(np.float32)[:L]
        features = torch.from_numpy(feat)

        # ── ss_labels (L,) ──
        labels    = list(map(int, str(row['ss_labels']).split(',')))
        ss_labels = torch.tensor(labels[:L], dtype=torch.long)

        # ── contact_map (L, L) — zeros if no file ──
        contact_map = torch.zeros(L, L, dtype=torch.float32)
        if self.contact_dir:
            cmap_path = os.path.join(self.contact_dir, f"{pid}.npz")
            if os.path.exists(cmap_path):
                cmap        = np.load(cmap_path)['contact_map'].astype(np.float32)
                contact_map = torch.from_numpy(cmap[:L, :L])

        return {
            'features':    features,     # (L, 1280)
            'ss_labels':   ss_labels,    # (L,)
            'contact_map': contact_map,  # (L, L)
            'seq_len':     L,
            'protein_id':  pid,
        }


def collate_fn(batch):
    """Pads variable-length proteins to max length in batch."""
    B     = len(batch)
    L_max = max(b['seq_len'] for b in batch)
    F     = batch[0]['features'].shape[-1]

    features    = torch.zeros(B, L_max, F)
    ss_labels   = torch.full((B, L_max), -1, dtype=torch.long)  # -1 = ignore_index
    contact_map = torch.zeros(B, L_max, L_max)
    mask        = torch.ones(B, L_max, dtype=torch.bool)        # True = padding

    for i, b in enumerate(batch):
        L = b['seq_len']
        features[i,    :L]    = b['features']
        ss_labels[i,   :L]    = b['ss_labels']
        contact_map[i, :L, :L] = b['contact_map']
        mask[i,        :L]    = False   # real residues

    return {
        'features':    features,     # (B, L_max, 1280)
        'ss_labels':   ss_labels,    # (B, L_max)
        'contact_map': contact_map,  # (B, L_max, L_max)
        'mask':        mask,         # (B, L_max) True=padding
    }


def get_dataloaders(fold: int, config: dict):
    """
    Returns (train_loader, val_loader) for a given CV fold.
    Called by Shivam's run_experiment.py.
    """
    df     = pd.read_csv(config['ss_csv'])
    splits = json.load(open(config['splits_json']))

    train_idx = splits['folds'][str(fold)]['train']
    val_idx   = splits['folds'][str(fold)]['val']

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = ProteinDataset(train_df, config['esm_dir'], config.get('contact_dir'))
    val_ds   = ProteinDataset(val_df,   config['esm_dir'], config.get('contact_dir'))

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.get('batch_size', 8),
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = config.get('num_workers', 15),
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.get('batch_size', 8),
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = config.get('num_workers', 15),
        pin_memory  = True,
    )

    print(f"[dataset] Fold {fold} → train={len(train_ds)}  val={len(val_ds)}")
    return train_loader, val_loader