"""
preprocess_ss.py  
===============================================
Parses the Princeton CB513 and CullPDB .npy files into clean CSVs
with protein sequences and Q3 secondary structure labels.

Input files (Princeton ICML2014 format):
  - cb513+profile_split1.npy.gz      → 514 proteins, shape (514, 39900)
  - cullpdb+profile_5926_filtered.npy.gz → 5926 proteins, shape (5926, 39900)

Both reshape to (N, 700, 57) where:
  [:, :, 0:22]  = amino acid one-hot  (order: A C E D G F I H K M L N Q P S R T W V Y X NoSeq)
  [:, :, 22:31] = SS 8-class one-hot  (order: L B E G I H S T NoSeq)
  [:, :, 31]    = sequence mask (1 = NoSeq / padding)

Output:
  data/processed/cb513_ss_labels.csv
  data/processed/cullpdb_ss_labels.csv
  Both have columns: protein_id, sequence, ss_labels (comma-separated Q3 ints), length

Usage:
  python -m foldnet.data.preprocess_ss \\
      --cb513   data/raw/cb513+profile_split1.npy \\
      --cullpdb data/raw/cullpdb+profile_5926_filtered.npy \\
      --out_dir data/processed
"""

import argparse
import os
import numpy as np
import pandas as pd 

# ── Princeton file layout ──────────────────────────────────────────────────────
SEQ_LEN       = 700    # all proteins zero-padded to this length
TOTAL_FEAT    = 57     # features per residue

# Amino acid order in the one-hot block [0:22]
AA_ORDER = list('ACEDGFIHKMLNQPSRTWVY') + ['X', 'NoSeq']

# SS8 label order in the one-hot block [22:31]
SS8_ORDER = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

# Map SS8 → Q3 integer classes  (0=Helix, 1=Strand, 2=Coil)
DSSP8_TO_Q3 = {
    'H': 0, 'G': 0, 'I': 0,   # all helix types → Helix
    'E': 1, 'B': 1,             # strand + bridge → Strand
    'L': 2, 'S': 2, 'T': 2,    # loop / bend / turn → Coil
}

# Non-standard amino acids to filter out
NON_STANDARD = {'X'}
MIN_LEN = 30    # discard sequences shorter than this


# ── Core parsing function ──────────────────────────────────────────────────────

def parse_npy_file(npy_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load a Princeton-format .npy file and return a clean DataFrame.
    Args:
        npy_path:     path to the .npy file (already gunzipped)
        dataset_name: prefix used for protein_id column (e.g. 'cb513')
    Returns:
        DataFrame with columns: protein_id, sequence, ss_labels, length
    """
    print(f"[preprocess_ss] Loading {npy_path} ...")
    raw = np.load(npy_path, allow_pickle=True, mmap_mode='r')
    data = raw.reshape(-1, SEQ_LEN, TOTAL_FEAT)
    N = len(data)
    print(f"[preprocess_ss] Loaded {N} proteins, shape {data.shape}")

    records = []
    skipped_short = 0
    skipped_nonstandard = 0

    for i in range(N):
        sample = data[i]   # (700, 57)

        # ── Find real sequence length (col 21 = 'NoSeq' flag in AA block) ──
        noseq_flag = sample[:, 21]          # 1.0 where padding
        real_mask  = noseq_flag == 0        # True = real residue
        real_len   = int(real_mask.sum())

        # ── Length filter ──
        if real_len < MIN_LEN:
            skipped_short += 1
            continue

        # ── Decode amino acid sequence ──
        aa_onehot = sample[:real_len, 0:22]          # (L, 22)
        aa_idx    = aa_onehot.argmax(axis=1)          # (L,)
        seq       = ''.join(AA_ORDER[j] for j in aa_idx)

        # ── Non-standard AA filter ──
        if any(aa in NON_STANDARD for aa in seq):
            skipped_nonstandard += 1
            continue

        # ── Decode SS labels (8-class → Q3) ──
        ss_onehot = sample[:real_len, 22:31]          # (L, 9)
        ss_idx    = ss_onehot.argmax(axis=1)           # (L,)
        ss8_chars = [SS8_ORDER[j] for j in ss_idx]
        q3_ints   = [DSSP8_TO_Q3.get(s, 2) for s in ss8_chars]  # unknown → Coil

        records.append({
            'protein_id': f'{dataset_name}_{i:04d}',
            'sequence':   seq,
            'ss_labels':  ','.join(map(str, q3_ints)),
            'length':     real_len,
        })

    df = pd.DataFrame(records)
    print(f"[preprocess_ss] {dataset_name}: {len(df)} kept | "
          f"{skipped_short} too short | {skipped_nonstandard} non-standard AA")
    if len(df) > 0:
        print(f"[preprocess_ss] Length  →  min={df.length.min()}  "
              f"max={df.length.max()}  mean={df.length.mean():.1f}")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cb513_path: str, cullpdb_path: str | None, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ── CB513 (test benchmark) ──
    df_cb513 = parse_npy_file(cb513_path, 'cb513')
    cb_out = os.path.join(out_dir, 'cb513_ss_labels.csv')
    df_cb513.to_csv(cb_out, index=False)
    print(f"[preprocess_ss] Saved → {cb_out}\n")

    # ── CullPDB (training data) ── optional
    if cullpdb_path and os.path.exists(cullpdb_path):
        df_cull = parse_npy_file(cullpdb_path, 'cull')
        cull_out = os.path.join(out_dir, 'cullpdb_ss_labels.csv')
        df_cull.to_csv(cull_out, index=False)
        print(f"[preprocess_ss] Saved → {cull_out}\n")

    print("[preprocess_ss] Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse CB513 / CullPDB .npy files → CSV')
    parser.add_argument('--cb513',   required=True, help='Path to cb513+profile_split1.npy')
    parser.add_argument('--cullpdb', default=None,  help='Path to cullpdb+profile_5926_filtered.npy')
    parser.add_argument('--out_dir', default='data/processed', help='Output directory')
    args = parser.parse_args()
    main(args.cb513, args.cullpdb, args.out_dir)