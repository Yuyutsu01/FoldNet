"""
preprocess_contacts.py  —  Shubham's Module (Step 5)
=====================================================
Downloads PDB files via RCSB HTTPS, extracts Cb coordinates,
computes binary contact maps, saves as .npz files.

Usage — CB513:
  python -m foldnet.data.preprocess_contacts \
      --csv     data/processed/cb513_ss_labels.csv \
      --out_dir data/processed/pdb_contact_maps \
      --mode    rcsb

Usage — CullPDB:
  python -m foldnet.data.preprocess_contacts \
      --csv     data/processed/cullpdb_pdbids.csv \
      --out_dir data/processed/pdb_contact_maps \
      --mode    direct
"""

import os
import time
import argparse
import requests
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, is_aa

CONTACT_THRESHOLD = 8.0
MIN_SEQ_SEP       = 6
RCSB_SEARCH_URL   = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_PDB_URL      = "https://files.rcsb.org/download/{}.pdb"


def get_cb_coords(chain):
    coords = []
    for res in chain.get_residues():
        if not is_aa(res, standard=True):
            continue
        try:
            atom = res['CA'] if res.resname == 'GLY' else res['CB']
            coords.append(atom.get_vector().get_array())
        except KeyError:
            try:
                coords.append(res['CA'].get_vector().get_array())
            except KeyError:
                continue
    if len(coords) < 10:
        return None
    return np.array(coords, dtype=np.float32)


def compute_contact_map(coords):
    L    = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    contact = (dist < CONTACT_THRESHOLD).astype(np.float32)
    idx = np.arange(L)
    sep = np.abs(idx[:, None] - idx[None, :])
    contact[sep < MIN_SEQ_SEP] = 0.0
    return contact


def download_pdb(pdb_id: str, cache_dir: str) -> str:
    """Download PDB file via HTTPS. Returns local file path or empty string."""
    pdb_file = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_file):
        return pdb_file
    try:
        url  = RCSB_PDB_URL.format(pdb_id)
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(pdb_file, 'w') as f:
                f.write(resp.text)
            return pdb_file
    except Exception:
        pass
    return ''


def process_one_chain(pdb_id: str, chain_id: str,
                      out_path: str, cache_dir: str) -> bool:
    if os.path.exists(out_path):
        return True
    try:
        pdb_file = download_pdb(pdb_id, cache_dir)
        if not pdb_file:
            return False

        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        model     = structure[0]

        chain_ids = [c.id for c in model.get_chains()]
        if chain_id not in chain_ids:
            if not chain_ids:
                return False
            chain_id = chain_ids[0]

        coords = get_cb_coords(model[chain_id])
        if coords is None:
            return False

        cmap = compute_contact_map(coords)
        np.savez_compressed(out_path, contact_map=cmap)
        return True

    except Exception as e:
        print(f"    [warn] {pdb_id}_{chain_id}: {e}")
        return False


def search_rcsb(sequence: str, identity: float = 0.95) -> list:
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": identity,
                "sequence_type": "protein",
                "value": sequence
            }
        },
        "return_type": "polymer_instance",
        "request_options": {
            "results_verbosity": "minimal",
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": 3}
        }
    }
    try:
        r = requests.post(RCSB_SEARCH_URL, json=query, timeout=15)
        if r.status_code != 200:
            return []
        hits = []
        for item in r.json().get('result_set', []):
            parts = item.get('identifier', '').split('.')
            if len(parts) == 2:
                hits.append({'pdb_id': parts[0], 'chain_id': parts[1]})
        return hits
    except Exception:
        return []


def main(csv_path, out_dir, mode, delay=0.25):
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, '_pdb_cache')
    os.makedirs(cache_dir, exist_ok=True)

    df      = pd.read_csv(csv_path)
    success = 0
    failed  = 0
    total   = len(df)

    print(f"[contacts] {total} proteins  mode={mode}  out={out_dir}")

    for i, row in df.iterrows():
        pid      = row['protein_id']
        out_path = os.path.join(out_dir, f"{pid}.npz")

        if os.path.exists(out_path):
            success += 1
            continue

        if mode == 'rcsb':
            seq  = row['sequence']
            print(f"[{i+1}/{total}] {pid} (len={len(seq)}) searching RCSB ...")
            hits = search_rcsb(seq, identity=0.95)
            if not hits:
                hits = search_rcsb(seq, identity=0.80)
            if not hits:
                print(f"  ✗ no match")
                failed += 1
                time.sleep(delay)
                continue
            pdb_id   = hits[0]['pdb_id']
            chain_id = hits[0]['chain_id']

        elif mode == 'direct':
            pdb_id   = str(row.get('pdb_id', '')).strip()
            chain_id = str(row.get('chain_id', 'A')).strip()
            if len(pdb_id) != 4:
                failed += 1
                continue
            print(f"[{i+1}/{total}] {pid} → {pdb_id}.{chain_id}")

        ok = process_one_chain(pdb_id, chain_id, out_path, cache_dir)

        if ok:
            success += 1
            if mode == 'rcsb':
                print(f"  ✓ {pdb_id}.{chain_id}")
        else:
            failed += 1
            if mode == 'rcsb':
                print(f"  ✗ download failed")

        time.sleep(delay)

        if (i + 1) % 50 == 0:
            print(f"\n--- {i+1}/{total} | ✓{success} ✗{failed} ---\n")

    print(f"\n[contacts] ✓ DONE — saved={success}  failed={failed}  total={total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     required=True)
    parser.add_argument('--out_dir', default='data/processed/pdb_contact_maps')
    parser.add_argument('--mode',    default='rcsb', choices=['rcsb', 'direct'])
    parser.add_argument('--delay',   type=float, default=0.25)
    args = parser.parse_args()
    main(args.csv, args.out_dir, args.mode, args.delay)
