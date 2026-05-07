"""
extract_cull_pdbids.py  —
========================================
The Princeton CullPDB .npy format does NOT store PDB IDs inside the array.
Instead, this script uses the RCSB sequence search API to match CullPDB
sequences back to their PDB chains.

Since CullPDB proteins ARE PDB structures (that's the whole point of the
dataset), matches at >=99% identity are almost always exact hits.

Run:
  python -m foldnet.data.extract_cull_pdbids \
      --csv   data/processed/cullpdb_ss_labels.csv \
      --out   data/processed/cullpdb_pdbids.csv \
      --limit 0

  --limit 0 = process all proteins (default)
  --limit 50 = process only first 50 (for quick testing)
"""

import os
import time
import argparse
import requests
import pandas as pd

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def search_rcsb_sequence(sequence: str, identity: float = 0.99) -> dict | None:
    """
    Query RCSB PDB for chains matching sequence at given identity.
    Returns {'pdb_id': '1ABC', 'chain_id': 'A'} or None.
    """
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
        r = requests.post(RCSB_SEARCH_URL, json=query, timeout=20)
        if r.status_code != 200:
            return None
        results = r.json().get("result_set", [])
        if not results:
            return None
        # identifier is like "1ABC.A"
        parts = results[0].get("identifier", "").split(".")
        if len(parts) == 2:
            return {"pdb_id": parts[0].upper(), "chain_id": parts[1]}
        return None
    except Exception:
        return None


def main(csv_path: str, out_path: str, limit: int = 0, delay: float = 0.3):
    df = pd.read_csv(csv_path)
    if limit > 0:
        df = df.head(limit)

    total = len(df)
    print(f"[pdbids] Searching RCSB for {total} CullPDB proteins ...")
    print(f"[pdbids] Estimated time: ~{total * delay / 60:.0f} min at {delay}s/request")

    records = []
    found   = 0
    missed  = 0

    for i, row in df.iterrows():
        pid = row["protein_id"]
        seq = row["sequence"]

        # Try at 99% identity first (exact match), then fall back to 95%
        hit = search_rcsb_sequence(seq, identity=0.99)
        if not hit:
            hit = search_rcsb_sequence(seq, identity=0.95)

        if hit:
            records.append({
                "protein_id": pid,
                "sequence":   seq,
                "length":     row["length"],
                "ss_labels":  row["ss_labels"],
                "pdb_id":     hit["pdb_id"],
                "chain_id":   hit["chain_id"],
            })
            found += 1
            if found % 50 == 1 or i < 10:
                print(f"  [{i+1}/{total}] {pid} -> {hit['pdb_id']}.{hit['chain_id']}")
        else:
            records.append({
                "protein_id": pid,
                "sequence":   seq,
                "length":     row["length"],
                "ss_labels":  row["ss_labels"],
                "pdb_id":     "",
                "chain_id":   "",
            })
            missed += 1
            if missed <= 5 or i < 10:
                print(f"  [{i+1}/{total}] {pid} -> no match")

        time.sleep(delay)

        if (i + 1) % 100 == 0:
            pct = found / (i + 1) * 100
            print(f"\n--- [{i+1}/{total}] found={found} missed={missed} ({pct:.0f}% match rate) ---\n")

    result = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\n[pdbids] Done - {found}/{total} matched ({found/total*100:.0f}%)")
    print(f"[pdbids] Saved -> {out_path}")
    print(f"\n[pdbids] Sample:")
    print(result[["protein_id", "pdb_id", "chain_id"]].head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default="foldnet/data/processed/cullpdb_ss_labels.csv")
    parser.add_argument("--out",   default="foldnet/data/processed/cullpdb_pdbids.csv")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N proteins (0=all). Use 10 to test first.")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between API calls (default 0.3)")
    args = parser.parse_args()
    main(args.csv, args.out, args.limit, args.delay)