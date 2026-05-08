import os
import json
import urllib.request
import numpy as np
import torch
from foldnet.models.foldnet import FoldNet
from foldnet.utils.predict import load_model

def fetch_pdb_data(pdb_id: str) -> str:
    """Fetches PDB file content from RCSB."""
    # Often PDB IDs in datasets are 5 chars like 1A3A_A, we need the first 4
    base_pdb = pdb_id[:4].lower()
    url = f"https://files.rcsb.org/download/{base_pdb}.pdb"
    try:
        response = urllib.request.urlopen(url)
        return response.read().decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not fetch PDB for {base_pdb} from RCSB. Error: {e}")
        return ""

def export_3d_json(protein_id: str, ss_pred: np.ndarray, contact_probs: np.ndarray, 
                   out_path: str, contact_threshold: float = 0.7, pdb_data: str = None):
    """
    Exports predicted SS and contacts into a JSON format expected by the 3D viewer.
    """
    if pdb_data is None:
        pdb_data = fetch_pdb_data(protein_id)
        
    seq_len = len(ss_pred)
    contacts = []
    
    # Extract contacts above threshold
    i_idx, j_idx = np.where(contact_probs > contact_threshold)
    for i, j in zip(i_idx, j_idx):
        if i < j:  # upper triangle only to avoid duplicates
            prob = float(contact_probs[i, j])
            contacts.append([int(i), int(j), prob])
            
    data = {
        "pdb_id": protein_id,
        "pdb_data": pdb_data,
        "ss_labels": ss_pred.tolist(),
        "contacts": contacts
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f)
        
    print(f"Exported 3D viewer data to {out_path}")

def run_export_for_protein(model_path: str, protein_id: str, features: torch.Tensor, 
                           out_dir: str = "results/3d_data"):
    """
    Helper function to run inference for a single protein and export it.
    Assumes features tensor is (1, L, feature_dim)
    """
    model = load_model(model_path)
    model.eval()
    
    with torch.no_grad():
        ss_logits, contact_probs = model(features)
        
        # We assume batch size 1
        ss_pred = torch.argmax(ss_logits[0], dim=-1).cpu().numpy()
        c_probs = torch.sigmoid(contact_probs[0]).cpu().numpy()
        
    out_path = os.path.join(out_dir, f"{protein_id}_viewer.json")
    export_3d_json(protein_id, ss_pred, c_probs, out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--pdb_id", type=str, required=True, help="PDB ID to test")
    parser.add_argument("--out", type=str, default="results/3d_data/test.json", help="Output JSON path")
    args = parser.parse_args()
    
    # Normally we would load features here. For standalone testing, 
    # if one wants to run it directly, they need the embeddings.
    print("For direct script usage, please ensure you pass valid features. Or use via run.py --visualise")
