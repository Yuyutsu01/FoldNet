import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import yaml
import numpy as np

from foldnet.data.dataset import get_dataloaders
from foldnet.utils.predict import load_model, predict
from foldnet.evaluation.metrics_ss import calculate_q3, calculate_mcc
from foldnet.evaluation.metrics_contacts import calculate_precision_at_l, calculate_long_range_precision

def main():
    parser = argparse.ArgumentParser(description="Export predictions for dashboard")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g. BiLSTM)")
    parser.add_argument("--config", type=str, default="configs/baseline_cnn.yaml", help="Path to config for dataloading")
    parser.add_argument("--out_dir", type=str, default="data/dashboard", help="Output directory")
    args = parser.parse_args()

    # Load data config
    with open(args.config, 'r') as f:
        config_full = yaml.safe_load(f)
        
    config = {}
    config.update(config_full.get('model', {}))
    config.update(config_full.get('training', {}))
    config.update(config_full.get('data', {}))
    
    if 'ss_csv' not in config:
        config['ss_csv'] = 'foldnet/data/processed/cb513_ss_labels.csv'
    if 'splits_json' not in config:
        config['splits_json'] = 'foldnet/data/processed/cb513_splits_5fold/splits.json'
    if 'esm_dir' not in config:
        config['esm_dir'] = 'foldnet/data/processed/basic_features'

    print("Loading test data...")
    _, val_loader = get_dataloaders(fold=0, config=config)

    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt)
    
    print("Running inference...")
    ss_p_list, ss_t_list, c_p_list, c_t_list, seq_lens, prot_ids, seqs = predict(model, val_loader)
    
    os.makedirs(os.path.join(args.out_dir, "proteins"), exist_ok=True)
    
    metadata = []
    
    # Process each protein
    print("Exporting data...")
    for i in range(len(prot_ids)):
        pid = prot_ids[i]
        L = seq_lens[i]
        
        ss_p = ss_p_list[i]
        ss_t = ss_t_list[i]
        c_p = c_p_list[i]
        c_t = c_t_list[i]
        
        q3 = calculate_q3(ss_p, ss_t)
        _, mcc = calculate_mcc(ss_p, ss_t)
        precisions = calculate_precision_at_l(c_p, c_t, L)
        p_L = precisions['Precision@L']
        lr_p = calculate_long_range_precision(c_p, c_t, L)
        
        # Format for JSON
        # Round contact maps to 3 decimal places to save space
        c_p_rounded = np.round(c_p, 3).tolist()
        c_t_list_format = c_t.tolist()
        
        protein_data = {
            "protein_id": pid,
            "sequence": seqs[i],
            "length": L,
            "metrics": {
                "Q3": float(q3),
                "MCC": float(mcc),
                "Precision@L": float(p_L),
                "LongRangePrecision": float(lr_p)
            },
            "true_ss": ss_t.tolist(),
            "pred_ss": ss_p.tolist(),
            "true_contacts": c_t_list_format,
            "pred_contacts": c_p_rounded
        }
        
        with open(os.path.join(args.out_dir, "proteins", f"{pid}.json"), "w") as f:
            json.dump(protein_data, f)
            
        metadata.append({
            "protein_id": pid,
            "length": L,
            "Q3": float(q3),
            "MCC": float(mcc),
            "Precision@L": float(p_L)
        })
        
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    print(f"Exported {len(prot_ids)} proteins to {args.out_dir}")

if __name__ == "__main__":
    main()
