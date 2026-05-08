import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import yaml
import pandas as pd
from tabulate import tabulate

from foldnet.data.dataset import get_dataloaders
from foldnet.utils.predict import load_model, predict
from foldnet.evaluation.metrics_ss import evaluate_metrics

def main():
    parser = argparse.ArgumentParser(description="Benchmark FoldNet Architectures")
    parser.add_argument("--cnn", type=str, help="Path to CNN checkpoint")
    parser.add_argument("--bilstm", type=str, help="Path to BiLSTM checkpoint")
    parser.add_argument("--transformer", type=str, help="Path to Transformer checkpoint")
    parser.add_argument("--config", type=str, default="configs/baseline_cnn.yaml", help="Path to config for dataloading")
    parser.add_argument("--out_csv", type=str, default="results/benchmark_report.csv", help="Output CSV path")
    parser.add_argument("--out_md", type=str, default="results/benchmark_report.md", help="Output MD path")
    args = parser.parse_args()

    # Load data
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
    # Usually we benchmark on a test set, here we use val_loader of fold 0 for demonstration
    _, val_loader = get_dataloaders(fold=0, config=config)

    models_to_test = {
        'CNN': args.cnn,
        'BiLSTM': args.bilstm,
        'Transformer': args.transformer
    }

    results = []

    for name, ckpt_path in models_to_test.items():
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"Skipping {name}: Checkpoint not found at {ckpt_path}")
            continue
            
        print(f"\nBenchmarking {name} from {ckpt_path}...")
        model = load_model(ckpt_path)
        
        ss_p, ss_t, c_p, c_t, seq_lens, prot_ids = predict(model, val_loader)
        metrics = evaluate_metrics(ss_p, ss_t, c_p, c_t, seq_lens)
        
        results.append({
            'Model': name,
            'Q3 (%)': metrics['Q3'],
            'MCC (Macro)': metrics['MCC_Macro'],
            'Precision@L': metrics['Precision@L'],
            'Precision@L/2': metrics['Precision@L/2'],
            'Precision@L/5': metrics['Precision@L/5'],
            'Long-Range Prec': metrics['LongRange_Precision']
        })

    if not results:
        print("No models were benchmarked.")
        return

    df = pd.DataFrame(results)
    
    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV to {args.out_csv}")
    
    # Save Markdown Table
    md_table = tabulate(df, headers='keys', tablefmt='github', showindex=False, floatfmt=".4f")
    with open(args.out_md, 'w') as f:
        f.write("# Architecture Benchmarking Report\n\n")
        f.write(md_table)
        f.write("\n")
        
    print(f"Saved Markdown to {args.out_md}")
    print("\n" + md_table)

if __name__ == "__main__":
    main()
