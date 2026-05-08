"""
FoldNet Main Entry Point
=======================
Usage:
python run.py --config configs/baseline_cnn.yaml --fold 0
tensorboard --logdir results/logs

"""

import os
import yaml
import argparse
import logging
import torch

from foldnet.data.dataset import get_dataloaders
from foldnet.models.train import train_foldnet
from foldnet.models.foldnet import FoldNet
from foldnet.utils.predict import predict
from foldnet.evaluation.metrics_ss import evaluate_metrics
from foldnet.evaluation.visualisation import create_visualisations
from scripts.export_for_3d import export_3d_json

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main():
    parser = argparse.ArgumentParser(description="Train FoldNet model")
    parser.add_argument("--config", type=str, default="configs/baseline_cnn.yaml", help="Path to YAML config file")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold to train (0-4)")
    parser.add_argument("--visualise", action="store_true", help="Run static and 3D visualisations after training/evaluation")
    parser.add_argument("--evaluate_only", action="store_true", help="Skip training and only run evaluation/visualisation")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (required for evaluate_only)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # 1. Load Configuration
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return

    with open(args.config, 'r') as f:
        config_full = yaml.safe_load(f)
    
    # Flatten config for easier passing
    config = {}
    config.update(config_full.get('model', {}))
    config.update(config_full.get('training', {}))
    config.update(config_full.get('experiment', {}))
    config.update(config_full.get('data', {}))
    
    # Ensure mandatory data paths are set if not in YAML
    # We use the defaults from our verified test_dataset setup
    if 'ss_csv' not in config:
        config['ss_csv'] = 'foldnet/data/processed/cb513_ss_labels.csv'
    if 'splits_json' not in config:
        config['splits_json'] = 'foldnet/data/processed/cb513_splits_5fold/splits.json'
    if 'esm_dir' not in config:
        # Default to basic features we just generated
        config['esm_dir'] = 'foldnet/data/processed/basic_features'
    
    # Ensure numeric types are correct (prevent string-to-float errors)
    for key in ['lr', 'lambda_contact', 'val_split']:
        if key in config and config[key] is not None:
            config[key] = float(config[key])
    for key in ['epochs', 'batch_size', 'accumulate_grad_batches', 'feature_dim', 'hidden_dim', 'num_classes']:
        if key in config and config[key] is not None:
            config[key] = int(config[key])
    
    # Override feature_dim if using basic features
    if 'basic_features' in config['esm_dir']:
        config['feature_dim'] = 25
        logger.info("Using basic features (dim=25)")

    logger.info(f"Starting Experiment: {config.get('name', 'unnamed')}")
    logger.info(f"Config: {args.config} | Fold: {args.fold}")

    # 2. Get DataLoaders
    try:
        train_loader, val_loader = get_dataloaders(fold=args.fold, config=config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 3. Model Loading / Training
    try:
        if args.evaluate_only:
            if not args.checkpoint or not os.path.exists(args.checkpoint):
                logger.error(f"Must provide a valid --checkpoint path for evaluate_only. Got: {args.checkpoint}")
                return
            logger.info(f"Loading checkpoint {args.checkpoint} for evaluation...")
            model = FoldNet.load_from_checkpoint(args.checkpoint)
            model.eval()
        else:
            trainer, model = train_foldnet(
                train_loader, 
                val_loader, 
                config,
                checkpoint_dir=config_full.get('paths', {}).get('checkpoint_dir', 'results/checkpoints'),
                logs_dir=config_full.get('paths', {}).get('logs_dir', 'results/logs')
            )
            logger.info("Training process completed successfully.")
        
        # 4. Visualisation
        if args.visualise:
            logger.info("Starting Visualisation Pipeline...")
            model.eval()
            
            # Run prediction on validation set
            ss_p, ss_t, c_p, c_t, seq_lens, prot_ids = predict(model, val_loader)
            metrics = evaluate_metrics(ss_p, ss_t, c_p, c_t, seq_lens)
            logger.info(f"Evaluation Metrics: Q3={metrics.get('Q3', 0):.2f}%, MCC={metrics.get('MCC_Macro', 0):.4f}")
            
            out_dir = config_full.get('paths', {}).get('vis_dir', 'results/visualisations')
            os.makedirs(out_dir, exist_ok=True)
            
            # Visualise the first 3 proteins in the validation set
            num_to_vis = min(3, len(seq_lens))
            for i in range(num_to_vis):
                prot_id = prot_ids[i]
                logger.info(f"Generating visualisations for {prot_id}")
                create_visualisations(ss_p[i], ss_t[i], c_p[i], c_t[i], seq_lens[i], prot_id, out_dir=out_dir)
                
                # Export JSON for 3D Viewer
                json_path = os.path.join(out_dir, f"{prot_id}_viewer.json")
                export_3d_json(prot_id, ss_p[i], c_p[i], json_path) 
                
            logger.info(f"Visualisations saved to {out_dir}")

    except Exception:
        logger.exception("Training failed with the following error:")

if __name__ == "__main__":
    main()
