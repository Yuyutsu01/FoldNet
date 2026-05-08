import torch
import numpy as np
from typing import List, Tuple, Dict
from foldnet.models.foldnet import FoldNet

def load_model(checkpoint_path: str, device: str = 'cpu') -> FoldNet:
    """
    Loads a FoldNet model from a PyTorch Lightning checkpoint.
    """
    # PyTorch Lightning's load_from_checkpoint automatically uses the saved hyperparams
    model = FoldNet.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model

def predict(model: FoldNet, dataloader: torch.utils.data.DataLoader, device: str = None):
    """
    Runs inference and extracts predictions and ground truths for evaluation.
    
    Returns:
        ss_pred_list: List of secondary structure predictions (1D arrays).
        ss_true_list: List of secondary structure true labels (1D arrays).
        contact_probs_list: List of predicted contact probability matrices.
        contact_true_list: List of true contact matrices.
        seq_lens: List of actual sequence lengths (excluding padding).
        protein_ids_list: List of protein IDs.
    """
    ss_pred_list = []
    ss_true_list = []
    contact_probs_list = []
    contact_true_list = []
    seq_lens = []
    protein_ids_list = []
    
    if device is None:
        device = next(model.parameters()).device
        
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            ss_labels = batch['ss_labels'].to(device)
            contact_map = batch['contact_map'].to(device)
            mask = batch.get('mask', None)
            batch_pids = batch.get('protein_ids', [f"protein_{i}" for i in range(features.size(0))])
            
            if mask is not None:
                mask = mask.to(device)
                
            ss_logits, contact_probs = model(features, mask=mask)
            
            # Convert logits to classes
            ss_preds = torch.argmax(ss_logits, dim=-1)
            
            # Convert contact logits to probabilities if BCEWithLogitsLoss was used
            # We see in foldnet.py it uses BCEWithLogitsLoss, meaning the output is logits
            # so we apply sigmoid to get probabilities.
            contact_probs = torch.sigmoid(contact_probs)
            
            batch_size = features.size(0)
            for i in range(batch_size):
                # Calculate valid sequence length (ignore padding where ss_labels == -1)
                valid_mask = ss_labels[i] != -1
                seq_len = valid_mask.sum().item()
                
                # If sequence has no valid residues, skip
                if seq_len == 0:
                    continue
                    
                seq_lens.append(seq_len)
                protein_ids_list.append(batch_pids[i])
                
                # Extract valid portions
                ss_p = ss_preds[i, valid_mask].cpu().numpy()
                ss_t = ss_labels[i, valid_mask].cpu().numpy()
                
                # For contact map, take the seq_len x seq_len upper-left submatrix
                c_p = contact_probs[i, :seq_len, :seq_len].cpu().numpy()
                c_t = contact_map[i, :seq_len, :seq_len].cpu().numpy()
                
                ss_pred_list.append(ss_p)
                ss_true_list.append(ss_t)
                contact_probs_list.append(c_p)
                contact_true_list.append(c_t)
                
    return ss_pred_list, ss_true_list, contact_probs_list, contact_true_list, seq_lens, protein_ids_list
