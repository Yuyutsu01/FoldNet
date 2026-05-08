import numpy as np
from typing import Dict

def calculate_precision_at_l(contact_probs: np.ndarray, contact_true: np.ndarray, seq_len: int) -> Dict[str, float]:
    """Calculate Precision@L, Precision@L/2, Precision@L/5."""
    i, j = np.triu_indices(seq_len, k=1)
    
    probs_1d = contact_probs[i, j]
    true_1d = contact_true[i, j]
    
    sorted_idx = np.argsort(probs_1d)[::-1]
    
    results = {}
    for fraction, name in [(1, 'L'), (2, 'L/2'), (5, 'L/5')]:
        k = max(1, seq_len // fraction)
        if len(sorted_idx) < k:
            k = len(sorted_idx)
        
        top_k_idx = sorted_idx[:k]
        if k > 0:
            precision = np.mean(true_1d[top_k_idx])
        else:
            precision = 0.0
        results[f'Precision@{name}'] = float(precision)
        
    return results

def calculate_long_range_precision(contact_probs: np.ndarray, contact_true: np.ndarray, seq_len: int, threshold: int = 24) -> float:
    """Calculate long-range precision (only pairs with |i-j| >= 24)."""
    i, j = np.triu_indices(seq_len, k=threshold)
    
    if len(i) == 0:
        return 0.0
        
    probs_1d = contact_probs[i, j]
    true_1d = contact_true[i, j]
    
    sorted_idx = np.argsort(probs_1d)[::-1]
    k = seq_len
    if len(sorted_idx) < k:
        k = len(sorted_idx)
        
    top_k_idx = sorted_idx[:k]
    if k > 0:
        return float(np.mean(true_1d[top_k_idx]))
    return 0.0
