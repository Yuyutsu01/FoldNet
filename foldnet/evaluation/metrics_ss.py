import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from typing import Dict, List, Tuple, Union

def calculate_q3(ss_pred: np.ndarray, ss_true: np.ndarray) -> float:
    """Calculate Q3 accuracy (percentage of correctly classified residues)."""
    return accuracy_score(ss_true, ss_pred) * 100.0

def calculate_mcc(ss_pred: np.ndarray, ss_true: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calculate per-class MCC and macro MCC."""
    classes = [0, 1, 2] # Helix, Sheet, Coil
    mccs = []
    
    for c in classes:
        pred_c = (ss_pred == c).astype(int)
        true_c = (ss_true == c).astype(int)
        if len(np.unique(true_c)) > 1 or len(np.unique(pred_c)) > 1:
            mcc = matthews_corrcoef(true_c, pred_c)
        else:
            mcc = 0.0
        mccs.append(mcc)
        
    macro_mcc = np.mean(mccs)
    return np.array(mccs), float(macro_mcc)

def calculate_confusion_matrix(ss_pred: np.ndarray, ss_true: np.ndarray) -> np.ndarray:
    """Calculate 3x3 confusion matrix."""
    return confusion_matrix(ss_true, ss_pred, labels=[0, 1, 2])

def evaluate_metrics(ss_pred_list: List[np.ndarray], ss_true_list: List[np.ndarray], 
                     contact_probs_list: List[np.ndarray], contact_true_list: List[np.ndarray],
                     seq_lens: List[int]) -> Dict[str, Union[float, np.ndarray]]:
    """Computes all metrics across a batch/list of proteins."""
    if not ss_pred_list:
        return {}
        
    from foldnet.evaluation.metrics_contacts import calculate_precision_at_l, calculate_long_range_precision
        
    all_ss_pred = np.concatenate(ss_pred_list)
    all_ss_true = np.concatenate(ss_true_list)
    
    q3 = calculate_q3(all_ss_pred, all_ss_true)
    per_class_mcc, macro_mcc = calculate_mcc(all_ss_pred, all_ss_true)
    cm = calculate_confusion_matrix(all_ss_pred, all_ss_true)
    
    p_L_list, p_L2_list, p_L5_list, long_range_list = [], [], [], []
    
    for probs, true, L in zip(contact_probs_list, contact_true_list, seq_lens):
        precisions = calculate_precision_at_l(probs, true, L)
        p_L_list.append(precisions['Precision@L'])
        p_L2_list.append(precisions['Precision@L/2'])
        p_L5_list.append(precisions['Precision@L/5'])
        long_range_list.append(calculate_long_range_precision(probs, true, L))
        
    return {
        'Q3': q3,
        'MCC_Macro': macro_mcc,
        'MCC_PerClass': per_class_mcc,
        'ConfusionMatrix': cm,
        'Precision@L': np.mean(p_L_list) if p_L_list else 0.0,
        'Precision@L/2': np.mean(p_L2_list) if p_L2_list else 0.0,
        'Precision@L/5': np.mean(p_L5_list) if p_L5_list else 0.0,
        'LongRange_Precision': np.mean(long_range_list) if long_range_list else 0.0
    }
