import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Secondary Structure Color Mapping:
# 0 = Helix (Red), 1 = Sheet (Yellow), 2 = Coil (Green)
SS_COLORS = {
    0: '#FF5555', # Bright Red
    1: '#FFFF55', # Bright Yellow
    2: '#55FF55'  # Bright Green
}
SS_LABELS = {0: 'H', 1: 'E', 2: 'C'}

def plot_secondary_structure(ss_pred: np.ndarray, ss_true: np.ndarray, seq_len: int, save_path: str):
    """Plots two rows of coloured bars (true labels vs predicted labels)."""
    fig, (ax_true, ax_pred) = plt.subplots(2, 1, figsize=(max(8, seq_len / 10), 2), sharex=True)
    
    for i in range(seq_len):
        true_class = ss_true[i]
        pred_class = ss_pred[i]
        ax_true.add_patch(plt.Rectangle((i, 0), 1, 1, color=SS_COLORS.get(true_class, '#AAAAAA')))
        ax_pred.add_patch(plt.Rectangle((i, 0), 1, 1, color=SS_COLORS.get(pred_class, '#AAAAAA')))
        
    ax_true.set_xlim(0, seq_len)
    ax_true.set_ylim(0, 1)
    ax_pred.set_ylim(0, 1)
    ax_true.set_yticks([])
    ax_pred.set_yticks([])
    ax_true.set_ylabel('True', rotation=0, labelpad=20, va='center')
    ax_pred.set_ylabel('Pred', rotation=0, labelpad=20, va='center')
    
    step = 20 if seq_len > 100 else 10
    xticks = np.arange(0, seq_len + 1, step)
    ax_pred.set_xticks(xticks)
    ax_pred.set_xticklabels(xticks)
    ax_pred.set_xlabel('Residue Index')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_contact_map(contact_probs: np.ndarray, contact_true: np.ndarray, save_path: str):
    """Plots side-by-side heatmaps of true binary contact map and predicted probabilities."""
    fig, (ax_true, ax_pred) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(contact_true, ax=ax_true, cmap='Greys', cbar=False, square=True)
    ax_true.set_title('True Contact Map')
    ax_true.set_xlabel('Residue j')
    ax_true.set_ylabel('Residue i')
    
    sns.heatmap(contact_probs, ax=ax_pred, cmap='viridis', cbar=True, square=True)
    ax_pred.set_title('Predicted Contact Probabilities')
    ax_pred.set_xlabel('Residue j')
    ax_pred.set_ylabel('Residue i')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_visualisations(ss_pred: np.ndarray, ss_true: np.ndarray, 
                          contact_probs: np.ndarray, contact_true: np.ndarray, 
                          seq_len: int, protein_id: str, out_dir: str = "results/visualisations"):
    """Wrapper function to generate all visualisations for a single protein."""
    os.makedirs(out_dir, exist_ok=True)
    
    ss_path = os.path.join(out_dir, f"{protein_id}_ss.png")
    plot_secondary_structure(ss_pred[:seq_len], ss_true[:seq_len], seq_len, ss_path)
    
    cm_path = os.path.join(out_dir, f"{protein_id}_contact.png")
    plot_contact_map(contact_probs[:seq_len, :seq_len], contact_true[:seq_len, :seq_len], cm_path)
    
    return ss_path, cm_path
