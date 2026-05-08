import os
import sys
import torch
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foldnet.data.dataset import ProteinDataset, collate_fn

def test_dataset():
    print("Testing ProteinDataset...")
    # Mock data
    df = pd.DataFrame({
        'protein_id': ['test1', 'test2'],
        'length': [10, 15],
        'ss_labels': ['0,1,2,0,1,2,0,1,2,0', '0,1,2,0,1,2,0,1,2,0,1,2,0,1,2']
    })
    
    # Create dummy embeddings
    esm_dir = "tests/dummy_esm"
    os.makedirs(esm_dir, exist_ok=True)
    np.save(os.path.join(esm_dir, "test1.npy"), np.random.randn(10, 1280))
    np.save(os.path.join(esm_dir, "test2.npy"), np.random.randn(15, 1280))
    
    dataset = ProteinDataset(df, esm_dir=esm_dir)
    assert len(dataset) == 2
    
    item = dataset[0]
    assert item['features'].shape == (10, 1280)
    assert item['ss_labels'].shape == (10,)
    assert item['seq_len'] == 10
    print("[PASS] Dataset loading")

def test_collate():
    print("Testing collate_fn...")
    batch = [
        {'features': torch.randn(10, 1280), 'ss_labels': torch.zeros(10), 'contact_map': torch.zeros(10, 10), 'seq_len': 10, 'protein_id': 'p1'},
        {'features': torch.randn(20, 1280), 'ss_labels': torch.zeros(20), 'contact_map': torch.zeros(20, 20), 'seq_len': 20, 'protein_id': 'p2'}
    ]
    
    collated = collate_fn(batch)
    assert collated['features'].shape == (2, 20, 1280)
    assert collated['ss_labels'].shape == (2, 20)
    assert collated['mask'].shape == (2, 20)
    assert collated['mask'][0, 15] == True  # Padding
    assert collated['mask'][1, 15] == False # Real
    print("[PASS] Collate function")

if __name__ == "__main__":
    try:
        test_dataset()
        test_collate()
        print("\nALL DATA TESTS PASSED! OK")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
