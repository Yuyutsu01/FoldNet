import os
import sys
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foldnet.models.foldnet import FoldNet

def test_model_forward():
    print("Testing FoldNet Forward Pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    encoders = ['cnn', 'bilstm', 'transformer']
    batch_size = 2
    seq_len = 100
    feat_dim = 1280
    
    for enc in encoders:
        print(f"Testing encoder: {enc}")
        model = FoldNet(encoder_type=enc, feature_dim=feat_dim, hidden_dim=256).to(device)
        model.eval()
        
        features = torch.randn(batch_size, seq_len, feat_dim).to(device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        
        ss_logits, contact_probs = model(features, mask=mask)
        
        # Check shapes
        assert ss_logits.shape == (batch_size, seq_len, 3)
        assert contact_probs.shape == (batch_size, seq_len, seq_len)
        print(f"[PASS] {enc} forward")

if __name__ == "__main__":
    try:
        test_model_forward()
        print("\nALL MODEL TESTS PASSED! OK")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
