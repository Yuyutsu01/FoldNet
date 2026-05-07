import torch
from foldnet.models.foldnet import FoldNet

def test_model_shapes():
    batch_size = 2
    L = 100
    feature_dim = 1280
    hidden_dim = 512
    num_classes = 3
    
    # Create dummy data
    features = torch.randn(batch_size, L, feature_dim)
    
    encoders = ['cnn', 'bilstm', 'transformer']
    
    for enc in encoders:
        print(f"\nTesting Encoder: {enc}")
        model = FoldNet(encoder_type=enc, feature_dim=feature_dim, hidden_dim=hidden_dim, num_classes=num_classes)
        
        ss_logits, contact_probs = model(features)
        
        print(f"SS Logits shape: {ss_logits.shape}")
        print(f"Contact Probs shape: {contact_probs.shape}")
        
        assert ss_logits.shape == (batch_size, L, num_classes)
        assert contact_probs.shape == (batch_size, L, L)
        print(f"Shape check PASSED for {enc}")

if __name__ == "__main__":
    test_model_shapes()
