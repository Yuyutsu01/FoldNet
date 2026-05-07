import torch
from torch.utils.data import DataLoader, Dataset
from foldnet.models.train import train_foldnet

class DummyProteinDataset(Dataset):
    def __init__(self, size=10, L=50, feature_dim=1280):
        self.size = size
        self.L = L
        self.feature_dim = feature_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'features': torch.randn(self.L, self.feature_dim),
            'ss_labels': torch.randint(0, 3, (self.L,)),
            'contact_map': torch.randint(0, 2, (self.L, self.L)).float(),
            'mask': torch.zeros(self.L, dtype=torch.bool)
        }

def test_training_step():
    # Configuration
    config = {
        'encoder_type': 'cnn',
        'feature_dim': 1280,
        'hidden_dim': 64,
        'num_classes': 3,
        'lambda_contact': 0.5,
        'lr': 1e-3,
        'epochs': 1,
        'accumulate_grad_batches': 1
    }
    
    # DataLoaders
    train_ds = DummyProteinDataset(size=4)
    val_ds = DummyProteinDataset(size=2)
    
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    
    print("Starting training test...")
    try:
        trainer, model = train_foldnet(train_loader, val_loader, config)
        print("Training test PASSED!")
    except Exception as e:
        print(f"Training test FAILED with error: {e}")

if __name__ == "__main__":
    test_training_step()
