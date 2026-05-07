import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from foldnet.models.foldnet import FoldNet

class SingleSampleDataset(Dataset):
    """
    A dataset with exactly one sample to test overfitting.
    """
    def __init__(self, L=50, feature_dim=1280):
        # Fixed random seed for reproducibility in overfitting test
        torch.manual_seed(42)
        self.features = torch.randn(L, feature_dim)
        self.ss_labels = torch.randint(0, 3, (L,))
        self.contact_map = torch.randint(0, 2, (L, L)).float()

    def __len__(self):
        return 1  # Exactly one sample

    def __getitem__(self, idx):
        return {
            'features': self.features,
            'ss_labels': self.ss_labels,
            'contact_map': self.contact_map
        }

def test_overfitting():
    print("Starting Overfitting Test (Goal: Loss -> 0)...")
    
    # 1. Setup Data
    dataset = SingleSampleDataset(L=30)
    train_loader = DataLoader(dataset, batch_size=1)
    
    # 2. Setup Model (using a small hidden dim for faster test)
    model = FoldNet(
        encoder_type='cnn',
        feature_dim=1280,
        hidden_dim=128,
        num_classes=3,
        lr=1e-3
    )
    
    # 3. Setup Trainer (Quiet mode, many epochs on 1 sample)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_checkpointing=False,
        logger=False, # Disable logging for a quick test
        enable_progress_bar=True,
        overfit_batches=1 # PyTorch Lightning built-in overfitting tool
    )
    
    # 4. Fit
    trainer.fit(model, train_loader)
    
    # 5. Check final loss
    final_loss = trainer.callback_metrics.get('train_loss_epoch')
    print(f"\nFinal Overfitting Loss: {final_loss:.6f}")
    
    if final_loss < 0.1:
        print("Overfitting test PASSED! The model can learn perfectly from data.")
    else:
        print("Overfitting test FAILED. Loss did not converge sufficiently.")

if __name__ == "__main__":
    test_overfitting()
