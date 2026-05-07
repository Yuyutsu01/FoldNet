import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from foldnet.models.foldnet import FoldNet

class SingleSampleDataset(Dataset):
    def __init__(self, L=50, feature_dim=1280):
        torch.manual_seed(42)
        self.features = torch.randn(L, feature_dim)
        self.ss_labels = torch.randint(0, 3, (L,))
        self.contact_map = torch.randint(0, 2, (L, L)).float()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            'features': self.features,
            'ss_labels': self.ss_labels,
            'contact_map': self.contact_map
        }

def run_overfit_test(encoder_type):
    print(f"\n{'='*40}")
    print(f"Testing Overfitting: ENCODER = {encoder_type.upper()}")
    print(f"{'='*40}")
    
    dataset = SingleSampleDataset(L=30)
    train_loader = DataLoader(dataset, batch_size=1)
    
    model = FoldNet(
        encoder_type=encoder_type,
        feature_dim=1280,
        hidden_dim=128,
        num_classes=3,
        lr=1e-3
    )
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        overfit_batches=1
    )
    
    trainer.fit(model, train_loader)
    
    final_loss = trainer.callback_metrics.get('train_loss_epoch')
    print(f"\nFinal Loss for {encoder_type}: {final_loss:.6f}")
    return final_loss

if __name__ == "__main__":
    results = {}
    for enc in ['cnn', 'bilstm', 'transformer']:
        loss = run_overfit_test(enc)
        results[enc] = loss
    
    print("\n" + "#"*40)
    print("FINAL RESULTS SUMMARY")
    print("#"*40)
    for enc, loss in results.items():
        status = "PASSED" if loss < 0.1 else "FAILED"
        print(f"{enc.upper():<12}: Loss = {loss:.6f} [{status}]")
