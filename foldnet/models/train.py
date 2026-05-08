import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from .foldnet import FoldNet
import torch.optim as nn_optim

# Enable TF32 TensorCore acceleration on RTX 40xx GPUs
torch.set_float32_matmul_precision('medium')

def train_foldnet(
    train_loader, 
    val_loader, 
    config, 
    checkpoint_dir='results/checkpoints',
    logs_dir='results/logs'
):
    """
    Master training function using PyTorch Lightning.
    """
    
    # 1. Initialize Model
    model = FoldNet(
        encoder_type=config.get('encoder_type', 'cnn'),
        feature_dim=config.get('feature_dim', 1280),
        hidden_dim=config.get('hidden_dim', 512),
        num_classes=config.get('num_classes', 3),
        lambda_contact=config.get('lambda_contact', 0.5),
        lr=config.get('lr', 1e-3)
    )

    # 2. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='foldnet-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 3. Trainer setup
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project='FoldNet', 
        name=config.get('name', f"foldnet-{config.get('encoder_type', 'cnn')}"),
        save_dir=logs_dir
    )
    
    # Note: Using 'gpu' if available, otherwise 'cpu'
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1
    
    # Precision: Use 16-bit mixed precision if on GPU to save memory
    precision = '16-mixed' if accelerator == 'gpu' else '32-true'

    trainer = pl.Trainer(
        max_epochs=config.get('epochs', 100),
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        default_root_dir=logs_dir,
        gradient_clip_val=1.0,
        # Accumulate gradients to simulate larger batch sizes on 6GB GPU
        accumulate_grad_batches=config.get('accumulate_grad_batches', 4)
    )

    # 4. Start Training
    trainer.fit(model, train_loader, val_loader)
    
    return trainer, model

if __name__ == "__main__":
    # This section is for basic testing of the training setup
    print("FoldNet training module initialized.")
