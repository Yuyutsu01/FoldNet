import torch
import torch.nn as nn
import pytorch_lightning as pl
from .encoders import CNNEncoder, BiLSTMEncoder, TransformerEncoder
from .heads import SecondaryStructureHead, ContactMapHead

class FoldNet(pl.LightningModule):
    def __init__(self, 
                 encoder_type='cnn', 
                 feature_dim=1280, 
                 hidden_dim=512, 
                 num_classes=3, 
                 lambda_contact=0.5,
                 lr=1e-3,
                 freeze_encoder=False):
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        
        # 1. Initialize Encoder
        if encoder_type == 'cnn':
            self.encoder = CNNEncoder(feature_dim, hidden_dim)
        elif encoder_type == 'bilstm':
            self.encoder = BiLSTMEncoder(feature_dim, hidden_dim)
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(feature_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        if freeze_encoder:
            self.freeze_encoder()
            
        # 2. Initialize Heads
        self.ss_head = SecondaryStructureHead(hidden_dim, num_classes)
        self.contact_head = ContactMapHead(hidden_dim)
        
        # 3. Training Parameters
        self.lambda_contact = lambda_contact
        self.lr = lr
        self.ss_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        # BCEWithLogitsLoss is autocast-safe (required for FP16 mixed precision on GPU)
        self.contact_criterion = nn.BCEWithLogitsLoss()

    def forward(self, features, mask=None):
        # features: (batch, L, feature_dim)
        if self.hparams.encoder_type == 'transformer':
            # Transformer expects src_key_padding_mask (True for padding)
            embeddings = self.encoder(features, src_key_padding_mask=mask)
        else:
            embeddings = self.encoder(features)
            
        ss_logits = self.ss_head(embeddings)
        contact_probs = self.contact_head(embeddings)
        
        return ss_logits, contact_probs

    def training_step(self, batch, batch_idx):
        features = batch['features']
        ss_labels = batch['ss_labels']
        contact_map = batch['contact_map']
        mask = batch.get('mask', None)
        
        ss_logits, contact_probs = self(features, mask=mask)
        
        # Flatten for loss calculation
        # ss_logits: (batch, L, 3) -> (batch*L, 3)
        # ss_labels: (batch, L) -> (batch*L)
        L = features.size(1)
        ss_loss = self.ss_criterion(ss_logits.view(-1, self.hparams.num_classes), ss_labels.view(-1))
        
        # Contact loss
        contact_loss = self.contact_criterion(contact_probs, contact_map.float())
        
        total_loss = ss_loss + self.lambda_contact * contact_loss
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        features = batch['features']
        ss_labels = batch['ss_labels']
        contact_map = batch['contact_map']
        mask = batch.get('mask', None)
        
        ss_logits, contact_probs = self(features, mask=mask)
        
        ss_loss = self.ss_criterion(ss_logits.view(-1, self.hparams.num_classes), ss_labels.view(-1))
        contact_loss = self.contact_criterion(contact_probs, contact_map.float())
        total_loss = ss_loss + self.lambda_contact * contact_loss
        
        self.log('val_loss', total_loss, prog_bar=True)
        
        # Save predictions for epoch-end metrics
        ss_preds = torch.argmax(ss_logits, dim=-1)
        c_probs = torch.sigmoid(contact_probs)
        
        batch_ss_p, batch_ss_t, batch_c_p, batch_c_t, batch_seq_lens = [], [], [], [], []
        
        for i in range(features.size(0)):
            valid_mask = ss_labels[i] != -1
            seq_len = valid_mask.sum().item()
            if seq_len == 0: continue
                
            batch_seq_lens.append(seq_len)
            batch_ss_p.append(ss_preds[i, valid_mask].cpu().numpy())
            batch_ss_t.append(ss_labels[i, valid_mask].cpu().numpy())
            batch_c_p.append(c_probs[i, :seq_len, :seq_len].cpu().numpy())
            batch_c_t.append(contact_map[i, :seq_len, :seq_len].cpu().numpy())
            
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
            
        self.validation_step_outputs.append({
            'ss_p': batch_ss_p, 'ss_t': batch_ss_t, 
            'c_p': batch_c_p, 'c_t': batch_c_t, 
            'seq_lens': batch_seq_lens
        })
        
        return total_loss
        
    def on_validation_epoch_end(self):
        if not hasattr(self, 'validation_step_outputs') or len(self.validation_step_outputs) == 0:
            return
            
        ss_p_list, ss_t_list, c_p_list, c_t_list, seq_lens = [], [], [], [], []
        
        for out in self.validation_step_outputs:
            ss_p_list.extend(out['ss_p'])
            ss_t_list.extend(out['ss_t'])
            c_p_list.extend(out['c_p'])
            c_t_list.extend(out['c_t'])
            seq_lens.extend(out['seq_lens'])
            
        from foldnet.evaluation.metrics_ss import evaluate_metrics
        metrics = evaluate_metrics(ss_p_list, ss_t_list, c_p_list, c_t_list, seq_lens)
        
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.log(f"val_{k}", v, prog_bar=(k == 'Q3'))
                
        # Optional: Log images to wandb
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger
            if isinstance(self.logger, WandbLogger):
                from foldnet.evaluation.visualisation import plot_secondary_structure, plot_contact_map
                import os
                import tempfile
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    ss_path = os.path.join(tmpdir, 'val_ss.png')
                    cm_path = os.path.join(tmpdir, 'val_cm.png')
                    
                    # Just plot the first protein in the validation set
                    plot_secondary_structure(ss_p_list[0], ss_t_list[0], seq_lens[0], ss_path)
                    plot_contact_map(c_p_list[0], c_t_list[0], cm_path)
                    
                    self.logger.experiment.log({
                        "val_ss_plot": wandb.Image(ss_path),
                        "val_contact_map": wandb.Image(cm_path),
                        "epoch": self.current_epoch
                    })
        except Exception as e:
            pass # Fail silently if wandb/matplotlib not available or not configured
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        # 5-epoch Linear Warmup followed by Cosine Annealing
        warmup_epochs = 5
        max_epochs = self.trainer.max_epochs
        
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs
        )
        
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[scheduler1, scheduler2], 
            milestones=[warmup_epochs]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined_scheduler,
                "interval": "epoch",
                "name": "learning_rate"
            }
        }

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen.")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen.")
