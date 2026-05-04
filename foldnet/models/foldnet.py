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
        self.ss_criterion = nn.CrossEntropyLoss()
        # Using binary cross entropy with weight to handle sparsity of contact maps
        self.contact_criterion = nn.BCELoss()

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
        ss_loss = self.ss_criterion(ss_logits.view(-1, 3), ss_labels.view(-1))
        
        # Contact loss
        contact_loss = self.contact_criterion(contact_probs, contact_map.float())
        
        total_loss = ss_loss + self.lambda_contact * contact_loss
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_ss_loss', ss_loss)
        self.log('train_contact_loss', contact_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        features = batch['features']
        ss_labels = batch['ss_labels']
        contact_map = batch['contact_map']
        mask = batch.get('mask', None)
        
        ss_logits, contact_probs = self(features, mask=mask)
        
        ss_loss = self.ss_criterion(ss_logits.view(-1, 3), ss_labels.view(-1))
        contact_loss = self.contact_criterion(contact_probs, contact_map.float())
        total_loss = ss_loss + self.lambda_contact * contact_loss
        
        self.log('val_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # Using Cosine Annealing with Warmup for better convergence
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
            'name': 'lr_scheduler'
        }
        return [optimizer], [scheduler]

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen.")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen.")
