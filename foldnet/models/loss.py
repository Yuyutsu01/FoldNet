import torch
import torch.nn as nn

class FoldNetLoss(nn.Module):
    """
    Multi-task loss for FoldNet: Secondary Structure (CE) + Contact Map (BCE).
    """
    def __init__(self, lambda_contact=0.5):
        super().__init__()
        self.lambda_contact = lambda_contact
        self.ss_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        # Using BCELoss for contact map probabilities
        self.contact_criterion = nn.BCELoss()

    def forward(self, ss_logits, ss_labels, contact_probs, contact_map):
        # 1. Secondary Structure Loss
        # Flatten: (batch, L, num_classes) -> (batch * L, num_classes)
        # Labels: (batch, L) -> (batch * L)
        ss_loss = self.ss_criterion(
            ss_logits.view(-1, ss_logits.size(-1)), 
            ss_labels.view(-1)
        )
        
        # 2. Contact Map Loss
        # Probabilities and Ground Truth are already (batch, L, L)
        contact_loss = self.contact_criterion(
            contact_probs, 
            contact_map.float()
        )
        
        # 3. Total Balanced Loss
        total_loss = ss_loss + self.lambda_contact * contact_loss
        
        return total_loss, ss_loss, contact_loss
