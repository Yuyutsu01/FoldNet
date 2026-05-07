import torch
import torch.nn as nn
import torch.nn.functional as F

class SecondaryStructureHead(nn.Module):
    def __init__(self, hidden_dim, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, L, hidden_dim)
        logits = self.classifier(x)
        return logits  # (batch, L, num_classes)

class ResidualBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ContactMapHead(nn.Module):
    def __init__(self, hidden_dim, num_layers=5):
        super().__init__()
        # Pairwise features will have 2 * hidden_dim channels
        input_channels = 2 * hidden_dim
        self.input_conv = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.res_blocks = nn.ModuleList([
            ResidualBlock2D(64) for _ in range(num_layers)
        ])
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, L, hidden_dim)
        batch_size, L, d = x.shape
        
        # Efficient pairwise feature construction: [E[i]; E[j]]
        # x_i: (batch, L, 1, d)
        x_i = x.unsqueeze(2).expand(-1, -1, L, -1)
        # x_j: (batch, 1, L, d)
        x_j = x.unsqueeze(1).expand(-1, L, -1, -1)
        
        # Pairwise features: (batch, L, L, 2*d)
        pairwise_features = torch.cat([x_i, x_j], dim=-1)
        
        # Prepare for 2D Conv: (batch, 2*d, L, L)
        pairwise_features = pairwise_features.permute(0, 3, 1, 2)
        
        out = self.input_conv(pairwise_features)
        for block in self.res_blocks:
            out = block(out)
        
        # (batch, 1, L, L) → (batch, L, L)
        logits = self.output_conv(out).squeeze(1)
        
        return logits  # raw logits — sigmoid applied by BCEWithLogitsLoss
