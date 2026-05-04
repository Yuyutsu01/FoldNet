import torch
import torch.nn as nn
import math

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)

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

class CNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=5):
        super().__init__()
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x shape: (batch, L, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, L)
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)
        return x.transpose(1, 2)  # (batch, L, hidden_dim)

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim // 2, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x shape: (batch, L, input_dim)
        out, _ = self.lstm(x)
        return out  # (batch, L, hidden_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, L, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nhead, hidden_dim * 4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch, L, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out  # (batch, L, hidden_dim)
