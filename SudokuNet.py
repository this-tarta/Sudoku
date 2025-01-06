import torch
import math

from torch import nn as nn

class SudokuConvBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, kernel_size: int, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        kA = kernel_size // 2
        kB = kA - 1 if kernel_size % 2 == 0 else kA
        self.block = nn.Sequential(
            padding_layer((kA,kB,kA,kB)),
            nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, stride=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block.forward(x)
    
class SudokuResBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, kernel_size: int, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        self.block = SudokuConvBlock(in_size, out_size, kernel_size, padding_layer)

    def forward(self, x):
        return x + self.block.forward(x)

class SudokuNet(nn.Module):
    def __init__(self, hidden_size: int, num_convs: int, kernel_size: int = 3, n: int = 3):
        super().__init__()
        self.n = n
        n2 = n * n
        self.net = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(1, n2, n2)),
            SudokuConvBlock(1, hidden_size, kernel_size),
            *[SudokuConvBlock(hidden_size, hidden_size, kernel_size) for _ in range(num_convs - 1)],
            nn.Conv2d(in_channels=hidden_size, out_channels=n2 + 1, kernel_size=1, stride=1),
            nn.Flatten(start_dim=2),
        )

    def forward(self, x):
        return self.net.forward((x / (self.n ** 2)) - 0.5)
    
class SudokuNetClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_convs: int, kernel_size: int = 3, n: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            SudokuNet(hidden_size, num_convs, kernel_size, n),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net.forward(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Positional encoding for Transformer models.

        Args:
            d_model (int): Dimensionality of the embedding space.
            max_len (int): Maximum sequence length for the positional encoding.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix to hold positional encodings of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Calculate positional encoding values
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])  # Odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension (1, max_len, d_model)
        
        # Register `pe` as a buffer so it won't be updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor `x`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SudokuTransformer(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, num_encoder_layers: int, ffw_size: int = 2048, dropout: float = 0.1, n: int = 3):
        super().__init__()
        self.num_classes = n * n + 1
        embed = nn.Linear(self.num_classes, embed_size)
        pos_encoding = PositionalEncoding(d_model=embed_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(embed_size, num_heads, dim_feedforward=ffw_size, dropout=dropout, activation='relu', batch_first=True)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False, norm=nn.LayerNorm(embed_size))
        proj = nn.Linear(embed_size, self.num_classes)
        sftmax = nn.Softmax(dim=-1)
        self.net = nn.Sequential(
            embed, pos_encoding, encoder, proj, sftmax
        )
        # remember to transpose after this

    def forward(self, x):
        x_1h = torch.nn.functional.one_hot(x.long(), self.num_classes).float()
        return self.net.forward(x_1h).transpose(-2, -1)

class TransformerResEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask = None, src_key_padding_mask = None, is_causal = False):
        return super().forward(src, src_mask, src_key_padding_mask, is_causal) + src