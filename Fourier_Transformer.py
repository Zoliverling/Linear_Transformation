import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

# Fourier Transform for the linear transformer encoder
class FourierTransform(nn.Module):
    def __init__(self):
        super(FourierTransform, self).__init__()

    def forward(self, x):
        # Apply the 2D Fourier transform to the last two dimensions
        return torch.fft.fft2(x).real


# Define the Positionwise Feed-Forward Network
class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

#Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.fourier_transform = FourierTransform()
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Applying Fourier Transform as a replacement for MHA
        src2 = self.fourier_transform(src)
        src = self.layernorm1(src + self.dropout(src2))

        # Positionwise Feedforward Network
        src2 = self.feed_forward(src)
        src = self.layernorm2(src + self.dropout(src2))

        return src

#Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # Self-attention on the targets
        tgt2 = self.layernorm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        # Attention over encoder's output
        tgt2 = self.layernorm2(tgt)
        tgt2 = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        # Positionwise feedforward
        tgt2 = self.layernorm3(tgt)
        tgt = tgt + self.dropout(self.feed_forward(tgt2))
        
        return tgt
    
class LinearTransformer(nn.Module):
    def __init__(self, feature_size, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(LinearTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.fourier_transform = FourierTransform()  # Fourier transform layer
        self.positional_encoder = nn.Embedding(1000, d_model)  # Customize based on max sequence length
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, feature_size)  # Adjust depending on your output size

    def forward(self, src, src_mask=None, tgt=None, tgt_mask=None):
        batch_size, seq_length, _ = src.size()

        # Positional Encoding
        positions = torch.arange(seq_length, device=src.device).unsqueeze(0).repeat(batch_size, 1)
        src = src + self.positional_encoder(positions)

        # Fourier Transform
        src = self.fourier_transform(src)

        # Encoder
        for layer in self.encoder_layers:
            src = layer(src, src_mask=src_mask)

        if tgt is not None:
            # If there is a target sequence (for tasks that use the decoder)
            tgt = tgt + self.positional_encoder(positions[:tgt.size(1), :])
            for layer in self.decoder_layers:
                tgt = layer(tgt, src, tgt_mask=tgt_mask)

            # Output layer for decoder output
            return self.output_layer(tgt)

        # Output layer to convert encoder output back to feature size (if only using the encoder)
        return self.output_layer(src)
    

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


