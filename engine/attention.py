"""
engine/attention.py

Attention built on F.scaled_dot_product_attention (flash / mem-efficient backend), causal masking, KV-cache read/write.  |  Phase 1 -> 2
"""



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024     # max sequence length (context window)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True          # GPT-2 uses bias in Linear/LayerNorm


class Attention(nn.Module):
    """Multi-head masked self-attention. """

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # One linear layer produces Q, K, V all at once (more efficient)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection after attention
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask: lower-triangular matrix of 1s.
        # Registered as a buffer so it moves with .to(device) but isn't a trained param.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x,cache=None,layer_idx=None):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Project to query, key, value, then split into heads
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))  # (B, nh, T, T)
        Tq = q.shape[2]
        Tk = k.shape[2]
        # Apply causal mask: block attention to future positions
        att = att.masked_fill(self.mask[:, :, Tk-Tq:Tk, :Tk] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                   # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back

        y = self.resid_dropout(self.c_proj(y))
        return y


