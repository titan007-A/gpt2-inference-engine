"""
engine/model.py

GPT-2 architecture: transformer blocks (LayerNorm + Attention + MLP + residuals), embeddings, final LayerNorm, tied LM head.  |  Phase 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.attention import Attention, GPT2Config   # attention.py se import


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: lm_head aur wte same weights share karte hain (GPT-2 ka standard)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, position_offset=0):
        B, T = input_ids.shape

        positions = torch.arange(
            position_offset, position_offset + T, device=input_ids.device
        ).unsqueeze(0)

        x = self.wte(input_ids) + self.wpe(positions)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
