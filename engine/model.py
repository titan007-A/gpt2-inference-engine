"""
engine/model.py

GPT-2 architecture: transformer blocks (LayerNorm + Attention + MLP + residuals), embeddings, final LayerNorm, tied LM head.  |  Phase 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from engine.attention import Attention   # attention.py se import

class MLP(nn.Module):
    def __init__(self,d_model):
        self.c_fc = nn.Linear(d_model,4*d_model)
        self.c_proj = nn.Linear(4*d_model,d_model)

    def forward(self,x):
        x=self.c_fc(x)
        x=F.gelu(x,approximate='tanh')
        x=self.c_proj(x)
        return x


class GPT2BLock(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model,num_heads)
        self.ln_2= nn.LayerNorm(d_model)
        self.mlp= MLP(d_model)

    def forward(self,x,layer_idx,cache=None,is_casual=True):
        x = x +self.attention(self.ln_1(x),layer_idx=layer_idx,cache =cache,is_casual=is_casual)
        x = x+self.mlp(x)
        return x


class GPT2Model(nn.Module):
    def __init__(self,vocab_size,n_positions,n_embd,n_layer,n_head):
        super().__init__()
        self.wte = nn.Embedding(vocab_size,n_embd)
        self.wpe = nn.Embedding(n_positions,n_embd)
        self.h = nn.ModuleList([
            GPT2BLock(n_embd,n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying: lm_head aur wte same weights share karte hain (GPT-2 ka standard)
        self.lm_head.weight = self.wte.weight

    def forward(self,input_ids,cache =None,position_offset=0):
        B,T = input_ids.shape

        positions = torch.arange(
            position_offset, position_offset + T, device=input_ids.device
        ).unsqueeze(0)

        x = self.wte(input_ids)+self.wpe(positions)

        for i, block in enumerate(self.h):
            x = block(x,layer_idx=i,cache=cache, is_casual=True)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
