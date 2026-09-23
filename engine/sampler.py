"""
engine/sampler.py

Token sampling: greedy, temperature, top-k, top-p (nucleus).  |  Phase 1
"""

import torch 
import torch.nn.functional as F

def greedy(logits):
    """logits (B,T,V) -> next token ids (B,1). Sirf last position mein se nikalne hai"""
    return logits[:,-1,:].argmax(dim=-1,keepdim=True)

def sample(logits,temperature=1.0,top_k=None,top_p=None,generator=None):
    """logits: (B, T, V) -> next token ids (B, 1), temperature / top-k / top-p ke saath."""
    if temperature==0:
      return greedy(logits)

    logits = logits[:,-1,:]/temperature

    if top_k is not None:
        kth = torch.topk(logits,top_k,dim=-1).values[:,-1:]
        logits = logits.masked_fill(logits<kth,float("-inf"))

    if top_p is not None:
        sorted_logits,sorted_idx = torch.sort(logits,dim=-1,descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        remove_sorted = (sorted_probs.cumsum(dim=-1)-sorted_probs)>top_p
        remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
