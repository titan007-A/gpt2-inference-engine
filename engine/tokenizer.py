"""
engine/tokenizer.py

Thin wrapper around the HuggingFace (Rust-backed) GPT-2 tokenizer: encode, decode, batch, special tokens.  |  Phase 1
"""
import torch 
from transformers import GPT2TokenizerFast


class Tokenizer:
    def __init__(self,name_or_path="gpt2"):
        self.tok = GPT2TokenizerFast.from_pretrained(name_or_path)
        self.eos_token_id = self.tok.eos_token_id
        self.vocab_size = self.tok.vocab_size


    def encode(self,text,device="cpu"):
        """text-> token ids"""
        ids = self.tok.encode(text)
        return torch.tensor([ids], dtype=torch.long,device=device)

    def decode(self,ids,device="cpu"):
        """list of token ids-> text"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tok.decode(ids)


