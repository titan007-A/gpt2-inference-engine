"""
tests/test_correctness.py

Parity tests: this engine's logits / greedy generation must match HuggingFace (torch.allclose).  |  Phase 1 (keep green afterwards)
"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from engine.weights import load_hf_weights

PROMPTS = ["Hello, are you there", "I live in bareilly", "Once upon a time"]

@pytest.fixture(scope="module")
def models():
    ours = load_hf_weights("gpt2")
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return ours, hf, tok

@pytest.mark.parametrize("prompt",PROMPTS)
def test_logits_match_hf(models,prompt):
    ours, hf, tok = models
    ids = tok(prompt,return_tensors="pt").input_ids

    with torch.no_grad():
        mine = ours(ids)
        ref = hf(ids).logits

    assert mine.shape == ref.shape
    assert torch.allclose(mine, ref, atol=1e-4), f"max diff = {(mine - ref).abs().max().item():.2e}"