"""
tests/test_correctness.py

Parity tests: this engine's logits / greedy generation must match HuggingFace (torch.allclose).  |  Phase 1 (keep green afterwards)
"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from engine.kv_cache import KVCache
from engine.tokenizer import Tokenizer
from engine.weights import load_hf_weights

PROMPTS = ["Hello, are you there", "I live in bareilly", "Once upon a time"]
MAX_NEW_TOKENS = 50

@pytest.fixture(scope="module")
def models():
    ours = load_hf_weights("gpt2")
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return ours, hf, tok

@pytest.fixture(scope="module")
def ours_tok():
    return Tokenizer()

@pytest.mark.parametrize("prompt",PROMPTS)
def test_logits_match_hf(models,prompt):
    ours, hf, tok = models
    ids = tok(prompt,return_tensors="pt").input_ids

    with torch.no_grad():
        mine = ours(ids)
        ref = hf(ids).logits

    assert mine.shape == ref.shape
    assert torch.allclose(mine, ref, atol=1e-4), f"max diff = {(mine - ref).abs().max().item():.2e}"


@pytest.mark.parametrize("prompt",PROMPTS)
def test_greedy_matches_hf(models,ours_tok,prompt):
    ours, hf, tok = models
    mine = ours.generate(prompt, MAX_NEW_TOKENS, ours_tok)

    ids = tok(prompt,return_tensors="pt").input_ids
    with torch.no_grad():
        # do_sample=False -> HF greedy. Hamara temperature=0 bhi greedy path leta hai.
        out = hf.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    ref = tok.decode(out[0])

    assert mine == ref, f"\nours: {mine!r}\nhf  : {ref!r}"


@pytest.mark.parametrize("prompt",PROMPTS)
def test_cache_matches_full_forward(models,prompt):
    """Ek-ek token cache ke saath -> har step ke logits poore forward se match karein."""
    ours, _, tok = models
    ids = tok(prompt,return_tensors="pt").input_ids

    with torch.inference_mode():
        full = ours(ids)
        cache = KVCache(ours.config.n_layer)
        for t in range(ids.shape[1]):
            step = ours(ids[:, t:t+1], cache=cache, position_offset=cache.length)
            a, b = step[:, -1, :], full[:, t, :]
            # atol dheela hai kyunki (1,hd)@(hd,t+1) aur (T,hd)@(hd,T) matmul alag BLAS
            # kernel pakadte hain -> fp32 reduction order alag. Asli cache bug ka diff
            # order-of-1 hota hai, 1e-4 nahi. Argmax wala assert sach me pakadta hai.
            assert torch.equal(a.argmax(-1), b.argmax(-1)), f"token {t} pe prediction badli"
            diff = (a - b).abs().max().item()
            assert diff < 1e-3, f"token {t} pe mismatch, max diff = {diff:.2e}"


@pytest.mark.parametrize("prompt",PROMPTS)
def test_prefill_then_decode_matches_full_forward(models,prompt):
    """generate() ka actual shape: poora prompt ek baar, phir aakhri token decode step."""
    ours, _, tok = models
    ids = tok(prompt,return_tensors="pt").input_ids
    T = ids.shape[1]

    with torch.inference_mode():
        full = ours(ids)
        cache = KVCache(ours.config.n_layer)
        ours(ids[:, :T-1], cache=cache, position_offset=0)
        step = ours(ids[:, T-1:], cache=cache, position_offset=cache.length)

    a, b = step[:, -1, :], full[:, -1, :]
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    diff = (a - b).abs().max().item()
    assert diff < 1e-3, f"max diff = {diff:.2e}"


    