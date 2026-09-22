"""
engine/weights.py

Load the HuggingFace GPT-2 checkpoint and convert it into this engine's module layout (name mapping, fused-QKV split, Conv1D -> Linear transpose, GELU-tanh).  |  Phase 1
"""

import torch
from transformers import GPT2LMHeadModel

from engine.attention import GPT2Config
from engine.model import GPT2Model

# HF stores these as Conv1D (in, out); nn.Linear wants (out, in) -> transpose
CONV1D_WEIGHTS = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

def load_hf_weights(model_name="gpt2"):
    hf= GPT2LMHeadModel.from_pretrained(model_name)

    config = GPT2Config(
        vocab_size=hf.config.vocab_size,
        block_size=hf.config.n_positions,
        n_layer=hf.config.n_layer,
        n_head=hf.config.n_head,
        n_embd=hf.config.n_embd,
        dropout=0.0,   # inference: no dropout
    )

    model = GPT2Model(config)

    new_sd={}
    for key,tensor in hf.state_dict().items():
        key = key.removeprefix("transformer.")
        if key.endswith((".attn.bias", ".attn.masked_bias")):   # old HF mask buffers, not weights
            continue
        if key.endswith(CONV1D_WEIGHTS):
            tensor = tensor.t()
        new_sd[key] = tensor

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    missing = [k for k in missing if not k.endswith("attn.mask")]
    assert not missing and not unexpected, (missing, unexpected)

    return model.eval()

