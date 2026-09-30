"""
main.py

Entry point: load configs/gpt2_config.yaml, then run a one-off generation or launch the serving layer.  |  all phases
"""

import sys

import torch

from engine.helper import load_config
from engine.tokenizer import Tokenizer
from engine.weights import load_hf_weights


def sampling_params(sampling):
    """config ki strategy ko generate() ke args mein badalta hai. greedy = temperature 0."""
    strategy = sampling["strategy"]
    if strategy == "greedy":
        return 0, None, None
    if strategy == "topk":
        return sampling["temperature"], sampling["top_k"], None
    if strategy == "topp":
        return sampling["temperature"], None, sampling["top_p"]
    raise ValueError(f"unknown sampling strategy: {strategy}")


def main():
    cfg = load_config()
    torch.manual_seed(cfg["sampling"]["seed"])

    model = load_hf_weights(cfg["model"]["hf_checkpoint"])
    tok = Tokenizer(cfg["paths"]["tokenizer"])

    prompt = sys.argv[1] if len(sys.argv) > 1 else cfg["benchmarks"]["prompt"]
    temperature, top_k, top_p = sampling_params(cfg["sampling"])

    out = model.generate(prompt, cfg["sampling"]["max_new_tokens"], tok,
                         temperature=temperature, top_k=top_k, top_p=top_p)
    print(out)


if __name__ == "__main__":
    main()
