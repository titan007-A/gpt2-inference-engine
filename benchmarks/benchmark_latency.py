"""
benchmarks/benchmark_latency.py

Single-request TTFT + TPOT, warmup + trials, report median.  |  Phase 8
"""

import statistics
import time

import torch

from engine.helper import load_config
from engine.tokenizer import Tokenizer
from engine.weights import load_hf_weights


def timed(fn, device):
    # CUDA kernels async launch hote hain, to sync ke bina timer kaam khatam hone ka
    # intezaar nahi karega aur number bakwaas aayega. CPU pe no-op hai.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def main():
    cfg = load_config()
    bench, sampling = cfg["benchmarks"], cfg["sampling"]
    max_new = sampling["max_new_tokens"]
    prompt = bench["prompt"]

    model = load_hf_weights(cfg["model"]["hf_checkpoint"], cfg["model"]["device"])
    tok = Tokenizer(cfg["paths"]["tokenizer"])
    param = next(model.parameters())
    device = param.device
    prompt_len = tok.encode(prompt).shape[1]

    for _ in range(bench["warmup_runs"]):
        model.generate(prompt, max_new, tok)

    ttfts, totals, n_gen = [], [], None
    for _ in range(bench["trials"]):
        # max_new_tokens=1 -> sirf prefill + ek sample (decode forward guard se skip hota hai),
        # yani ye seedha TTFT hai bina generate() ko instrument karne ki zaroorat ke
        t, _ = timed(lambda: model.generate(prompt, 1, tok), device)
        ttfts.append(t)

        t, out = timed(lambda: model.generate(prompt, max_new, tok), device)
        totals.append(t)
        # EOS pe loop pehle break kar sakta hai, to asli count nikalo warna TPOT galat aayega
        n_gen = tok.encode(out).shape[1] - prompt_len

    ttft = statistics.median(ttfts)
    total = statistics.median(totals)
    tpot = (total - ttft) / (n_gen - 1)

    print(f"device     {device}   dtype {param.dtype}   torch {torch.__version__}")
    print(f"prompt     {prompt!r}  ({prompt_len} tokens)")
    print(f"generated  {n_gen} tokens   trials {bench['trials']}   warmup {bench['warmup_runs']}")
    print(f"strategy   {sampling['strategy']}   kv_cache {cfg['kv_cache']['mode']}")
    print()
    print(f"TTFT       {ttft * 1000:8.1f} ms")
    print(f"TPOT       {tpot * 1000:8.1f} ms   ({1 / tpot:.1f} tok/s)")
    print(f"total      {total * 1000:8.1f} ms")


if __name__ == "__main__":
    main()
