# GPT-2 Optimized Inference Engine — Project Roadmap

## 1. Project Goal

Ek custom inference engine banana jo GPT-2 model ko run kare **base HuggingFace `model.generate()` se zyada fast**, using:
- Custom attention + KV cache implementation
- Memory-efficient caching
- Batching (multiple requests ek saath)
- Quantization
- Serving layer (API se access karne ke liye)

End result: ek chhota, self-built version of what vLLM / TGI / TensorRT-LLM production mein karte hain — bas GPT-2 scale par.

---

## 2. High-Level Architecture

```
Client Request
      |
      v
[ Serving Layer (FastAPI) ]
      |
      v
[ Request Queue + Scheduler ] --> continuous batching decide karta hai
      |
      v
[ Inference Engine Core ]
      |-- Tokenizer
      |-- Model (GPT-2 blocks: LayerNorm + Attention + MLP)
      |-- KV Cache Manager
      |-- Attention (SDPA / Flash backend)
      |-- Sampler (greedy / top-k / top-p)
      |
      v
[ Streaming Output ] --> token-by-token response client ko
```

---

## 3. Project Structure

```
gpt2-inference-engine/
├── engine/
│   ├── __init__.py
│   ├── model.py              # GPT2 architecture (blocks, attention, MLP)
│   ├── weights.py             # HF checkpoint load + convert karne wala code
│   ├── kv_cache.py            # optimized KV cache class (pre-allocated)
│   ├── attention.py           # attention function (SDPA/flash backend)
│   ├── sampler.py              # greedy, top-k, top-p, temperature sampling
│   ├── tokenizer.py            # tokenizer wrapper
│   └── batching.py              # continuous/dynamic batching scheduler
│
├── serving/
│   ├── server.py               # FastAPI server, streaming responses
│   ├── request_queue.py        # incoming requests ko batch mein group karna
│   └── scheduler.py            # kaunsi requests ek saath process hongi
│
├── optimizations/
│   ├── quantization.py         # int8/int4 quantization logic
│   ├── cuda_graphs.py          # CUDA graph capture (decode step)
│   └── speculative.py          # speculative decoding (optional, advanced)
│
├── benchmarks/
│   ├── benchmark_latency.py    # single request latency
│   ├── benchmark_throughput.py # concurrent throughput
│   └── compare_hf.py           # apna engine vs HF baseline
│
├── tests/
│   └── test_correctness.py     # HF ke outputs se match karte hain ya nahi
│
├── configs/
│   └── gpt2_config.yaml
│
└── main.py                     # entry point
```

---

## 4. Phase-Wise Roadmap

### Phase 1 — Correctness Foundation
**Goal:** Apna GPT-2 implementation bane, aur HuggingFace ke outputs se exactly match kare.

Tasks:
- GPT-2 block banao (LayerNorm + Attention + MLP + residuals)
- HuggingFace pretrained weights load karke apni class mein daalo (Q/K/V split, transpose, GELU-tanh)
- `torch.allclose()` se HF ke logits ke saath verify karo
- Basic KV cache (dynamic, `torch.cat` based) ke saath ek simple generation loop chalao

**Deliverable:** Apna model HF jaisa hi output de raha ho, cache ke saath ek chhota text generate ho raha ho.

**Estimated time:** 1–1.5 weeks

---

### Phase 2 — Single-Request Optimization
**Goal:** Ek hi request ko jitna fast ho sake process karna.

Tasks:
- KV cache ko **pre-allocated fixed-size tensor** mein convert karo (`torch.cat` hatao)
- `F.scaled_dot_product_attention` confirm karo flash-attention backend use kar raha hai
- Model ko fp16/bf16 mein convert karo
- `torch.inference_mode()` sab jagah lagao
- CUDA graph capture karo decode step ke liye (Python overhead kam karne ke liye)

**Deliverable:** Same output quality, lekin measurable latency improvement HF baseline se.

**Estimated time:** 1–2 weeks

---

### Phase 3 — Batching
**Goal:** Multiple requests ek saath handle karna, GPU utilization maximize karna.

Tasks:
- Static batching implement karo (fixed batch size)
- Continuous batching scheduler banao — jab ek request khatam ho, naya request turant slot le le
- Padding/masking sahi se handle karo variable-length sequences ke liye

**Deliverable:** Throughput (tokens/sec across requests) single-request se kaafi zyada.

**Estimated time:** 1.5–2 weeks

---

### Phase 4 — Memory Management
**Goal:** KV cache memory ko efficient banana jab alag-alag length ki requests ek saath chal rahi hon.

Tasks:
- Per-request dynamic cache sizing (fixed max allocate mat karo har request ke liye)
- (Optional, advanced) PagedAttention jaisa block-based cache — vLLM ka core idea, chhote "pages" mein memory allocate karna

**Deliverable:** Zyada concurrent requests handle ho paayein bina OOM ke.

**Estimated time:** 1–2 weeks (paging skip karo to 0.5 week)

---

### Phase 5 — Quantization
**Goal:** Memory footprint kam karna, possibly speed bhi improve karna.

Tasks:
- INT8 quantization implement karo (`bitsandbytes` library se easy integration)
- Accuracy loss check karo (quantized vs fp16 output comparison)

**Deliverable:** Model chhoti memory mein chal raha ho, minimal quality loss ke saath.

**Estimated time:** 1 week

---

### Phase 6 — Speculative Decoding (Optional/Advanced)
**Goal:** Decode step ki wall-clock speed badhana.

Tasks:
- Chhota draft model (distilled GPT-2 ya smaller variant) use karo multiple tokens guess karne ke liye
- Bada model unhe parallel verify kare, galat guesses discard ho

**Deliverable:** Decode throughput mein additional speedup, bina quality kharab kiye.

**Estimated time:** 1.5–2 weeks (skip kar sakte ho agar time kam hai — ye "nice-to-have" hai)

---

### Phase 7 — Serving Layer
**Goal:** Engine ko ek usable API ke roop mein expose karna.

Tasks:
- FastAPI server banao, request/response schema define karo
- Streaming responses (Server-Sent Events ya WebSocket) — token-by-token output
- Request queue + scheduler serving layer se integrate karo

**Deliverable:** `curl`/Postman se hit kar sako aur streaming text response mile.

**Estimated time:** 1–1.5 weeks

---

### Phase 8 — Benchmarking & Validation
**Goal:** Prove karna ki engine actually optimized hai.

Metrics track karo:
- **TTFT** (Time to First Token) — prefill speed
- **TPOT** (Time Per Output Token) — decode speed
- **Throughput** (tokens/sec across concurrent requests)

Compare karo:
- Plain HF `model.generate()` (baseline)
- Apna engine — har phase ke baad progressively

**Estimated time:** Ongoing, parallel to har phase (dedicate karo ~0.5 week final polish ke liye)

---

## 5. Overall Timeline Summary

| Phase | Focus | Duration |
|---|---|---|
| 1 | Correctness foundation | 1–1.5 weeks |
| 2 | Single-request optimization | 1–2 weeks |
| 3 | Batching | 1.5–2 weeks |
| 4 | Memory management | 1–2 weeks |
| 5 | Quantization | 1 week |
| 6 | Speculative decoding (optional) | 1.5–2 weeks |
| 7 | Serving layer | 1–1.5 weeks |
| 8 | Benchmarking (parallel/final) | 0.5 week |

**Total (Phase 6 ke saath):** ~9–12.5 weeks
**Total (Phase 6 skip karke, MVP-focused):** ~7.5–10.5 weeks

*Ye estimates part-time/learning-pace ke hisaab se hain. Agar full-time daily kaam karoge, ye timeline lagbhag aadha ho sakta hai.*

---

## 6. Priority Order (agar time kam ho)

1. **Phase 1** — must, bina iske kuch bhi valid nahi
2. **Phase 2** — must, sabse zyada bang-for-buck (fast wins)
3. **Phase 3** — high impact agar multiple requests handle karni hain
4. **Phase 7** — must agar "engine" ko usable banana hai
5. **Phase 4, 5, 6** — incremental gains, nice-to-have, jitna time bache utna karo

---

## 7. Tech Stack

| Component | Tool |
|---|---|
| Core framework | PyTorch |
| Attention backend | `F.scaled_dot_product_attention` |
| Serving | FastAPI |
| Tokenizer | HuggingFace `tokenizers` (Rust-backed) |
| Quantization | `bitsandbytes` |
| Profiling | `torch.profiler`, `nvidia-smi`, NVIDIA Nsight (`nsys`) |

---

## 8. Notes / Risks

- **Correctness pehle, speed baad mein** — agar Phase 1 mein hi mismatch hai HF outputs se, aage sab kaam waste jaayega.
- GPT-2 chhota model hai (124M–1.5B params) — quantization/speculative decoding ka benefit utna dramatic nahi dikhega jitna bade LLMs (7B+) mein, lekin implementation practice ke liye valuable hai.
- Continuous batching aur memory paging sabse zyada engineering-heavy parts hain — inme sabse zyada time lag sakta hai estimate se.
- Har phase ke baad benchmark zaroor chalao — "optimize" claim karne se pehle number dikhna chahiye.
