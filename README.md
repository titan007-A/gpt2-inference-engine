# GPT-2 Optimized Inference Engine

A from-scratch, optimized inference engine for GPT-2: custom attention + KV cache,
continuous batching, quantization, and a streaming serving layer. A GPT-2-scale
re-implementation of what vLLM / TGI / TensorRT-LLM do in production.

## Start here

Follow **`GPT2_Inference_Engine_Build_Guide.docx`** (in the project root) top to bottom.
The original plan is in `gpt2_inference_engine_roadmap.md`.

## Setup

```powershell
# from the project root
.venv\Scripts\Activate.ps1          # PowerShell  (cmd: .venv\Scripts\activate.bat)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## Layout

| Path | What lives here |
|---|---|
| `engine/`        | model, weights, KV cache, attention, sampler, tokenizer, batching |
| `serving/`       | FastAPI server, request queue, scheduler |
| `optimizations/` | quantization, CUDA graphs, speculative decoding |
| `benchmarks/`    | latency, throughput, HuggingFace comparison |
| `tests/`         | correctness (parity with HuggingFace) |
| `configs/`       | `gpt2_config.yaml` |
| `main.py`        | entry point |

## Ground rules

1. **Correctness before speed** - Phase 1 parity with HuggingFace must hold before anything else.
2. **Benchmark after every phase** - record TTFT, TPOT, throughput; show the number before claiming "optimized".
