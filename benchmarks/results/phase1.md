# Phase 1 — Correctness Foundation: conclusions

Machine: RTX 5050 Laptop GPU, torch 2.14.0+cu130, Windows. Model: gpt2 (124M), fp32.
Sab numbers `benchmarks/benchmark_latency.py` se, median of 10 trials, 3 warmup runs.

Reproduce karne ke liye (project root se, `.venv` activate karke):

```powershell
python -m pytest -k correctness -q      # 12 tests
python benchmarks\benchmark_latency.py  # TTFT / TPOT
python main.py                          # ek sample
```

---

## 1. Definition of Done — status

| Item | Status |
|---|---|
| `pytest -k correctness` passes | ✅ 12 passed |
| Greedy output == HF greedy, 3+ prompts | ✅ 50 tokens, 3 prompts, exact string match |
| `main.py` prints a short generated sample | ✅ config-driven, CLI prompt bhi leta hai |

Extra checks jo DoD mein nahi the par kar liye:

- KV cache path == single full forward (token-by-token, aur prefill+decode dono shapes)
- GPU output CPU output se bilkul identical (greedy, 2 prompts)

## 2. Baseline latency

| | CPU (fp32) | GPU (fp32) |
|---|---|---|
| TTFT | 37.1 ms | **6.3 ms** |
| TPOT | 32.5 ms (30.8 tok/s) | **6.5 ms (154.8 tok/s)** |
| total, 128 tokens | 4162.5 ms | **826.4 ms** |

Prompt `"Once upon a time"` (4 tokens), greedy, dynamic KV cache.

`total ≈ TTFT + TPOT × (tokens - 1)` — GPU: `6.3 + 6.5 × 127 = 831.8` vs measured `826.4`. Formula check out hota hai.

~5x ka ye antar **sirf device se** aaya hai. Koi code optimization nahi hui — `load_hf_weights` model ko GPU pe daalne lagi, bas. Phase 2 ka kaam ab isi GPU baseline ke against measure hoga.

## 3. TPOT overhead-bound hai, compute-bound nahi

6.5 ms per token GPT-2 124M ke liye GPU pe bahut slow hai. Ek decode step ka math ~0.25 GFLOP hai; RTX 5050 fp32 mein several TFLOP/s karta hai — yani pure compute 1 ms se kaafi neeche hona chahiye.

Matlab 6.5 ms ka bada hissa **per-step overhead** hai. Kaun dominate kar raha hai ye **abhi profile nahi kiya**, candidates:

- Har step ~120+ kernel launches (12 layers × ~10 ops), har ek pe Python dispatch
- `kv_cache.py` ka `torch.cat` — har step, har layer pe poora cache reallocate + copy
- `model.py` ka `next_token.item()` — har step pe GPU→CPU sync force karta hai

## 4. KV cache ka asli fayda (measured)

Cache on vs off, same prompt, greedy, median of 5 trials. "Off" matlab har step pe poori sequence dobara forward.

| tokens | GPU speedup | CPU speedup |
|---|---|---|
| 64 | 1.09x | 1.73x |
| 128 | 1.30x | 2.75x |
| 256 | 1.76x | — |
| 512 | 1.60x | — |

Output dono paths mein identical raha (har row pe verified).

**Ye theory se kam hai** — cache O(n²) FLOPs ko O(n) karta hai, 128 tokens pe ~64x kam math. Phir 1.30x kyun?

Kyunki **batch=1 pe 124M model memory-bandwidth-bound hai, compute-bound nahi.** Har decode step mein poore 124M weights GPU memory se padhne padte hain — chahe us step mein 1 token bhejo ya 128. Weight-read ka cost dono cases mein same hai. No-cache version ~64x zyada FLOPs karta hai, par GPU ke compute units waise bhi idle the, to wo extra math lagbhag free hai. Cache us kaam ko hata raha hai jo bottleneck hi nahi tha.

CPU compute-bound hai, isliye wahan same 128 tokens pe 2.75x milta hai vs GPU ka 1.30x.

**512 pe speedup gir gaya (1.76 → 1.60)**, jo ulta hona chahiye tha. Shak `torch.cat` pe hai: 512 tokens pe wo per step 12 layers × 2 tensors × 512 × 768 floats copy karta hai, yani cache khud O(T²) memory traffic paida karke apna fayda kha raha hai. Phase 2 ka pre-allocated cache isko confirm ya reject karega.

## 5. fp32 noise vs asli bug — tolerance kaise set ki

Cache path ko full forward se compare karte waqt logits mein `1.6e-4` ka diff mila. Pehle bug lagta tha, nahi tha:

- Logits ki magnitude ~120-146 hai, to relative diff ~`1e-6` — normal fp32 accumulation noise 12 layers ke through
- Har position pe **argmax identical** tha
- Diff `t` ke saath badh bhi nahi raha tha, yani drift accumulate nahi ho raha tha
- Wajah: incremental path `(1,hd)@(hd,t+1)` matmul karta hai, full forward `(T,hd)@(hd,T)` — alag shapes, alag BLAS kernels, alag fp32 reduction order. Mathematically same, numerically nahi.

Tolerance `1e-3` set ki. Validate karne ke liye jaan boojh ke bug daala (`position_offset=0` — classic KV cache galti) aur diff `5.6e+01` aaya. Yani asli bug aur noise ke beech ~5 orders of magnitude ka gap hai, `1e-3` aaram se beech mein baithta hai.

Sabak: threshold dheela karne se pehle check karo ki failure noise hai ya bug — aur threshold set karne ke baad inject-a-bug se verify karo ki test kuch pakadta bhi hai.

## 6. Phase 2 ke liye open items

Priority order, findings ke hisaab se:

0. **Pehle profile karo** (`torch.profiler`, decode step). Section 3 kehta hai overhead dominate kar raha hai par kaunsa, ye pata nahi. Iske bina Phase 2 andhere mein optimize karna hoga.
1. **Pre-allocated KV cache** — `torch.cat` hatao, `max_seq_len` ka fixed tensor allocate karke index se likho. Section 4 ka 512-token dip seedha isi pe point karta hai.
2. **Attention ko `F.scaled_dot_product_attention` pe le jao.** Note: `attention.py` ka docstring already ye claim karta hai, par code abhi manual matmul + `masked_fill` + softmax karta hai aur poora `(B,nh,T,T)` attention matrix materialize karta hai. SDPA `is_causal=True` ke saath isse avoid karta hai aur flash/mem-efficient backend pakadta hai.
3. **fp16/bf16.** Section 4 ke hisaab se decode bandwidth-bound hai, to weights ke bytes aadhe karna seedha usi bottleneck pe lagta hai — yahan sabse bada win expect karo.
4. **CUDA graphs** decode step ke liye — kernel launch overhead pe. Potential sabse zyada, complexity bhi sabse zyada, to 1-3 ke baad.
5. `next_token.item()` ka per-step sync measure karo — hatane layak hai ya noise mein gum hai.

Phase 2 ka DoD: output quality same (ya fp16 ke liye tay ki gayi tolerance ke andar), aur TTFT/TPOT is GPU baseline se better — **numbers record karke**.

Ek cheez jo abhi missing hai: `benchmarks/compare_hf.py` khaali stub hai, to humne apne engine ko HF `model.generate()` ke against kabhi naapa nahi. "Optimized" claim karne se pehle wo baseline chahiye.
