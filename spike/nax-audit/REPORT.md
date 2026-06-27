# Phase-7 NAX audit — W8A8 spike GO/NO-GO gate (task #315)

**Date:** 2026-06-26
**Author:** spike/nax-audit agent
**Host:** Apple M3 Ultra, Mac Studio (Mac15,14), 256 GB, macOS 26.5.1, MLX 0.29.3 installed (pyproject floor `mlx>=0.31.2`)

---

## TL;DR — verdict

> ### **KILL** the 2-3 wk W8A8 + fused-kernel framing in task #315.
>
> The Apple Silicon int8 GEMM hardware path is **M5+ only** (Metal 4
> `mpp::tensor_ops::matmul2d` via `cooperative_tensor`). Our primary target
> box is **M3 Ultra**, which does not have it. MLX itself currently refuses
> integer matmul at the API surface, and Apple's MLX maintainer has stated
> publicly (2024) that a both-quantized matmul is not planned in the near
> future. A reference implementation for the M5+ case already exists
> ([Cider](https://github.com/Mininglamp-AI/cider), MIT) and self-reports
> 1.2-1.9× prefill / 7.4% end-to-end VLM decode — i.e. **a fraction of the
> 1.3-1.5× decode win task #315 assumes**, and the implementation cost
> there is days, not weeks.
>
> **Recommended next action:** close task #315 in its current form.
> Replace with a small ticket: *"Evaluate Cider integration behind an M5+
> runtime cap-check"* (~3 day spike), gated on enough M5 hardware in
> production to matter. On M3/M4 (the actual installed base today) there
> is no W8A8 win available **at any engineering cost** until Apple ships
> Metal 4 tensor ops to older silicon, which they have not announced.

Two facts drove this:

1. **Hardware:** INT8×INT8 → INT32 GEMM via Metal 4 cooperative_tensor is M5+ only. M3 Ultra and M4 GPUs do not expose simdgroup int8 matrix ops — every Metal GEMM kernel in the installed `mlx.metallib` is templated on float types (`simdgroup_matrix_8x8_multiply_accumulate.v64f32...`), with zero int8 / int32 variants.
2. **API:** `mx.matmul(int8, int8)` is hard-rejected by MLX: *"Only inexact types are supported but int8 and int8 were provided"*. `mx.quantized_matmul` silently accepts an int8 `x` input but is not faster than float-activation — our bench shows **0.450 ms vs 0.439 ms on the same shape on M3 Ultra** (1.03×, within noise). It is weight-only and upcasts internally.

---

## Findings

### 1. MLX version pin

| Source                          | Pin              |
|---------------------------------|------------------|
| `pyproject.toml`                | `mlx>=0.31.2`    |
| `pyproject.toml`                | `mlx-lm>=0.31.3` |
| `pyproject.toml` (vision extra) | `mlx-vlm>=0.6.3` |
| Installed on probe box          | `mlx 0.29.3`     |

> Probe ran on `mlx 0.29.3` because the host Python lags the pin. None of the audit's negative findings change at 0.31.2 — the upstream symbols probed (`mx.matmul`, `mx.quantized_matmul`, `mx.fast.*`) and the error-string identifying integer matmul as unsupported are still present in 0.31.x. The NAX-specific kernel (`qmm_t_nax`) was added in the 0.31 line per [#3584](https://github.com/ml-explore/mlx/issues/3584), but is still **weight-only** quantization.

### 2. AMX int8 dot product exposure in MLX — NO

- `mx.matmul` rejects all integer dtypes outright (confirmed by probe; see §Bench). Source: `[matmul] Only inexact types are supported`.
- No public `mx.fast.matmul_int8` / `mx.fast.dot_int8` / equivalent. `dir(mx.fast)` is exactly: `cuda_kernel, layer_norm, metal_kernel, precompiled_cuda_kernel, rms_norm, rope, scaled_dot_product_attention`.
- No `mx.int_matmul` / `mx.imatmul` / `mx.matmul_int` symbol.
- Searching the installed `libmlx.dylib` for `int8.*matmul`, `matmul.*int8`, `amx`, `nax`, `simdgroup_matrix.*int` — **zero hits**. The only `int8` references are dtype storage/cast and template `int8_t` in unrelated reductions.
- Searching the installed `mlx.metallib`: every `simdgroup_matrix_*_multiply_accumulate` kernel name has the suffix `.v64f32.v64f32.v64f32.v64f32`. There is no int8 GEMM kernel anywhere in the metallib.
- **Critical maintainer statement** ([ml-explore/mlx#1293](https://github.com/ml-explore/mlx/issues/1293), 2024-07-28 — Apple's @angeloskath): *"A matmul kernel where both matrices are quantized is not currently implemented in MLX. […] **I don't think we plan to implement this in the near future.**"* Closed `wontfix`.
- Note on AMX scope: AMX is the CPU coprocessor reachable via Accelerate / `vDSP`. MLX's `mx.matmul` defaults to the GPU stream, where AMX is **not reachable** at all. Even if AMX could be reached on a CPU stream, MLX still has no int8 GEMM dispatch.

### 3. NAX (M5 Neural Accelerator) — partial, weight-only, not for us

- **Apple position** ([ml-explore/mlx#2693](https://github.com/ml-explore/mlx/issues/2693), 2025-10-22 — Apple's @awni): *"Neural accelerator support is a work in progress. We will be evaluating additional Metal 4 features on a case-by-case basis and include what makes sense. FYI AMX != Neural Accelerators."*
- A NAX-aware kernel `qmm_t_nax` exists in MLX 0.31.x (referenced in [#3584](https://github.com/ml-explore/mlx/issues/3584)) — but it is **`qmm` = quantized matmul = weight-only**. The activation is still fp16/bf16. NAX shipping in MLX has not changed the W-vs-WA situation.
- NAX shows active regression bugs at M=96–128 ([#3584](https://github.com/ml-explore/mlx/issues/3584), open: 1.5-1.8× slower than fp16) — i.e. you can today be *slower* on NAX than on a plain fp16 GEMM. Not a quiet path to ship on.
- Even at parity, MLX on M5 is reported 10-20% slower than PyTorch MPS for BF16 GEMM ([#3196](https://github.com/ml-explore/mlx/issues/3196), open).

### 4. What `mx.quantized_matmul` already gives us — W-only

Signature: `quantized_matmul(x, w_q, scales, biases, transpose=True, group_size=64, bits=4)`. The kernel is templated on the input dtype `x` and the error path enforces *"only floating types are supported"* (confirmed via string-table dump of `libmlx.dylib`). When given an `int8` `x` it accepts silently — almost certainly upcasting. Our bench measures **no speedup** from int8 `x` (0.450 ms vs 0.439 ms; see §Bench).

The wins delivered by `quantized_matmul` today come from packing the weight into 4-/8-bit and saving DRAM bandwidth, not from any int8 compute path. Decoder shape on M3 Ultra: W8 is 1.40× over fp16 dense, W4 is 1.45× — both essentially bandwidth-bound, with int8 packing already giving us most of the win that "true" W8A8 would.

### 5. Reference implementations in the wild

- **Cider** — [Mininglamp-AI/cider](https://github.com/Mininglamp-AI/cider) (MIT, MLX ≥ 0.31). Online activation quantization for W8A8/W4A8 + optimized SDPA, as MLX custom primitives. Uses `mpp::tensor_ops::matmul2d(16, 32, 16)` via Metal 4 `cooperative_tensor`. **M5+ only**; on M4 and earlier it installs as pure-Python and `is_available()` returns `False`. Self-reported numbers: 1.2-1.9× prefill, 1.6× SDPA at 32K, **7.4% end-to-end decode** on Qwen3-VL-4B.
- **Rigel** — arxiv 2606.12765, *Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU*. Confirms M4 Max has tensor cores but the int8 path documented in MSL 4 is not exposed there.
- **llama.cpp** — [PR #16634](https://github.com/ggml-org/llama.cpp/pull/16634) "initial Metal4 tensor API support" — confirms the tensor-API gate is on Metal 4 / M5.
- **Apple WWDC25 §298** ("Explore LLMs on Apple silicon with MLX") and WWDC25 §262 ("Combine Metal 4 ML and graphics") — Apple themselves frame Metal 4 tensor ops as the M5 story; no backport mentioned.
- **mlx-examples** — no W8A8 in tree as of audit date.

### 6. Hardware-path verdict per platform

| Platform               | Hardware path                       | MLX exposes it?                | Verdict for #315 work                                   |
|------------------------|-------------------------------------|--------------------------------|---------------------------------------------------------|
| M2 (dev/laptop family) | No GPU int8 GEMM at all             | No                             | **NO-GO**                                               |
| **M3 Ultra (primary)** | **No GPU int8 GEMM**                | **No**                         | **NO-GO** — no win available at any cost                |
| M4 / M4 Max            | No Metal 4 cooperative_tensor int8  | No                             | **NO-GO**                                               |
| M5 / M5 Pro / M5 Max   | Yes (`mpp::tensor_ops::matmul2d`)   | Indirectly via Cider's custom primitives; MLX core does not expose int8×int8 | **CONDITIONAL GO** via Cider integration (≤3 days), **NOT** a 2-3 week novel-kernel spike |

### 7. Cost / risk reassessment for task #315

The task description estimates *2-3 weeks for W8A8 + fused Metal kernel; 1.3-1.5× decode*. With what we found:

- On M3/M4: there is no hardware path and Apple has not announced one. *Effort to deliver the stated win: indeterminate (years; depends on Apple).* This is not an estimating problem, it's a **physical impossibility** until either Apple backports Metal 4 tensor ops or we ship custom GPU shader code that doesn't use tensor cores — which won't yield 1.3-1.5× because we'd be back on simdgroup_matrix fp16.
- On M5+: hardware exists, **but**:
  - The 1.3-1.5× *decode* projection in #315 is not supported by any public W8A8-on-Apple-Silicon number. Cider's self-reported end-to-end decode win on Qwen3-VL-4B is **7.4%**, not 30-50%. Prefill is where the win lives (1.2-1.9×) — and prefill is not our decode-bound serving bottleneck.
  - A working MIT-licensed reference (Cider) already exists; novel kernel work is wasted effort.
  - A correct M5+ spike is **integrate-and-A/B Cider behind a runtime cap-check ≈ 3 days**, not write a kernel ≈ 2-3 weeks.
- Risk profile: NAX-related regressions are still landing in MLX 0.31.x ([#3584](https://github.com/ml-explore/mlx/issues/3584), [#3196](https://github.com/ml-explore/mlx/issues/3196), [#3534](https://github.com/ml-explore/mlx/issues/3534), [#3656](https://github.com/ml-explore/mlx/issues/3656), [#3702](https://github.com/ml-explore/mlx/issues/3702)). Building on top of an actively-regressing kernel surface, on a platform with thin installed base, with a 7.4% expected win — bad EV.

---

## Bench probe (raw, M3 Ultra)

Script: `spike/nax-audit/probe.py`. Shape `(M=32, K=4096, N=4096)`, repeat 20 + 3 warmup, median wall-clock.

```
=== Bench ===
  matmul bf16 dense         median=   0.645 ms  min=   0.575 ms
  matmul f16 dense          median=   0.616 ms  min=   0.573 ms
  quantized_matmul W8 f16   median=   0.439 ms  min=   0.433 ms
  quantized_matmul W4 f16   median=   0.425 ms  min=   0.383 ms

=== int8 activation probe (does MLX have a real i8 path?) ===
  quantized_matmul W8 i8    median=   0.450 ms  min=   0.439 ms
  observation: W8+i8 path took 0.450 ms (vs W8+f16 0.439 ms)
  >>> int8 activation NOT faster — almost certainly software upcast

=== mx.matmul integer support (the deciding test) ===
  int8  @ int8 : REJECTED — [matmul] Only inexact types are supported but int8 and int8 were provided which results in int8, which is not a floating point type.
  int32 @ int32: REJECTED — [matmul] Only inexact types are supported but int32 and int32 were provided which results in int32, which is not a floating point type.

=== Speedup summary (vs f16 dense matmul) ===
  bf16          :  0.96x
  W8 (f16 act)  :  1.40x
  W4 (f16 act)  :  1.45x
```

Interpretation: the *already-shipping* W8 weight-only path delivers **1.40×** on M3 Ultra at decoder shape — almost the same as the 1.3-1.5× the task #315 hypothesis attributes to W8A8. There is **no further headroom from int8 activations on this box**.

(Full output: `spike/nax-audit/probe_output.txt`.)

---

## Recommendation

1. **Close task #315** in its current 2-3 wk fused-kernel framing. The work has no hardware target on rapid-mlx's primary box (M3 Ultra), MLX has no API surface for it, and Apple's MLX team explicitly deprioritized it.
2. **Replace with a small follow-up:** *"Evaluate Cider for W8A8 on M5+ behind a runtime cap-check"* — ≤ 3 day spike, gated on M5+ representation in production.
3. **Update ROADMAP / 0.8TODO.md** to reflect that the W8A8 win on the dominant install base (M2/M3/M4) is unavailable until Apple's posture changes. Keep watching [ml-explore/mlx#2693](https://github.com/ml-explore/mlx/issues/2693) for any signal of a non-M5 path.
4. **Bank the W8 / W4 bandwidth win we already have.** The bench shows 1.40-1.45× is *already shipping* via `mx.quantized_matmul`. Make sure the serving path is opting into W8 where the quality budget allows.

---

## Sources

- [ml-explore/mlx#2693 — Metal 4 / M5 support (Tensor APIs to program GPU Neural Accelerators)](https://github.com/ml-explore/mlx/issues/2693)
- [ml-explore/mlx#1293 — How can we enable w4a8 GEMM in MLX? (wontfix)](https://github.com/ml-explore/mlx/issues/1293)
- [ml-explore/mlx#3584 — qmm_splitk dispatch regression on NAX](https://github.com/ml-explore/mlx/issues/3584)
- [ml-explore/mlx#3196 — NA/M5 matmul slower than PyTorch MPS BF16](https://github.com/ml-explore/mlx/issues/3196)
- [ml-explore/mlx#3534 — M5 float32 precision issue](https://github.com/ml-explore/mlx/issues/3534)
- [ml-explore/mlx#3656 — m5 max quantized_matmul artifacts](https://github.com/ml-explore/mlx/issues/3656)
- [ml-explore/mlx#3702 — A19 GPU returns wrong float32 results](https://github.com/ml-explore/mlx/issues/3702)
- [ml-explore/mlx#3764 — Add small-batch quantized matvec kernel (merged 2026-06-26)](https://github.com/ml-explore/mlx/pull/3764)
- [Mininglamp-AI/cider — W8A8/W4A8 on Apple Silicon (MIT)](https://github.com/Mininglamp-AI/cider)
- [Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on M4 Max](https://arxiv.org/html/2606.12765v1)
- [ggml-org/llama.cpp#16634 — initial Metal 4 tensor API support](https://github.com/ggml-org/llama.cpp/pull/16634)
- [Apple WWDC25 §298 — Explore LLMs on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)
- [Apple WWDC25 §262 — Combine Metal 4 ML and graphics](https://developer.apple.com/videos/play/wwdc2025/262/)
- [Apple WWDC26 §330 — Optimize custom ML ops with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/)
- [Metal Shading Language Specification v4 (PDF)](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
