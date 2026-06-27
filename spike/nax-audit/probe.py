"""Phase-7 NAX audit bench probe.

Times matmul paths on Apple Silicon to detect whether MLX exposes a
true int8 fast-path (W8A8) versus a software upcast.

We measure:
  baseline_f16      mx.matmul(f16, f16)                              # fp16 dense
  baseline_bf16     mx.matmul(bf16, bf16)
  qm_W8_f16         mx.quantized_matmul(f16_x, W8_q, ...)            # W-only int8, A is fp16
  qm_W8_i8          mx.quantized_matmul(int8_x, W8_q, ...)           # A passed as int8 (probe)
  qm_W4_f16         mx.quantized_matmul(f16_x, W4_q, ...)            # W-only int4
  mlx_int_matmul    mx.matmul(int8, int8)                            # expected to fail
"""

from __future__ import annotations

import time
import statistics
import platform
import subprocess

import mlx.core as mx


REPS = 20
WARMUP = 3
M, K, N = 32, 4096, 4096  # decode-shape-ish: short M, square KxN


def t(fn, label):
    # warmup
    for _ in range(WARMUP):
        y = fn()
        mx.eval(y)
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples) * 1000.0
    lo = min(samples) * 1000.0
    print(f"  {label:24s}  median={med:8.3f} ms  min={lo:8.3f} ms  shape=({M}x{K})@({K}x{N})")
    return med


def main():
    print("=== Host ===")
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = "?"
    print(f"  chip:     {chip}")
    print(f"  platform: {platform.platform()}")
    # MLX version — no __version__ on the C++ side; we just print the pin
    print(f"  mlx:      installed version sourced from `pip show mlx`")

    print()
    print("=== Setup ===")
    x_f16 = mx.random.normal((M, K)).astype(mx.float16)
    x_bf16 = mx.random.normal((M, K)).astype(mx.bfloat16)
    # weight as (N, K) for transpose=True semantic in quantized_matmul
    w_full = mx.random.normal((N, K)).astype(mx.float16)
    w_full_bf = w_full.astype(mx.bfloat16)

    w_q8, s8, b8 = mx.quantize(w_full, group_size=64, bits=8)
    w_q4, s4, b4 = mx.quantize(w_full, group_size=64, bits=4)

    x_i8 = (x_f16 * 64).astype(mx.int8)  # simulated int8 activation

    print(f"  x_f16:   shape={x_f16.shape} dtype={x_f16.dtype}")
    print(f"  x_bf16:  shape={x_bf16.shape} dtype={x_bf16.dtype}")
    print(f"  x_i8:    shape={x_i8.shape} dtype={x_i8.dtype}")
    print(f"  w_full:  shape={w_full.shape} dtype={w_full.dtype}")
    print(f"  w_q8:    shape={w_q8.shape} dtype={w_q8.dtype}  (packed in uint32)")
    print(f"  w_q4:    shape={w_q4.shape} dtype={w_q4.dtype}  (packed in uint32)")

    mx.eval(x_f16, x_bf16, x_i8, w_full, w_full_bf, w_q8, s8, b8, w_q4, s4, b4)

    print()
    print("=== Bench ===")
    bf16  = t(lambda: mx.matmul(x_bf16, w_full_bf.T),                                        "matmul bf16 dense")
    f16   = t(lambda: mx.matmul(x_f16,  w_full.T),                                           "matmul f16 dense")
    qm_w8 = t(lambda: mx.quantized_matmul(x_f16, w_q8, s8, b8, bits=8, transpose=True),      "quantized_matmul W8 f16")
    qm_w4 = t(lambda: mx.quantized_matmul(x_f16, w_q4, s4, b4, bits=4, transpose=True),      "quantized_matmul W4 f16")

    print()
    print("=== int8 activation probe (does MLX have a real i8 path?) ===")
    # int8 x with quantized_matmul — what happens?
    try:
        qm_w8_i8 = t(lambda: mx.quantized_matmul(x_i8, w_q8, s8, b8, bits=8, transpose=True), "quantized_matmul W8 i8")
        print(f"  observation: W8+i8 path took {qm_w8_i8:.3f} ms (vs W8+f16 {qm_w8:.3f} ms)")
        if qm_w8_i8 < qm_w8 * 0.85:
            print("  >>> int8 activation MAY be faster — possible fast-path")
        else:
            print("  >>> int8 activation NOT faster — almost certainly software upcast")
    except Exception as e:
        print(f"  W8+i8 FAILED: {type(e).__name__}: {e}")

    # The smoking gun: try int matmul directly
    print()
    print("=== mx.matmul integer support (the deciding test) ===")
    for label, a, b_ in [
        ("int8 @ int8",   mx.zeros((M, K), dtype=mx.int8),  mx.zeros((K, N), dtype=mx.int8)),
        ("int32 @ int32", mx.zeros((M, K), dtype=mx.int32), mx.zeros((K, N), dtype=mx.int32)),
    ]:
        try:
            y = mx.matmul(a, b_)
            mx.eval(y)
            print(f"  {label}: OK, dtype={y.dtype}")
        except Exception as e:
            print(f"  {label}: REJECTED — {str(e)[:160]}")

    print()
    print("=== Speedup summary (vs f16 dense matmul) ===")
    print(f"  bf16          : {f16/bf16:5.2f}x")
    print(f"  W8 (f16 act)  : {f16/qm_w8:5.2f}x")
    print(f"  W4 (f16 act)  : {f16/qm_w4:5.2f}x")


if __name__ == "__main__":
    main()
