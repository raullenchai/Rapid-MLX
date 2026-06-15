# SPDX-License-Identifier: Apache-2.0
"""Standardized B=1 benchmark for community submissions.

Locks every comparability parameter so a number submitted from one
user's machine can be compared 1:1 with a number submitted from
another's. The defaults mirror llama.cpp's ``llama-bench``
(``-p 512 -n 128 -r 5`` short; long bucket extends to pp=2048,
tg=512 for context-length sensitivity) because that's the de-facto
standard on r/LocalLLaMA / Apple Silicon discussion threads, so
Rapid-MLX numbers slot straight into existing comparison tables.

Hardcoded:

- 2 buckets: short (pp512, tg128) + long (pp2048, tg512)
- 5 measured rounds + 1 warmup discarded
- Greedy decode by default (temp=0, top_p=1) — the apples-to-apples
  baseline. ``sampling='sampled'`` provides the temp=0.7/top_p=0.9
  real-world line as a second, separately-bucketed submission.
- Synthetic random token prompt seeded by ``PROMPT_SEED`` so every
  submitter sends the exact same bytes through the model.
- Decode TPS formula: ``output_tokens / (t_end − t_first_token)``,
  i.e. vLLM TPOT / mlx_lm ``generation_tps`` semantics — excludes
  prefill so the number is the steady-state decoder speed, not a
  conflated end-to-end rate.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import time
from dataclasses import dataclass

from . import PROMPT_SEED, SCHEMA_VERSION

# Lock the comparability axes. Match the schema ``const`` values
# exactly — any drift here either fails schema validation in CI or
# corrupts the database.
ROUNDS_MEASURED: int = 5
ROUNDS_WARMUP: int = 1
SHORT_PROMPT_TOKENS: int = 512
SHORT_MAX_TOKENS: int = 128
LONG_PROMPT_TOKENS: int = 2048
LONG_MAX_TOKENS: int = 512


@dataclass(frozen=True)
class RoundResult:
    decode_tps: float
    prefill_tps: float
    ttft_ms: float
    peak_ram_mb: int | None = None


@dataclass(frozen=True)
class BucketResult:
    rounds_raw: list[RoundResult]

    @property
    def decode_stat(self) -> dict[str, float]:
        return _stat([r.decode_tps for r in self.rounds_raw])

    @property
    def prefill_stat(self) -> dict[str, float]:
        return _stat([r.prefill_tps for r in self.rounds_raw])

    @property
    def ttft_stat(self) -> dict[str, float]:
        return _stat([r.ttft_ms for r in self.rounds_raw])

    def to_schema_dict(self) -> dict:
        return {
            "decode_tps": self.decode_stat,
            "prefill_tps": self.prefill_stat,
            "ttft_ms": self.ttft_stat,
            "rounds_raw": [
                {
                    "decode_tps": r.decode_tps,
                    "prefill_tps": r.prefill_tps,
                    "ttft_ms": r.ttft_ms,
                }
                for r in self.rounds_raw
            ],
        }


@dataclass(frozen=True)
class BenchResult:
    short: BucketResult
    long: BucketResult
    peak_ram_mb: int | None
    prompt_hash: str  # SHA256[:16] of actual prompt tokens fed
    sampling: str  # "greedy" or "sampled"


def _stat(values: list[float]) -> dict[str, float]:
    """median / min / max / stddev — the four columns the schema requires.

    Single-value lists get stddev=0 (sample stdev would raise on n<2).
    """
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "stddev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def _build_synthetic_prompt(
    tokenizer,
    target_tokens: int,
    seed: int,
) -> tuple[str, list[int]]:
    """Generate a deterministic random-token prompt of length ``target_tokens``.

    Matches llama.cpp's ``llama-bench`` approach: random tokens from
    the model's own vocab give bit-exact reproducibility (same seed +
    same tokenizer ⇒ same prompt every time) while exercising the
    prefill path with realistic token distributions.

    Returns ``(prompt_text, prompt_token_ids)``. The text is what we
    feed the engine; the IDs are what we hash for ``prompt_hash``.
    """
    rng = random.Random(seed ^ target_tokens)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(
        getattr(tokenizer, "get_vocab", lambda: {})() or {}
    )
    if vocab_size < 1000:
        raise RuntimeError(
            "tokenizer vocab too small to sample synthetic prompts "
            f"(vocab_size={vocab_size}); is the tokenizer loaded?"
        )
    # Cap to a "safe" range — avoid the very high IDs which are often
    # reserved/special tokens that some models don't decode cleanly.
    cap = min(vocab_size, 100_000)
    # Reserve 0..255 too (often byte-fallback tokens) to keep the
    # prompt's character distribution more model-like.
    ids = [rng.randint(256, cap - 1) for _ in range(target_tokens)]
    # Decode to text; the engine will re-tokenize. The re-tokenized
    # length may differ from ``target_tokens`` by 1-2 due to BPE merges,
    # but the schema fields ``buckets_spec.short.prompt_tokens`` etc.
    # are the *intended* targets — actual observed ``prompt_tokens``
    # comes from the engine's reported ``prompt_tokens`` and is what
    # divides into ``prefill_tps``.
    text = tokenizer.decode(ids)
    return text, ids


def _prompt_hash(token_ids_short: list[int], token_ids_long: list[int]) -> str:
    """SHA256[:16] of the concatenated prompt token IDs.

    The aggregator can verify this matches by re-running the same
    seed against the same tokenizer (no network round-trip needed).
    A non-matching hash is a clear signal of tampering or a
    seed/tokenizer drift bug — both worth flagging.
    """
    payload = (
        ",".join(str(t) for t in token_ids_short)
        + "|"
        + ",".join(str(t) for t in token_ids_long)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def _run_one_round(
    engine,
    prompt_text: str,
    sampling_params,
    target_prompt_tokens: int,
) -> RoundResult:
    """Drive one bench round through ``AsyncEngineCore`` and capture timing.

    All ``time.perf_counter()`` calls happen on the same monotonic
    clock so the deltas are meaningful across awaits.
    """
    t_start = time.perf_counter()
    t_first_token: float | None = None
    last_output = None

    rid = await engine.add_request(prompt_text, sampling_params)
    async for out in engine.stream_outputs(rid, timeout=180):
        if t_first_token is None and out.new_token_ids:
            t_first_token = time.perf_counter()
        last_output = out
        if out.finished:
            break

    t_end = time.perf_counter()

    if t_first_token is None or last_output is None:
        raise RuntimeError(
            "bench round produced no tokens before timeout / EOF — "
            "the engine returned an empty stream, which usually means "
            "the model failed to load or sampling produced zero output"
        )

    prompt_tokens_actual = last_output.prompt_tokens or target_prompt_tokens
    completion_tokens = last_output.completion_tokens or len(
        last_output.output_token_ids
    )

    prefill_window_s = max(t_first_token - t_start, 1e-6)
    decode_window_s = max(t_end - t_first_token, 1e-6)

    return RoundResult(
        prefill_tps=prompt_tokens_actual / prefill_window_s,
        decode_tps=completion_tokens / decode_window_s,
        ttft_ms=(t_first_token - t_start) * 1000.0,
    )


async def _run_bucket(
    engine,
    tokenizer,
    sampling_params_factory,
    target_prompt_tokens: int,
    max_tokens: int,
) -> tuple[BucketResult, list[int]]:
    """Run one bucket: warmup + 5 measured rounds.

    Returns the measured rounds plus the actual prompt token IDs used
    (so callers can hash them for ``prompt_hash``).
    """
    prompt_text, prompt_ids = _build_synthetic_prompt(
        tokenizer, target_prompt_tokens, PROMPT_SEED
    )
    sampling = sampling_params_factory(max_tokens)

    # Warmup rounds (discarded — first-pass JIT + kernel cache warm-up
    # dominates these and would skew the median).
    for _ in range(ROUNDS_WARMUP):
        await _run_one_round(engine, prompt_text, sampling, target_prompt_tokens)

    measured: list[RoundResult] = []
    for _ in range(ROUNDS_MEASURED):
        measured.append(
            await _run_one_round(
                engine, prompt_text, sampling, target_prompt_tokens
            )
        )

    return BucketResult(rounds_raw=measured), prompt_ids


def _make_sampling_params_factory(sampling: str):
    """Build a callable ``max_tokens -> SamplingParams``."""
    from vllm_mlx.request import SamplingParams

    if sampling == "greedy":
        # temp=0 + top_p=1 ⇒ argmax. Deterministic, matches llama-bench.
        def factory(max_tokens: int):
            return SamplingParams(
                max_tokens=max_tokens, temperature=0.0, top_p=1.0, top_k=0
            )
    elif sampling == "sampled":
        # Real-world sampling. Matches Rapid-MLX's serve defaults
        # (config.py SamplingParams.temperature=0.7, top_p=0.9).
        def factory(max_tokens: int):
            return SamplingParams(
                max_tokens=max_tokens, temperature=0.7, top_p=0.9, top_k=0
            )
    else:
        raise ValueError(f"unknown sampling mode: {sampling!r}")
    return factory


async def run_standardized_bench(
    engine,
    tokenizer,
    sampling: str = "greedy",
) -> BenchResult:
    """Run the full short+long standardized bench against a loaded engine.

    The caller owns engine lifecycle (the CLI's bench command opens
    ``AsyncEngineCore`` in an ``async with`` block; we just drive
    rounds through it).
    """
    if sampling not in ("greedy", "sampled"):
        raise ValueError(f"sampling must be 'greedy' or 'sampled', got {sampling!r}")

    factory = _make_sampling_params_factory(sampling)

    short_result, short_ids = await _run_bucket(
        engine, tokenizer, factory, SHORT_PROMPT_TOKENS, SHORT_MAX_TOKENS
    )
    long_result, long_ids = await _run_bucket(
        engine, tokenizer, factory, LONG_PROMPT_TOKENS, LONG_MAX_TOKENS
    )

    peak_ram = _read_peak_ram_mb()

    return BenchResult(
        short=short_result,
        long=long_result,
        peak_ram_mb=peak_ram,
        prompt_hash=_prompt_hash(short_ids, long_ids),
        sampling=sampling,
    )


def _read_peak_ram_mb() -> int | None:
    """Peak Metal-backed memory in MiB, if mlx exposes it.

    Returns ``None`` if mlx isn't installed or the API path differs in
    a future version — the schema allows null and the rest of the
    submission still goes through.
    """
    try:
        import mlx.core as mx

        # mlx.core.metal.get_peak_memory() returns bytes
        peak = getattr(mx, "metal", None)
        if peak is None:
            return None
        getter = getattr(peak, "get_peak_memory", None)
        if getter is None:
            return None
        bytes_ = int(getter())
        return bytes_ // (1024 * 1024)
    except (ImportError, AttributeError, ValueError, OSError):
        return None


def standardized_config_dict(sampling: str, prompt_hash: str) -> dict:
    """Build the ``config`` block for the JSON submission.

    Mirrors ``schema.json#/properties/config`` exactly — every field
    is a const enforced by the schema. The bench runner is the only
    code path that writes these; if the user passed manual knobs
    those are ignored by ``--submit``.
    """
    return {
        "rounds": ROUNDS_MEASURED,
        "warmup_rounds": ROUNDS_WARMUP,
        "sampling": sampling,
        "buckets_spec": {
            "short": {
                "prompt_tokens": SHORT_PROMPT_TOKENS,
                "max_tokens": SHORT_MAX_TOKENS,
            },
            "long": {
                "prompt_tokens": LONG_PROMPT_TOKENS,
                "max_tokens": LONG_MAX_TOKENS,
            },
        },
        "prompt_hash": prompt_hash,
    }


__all__ = [
    "BenchResult",
    "BucketResult",
    "RoundResult",
    "SCHEMA_VERSION",
    "run_standardized_bench",
    "standardized_config_dict",
]
