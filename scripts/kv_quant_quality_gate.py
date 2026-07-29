#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KV-quant differential quality gate (local, advisory).

Runs the SAME model twice on the SAME prompts — an unquantized-KV BASELINE and a
quantized-KV CANDIDATE (``int8`` / ``int4``) — and measures how well the
candidate AGREES WITH ITS OWN unquantized baseline. This differential framing
isolates *quantization-induced* degradation from a model's plain inability at a
prompt: the gate only faults a candidate the quantized cache made worse than its
own full-precision self. (The baseline dtype is the model's native KV dtype —
usually bf16 or fp16 — and is detected and reported, not assumed.)

Today, ``vllm_mlx/kv_cache_dtype.py`` decides whether int4/int8 KV is "safe" via
a hand-written empirical safelist. This harness produces the *measured* signal
that list currently lacks.

It runs entirely locally (no external-service calls). Like any inference tool it
loads the model through the standard HuggingFace cache, which fetches the weights
once if they are not already cached; pass ``--offline`` to forbid any network
fetch (the load then fails fast if the model isn't fully cached). The bundled
tests never download — they are cache-only and skip when the model is absent.

It ships **advisory** (measure-first, mirroring the ``diff_coverage`` pattern):
it prints a full report + PASS/FAIL but exits ``0`` regardless. Pass
``--enforce`` to make a FAIL exit non-zero (for a future promotion to a blocking
gate once thresholds are calibrated on fleet data).

All *scoring* lives in the pure, hermetically-tested :mod:`vllm_mlx.kv_quant_gate`;
this file only drives inference and prints. Chip-tier gating of the optional NIAH
metric uses :mod:`vllm_mlx.chip_tier`.

Usage:
    python -m scripts.kv_quant_quality_gate qwen3.5-4b-4bit
    python -m scripts.kv_quant_quality_gate mlx-community/Qwen3-0.6B-4bit \\
        --candidate int8 int4 --max-tokens 48 --json-out report.json
    python -m scripts.kv_quant_quality_gate <model> --niah --enforce --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Allow ``python scripts/kv_quant_quality_gate.py`` (not just ``-m``) by putting
# the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_mlx.chip_tier import classify_chip_tier, detect_chip_tier  # noqa: E402
from vllm_mlx.kv_cache_dtype import dtype_to_quantization_bits  # noqa: E402
from vllm_mlx.kv_quant_gate import (  # noqa: E402
    AgreementResult,
    LogitDivergence,
    build_report,
    default_thresholds,
    greedy_agreement_rate,
    logit_divergence,
    structured_output_retention,
)
from vllm_mlx.quantized_batch_cache import (  # noqa: E402
    probe_kv_head_dims,
    resolve_kv_quantization,
)

# Built-in prompt set. A couple explicitly ask for JSON so the structured-output
# retention metric has attributable prompts. Kept small — this is an M-sized
# advisory tool, not a benchmark suite.
_DEFAULT_PROMPTS: list[dict[str, str]] = [
    {"kind": "text", "prompt": "Explain what a KV cache is in two sentences."},
    {"kind": "text", "prompt": "List three prime numbers greater than 100."},
    {"kind": "text", "prompt": "Write a haiku about unified memory."},
    {
        "kind": "json",
        "prompt": (
            "Respond with ONLY a JSON object with keys "
            '"name" (string) and "age" (number) for a fictional person. '
            "No prose, no code fence."
        ),
    },
    {
        "kind": "json",
        "prompt": (
            "Return ONLY a JSON array of three city names as strings. "
            "No prose, no code fence."
        ),
    },
]

# NIAH needle prompt scaffold. Deliberately modest length — the gate is advisory
# and RAM-tier-gated; a full long-context sweep is out of MVP scope.
_NIAH_MAGIC = "7431905"
_NIAH_MIN_RAM_GB = 32.0


def _greedy_sampler():
    import mlx.core as mx

    return lambda logprobs: mx.argmax(logprobs, axis=-1)


def _eos_ids(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return {int(i) for i in ids}
    eos = getattr(tokenizer, "eos_token_id", None)
    return {int(eos)} if eos is not None else set()


def _encode_prompt(tokenizer, text: str) -> list[int]:
    """Encode a user turn, applying the chat template when one exists."""
    if getattr(tokenizer, "chat_template", None):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(text))


def _is_kv_quant_incompat(exc: BaseException) -> bool:
    """True iff ``exc`` is MLX's "head_dim not divisible by group size" error.

    ``mx.quantize`` raises a ``ValueError`` like "The last dimension of the matrix
    needs to be divisible by the quantization group size 64 ..." when the head_dim
    is incompatible. We match on that shape so the gate can degrade to an advisory
    skip for THIS failure only, never swallowing an unrelated error.
    """
    msg = str(exc).lower()
    return "group size" in msg and "divisible" in msg


def _decode_text(tokenizer, token_ids: list[int]) -> str:
    """Decode generated tokens to plain text, dropping special/control tokens.

    Greedy generation stops ON the EOS token (it is appended before the break),
    so a naive ``decode`` leaves a trailing control token like ``<|im_end|>`` in
    the string — which breaks the strict whole-output JSON check. We pass
    ``skip_special_tokens=True`` and fall back gracefully for the rare tokenizer
    that does not accept the kwarg.
    """
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _enable_hf_offline() -> None:
    """Force HuggingFace offline mode RELIABLY, whatever the import order.

    Setting the env vars alone is not enough: ``huggingface_hub.constants``
    snapshots ``HF_HUB_OFFLINE`` from the environment ONCE at import time, and
    ``is_offline_mode()`` returns that frozen module global. A programmatic
    caller that already imported ``huggingface_hub`` (while online) would have
    the snapshot pinned to ``False``, so a later env-var change is silently
    ignored and the "no network fetch" contract breaks. We therefore ALSO patch
    the already-imported constant to ``True`` — ``is_offline_mode()`` reads it at
    call time, so the load fails fast on an uncached model either way.
    """
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        # huggingface_hub not importable — the env vars are still set for any
        # process that imports it fresh later.
        pass


def _run_generation(
    model,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    kv_bits: int | None,
    kv_group_size: int,
    eos_ids: set[int],
    collect_logprobs: bool,
    stop_on_eos: bool,
):
    """Greedy (temp=0) generation, capturing tokens and optional per-step logprobs.

    Returns ``(tokens, logprobs_list, cache)``. ``logprobs_list`` holds one
    float32 log-probability vector per generated step when ``collect_logprobs``,
    else is empty. ``cache`` is the (possibly now-quantized) prompt cache — after
    a ``kv_bits`` run mlx-lm has quantized it in place, so it measures the
    candidate footprint.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    kwargs: dict = {
        "max_tokens": max_tokens,
        "prompt_cache": cache,
        "sampler": _greedy_sampler(),
        "kv_group_size": kv_group_size,
    }
    if kv_bits is not None:
        kwargs["kv_bits"] = kv_bits

    tokens: list[int] = []
    logprobs_list: list[np.ndarray] = []
    for token, logprobs in generate_step(mx.array(prompt_ids), model, **kwargs):
        tid = int(token.item()) if hasattr(token, "item") else int(token)
        tokens.append(tid)
        if collect_logprobs:
            # mlx compute dtype is often bf16; np.asarray can't read its buffer
            # (PEP-3118 itemsize mismatch), so cast to float32 in MLX first.
            logprobs_list.append(np.array(logprobs.astype(mx.float32)).reshape(-1))
        if stop_on_eos and tid in eos_ids:
            break
    return tokens, logprobs_list, cache


def _kv_cache_bytes(cache) -> int:
    """Sum the FULL allocated bytes of every KV array in a prompt cache.

    Consistent across cache types: a plain ``KVCache`` exposes ``keys`` /
    ``values`` as single arrays; a ``QuantizedKVCache`` exposes them as
    ``(packed, scales, biases)`` tuples. We sum ``.nbytes`` (shape+dtype
    metadata — no lazy-eval spike) over ALL of them WITHOUT trimming to the used
    offset.

    Deliberately NOT ``vllm_mlx.memory_cache.estimate_kv_cache_memory``: that
    helper trims a ``KVCache`` to its used ``offset`` (via ``.state``) but counts
    a ``QuantizedKVCache``'s full padded buffer, which would skew a differential
    bf16-vs-quantized comparison. Both caches here reach the SAME offset and the
    SAME step-padded buffer, so a full-buffer byte sum is the apples-to-apples
    measurement.
    """
    total = 0
    for layer in cache:
        if layer is None:
            continue
        for attr in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if attr is None:
                continue
            arrays = attr if isinstance(attr, (list, tuple)) else (attr,)
            for a in arrays:
                if a is not None and hasattr(a, "nbytes"):
                    total += int(a.nbytes)
    return total


def _baseline_kv_dtype(cache) -> str:
    """Return the element dtype of a NON-quantized KV cache, e.g. ``"bfloat16"``.

    The baseline cache's dtype follows the loaded model/runtime — it is NOT
    guaranteed to be bf16 (many mlx checkpoints are fp16). Reading the actual
    ``keys`` array dtype keeps the report honest instead of hardcoding a label.
    Returns ``"unknown"`` if it can't be read.
    """
    for layer in cache:
        keys = getattr(layer, "keys", None)
        # A plain KVCache exposes ``keys`` as a single array (not a tuple, which
        # only a QuantizedKVCache uses).
        if keys is not None and not isinstance(keys, (list, tuple)):
            dtype = getattr(keys, "dtype", None)
            if dtype is not None:
                return str(dtype).replace("mlx.core.", "")
    return "unknown"


def _measure_kv_bytes(
    model,
    prompt_ids: list[int],
    *,
    mem_tokens: int,
    kv_bits: int | None,
    kv_group_size: int,
) -> tuple[int, str]:
    """Measure KV-cache footprint (bytes) and element dtype after a fixed run.

    Fixed length (no EOS stop) so baseline and candidate reach the same offset
    and the byte comparison is apples-to-apples (see :func:`_kv_cache_bytes`).
    The returned dtype string is meaningful for the (unquantized) baseline; for a
    quantized run it reflects the packed representation and the caller ignores it.
    """
    _, _, cache = _run_generation(
        model,
        prompt_ids,
        max_tokens=mem_tokens,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        eos_ids=set(),
        collect_logprobs=False,
        stop_on_eos=False,
    )
    return _kv_cache_bytes(cache), _baseline_kv_dtype(cache)


def _combine_agreement(results: list[AgreementResult]) -> AgreementResult:
    total = sum(r.total for r in results)
    matched = sum(r.matched for r in results)
    rate = (matched / total) if total > 0 else 1.0
    return AgreementResult(
        total=total, matched=matched, rate=rate, first_divergence_index=None
    )


def _combine_logits(results: list[LogitDivergence]) -> LogitDivergence:
    steps = sum(r.compared_steps for r in results)
    if steps == 0:
        return LogitDivergence(
            compared_steps=0, mean_kl=0.0, max_kl=0.0, top1_agreement_rate=1.0
        )
    mean_kl = sum(r.mean_kl * r.compared_steps for r in results) / steps
    top1 = sum(r.top1_agreement_rate * r.compared_steps for r in results) / steps
    max_kl = max(r.max_kl for r in results)
    return LogitDivergence(
        compared_steps=steps,
        mean_kl=mean_kl,
        max_kl=max_kl,
        top1_agreement_rate=top1,
    )


def _maybe_run_niah(
    model,
    tokenizer,
    chip,
    *,
    enabled: bool,
    max_tokens: int,
    kv_bits: int,
    kv_group_size: int,
    eos_ids: set[int],
    total_ram_gb: float | None,
) -> dict:
    """Optional needle-in-a-haystack retrieval metric, RAM/compute-tier-gated.

    Gated on the chip tier (#5): only runs on an M3-or-newer chip with enough
    RAM. A low-RAM M1/M2 skips it cleanly (the long-ish context is the expensive
    part). This is the concrete #5 -> #4 wiring; the metric itself is kept modest
    (single moderate-context needle) per the advisory MVP scope.
    """
    if not enabled:
        return {"status": "skipped", "reason": "not requested (pass --niah)"}
    if not chip.is_m3_or_newer:
        return {
            "status": "skipped",
            "reason": f"chip tier below M3 (gen={chip.generation}) — NIAH gated off",
        }
    # Fail-closed on RAM: run the expensive long-context pass ONLY when RAM was
    # successfully detected AND meets the minimum. An unknown (``None``) capacity
    # must SKIP — never assume enough headroom on a host we couldn't measure.
    if total_ram_gb is None:
        return {
            "status": "skipped",
            "reason": "RAM capacity undetected — NIAH gated off (fail-closed)",
        }
    if total_ram_gb < _NIAH_MIN_RAM_GB:
        return {
            "status": "skipped",
            "reason": f"RAM {total_ram_gb:.0f}GB < {_NIAH_MIN_RAM_GB:.0f}GB gate",
        }

    filler = "The garden was quiet in the early morning light. " * 60
    needle = f"The secret access code is {_NIAH_MAGIC}. "
    question = (
        "\n\nBased on the text above, what is the secret access code? "
        "Answer with only the number."
    )
    context = filler[: len(filler) // 2] + needle + filler[len(filler) // 2 :]
    prompt_ids = _encode_prompt(tokenizer, context + question)

    base_tokens, _, _ = _run_generation(
        model,
        prompt_ids,
        max_tokens=max_tokens,
        kv_bits=None,
        kv_group_size=kv_group_size,
        eos_ids=eos_ids,
        collect_logprobs=False,
        stop_on_eos=True,
    )
    cand_tokens, _, _ = _run_generation(
        model,
        prompt_ids,
        max_tokens=max_tokens,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        eos_ids=eos_ids,
        collect_logprobs=False,
        stop_on_eos=True,
    )
    base_found = _NIAH_MAGIC in tokenizer.decode(base_tokens)
    cand_found = _NIAH_MAGIC in tokenizer.decode(cand_tokens)
    if not base_found:
        return {
            "status": "skipped",
            "reason": "baseline itself missed the needle — not attributable",
            "baseline_found": False,
            "candidate_found": cand_found,
        }
    return {
        "status": "pass" if cand_found else "fail",
        "reason": "candidate retained needle"
        if cand_found
        else "candidate lost needle",
        "baseline_found": True,
        "candidate_found": cand_found,
    }


def run_gate(
    *,
    model_arg: str,
    candidate_dtypes: list[str],
    prompts: list[dict[str, str]],
    max_tokens: int,
    mem_tokens: int,
    kv_group_size: int,
    advisory: bool,
    run_niah: bool,
    json_out: Path | None,
    offline: bool = False,
) -> int:
    """Load the model, run every candidate dtype, print + optionally dump reports.

    Returns a process exit code. Advisory runs always return 0; enforced runs
    return 1 if any candidate's overall verdict is FAIL.

    When ``offline`` is set, HuggingFace offline mode is forced before the load
    (env vars AND the already-imported ``constants.HF_HUB_OFFLINE``, see
    :func:`_enable_hf_offline`) so an uncached model fails fast instead of
    triggering a network fetch — reliably even if the hub was imported earlier.

    Raises:
        ValueError: If ``max_tokens`` / ``mem_tokens`` / ``kv_group_size`` are not
            strictly positive, or if ``prompts`` / ``candidate_dtypes`` is empty.
            A non-positive token budget yields empty generations that would score
            a vacuous ``1.0`` agreement, and an empty candidate list would let an
            enforced gate "pass" having measured nothing — both must never slip
            through. Validated up front, before any model load.
    """
    if max_tokens <= 0 or mem_tokens <= 0 or kv_group_size <= 0:
        raise ValueError(
            "max_tokens, mem_tokens, and group_size must be strictly positive "
            f"(got max_tokens={max_tokens}, mem_tokens={mem_tokens}, "
            f"group_size={kv_group_size})"
        )
    if not prompts:
        raise ValueError("prompts must be a non-empty list")
    if not candidate_dtypes:
        # An empty candidate list would run the loop zero times and let an
        # --enforce invocation return success having measured NOTHING.
        raise ValueError("candidate_dtypes must be a non-empty list")

    if offline:
        # Forbid any network fetch — the load then fails fast if the model isn't
        # fully cached (mlx_lm.load -> snapshot_download raises OfflineModeIsEnabled).
        _enable_hf_offline()

    from mlx_lm import load

    from vllm_mlx.model_aliases import resolve_model

    hf_path = resolve_model(model_arg)
    print(f"[kv-quant-gate] loading {model_arg} -> {hf_path} ...", file=sys.stderr)
    model, tokenizer = load(hf_path)
    eos_ids = _eos_ids(tokenizer)

    # KV-quant group-size compatibility (issue #1294). ``mx.quantize`` requires
    # BOTH the key and value head dims to be exact multiples of the group size; a
    # model whose head_dim isn't divisible by the requested size (e.g. Phi-3's
    # head_dim=96 vs the default 64) would otherwise abort the whole gate with a
    # raw MLX ValueError mid-generation. Resolve a compatible size up front with
    # the SAME probe + policy the live serve path uses (``probe_kv_head_dims`` +
    # ``resolve_kv_quantization``), so the gate measures the exact config serving
    # would run — including MLA models whose K/V head dims differ. Baseline runs
    # are unquantized, so this only affects the candidate generations.
    k_head_dim, v_head_dim = probe_kv_head_dims(model)
    resolved, live_disabled = resolve_kv_quantization(
        k_head_dim, v_head_dim, kv_group_size
    )
    if live_disabled:
        # Serving would keep this model's KV cache bf16 — either its K/V head dims
        # divide no supported group size (e.g. head_dim=80) or they couldn't be
        # probed. Either way there is no live KV-quant configuration to measure,
        # so skip rather than test a config serving never runs (or crash on an
        # incompatible mx.quantize). Mirrors resolve_kv_quantization exactly.
        if k_head_dim is not None and v_head_dim is not None:
            reason = (
                f"KV head dims (k={k_head_dim}, v={v_head_dim}) divide no "
                "supported group size (128/64/32)"
            )
        else:
            reason = "KV head dims could not be probed"
        print(
            f"[kv-quant-gate] {reason}; serving keeps this model's KV cache bf16 "
            "— nothing to measure.",
            file=sys.stderr,
        )
        if json_out is not None:
            json_out.write_text(json.dumps([], indent=2))
            print(f"[kv-quant-gate] wrote JSON report -> {json_out}", file=sys.stderr)
        # Advisory: nothing to measure, exit 0. Enforced: a gate that measured
        # NOTHING must not read as success (mirrors the empty-candidate guard).
        return 0 if advisory else 1
    if resolved != kv_group_size:
        print(
            f"[kv-quant-gate] adjusted --group-size {kv_group_size} -> {resolved} "
            f"to satisfy this model's KV head dims (k={k_head_dim}, v={v_head_dim}; "
            "mx.quantize constraint).",
            file=sys.stderr,
        )
        kv_group_size = resolved

    try:
        chip = detect_chip_tier()
    except Exception:
        chip = classify_chip_tier(None)
    chip_dict = asdict(chip)

    total_ram_gb: float | None
    try:
        from vllm_mlx.optimizations import get_system_memory_gb

        total_ram_gb = get_system_memory_gb()
    except Exception:
        total_ram_gb = None

    json_prompts = [p for p in prompts if p.get("kind") == "json"]
    mem_prompt_ids = _encode_prompt(tokenizer, prompts[0]["prompt"])

    # The baseline is candidate-INDEPENDENT — the bf16 generation, its logprobs,
    # its JSON text, and its KV footprint do not change with the candidate dtype.
    # Run it ONCE here (the baseline generation is the dominant inference cost;
    # recomputing it inside the candidate loop would double the work of the
    # default two-dtype ``int8 int4`` run) and reuse it for every candidate.
    baseline_runs: list[dict] = []
    for spec in prompts:
        prompt_ids = _encode_prompt(tokenizer, spec["prompt"])
        base_tokens, base_lp, _ = _run_generation(
            model,
            prompt_ids,
            max_tokens=max_tokens,
            kv_bits=None,
            kv_group_size=kv_group_size,
            eos_ids=eos_ids,
            collect_logprobs=True,
            stop_on_eos=True,
        )
        baseline_runs.append(
            {
                "prompt_ids": prompt_ids,
                "kind": spec.get("kind"),
                "tokens": base_tokens,
                "logprobs": base_lp,
                # Baseline JSON text is candidate-independent too — decode once.
                "text": (
                    _decode_text(tokenizer, base_tokens)
                    if spec.get("kind") == "json"
                    else None
                ),
            }
        )

    base_bytes, baseline_kv_dtype = _measure_kv_bytes(
        model,
        mem_prompt_ids,
        mem_tokens=mem_tokens,
        kv_bits=None,
        kv_group_size=kv_group_size,
    )

    any_fail = False
    reports: list[dict] = []
    for cdtype in candidate_dtypes:
        _, cand_bits = dtype_to_quantization_bits(cdtype)
        thresholds = default_thresholds(cdtype)

        agreements: list[AgreementResult] = []
        divergences: list[LogitDivergence] = []
        retention_pairs: list[tuple[str, str]] = []

        # Safety net for the residual case the up-front resolve can't cover — a
        # model whose head_dim we couldn't infer, or one with mixed per-layer head
        # dims (e.g. a k_pe rope split). If ``mx.quantize`` still rejects the
        # head_dim, skip THIS candidate with an advisory rather than crash the
        # whole gate (issue #1294). Only the divisibility error is caught.
        try:
            for base in baseline_runs:
                cand_tokens, cand_lp, _ = _run_generation(
                    model,
                    base["prompt_ids"],
                    max_tokens=max_tokens,
                    kv_bits=cand_bits,
                    kv_group_size=kv_group_size,
                    eos_ids=eos_ids,
                    collect_logprobs=True,
                    stop_on_eos=True,
                )
                ag = greedy_agreement_rate(base["tokens"], cand_tokens)
                agreements.append(ag)
                # Same-context prefix: up to AND INCLUDING the first divergence
                # step (its distributions still share a context), else the whole
                # overlap.
                compare_len = (
                    ag.first_divergence_index + 1
                    if ag.first_divergence_index is not None
                    else ag.total
                )
                divergences.append(
                    logit_divergence(base["logprobs"], cand_lp, compare_len=compare_len)
                )
                if base["kind"] == "json":
                    # Candidate decoded with skip_special_tokens so a trailing EOS
                    # control token can't break the strict whole-output JSON parse.
                    retention_pairs.append(
                        (base["text"], _decode_text(tokenizer, cand_tokens))
                    )

            cand_bytes, _ = _measure_kv_bytes(
                model,
                mem_prompt_ids,
                mem_tokens=mem_tokens,
                kv_bits=cand_bits,
                kv_group_size=kv_group_size,
            )
        except ValueError as exc:
            if not _is_kv_quant_incompat(exc):
                raise
            print(
                f"[kv-quant-gate] skipping {cdtype} candidate — this model's "
                f"head_dim is incompatible with KV-quant group size "
                f"{kv_group_size} ({exc}).",
                file=sys.stderr,
            )
            continue
        from vllm_mlx.kv_quant_gate import memory_delta

        niah = _maybe_run_niah(
            model,
            tokenizer,
            chip,
            enabled=run_niah,
            max_tokens=max_tokens,
            kv_bits=cand_bits,
            kv_group_size=kv_group_size,
            eos_ids=eos_ids,
            total_ram_gb=total_ram_gb,
        )

        report = build_report(
            model=model_arg,
            hf_path=hf_path,
            baseline_dtype=baseline_kv_dtype,
            candidate_dtype=cdtype,
            num_prompts=len(prompts),
            advisory=advisory,
            chip=chip_dict,
            thresholds=thresholds,
            agreement=_combine_agreement(agreements),
            logits=_combine_logits(divergences),
            retention=structured_output_retention(retention_pairs)
            if json_prompts
            else structured_output_retention([]),
            memory=memory_delta(base_bytes, cand_bytes),
            niah=niah,
        )
        print(report.human_summary())
        print()
        reports.append(report.to_dict())
        if report.overall == "FAIL":
            any_fail = True

    if json_out is not None:
        json_out.write_text(json.dumps(reports, indent=2))
        print(f"[kv-quant-gate] wrote JSON report -> {json_out}", file=sys.stderr)

    if advisory:
        return 0
    # Enforced: a run where every candidate was skipped as incompatible measured
    # NOTHING and must not report success (mirrors the empty-candidate guard).
    if not reports:
        print(
            "[kv-quant-gate] --enforce: no candidate could be measured "
            "(all incompatible with this model's head_dim).",
            file=sys.stderr,
        )
        return 1
    return 1 if any_fail else 0


def _load_prompts(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return list(_DEFAULT_PROMPTS)
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("prompts file must be a non-empty JSON array")
    prompts: list[dict[str, str]] = []
    for entry in data:
        if isinstance(entry, str):
            prompts.append({"kind": "text", "prompt": entry})
        elif isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
            prompts.append(
                {"kind": str(entry.get("kind", "text")), "prompt": entry["prompt"]}
            )
        else:
            raise ValueError(f"bad prompt entry: {entry!r}")
    return prompts


def _positive_int(value: str) -> int:
    """argparse type: accept only strictly-positive integers."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("model", help="alias or HF path (e.g. qwen3.5-4b-4bit)")
    p.add_argument(
        "--candidate",
        nargs="+",
        default=["int8", "int4"],
        choices=["int8", "int4"],
        help="candidate KV dtype(s) to test against the bf16 baseline",
    )
    p.add_argument("--max-tokens", type=_positive_int, default=48)
    p.add_argument(
        "--mem-tokens",
        type=_positive_int,
        default=128,
        help="fixed generation length for the memory-delta measurement",
    )
    p.add_argument("--group-size", type=_positive_int, default=64)
    p.add_argument("--prompts-file", type=Path, default=None)
    p.add_argument(
        "--niah",
        action="store_true",
        help="run the optional NIAH retrieval metric (RAM/chip-tier-gated)",
    )
    p.add_argument(
        "--enforce",
        action="store_true",
        help="exit non-zero on FAIL (default: advisory, always exit 0)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE before load (cached models only)",
    )
    p.add_argument("--json-out", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    prompts = _load_prompts(args.prompts_file)
    return run_gate(
        model_arg=args.model,
        candidate_dtypes=list(dict.fromkeys(args.candidate)),
        prompts=prompts,
        max_tokens=args.max_tokens,
        mem_tokens=args.mem_tokens,
        kv_group_size=args.group_size,
        advisory=not args.enforce,
        run_niah=args.niah,
        json_out=args.json_out,
        offline=args.offline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
