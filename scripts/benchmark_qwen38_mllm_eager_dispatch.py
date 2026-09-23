#!/usr/bin/env python3
"""Offline A/B qualification for Qwen3.8 MLLM eager layer submission.

This is a benchmark-only experiment.  It loads one already verified local
snapshot through the serialized MLLM engine, then reversibly wraps the exact
mlx-vlm Qwen3.5 dense decoder-layer class.  Production defaults are untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import inspect
import json
import mimetypes
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
MIB = 1024 * 1024
EXPECTED_PAIRS = 6
EXPECTED_TOKENS = 256
EXPECTED_DECODER_MODULE = "mlx_vlm.models.qwen3_5.language"
EXPECTED_DECODER_CLASS = "Qwen3_5DecoderLayer"
EXPECTED_MLX_VLM_VERSION = "0.7.1"


@dataclass(frozen=True)
class ExactArtifact:
    repo_id: str
    revision: str
    verification_id: str
    config_sha256: str
    index_sha256: str
    shard_blobs: tuple[tuple[str, str], ...]


QWEN38_ARTIFACT = ExactArtifact(
    repo_id="rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
    revision="aa985c29ff5b334cbfdcbbc787d47e66e9d9e456",
    verification_id=(
        "hf-snapshot-sha256:"
        "360a8c76fc60c254c595442d16157eb76868033d328d5df9f41344c83bf77a66"
    ),
    config_sha256="48e2fc64017e1d7472373ce012ba1db172319e5fc1e4f81ccc97fea523b00d1b",
    index_sha256="47d389bc0826b9f8d0ad495d0c4bf89d7e9398a50b77ae35771dde96e9db9229",
    shard_blobs=(
        (
            "model-00001-of-00003.safetensors",
            "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
        ),
        (
            "model-00002-of-00003.safetensors",
            "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
        ),
        (
            "model-00003-of-00003.safetensors",
            "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
        ),
    ),
)


PROMPTS = (
    (
        "coding",
        "Write a typed Python merge_intervals function and state its complexity.",
    ),
    ("json", 'Return JSON only: {"risk": string, "mitigations": [three strings]}.'),
    ("tool", 'Return only tool arguments for weather(city="Tokyo", unit="celsius").'),
    ("reasoning", "An item is discounted 20%, then taxed 8%, ending at $86.40. Solve."),
    ("creative", "Write a vivid scene about a lighthouse whose fog carries memories."),
    ("coding", "Implement an iterative binary search in Python with type hints."),
)


def _canonical_json_sha(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _inspect_exact_artifact(
    snapshot: Path, expected: ExactArtifact = QWEN38_ARTIFACT
) -> dict[str, Any]:
    """Recompute the B0 identity from an already cached canonical HF snapshot."""
    snapshot = snapshot.expanduser().resolve()
    expected_repo_dir = "models--" + expected.repo_id.replace("/", "--")
    if snapshot.name != expected.revision or snapshot.parent.name != "snapshots":
        raise ValueError("snapshot is not the expected canonical HF revision path")
    if snapshot.parent.parent.name != expected_repo_dir:
        raise ValueError("snapshot repo identity does not match the pinned B0 artifact")

    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config = _read_json(config_path)
    index = _read_json(index_path)
    config_sha = _canonical_json_sha(config)
    index_sha = _canonical_json_sha(index)
    if config_sha != expected.config_sha256 or index_sha != expected.index_sha256:
        raise ValueError("snapshot config/index digest does not match pinned B0 truth")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("weight index has no weight_map")
    indexed_shards = sorted(set(weight_map.values()))
    expected_shards = sorted(name for name, _ in expected.shard_blobs)
    if indexed_shards != expected_shards:
        raise ValueError("weight index shard set does not match pinned B0 truth")

    file_identities: list[tuple[str, str]] = []
    blob_root = snapshot.parent.parent / "blobs"
    for name, blob in sorted(expected.shard_blobs):
        shard = snapshot / name
        if not shard.is_symlink():
            raise ValueError(
                f"weight shard must be a canonical HF blob symlink: {name}"
            )
        resolved = shard.resolve(strict=True)
        if resolved.parent != blob_root.resolve() or resolved.name != blob:
            raise ValueError(f"weight shard blob identity mismatch: {name}")
        file_identities.append((name, f"hf_blob:{blob}"))

    identity = {
        "repo_id": expected.repo_id,
        "revision": expected.revision,
        "subfolder": None,
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "shards": expected_shards,
        "file_identities": file_identities,
    }
    verification_id = "hf-snapshot-sha256:" + _canonical_json_sha(identity)
    if verification_id != expected.verification_id:
        raise ValueError("recomputed B0 verification id mismatch")

    text = config.get("text_config", config)
    layer_types = list(text.get("layer_types", []))
    required = {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "full_attention_interval": 4,
    }
    if any(int(text.get(key, -1)) != value for key, value in required.items()):
        raise ValueError("artifact geometry is not exact Qwen3.8-27B truth")
    if layer_types and (
        layer_types.count("linear_attention") != 48
        or layer_types.count("full_attention") != 16
    ):
        raise ValueError("artifact layer-type geometry is not 48 linear / 16 full")
    quant = config.get("quantization", config.get("quantization_config", {}))
    if int(quant.get("bits", -1)) != 4 or int(quant.get("group_size", -1)) != 64:
        raise ValueError("artifact is not the exact 4-bit/group-64 quantization")
    return {
        "verified": True,
        "repo_id": expected.repo_id,
        "revision": expected.revision,
        "verification_id": verification_id,
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "shard_identities": dict(file_identities),
    }


def _language_layers(language_model: Any) -> list[Any]:
    layers = getattr(language_model, "layers", None)
    if layers is None:
        layers = getattr(getattr(language_model, "model", None), "layers", None)
    return list(layers or [])


def _qualify_loaded_model(
    language_model: Any,
    decoder_class: type,
    *,
    enforce_origin: bool = True,
) -> dict[str, Any]:
    if enforce_origin and (
        decoder_class.__module__ != EXPECTED_DECODER_MODULE
        or decoder_class.__name__ != EXPECTED_DECODER_CLASS
    ):
        raise RuntimeError("loaded decoder class is not exact mlx-vlm Qwen3.5 dense")
    if enforce_origin:
        parameters = tuple(inspect.signature(decoder_class.__call__).parameters)
        expected_parameters = (
            "self",
            "x",
            "mask",
            "cache",
            "position_ids",
            "position_embeddings",
        )
        if parameters != expected_parameters:
            raise RuntimeError("loaded decoder call ABI does not match pinned mlx-vlm")
    layers = _language_layers(language_model)
    if len(layers) != 64 or any(type(layer) is not decoder_class for layer in layers):
        raise RuntimeError("loaded model is not exactly 64 homogeneous decoder layers")
    linear = sum(bool(getattr(layer, "is_linear", False)) for layer in layers)
    if linear != 48:
        raise RuntimeError("loaded model is not exactly 48 linear / 16 full layers")
    args = getattr(language_model, "args", None)
    if args is None:
        args = getattr(getattr(language_model, "model", None), "args", None)
    if int(getattr(args, "hidden_size", -1)) != 5120:
        raise RuntimeError("loaded model hidden size is not 5120")
    if int(getattr(args, "num_hidden_layers", -1)) != 64:
        raise RuntimeError("loaded model layer count metadata is not 64")
    if int(getattr(args, "full_attention_interval", -1)) != 4:
        raise RuntimeError("loaded model full-attention interval is not 4")
    exact_args = {
        "model_type": "qwen3_5_text",
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
    }
    for field, expected in exact_args.items():
        if getattr(args, field, None) != expected:
            raise RuntimeError(f"loaded model {field} is not exact Qwen3.8 truth")
    return {
        "qualified": True,
        "decoder_class": f"{decoder_class.__module__}.{decoder_class.__name__}",
        "hidden_size": 5120,
        "layers": 64,
        "linear_layers": 48,
        "full_layers": 16,
        **exact_args,
        "layer_object_ids": [id(layer) for layer in layers],
    }


class EagerLayerPatch:
    """Reversible instance-allowlisted wrapper around one exact class."""

    def __init__(self, decoder_class: type, layers: list[Any], mx: Any):
        self.decoder_class = decoder_class
        self.layer_indexes = {id(layer): index for index, layer in enumerate(layers)}
        self.mx = mx
        self.original = decoder_class.__call__
        self.hits = 0
        self.layer_hits = [0] * len(layers)
        self.installed = False
        self.candidate_enabled = False
        self.wrapped: Any | None = None

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("eager patch already installed")
        owner = self
        original = self.original

        def wrapped(layer: Any, *args: Any, **kwargs: Any) -> Any:
            output = original(layer, *args, **kwargs)
            shape = tuple(getattr(output, "shape", ()))
            layer_index = owner.layer_indexes.get(id(layer))
            if layer_index is not None and shape == (1, 1, 5120):
                owner.mx.async_eval(output)
                owner.hits += 1
                owner.layer_hits[layer_index] += 1
            return output

        self.wrapped = wrapped
        # Installation records the wrapper but leaves baseline byte-for-byte on
        # the original class method.  The request boundary performs the toggle.
        self.decoder_class.__call__ = self.original
        self.installed = True

    def set_candidate(self, enabled: bool) -> None:
        if not self.installed or self.wrapped is None:
            raise RuntimeError("eager patch is not installed")
        self.decoder_class.__call__ = self.wrapped if enabled else self.original
        self.candidate_enabled = enabled

    def close(self) -> None:
        if self.installed:
            self.decoder_class.__call__ = self.original
            self.installed = False
        self.candidate_enabled = False

    def __enter__(self) -> EagerLayerPatch:
        self.install()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _delta_token_ids(output: Any, seen: int) -> list[int]:
    raw = getattr(output, "tokens", None)
    if raw is None:
        raw = getattr(output, "token_ids", None)
    ids = [int(value) for value in (raw or [])]
    completed = int(getattr(output, "completion_tokens", seen + len(ids)))
    newly_completed = max(0, completed - seen)
    if newly_completed == 0:
        return []
    if len(ids) == newly_completed:
        return ids
    if len(ids) == completed:
        return ids[-newly_completed:]
    raise RuntimeError("stream token payload is neither delta nor cumulative")


def _cv(values: list[float]) -> float:
    if not values or statistics.mean(values) == 0:
        return float("inf")
    return (
        statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0
    )


def _media_sequence_pass(
    before: dict[str, Any],
    middle: dict[str, Any],
    after: dict[str, Any],
    expected: str,
) -> bool:
    expected = expected.casefold()

    def eager_engaged(sample: dict[str, Any]) -> bool:
        layer_hits = sample.get("eager_layer_hits", [])
        return bool(
            int(sample.get("eager_hits", 0)) > 0
            and len(layer_hits) == 64
            and min(layer_hits) > 0
            and len(set(layer_hits)) == 1
            and int(sample["eager_hits"]) == sum(int(hits) for hits in layer_hits)
        )

    return bool(
        expected
        and expected in before.get("text", "").casefold()
        and expected in after.get("text", "").casefold()
        and before.get("token_sha256") == after.get("token_sha256")
        and before.get("text_sha256") == after.get("text_sha256")
        and middle.get("text")
        and int(before.get("singleton_batch_delta", 0)) > 0
        and int(middle.get("singleton_batch_delta", 0)) > 0
        and int(after.get("singleton_batch_delta", 0)) > 0
        and eager_engaged(before)
        and eager_engaged(middle)
        and eager_engaged(after)
    )


def evaluate_gates(
    receipt: dict[str, Any], memory_limit: int = 64 * MIB
) -> dict[str, Any]:
    strata = receipt.get("pairs", [])
    complete = len(strata) == EXPECTED_PAIRS and all(
        len(stratum.get("baseline", [])) == 2 and len(stratum.get("candidate", [])) == 2
        for stratum in strata
    )
    samples = [
        sample
        for stratum in strata
        for arm in ("baseline", "candidate")
        for sample in stratum.get(arm, [])
    ]
    stratum_ratios: list[float] = []
    paired_ratios: list[float] = []
    ttft_ratios: list[float] = []
    active_deltas: list[int] = []
    peak_deltas: list[int] = []
    for stratum in strata:
        baseline = stratum.get("baseline", [])
        candidate = stratum.get("candidate", [])
        if len(baseline) != 2 or len(candidate) != 2:
            continue
        baseline_tps = [float(sample.get("decode_tps", 0)) for sample in baseline]
        candidate_tps = [float(sample.get("decode_tps", 0)) for sample in candidate]
        if min(baseline_tps) > 0:
            stratum_ratios.append(
                statistics.median(candidate_tps) / statistics.median(baseline_tps)
            )
            paired_ratios.extend(
                candidate_tps[index] / baseline_tps[index] for index in range(2)
            )
        baseline_ttft = [float(sample.get("ttft_s", 0)) for sample in baseline]
        candidate_ttft = [float(sample.get("ttft_s", 0)) for sample in candidate]
        if min(baseline_ttft) > 0:
            ttft_ratios.append(
                statistics.median(candidate_ttft) / statistics.median(baseline_ttft)
            )
        for index in range(2):
            active_deltas.append(
                int(candidate[index]["memory"]["active_bytes"])
                - int(baseline[index]["memory"]["active_bytes"])
            )
            peak_deltas.append(
                int(candidate[index]["memory"]["peak_bytes"])
                - int(baseline[index]["memory"]["peak_bytes"])
            )
    exact = all(
        len(
            {
                sample.get("token_sha256")
                for arm in ("baseline", "candidate")
                for sample in stratum.get(arm, [])
            }
        )
        == 1
        for stratum in strata
    )
    full_length = all(
        int(sample.get("completion_tokens", -1)) == EXPECTED_TOKENS
        for sample in samples
    )
    singleton = all(
        int(sample.get("singleton_batch_delta", 0)) > 0 for sample in samples
    )
    eager_exact = all(
        (
            int(sample.get("eager_hits", -1)) == 0
            and len(sample.get("eager_layer_hits", [])) == 64
            and not any(sample["eager_layer_hits"])
        )
        if arm == "baseline"
        else (
            int(sample.get("eager_hits", -1))
            == 64 * int(sample.get("completion_tokens", -1))
            and len(sample.get("eager_layer_hits", [])) == 64
            and all(
                int(hits) == int(sample.get("completion_tokens", -1))
                for hits in sample["eager_layer_hits"]
            )
        )
        for stratum in strata
        for arm in ("baseline", "candidate")
        for sample in stratum.get(arm, [])
    )
    checks = {
        "six_complete_prompt_strata": complete,
        "all_twenty_four_samples_are_256_tokens": complete and full_length,
        "median_decode_speedup_gte_1_03": len(stratum_ratios) == EXPECTED_PAIRS
        and statistics.median(stratum_ratios) >= 1.03,
        "five_of_six_strata_positive": sum(ratio > 1.0 for ratio in stratum_ratios)
        >= 5,
        "exact_token_ids_all_runs_per_stratum": complete and exact,
        "paired_ratio_cv_lte_0_05": len(paired_ratios) == 12
        and _cv(paired_ratios) <= 0.05,
        "median_ttft_ratio_lte_1_10": len(ttft_ratios) == EXPECTED_PAIRS
        and statistics.median(ttft_ratios) <= 1.10,
        "active_delta_lte_64_mib": bool(active_deltas)
        and max(active_deltas) <= memory_limit,
        "isolated_peak_delta_lte_64_mib": bool(peak_deltas)
        and max(peak_deltas) <= memory_limit,
        "singleton_fastpath_engaged": singleton,
        "exact_candidate_hits_per_layer_and_zero_baseline": complete and eager_exact,
        "same_model_and_executor": bool(
            receipt.get("identity", {}).get("same_model")
            and receipt.get("identity", {}).get("same_executor")
        ),
        "prefix_cache_disabled": receipt.get("configuration", {}).get("prefix_cache")
        is False,
        "greedy_thinking_off": receipt.get("configuration", {}).get("temperature")
        == 0.0
        and receipt.get("configuration", {}).get("thinking") is False,
        "measured_ignore_eos_enabled": receipt.get("configuration", {}).get(
            "measured_ignore_eos"
        )
        is True,
        "artifact_exact_b0_verified": bool(receipt.get("artifact", {}).get("verified")),
        "source_clean": receipt.get("source", {}).get("dirty") is False,
        "source_tree_match": receipt.get("source", {}).get("source_tree_match") is True,
        "vlm_media_text_media_recovery": bool(
            receipt.get("media_recovery", {}).get("pass")
        ),
        "no_errors": not receipt.get("errors"),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "metrics": {
            "stratum_speedup_median": statistics.median(stratum_ratios)
            if stratum_ratios
            else None,
            "positive_strata": sum(ratio > 1.0 for ratio in stratum_ratios),
            "paired_ratio_cv": _cv(paired_ratios),
            "ttft_ratio_median": statistics.median(ttft_ratios)
            if ttft_ratios
            else None,
            "max_active_delta_bytes": max(active_deltas) if active_deltas else None,
            "max_peak_delta_bytes": max(peak_deltas) if peak_deltas else None,
        },
    }


async def _worker_memory(engine: Any, *, reset: bool = False) -> dict[str, int]:
    def capture() -> dict[str, int]:
        import mlx.core as mx

        mx.synchronize()
        if reset:
            reset_peak = getattr(mx, "reset_peak_memory", None)
            if not callable(reset_peak):
                raise RuntimeError("mlx.core.reset_peak_memory is required")
            reset_peak()
            mx.synchronize()
        return {
            "active_bytes": int(mx.get_active_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
        }

    return await engine.execute_on_model_worker(capture)


async def _run_sample(
    engine: Any,
    patch: EagerLayerPatch,
    *,
    mode: str,
    prompt: str,
    max_tokens: int,
    messages: list[dict[str, Any]] | None = None,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    patch.set_candidate(mode == "candidate")
    hits_before = patch.hits
    layer_hits_before = list(patch.layer_hits)
    await _worker_memory(engine, reset=True)
    stats_before = dict(engine.get_stats().get("batch_generator", {}))
    started = time.perf_counter()
    first_at: float | None = None
    token_ids: list[int] = []
    final = None
    async for output in engine.stream_chat(
        messages=messages or [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
        ignore_eos=ignore_eos,
    ):
        delta = _delta_token_ids(output, len(token_ids))
        if delta and first_at is None:
            first_at = time.perf_counter()
        token_ids.extend(delta)
        final = output
    ended = time.perf_counter()
    if final is None or first_at is None or not token_ids:
        raise RuntimeError("request produced no timed token output")
    memory = await _worker_memory(engine)
    stats_after = dict(engine.get_stats().get("batch_generator", {}))
    text = (
        getattr(final, "raw_text", None) or getattr(final, "text", None) or ""
    ).strip()
    decode_s = max(ended - first_at, 1e-12)
    return {
        "mode": mode,
        "completion_tokens": len(token_ids),
        "token_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "ttft_s": first_at - started,
        "elapsed_s": ended - started,
        "decode_tps": max(len(token_ids) - 1, 0) / decode_s,
        "singleton_batch_delta": int(stats_after.get("singleton_batches", 0))
        - int(stats_before.get("singleton_batches", 0)),
        "eager_hits": patch.hits - hits_before,
        "eager_layer_hits": [
            after - before for after, before in zip(patch.layer_hits, layer_hits_before)
        ],
        "memory": memory,
    }


def _data_url(path: Path) -> tuple[str, dict[str, Any]]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return (
        f"data:{mime};base64,{base64.b64encode(raw).decode()}",
        {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mime_type": mime,
            "bytes": len(raw),
        },
    )


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    artifact = _inspect_exact_artifact(args.model)
    mlx_vlm_version = importlib.metadata.version("mlx-vlm")
    if mlx_vlm_version != EXPECTED_MLX_VLM_VERSION:
        raise RuntimeError(
            f"benchmark requires mlx-vlm=={EXPECTED_MLX_VLM_VERSION}, "
            f"found {mlx_vlm_version}"
        )
    from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer

    import rapid_mlx
    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.scheduler import SchedulerConfig

    source_root = SCRIPT.parent.parent.resolve()
    try:
        Path(rapid_mlx.__file__).resolve().relative_to(source_root)
        source_tree_match = True
    except ValueError:
        source_tree_match = False
    if not source_tree_match:
        raise RuntimeError("imported rapid_mlx is not from this benchmark source tree")

    engine = BatchedEngine(
        str(args.model.expanduser().resolve()),
        force_mllm=True,
        no_hybrid=True,
        no_spec_decode=True,
        stream_interval=1,
        scheduler_config=SchedulerConfig(
            enable_prefix_cache=False,
            mllm_singleton_fastpath="auto",
        ),
    )
    errors: list[str] = []
    patch: EagerLayerPatch | None = None
    try:
        await engine.start()
        language_model = getattr(engine._model, "language_model", engine._model)
        loaded = _qualify_loaded_model(language_model, Qwen3_5DecoderLayer)
        import mlx.core as mx

        patch = EagerLayerPatch(
            Qwen3_5DecoderLayer, _language_layers(language_model), mx
        )
        patch.install()
        identity_before = {
            "model": id(language_model),
            "executor": id(engine._model_load_executor),
        }

        for _ in range(2):
            for mode in ("baseline", "candidate", "candidate", "baseline"):
                await _run_sample(
                    engine, patch, mode=mode, prompt=PROMPTS[0][1], max_tokens=32
                )

        pairs = []
        for index, (category, prompt) in enumerate(PROMPTS):
            order = (
                ("baseline", "candidate", "candidate", "baseline")
                if index % 2 == 0
                else ("candidate", "baseline", "baseline", "candidate")
            )
            samples = {"baseline": [], "candidate": []}
            for mode in order:
                samples[mode].append(
                    await _run_sample(
                        engine,
                        patch,
                        mode=mode,
                        prompt=prompt,
                        max_tokens=EXPECTED_TOKENS,
                        ignore_eos=True,
                    )
                )
            pairs.append(
                {
                    "index": index + 1,
                    "category": category,
                    "order": list(order),
                    **samples,
                }
            )

        image_url, image_identity = _data_url(args.image_path)
        media_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.image_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        media_first = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt="",
            max_tokens=64,
            messages=media_messages,
        )
        middle = await _run_sample(
            engine, patch, mode="candidate", prompt=PROMPTS[3][1], max_tokens=64
        )
        media_after = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt="",
            max_tokens=64,
            messages=media_messages,
        )
        media_recovery = {
            "required": True,
            "image": image_identity,
            "expected": args.image_expect,
            "before": {
                key: value for key, value in media_first.items() if key != "text"
            },
            "middle_text": {
                key: value for key, value in middle.items() if key != "text"
            },
            "after": {
                key: value for key, value in media_after.items() if key != "text"
            },
            "pass": _media_sequence_pass(
                media_first, middle, media_after, args.image_expect
            ),
        }
        identity_after = {
            "model": id(getattr(engine._model, "language_model", engine._model)),
            "executor": id(engine._model_load_executor),
        }
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT.parent.parent,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        source_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=SCRIPT.parent.parent,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        )
        package_versions = {}
        for package in ("mlx", "mlx-lm", "mlx-vlm", "rapid-mlx"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        physical_memory = None
        try:
            physical_memory = int(os.sysconf("SC_PHYS_PAGES")) * int(
                os.sysconf("SC_PAGE_SIZE")
            )
        except (OSError, ValueError):
            pass
        receipt = {
            "schema_version": 1,
            "methodology_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "source": {
                "git_commit": source_commit,
                "dirty": source_dirty,
                "source_tree_match": source_tree_match,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "physical_memory_bytes": physical_memory,
                "packages": package_versions,
            },
            "artifact": artifact,
            "loaded_model": {
                key: value for key, value in loaded.items() if key != "layer_object_ids"
            },
            "configuration": {
                "pairs": EXPECTED_PAIRS,
                "tokens": EXPECTED_TOKENS,
                "warmup_cycles": 2,
                "warmup_order": ["baseline", "candidate", "candidate", "baseline"],
                "measured_orders": ["ABBA", "BAAB"],
                "runs_per_arm_per_stratum": 2,
                "prefix_cache": False,
                "singleton_fastpath": "auto",
                "temperature": 0.0,
                "top_p": 1.0,
                "thinking": False,
                "measured_ignore_eos": True,
                "stream_interval": 1,
            },
            "identity": {
                "before": identity_before,
                "after": identity_after,
                "same_model": identity_before["model"] == identity_after["model"],
                "same_executor": identity_before["executor"]
                == identity_after["executor"],
            },
            "pairs": pairs,
            "media_recovery": media_recovery,
            "errors": errors,
        }
        receipt["gates"] = evaluate_gates(receipt, args.max_memory_delta_mib * MIB)
        return receipt
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if patch is not None:
            patch.close()
        await engine.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="exact local canonical HF snapshot; never downloaded",
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--image-expect", required=True)
    parser.add_argument(
        "--image-prompt", default="Describe the main visible subject in one sentence."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-memory-delta-mib", type=int, default=64)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.image_expect.strip():
        parser.error("--image-expect must be non-blank")
    if not args.model.is_dir():
        parser.error("--model must be an existing local snapshot directory")
    if not args.image_path.is_file():
        parser.error("--image-path must be an existing local file")
    if args.max_memory_delta_mib < 0:
        parser.error("--max-memory-delta-mib must be non-negative")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        receipt = asyncio.run(run_benchmark(args))
        status = 0 if receipt["gates"]["pass"] else 1
    except Exception as exc:
        receipt = {
            "schema_version": 1,
            "methodology_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "artifact": {"verified": False},
            "errors": [f"{type(exc).__name__}: {exc}"],
            "gates": {"pass": False, "checks": {"setup": False}},
        }
        status = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "pass": receipt["gates"]["pass"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
