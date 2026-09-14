# SPDX-License-Identifier: Apache-2.0
"""Explicit serial server for the qualified native Qwen3.6 MTP runtime."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import HTTPException

from .eligibility import NativeMTPPair
from .runtime import QUALIFIED_MLX_VLM_VERSION, load_runtime

logger = logging.getLogger(__name__)


def _validate_greedy_request(request: Any) -> None:
    temperature = getattr(request, "temperature", None)
    if temperature not in (None, 0, 0.0):
        raise HTTPException(
            status_code=400,
            detail="Native MTP currently supports greedy decoding only; set temperature=0.",
        )
    unsupported = {}
    repetition_penalty = getattr(request, "repetition_penalty", None)
    if repetition_penalty not in (None, 1, 1.0):
        unsupported["repetition_penalty"] = repetition_penalty
    for name in ("presence_penalty", "frequency_penalty"):
        value = getattr(request, name, None)
        if value not in (None, 0, 0.0):
            unsupported[name] = value
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise HTTPException(
            status_code=400,
            detail=f"Native MTP does not support sampling penalties: {names}.",
        )
    if getattr(request, "logprobs", None) not in (None, False) or getattr(
        request, "top_logprobs", None
    ) not in (None, 0):
        raise HTTPException(
            status_code=400,
            detail="Native MTP does not support logprobs.",
        )


def run_native_mtp_server(
    *,
    pair: NativeMTPPair,
    host: str,
    port: int,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    uvicorn_log_level: str,
    no_thinking: bool = False,
    api_key: str | None = None,
    rate_limit: int = 0,
    max_request_bytes: int = 8 * 1024 * 1024,
    body_receive_timeout_seconds: float = 15.0,
    default_timeout: float = 1800.0,
    max_concurrent_requests: int = 256,
    cors_policy: Any | None = None,
    tool_call_parser: str | None = "qwen3_coder_xml",
    reasoning_parser_name: str | None = "qwen3",
) -> None:
    """Load the immutable target/drafter pair and run the serial API server."""

    try:
        import uvicorn
        from mlx_vlm import load
    except ImportError as exc:
        raise RuntimeError(
            "native MTP requires "
            f"mlx-vlm {QUALIFIED_MLX_VLM_VERSION}; install rapid-mlx[mtp]"
        ) from exc

    # Reuse the established serial speculative server's thread-affine API and
    # lifecycle boundary. It already owns auth, admission, deadlines,
    # cancellation, stream backpressure, and OpenAI response shaping.
    from ..dflash.server import _build_app, _dflash_executor

    def _load_all():
        started = time.perf_counter()
        model, processor = load(pair.target_repo, revision=pair.target_revision)
        runtime = load_runtime(
            pair.drafter_repo,
            target_revision=pair.target_revision,
            drafter_revision=pair.drafter_revision,
            block_size=pair.block_size,
            expected_model_type=pair.drafter_model_type,
        )
        # The request-time MTP reset also binds, but doing it here turns an
        # incompatible target/sidecar pair into a deterministic startup error
        # on the same thread that owns generation.
        runtime.drafter.bind(model)
        logger.info(
            "Native MTP target and sidecar loaded in %.1fs",
            time.perf_counter() - started,
        )
        return model, processor, runtime

    model, processor, runtime = _dflash_executor.submit(_load_all).result()

    def _generation_kwargs(*, max_tokens: int, temperature: float, top_p: float):
        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "draft_model": runtime.drafter,
            "draft_kind": runtime.kind,
            "draft_block_size": runtime.block_size,
        }

    app = _build_app(
        model=model,
        processor=processor,
        runtime=runtime,
        served_model_name=served_model_name,
        default_max_tokens=default_max_tokens,
        cors_origins=cors_origins,
        no_thinking=no_thinking,
        api_key=api_key,
        rate_limit=rate_limit,
        max_request_bytes=max_request_bytes,
        body_receive_timeout_seconds=body_receive_timeout_seconds,
        default_timeout=default_timeout,
        max_concurrent_requests=max_concurrent_requests,
        cors_policy=cors_policy,
        tool_call_parser=tool_call_parser,
        reasoning_parser_name=reasoning_parser_name,
        generation_kwargs_fn=_generation_kwargs,
        validate_request_fn=_validate_greedy_request,
        backend_name="Native MTP",
    )

    host_display = "localhost" if host == "0.0.0.0" else host
    print(f"  Ready: http://{host_display}:{port}/v1  (Native MTP mode)")
    print(f"  Model: {served_model_name}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
    )


__all__ = ["_validate_greedy_request", "run_native_mtp_server"]
