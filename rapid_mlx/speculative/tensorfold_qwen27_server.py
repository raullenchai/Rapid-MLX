# SPDX-License-Identifier: Apache-2.0
"""Experimental TensorFold provider for Rapid's serial text HTTP server."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import queue
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

from rapid_mlx.request import RequestOutput

from .tensorfold_qwen27 import TensorFoldQwen27Backend, validate_request

_MAX_CONCURRENT_REQUESTS = 1


class _StableIncrementalText:
    """Use MLX-LM's line-buffered decoder so emitted text needs no revision.

    Some tokenizers clean whitespace differently after the following token is
    known. Decoding the entire prefix and slicing it therefore corrupts SSE
    output when the newly decoded prefix is not string-prefix-stable. This is
    the same line-buffering contract used by Rapid's native scheduler.
    """

    def __init__(self, tokenizer: Any, lock: Any) -> None:
        self._lock = lock
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

        if not hasattr(tokenizer, "clean_up_tokenization_spaces"):
            tokenizer.clean_up_tokenization_spaces = False
        self._decoder = NaiveStreamingDetokenizer(tokenizer)

    def extend(self, tokens: list[int]) -> str:
        with self._lock:
            for token in tokens:
                self._decoder.add_token(int(token))
            return str(self._decoder.text)

    def finalize(self) -> str:
        with self._lock:
            self._decoder.finalize()
            return str(self._decoder.text)


@dataclass(frozen=True)
class ProviderChunk:
    text: str
    token: int
    generation_tokens: int
    prompt_tokens: int


@dataclass(frozen=True)
class ProviderResult:
    text: str
    tokens: list[int]
    generation_tokens: int
    prompt_tokens: int


class TensorFoldRequestProvider:
    """Map one TensorFold ``ChatJob`` onto Rapid request/output objects.

    TensorFold owns scheduling and cache state.  This boundary observes its
    token chunks before decoding, which avoids reconstructing token IDs from
    text and gives qualification runs an exact token sequence to hash.
    """

    def __init__(
        self, backend: TensorFoldQwen27Backend, *, audit_path: str | None = None
    ) -> None:
        self.backend = backend
        self._audit_path = audit_path
        self._audit_lock = threading.Lock()
        self.last_token_ids: list[int] = []
        self.last_outputs: list[RequestOutput] = []

    def _outputs(self, prompt: str, **kwargs: Any) -> Iterator[RequestOutput]:
        from tensorfold.engine.lane_engine import SuffixLookupProposer
        from tensorfold.server.cancellation import Cancellation
        from tensorfold.server.scheduler import ChatJob
        from tensorfold.server.stopping import StopPolicy

        app = self.backend._app
        if app is None:
            raise RuntimeError("TensorFold backend is closed")
        tokenizer = app.tokenizer
        with app.tokenizer_lock:
            prompt_ids = [int(x) for x in tokenizer.encode(prompt)]
        fields = {
            key: kwargs[key]
            for key in ("top_p", "top_k", "min_p", "seed", "stop")
            if kwargs.get(key) is not None
        }
        validate_request(sampling={k: v for k, v in fields.items() if k != "stop"})
        stops = StopPolicy(fields, tokenizer, app.tokenizer_lock, app.stop_ids)
        temperature = float(kwargs.get("temperature", 0.0))
        cancellation = Cancellation()
        request_id = f"rapid-{uuid.uuid4().hex[:12]}"
        job = ChatJob(
            job_id=request_id,
            prompt_ids=prompt_ids,
            max_tokens=max(1, int(kwargs.get("max_tokens", 1))),
            temperature=temperature,
            sampling=app._resolve_sampling(fields, temperature, prompt_ids),
            drafts=True,
            ignore_eos=stops.ignore_eos,
            stop_check=stops if stops.strings else None,
            cancellation=cancellation,
            proposer=SuffixLookupProposer(min_match=app.min_match),
        )
        thinking_budget = int(kwargs.get("thinking_budget") or 0)
        if thinking_budget > 0:
            job.think_close, job.think_end = app._think_close()
            job.think_budget = thinking_budget if job.think_end >= 0 else 0
        collected: list[int] = []
        decoded = ""
        incremental = _StableIncrementalText(tokenizer, app.tokenizer_lock)
        outputs: list[RequestOutput] = []
        app.scheduler.submit(job)
        try:
            while True:
                try:
                    chunk = job.chunks.get(timeout=0.05)
                except queue.Empty:
                    cancellation.check()
                    continue
                if chunk is None:
                    break
                chunk_tokens = [int(token) for token in chunk]
                fresh: list[int] = []
                for token in chunk_tokens:
                    if token in stops.eos_ids:
                        break
                    fresh.append(token)
                current = stops.visible(incremental.extend(fresh), partial=True)
                if not current.startswith(decoded):
                    raise RuntimeError(
                        "TensorFold streaming decoder produced a non-monotonic prefix"
                    )
                delta = current[len(decoded) :]
                delta_index = len(fresh) - 1
                for index, token in enumerate(chunk_tokens):
                    collected.append(token)
                    token_delta = delta if index == delta_index else ""
                    if token_delta:
                        decoded = current
                    output = RequestOutput(
                        request_id=request_id,
                        new_token_ids=[token],
                        new_text=token_delta,
                        output_token_ids=list(collected),
                        output_text=decoded,
                        prompt_tokens=len(prompt_ids),
                        completion_tokens=len(collected),
                    )
                    outputs.append(output)
                    yield output
            if job.error is not None:
                raise job.error
            # Flush text held while it could still be a partial stop string and
            # verify that the streamed surface exactly reconstructs the final
            # batch decode. SSE cannot retract bytes, so fail closed if a future
            # tokenizer violates the contextual decoder contract.
            incremental.finalize()
            content_ids = list(collected)
            while content_ids and content_ids[-1] in stops.eos_ids:
                content_ids.pop()
            with app.tokenizer_lock:
                final_text = stops.visible(tokenizer.decode(content_ids))
            if not final_text.startswith(decoded):
                raise RuntimeError(
                    "TensorFold streamed text does not match final tokenizer decode"
                )
            if final_delta := final_text[len(decoded) :]:
                decoded = final_text
                output = RequestOutput(
                    request_id=request_id,
                    new_token_ids=[],
                    new_text=final_delta,
                    output_token_ids=list(collected),
                    output_text=decoded,
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(collected),
                )
                outputs.append(output)
                yield output
            finish = job.stream.finish_reason if job.stream is not None else "length"
            terminal = RequestOutput(
                request_id=request_id,
                output_token_ids=list(collected),
                output_text=decoded,
                finished=True,
                finish_reason=finish,
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(collected),
                cached_tokens=int(job.cached_tokens),
            )
            outputs.append(terminal)
        finally:
            cancellation.cancel()
            self.last_token_ids = list(collected)
            self.last_outputs = outputs
            if self._audit_path:
                encoded = ",".join(str(token) for token in collected).encode()
                record = {
                    "request_id": request_id,
                    "token_ids": collected,
                    "token_sha256": hashlib.sha256(encoded).hexdigest(),
                }
                with (
                    self._audit_lock,
                    open(self._audit_path, "a", encoding="utf-8") as handle,
                ):
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def stream_generate(
        self, _model: Any, _processor: Any, prompt: str, **kwargs: Any
    ) -> Iterator[ProviderChunk]:
        for output in self._outputs(prompt, **kwargs):
            if output.finished:
                continue
            yield ProviderChunk(
                text=output.new_text,
                token=(output.new_token_ids[0] if output.new_token_ids else -1),
                generation_tokens=output.completion_tokens,
                prompt_tokens=output.prompt_tokens,
            )

    def generate(
        self, _model: Any, _processor: Any, prompt: str, **kwargs: Any
    ) -> ProviderResult:
        text = ""
        prompt_tokens = 0
        for output in self._outputs(prompt, **kwargs):
            text = output.output_text
            prompt_tokens = output.prompt_tokens
        return ProviderResult(
            text, list(self.last_token_ids), len(self.last_token_ids), prompt_tokens
        )


def validate_http_request(
    request: Any, *, supports_reasoning_budget: bool = False
) -> None:
    from fastapi import HTTPException

    try:
        validate_request(
            tools=getattr(request, "tools", None),
            response_format=getattr(request, "response_format", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if any(
        not isinstance(getattr(message, "content", None), str)
        for message in request.messages
    ):
        raise HTTPException(
            status_code=400,
            detail="TensorFold profiles support text message content only",
        )
    # OpenAI-compatible clients commonly serialize neutral penalty values
    # instead of omitting them. TensorFold does not implement these
    # processors, but accepting their mathematical identities is equivalent
    # to omission and avoids rejecting an otherwise ordinary Desktop turn.
    # Any value that would change logits remains an explicit 400.
    unsupported = []
    neutral_penalties = {
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    for name, neutral in neutral_penalties.items():
        value = getattr(request, name, None)
        if value is not None and value != neutral:
            unsupported.append(name)

    # Rapid's shared prompt renderer supports the one boolean thinking switch.
    # Keep every other template kwarg fail-closed because TensorFold has not
    # qualified those semantics.
    template_kwargs = getattr(request, "chat_template_kwargs", None)
    if template_kwargs not in (None, {}):
        if not (
            isinstance(template_kwargs, dict)
            and set(template_kwargs) == {"enable_thinking"}
            and isinstance(template_kwargs["enable_thinking"], bool)
        ):
            unsupported.append("chat_template_kwargs")

    unsupported.extend(
        name
        for name in (
            "logit_bias",
            "top_logprobs",
            "video_fps",
            "video_max_frames",
            "reasoning_effort",
            "parallel_tool_calls",
            "tool_choice",
        )
        if getattr(request, name, None) not in (None, {})
    )
    if (
        not supports_reasoning_budget
        and getattr(request, "reasoning_max_tokens", None) is not None
    ):
        unsupported.append("reasoning_max_tokens")
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail="TensorFold profile does not support: " + ", ".join(unsupported),
        )


def generation_kwargs(
    *, max_tokens: int, temperature: float, top_p: float, request: Any
) -> dict[str, Any]:
    kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": getattr(request, "top_k", None),
        "min_p": getattr(request, "min_p", None),
        "seed": getattr(request, "seed", None),
        "stop": getattr(request, "stop", None),
    }
    if (budget := getattr(request, "reasoning_max_tokens", None)) is not None:
        kwargs["thinking_budget"] = budget
    return kwargs


def render_prompt(
    processor: Any, _model: Any, request: Any, *, enable_thinking: bool, **_ignored: Any
) -> str:
    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    return cast(
        str,
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ),
    )


def run_tensorfold_qwen27_server(
    *,
    main_model_repo: str,
    main_model_revision: str | None,
    drafter_repo: str,
    drafter_revision: str | None,
    host: str,
    port: int,
    port_explicit: bool | None,
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
    tool_call_parser: str | None = None,
    reasoning_parser_name: str | None = "qwen3",
    default_reasoning_effort: str | None = None,
    backend_class: Any = TensorFoldQwen27Backend,
    profile_id: str = "qwen3.8-27b-tensorfold",
    backend_label: str = "TensorFold Qwen3.8-27B",
    method: str = "dflash",
    algorithm: str = "dflash2",
    fallback_model: str = "qwen3.8-27b-4bit",
    min_memory_gb: int = 48,
    runtime_extra: str = "tensorfold-qwen27",
    target_repository: str = "Vontra/Qwen3.8-27B-MLX-4bit",
    target_revision: str = "70ae7fac63274ff2eac54152031433374cb80f2f",
    paired_repository: str | None = "z-lab/Qwen3.8-27B-DFlash2",
    paired_revision: str | None = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
    supports_reasoning_budget: bool = False,
    **_ignored: Any,
) -> None:
    """Load the pinned pair and serve the experimental serial text API."""
    if main_model_revision is not None or drafter_revision is not None:
        raise RuntimeError("tensorfold backend requires pinned local snapshot paths")
    if tool_call_parser is not None:
        raise RuntimeError("tensorfold backend does not support tool parsing")
    backend = backend_class.load(
        main_model_repo,
        drafter_repo,
        served_name=served_model_name,
        max_tokens=default_max_tokens,
    )
    # Qualification-only token evidence. Unset by default so production
    # requests never persist generated token IDs.
    provider = TensorFoldRequestProvider(
        backend, audit_path=os.environ.get("RAPID_MLX_TENSORFOLD_AUDIT_PATH")
    )
    from rapid_mlx.api.models import ModelInfo, SpeculativeDecodingInfo
    from rapid_mlx.speculative.dflash.server import _build_app

    speculative_info = SpeculativeDecodingInfo(
        configured=True,
        method=method,
        runtime_state="active",
        backend="tensorfold",
        unsupported_features=["tools", "media", "grammar"],
    )
    model_info = ModelInfo(
        id=served_model_name,
        capabilities=["text", "experimental"],
        reasoning_parser=reasoning_parser_name,
        speculative_decoding=speculative_info,
        fallback_model=fallback_model,
        min_memory_gb=min_memory_gb,
    )

    app = _build_app(
        model=None,
        processor=backend._app.tokenizer,
        runtime=SimpleNamespace(
            algorithm=algorithm,
            drafter_repo=paired_repository,
            target_revision=target_revision,
            drafter_revision=paired_revision,
        ),
        served_model_name=served_model_name,
        default_max_tokens=default_max_tokens,
        cors_origins=cors_origins,
        no_thinking=no_thinking,
        api_key=api_key,
        rate_limit=rate_limit,
        max_request_bytes=max_request_bytes,
        body_receive_timeout_seconds=body_receive_timeout_seconds,
        default_timeout=default_timeout,
        # TensorFold exposes one serial lane. Keep admission aligned with that
        # runtime capacity even when the shared CLI default is much larger.
        max_concurrent_requests=_MAX_CONCURRENT_REQUESTS,
        cors_policy=cors_policy,
        tool_call_parser=None,
        reasoning_parser_name=reasoning_parser_name,
        default_reasoning_effort=default_reasoning_effort,
        stream_generate_fn=provider.stream_generate,
        generate_fn=provider.generate,
        render_prompt_fn=render_prompt,
        generation_kwargs_fn=generation_kwargs,
        generation_kwargs_with_request=True,
        validate_request_fn=functools.partial(
            validate_http_request,
            supports_reasoning_budget=supports_reasoning_budget,
        ),
        backend_name=backend_label,
        speculative_info=speculative_info,
        model_info=model_info,
        runtime_status_extra={
            "profile": {
                "id": profile_id,
                "mode": "accelerated",
                "fallback_mode": "normal",
                "compatibility": {
                    "state": "ready",
                    "reason": None,
                    "action": None,
                },
                "runtime": {
                    "extra": runtime_extra,
                    "installed": True,
                },
                "models": {
                    "target": {
                        "repository": target_repository,
                        "revision": target_revision,
                        "ready": True,
                    },
                    **(
                        {
                            "drafter": {
                                "repository": paired_repository,
                                "revision": paired_revision,
                                "ready": True,
                            }
                        }
                        if paired_repository is not None
                        else {}
                    ),
                },
                "capabilities": {
                    "text_chat": True,
                    "streaming": True,
                    "tools": False,
                    "media": False,
                    "grammar": False,
                    "max_concurrency": _MAX_CONCURRENT_REQUESTS,
                },
            }
        },
    )

    @app.on_event("shutdown")
    def _close_backend() -> None:
        backend.close()

    from rapid_mlx._uvicorn import run_uvicorn

    run_uvicorn(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
        port_explicit=port_explicit,
    )
