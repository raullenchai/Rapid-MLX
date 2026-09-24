# SPDX-License-Identifier: Apache-2.0
"""Serial OpenAI-compatible server for the qualified LFM DSpark pair."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import HTTPException

from rapid_mlx.api.models import (
    ChatCompletionRequest,
    CompanionSpeculativeDecodingInfo,
    ModelInfo,
)

from .artifacts import CompanionDSparkArtifacts, download_companion_artifacts
from .eligibility import CompanionDSparkPair
from .runtime import load_runtime

logger = logging.getLogger(__name__)


def _validate_greedy_request(request: ChatCompletionRequest) -> None:
    """Reject every request shape outside the qualified greedy contract."""

    # Keep every malformed/unsupported content block on the request-policy
    # side of the render-worker boundary.  This shared validator is scoped to
    # request shape and serving-lane capability, so its ValueError contract is
    # safe to translate here; unrelated renderer exceptions remain 5xx.
    from rapid_mlx.api.utils import validate_content_blocks_for_capabilities

    try:
        validate_content_blocks_for_capabilities(
            request.messages,
            model_name=request.model,
            allow_image=True,
            allow_video=False,
            allow_audio=False,
        )
    except ValueError as exc:
        message = str(exc)
        code = (
            "unsupported_content_type"
            if " does not support " in message
            else "invalid_request"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": code,
                    "param": "messages.content",
                }
            },
        ) from exc

    unsupported: list[str] = []
    if request.temperature not in (None, 0, 0.0):
        unsupported.append("temperature")
    if request.top_p not in (None, 1, 1.0):
        unsupported.append("top_p")
    if request.top_k not in (None, 0):
        unsupported.append("top_k")
    if request.min_p not in (None, 0, 0.0):
        unsupported.append("min_p")
    if request.repetition_penalty not in (None, 1, 1.0):
        unsupported.append("repetition_penalty")
    if request.presence_penalty not in (None, 0, 0.0):
        unsupported.append("presence_penalty")
    if request.frequency_penalty not in (None, 0, 0.0):
        unsupported.append("frequency_penalty")
    if request.seed is not None:
        unsupported.append("seed")
    if request.stop:
        unsupported.append("stop")
    if request.video_fps is not None or request.video_max_frames is not None:
        unsupported.append("video parameters")
    if (
        request.tools
        or request.functions
        or request.tool_choice is not None
        or request.function_call is not None
        or request.parallel_tool_calls is not None
    ):
        unsupported.append("tools")
    if request.logprobs or request.top_logprobs not in (None, 0):
        unsupported.append("logprobs")
    if request.logit_bias:
        unsupported.append("logit_bias")
    if request.response_format is not None:
        unsupported.append("response_format")
    if request.chat_template_kwargs:
        unsupported.append("chat_template_kwargs")
    if request.reasoning_max_tokens is not None or request.reasoning_effort is not None:
        unsupported.append("reasoning")
    if request.enable_thinking is True:
        unsupported.append("enable_thinking")
    if unsupported:
        names = ", ".join(sorted(set(unsupported)))
        raise HTTPException(
            status_code=400,
            detail=(
                "LFM companion DSpark is qualified for unprocessed greedy "
                f"decoding only; unsupported request field(s): {names}."
            ),
        )


def _validate_companion_request(
    request: ChatCompletionRequest,
    *,
    served_model_name: str,
) -> None:
    """Apply the qualified request policy and bind it to the loaded target."""

    if request.model != served_model_name:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model `{request.model}` does not exist.",
                    "type": "not_found_error",
                    "code": "model_not_found",
                    "param": "model",
                }
            },
        )
    _validate_greedy_request(request)


def _build_companion_model_info(
    *,
    pair: CompanionDSparkPair,
    served_model_name: str,
) -> ModelInfo:
    """Build the one model-card truth shared by every companion endpoint."""

    speculative_info = CompanionSpeculativeDecodingInfo(
        configured=True,
        method="dspark",
        runtime_state="active",
        target_model=pair.target_repo,
        drafter_model=pair.drafter_repo,
        target_revision=pair.target_revision,
        drafter_revision=pair.drafter_revision,
        num_speculative_tokens=pair.num_speculative_tokens,
        draft_block_size=pair.draft_block_size,
    )
    return ModelInfo(
        id=served_model_name,
        modality="image",
        serving_lane="vision",
        serving_lane_reason="qualified_companion_dspark",
        capabilities=["text", "vision"],
        speculative_decoding=speculative_info,
    )


def _prepare_multimodal_prompt(
    processor: Any,
    model: Any,
    request: ChatCompletionRequest,
    *,
    enable_thinking: bool | None = None,
):
    """Render native multimodal messages and retain their image payloads."""

    from mlx_vlm.prompt_utils import apply_chat_template
    from requests import RequestException

    from rapid_mlx.api.utils import validate_content_blocks_for_capabilities
    from rapid_mlx.models.mllm import FileSizeExceededError, process_image_input
    from rapid_mlx.speculative.dflash.server import PreparedPrompt

    validate_content_blocks_for_capabilities(
        request.messages,
        model_name=request.model,
        allow_image=True,
        allow_video=False,
        allow_audio=False,
    )
    messages: list[dict[str, Any]] = []
    images: list[str] = []

    def _process_image(image_ref: Any) -> str:
        try:
            return process_image_input(image_ref)
        # RemoteMediaFetchError intentionally is not repeated here: it derives
        # from ValueError.  FileSizeExceededError is its independent sibling.
        # Keep this catch immediately around media processing so a ValueError
        # from apply_chat_template (or any other renderer bug) remains a 500.
        except (ValueError, FileSizeExceededError, RequestException) as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Invalid image input.",
                        "type": "invalid_request_error",
                        "code": "invalid_image",
                        "param": "messages.content",
                    }
                },
            ) from exc

    for raw_message in request.messages:
        message = raw_message.model_dump(exclude_none=True)
        content = raw_message.content
        if isinstance(content, list):
            native_parts: list[dict[str, Any]] = []
            for raw_part in content:
                part = (
                    raw_part.model_dump(exclude_none=True)
                    if hasattr(raw_part, "model_dump")
                    else dict(raw_part)
                )
                kind = part.get("type")
                if kind in ("text", "input_text", "output_text"):
                    native_parts.append({"type": "text", "text": part.get("text", "")})
                elif kind in ("image_url", "input_image"):
                    image_ref = part.get("image_url")
                    if isinstance(image_ref, dict):
                        image_ref = image_ref.get("url")
                    images.append(_process_image(image_ref))
                    native_parts.append({"type": "image"})
            message["content"] = native_parts
        messages.append(message)

    template_kwargs: dict[str, Any] = {
        "num_images": len(images),
        "num_audios": 0,
        "enable_thinking": bool(enable_thinking),
    }
    prompt = apply_chat_template(processor, model.config, messages, **template_kwargs)
    generation_kwargs = {"image": images} if images else {}
    return PreparedPrompt(prompt=prompt, generation_kwargs=generation_kwargs)


def run_companion_dspark_server(
    *,
    pair: CompanionDSparkPair,
    artifacts: CompanionDSparkArtifacts | None,
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
    reasoning_parser_name: str | None = None,
) -> None:
    """Load the immutable pair before binding and run the serial API."""

    from rapid_mlx._uvicorn import run_uvicorn
    from rapid_mlx.speculative.dflash.server import _build_app, _dflash_executor
    from rapid_mlx.telemetry.server_start import failure_stage

    def _load_all():
        started = time.perf_counter()
        resolved = artifacts or download_companion_artifacts(pair)
        loaded = load_runtime(pair, resolved)
        logger.info(
            "LFM companion DSpark target and drafter loaded in %.1fs",
            time.perf_counter() - started,
        )
        return loaded

    # Any download, compatibility, attach, or model-load failure escapes this
    # boundary before uvicorn binds. There is deliberately no baseline fallback.
    with failure_stage("prepare"):
        model, processor, runtime = _dflash_executor.submit(_load_all).result()

    def _generation_kwargs(*, max_tokens: int, temperature: float, top_p: float):
        # Request validation guarantees these sampling values. Assert again at
        # the final call boundary so a programmatic app caller cannot attach the
        # drafter to an unqualified sampling request.
        if temperature not in (0, 0.0) or top_p not in (1, 1.0):
            raise RuntimeError("companion DSpark generation must remain greedy")
        return {
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "draft_model": runtime.drafter,
            "draft_kind": runtime.kind,
            "draft_block_size": runtime.draft_block_size,
        }

    model_info = _build_companion_model_info(
        pair=pair,
        served_model_name=served_model_name,
    )
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
        tool_call_parser=None,
        reasoning_parser_name=reasoning_parser_name,
        render_prompt_fn=_prepare_multimodal_prompt,
        generation_kwargs_fn=_generation_kwargs,
        validate_request_fn=lambda request: _validate_companion_request(
            request,
            served_model_name=served_model_name,
        ),
        backend_name="LFM DSpark",
        model_info=model_info,
        strict_openai_streaming=True,
    )

    host_display = "localhost" if host == "0.0.0.0" else host

    def _print_ready() -> None:
        print(f"  Ready: http://{host_display}:{port}/v1  (LFM DSpark mode)")
        print(f"  Model: {served_model_name}")

    run_uvicorn(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
        on_server_accepting=_print_ready,
    )


__all__ = [
    "_build_companion_model_info",
    "_validate_companion_request",
    "_prepare_multimodal_prompt",
    "_validate_greedy_request",
    "run_companion_dspark_server",
]
