"""Dedicated serial OpenAI-compatible server for qualified DSpark K4."""

from __future__ import annotations

from functools import partial
from typing import Any

from .artifacts import (
    MTP_REPO,
    MTP_REVISION,
    TARGET_REVISION,
    download_mtp_snapshot,
    download_target_snapshot,
    require_product_memory,
)
from .serving import (
    generate,
    generation_kwargs,
    load_product_runtime,
    render_prompt,
    stream_generate,
    validate_request,
)


def run_server(
    *,
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
    reasoning_parser_name: str | None = "deepseek_v4",
) -> None:
    import uvicorn

    from vllm_mlx.speculative.dflash.server import _build_app, _dflash_executor

    require_product_memory()
    target_path = download_target_snapshot()
    mtp_path = download_mtp_snapshot()

    def _load_all():
        return load_product_runtime(
            str(target_path),
            str(mtp_path),
            target_revision=TARGET_REVISION,
            mtp_revision=MTP_REVISION,
            mtp_identity=MTP_REPO,
        )

    model, tokenizer, runtime = _dflash_executor.submit(_load_all).result()
    app = _build_app(
        model=model,
        processor=tokenizer,
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
        render_prompt_fn=render_prompt,
        stream_generate_fn=partial(stream_generate, runtime=runtime),
        generate_fn=partial(generate, runtime=runtime),
        generation_kwargs_fn=generation_kwargs,
        validate_request_fn=validate_request,
        backend_name="DeepSeek V4.1 DSpark K4",
    )
    print()
    host_display = "localhost" if host == "0.0.0.0" else host
    print(f"  Ready: http://{host_display}:{port}/v1  (DSpark K4 serial mode)")
    print(f"  Docs:  http://{host_display}:{port}/docs")
    print()
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
    )
