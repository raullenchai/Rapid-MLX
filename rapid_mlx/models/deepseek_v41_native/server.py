"""OpenAI-compatible server with an opt-in draft-off batching lane."""

from __future__ import annotations

from functools import partial
from typing import Any

from .artifacts import (
    download_mtp_snapshot,
    download_target_snapshot,
    require_product_memory,
    REAP,
    product_artifact,
    resolve_product_paths,
    take_resolved_product_paths,
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
    port_explicit: bool | None = None,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    uvicorn_log_level: str,
    no_thinking: bool = False,
    default_reasoning_effort: str | None = None,
    api_key: str | None = None,
    rate_limit: int = 0,
    max_request_bytes: int = 8 * 1024 * 1024,
    body_receive_timeout_seconds: float = 15.0,
    default_timeout: float = 1800.0,
    max_concurrent_requests: int = 256,
    lanes: int = 1,
    cors_policy: Any | None = None,
    reasoning_parser_name: str | None = "deepseek_v4",
    model_name: str = REAP.alias,
) -> None:
    import uvicorn

    from rapid_mlx.speculative.dflash.server import _build_app, _dflash_executor

    from rapid_mlx.telemetry.server_start import failure_stage

    from .lanes import request_lanes

    lanes = request_lanes(lanes)
    backend = None
    batch_options: dict[str, Any] = {}
    artifact = product_artifact(model_name)
    if artifact is None:
        raise ValueError(f"unknown V4.1 product artifact: {model_name}")
    if artifact == REAP:
        require_product_memory()
    else:
        require_product_memory(artifact)
    with failure_stage("download"):
        if artifact == REAP:
            # The original lane never takes the local qualification override.
            local = False
            target_path = download_target_snapshot()
            mtp_path = download_mtp_snapshot()
        else:
            # Reuse the CLI's verified prefetch (and its single override
            # warning); otherwise resolve and verify here.
            resolved = take_resolved_product_paths(artifact)
            if resolved is None:
                resolve_product_paths(artifact)
                resolved = take_resolved_product_paths(artifact)
            assert resolved is not None
            target_path, mtp_path, local = resolved

    def _load_all():
        target_revision = "local-unpinned" if local else artifact.target_revision
        mtp_revision = "local-unpinned" if local else artifact.mtp_revision
        assert target_revision is not None and mtp_revision is not None
        return load_product_runtime(
            str(target_path),
            str(mtp_path),
            target_revision=target_revision,
            mtp_revision=mtp_revision,
            mtp_identity=str(mtp_path) if local else artifact.mtp_repo,
            **({"mixed_target_precision": True} if artifact.draft_subfolder else {}),
        )

    # Product runtime construction materializes both downloaded checkpoints.
    with failure_stage("prepare"):
        if lanes == 1:
            model, tokenizer, runtime = _dflash_executor.submit(_load_all).result()
        else:
            from types import SimpleNamespace

            from rapid_mlx.api.models import SpeculativeDecodingInfo
            from rapid_mlx.speculative.dflash.server import BackendExecutionPolicy

            def _load_batch():
                model, tokenizer, draft_runtime = _load_all()
                return (
                    model,
                    tokenizer,
                    SimpleNamespace(target_revision=draft_runtime.target_revision),
                )

            from .batch_scheduler import BatchBackend

            backend = BatchBackend(lanes=lanes, load_fn=_load_batch)
            try:
                model, tokenizer = backend.model, backend.tokenizer
                runtime = SimpleNamespace(
                    algorithm="greedy-batched",
                    drafter_repo=None,
                    target_revision=backend.runtime.target_revision,
                    drafter_revision=None,
                )
                # Construction loads the paired checkpoint once, but this lane
                # performs target-only decoding and retains no draft runtime.
                del backend.runtime
                batch_options = {
                    "execution_policy": BackendExecutionPolicy(lanes, backend.executor),
                    "generate_fn_accepts_cancel_event": True,
                    "speculative_info": SpeculativeDecodingInfo(
                        configured=False,
                        method=None,
                        runtime_state="unavailable",
                    ),
                    "runtime_status_extra": {"mode": "batched", "lanes": lanes},
                }
            except BaseException:
                backend.close()
                raise
    try:
        app = _build_app(
            model=model,
            processor=tokenizer,
            runtime=runtime,
            served_model_name=served_model_name,
            default_max_tokens=default_max_tokens,
            cors_origins=cors_origins,
            no_thinking=no_thinking,
            default_reasoning_effort=default_reasoning_effort,
            api_key=api_key,
            rate_limit=rate_limit,
            max_request_bytes=max_request_bytes,
            body_receive_timeout_seconds=body_receive_timeout_seconds,
            default_timeout=default_timeout,
            max_concurrent_requests=lanes if backend else max_concurrent_requests,
            cors_policy=cors_policy,
            tool_call_parser=None,
            reasoning_parser_name=reasoning_parser_name,
            render_prompt_fn=render_prompt,
            stream_generate_fn=backend.stream_generate
            if backend
            else partial(stream_generate, runtime=runtime),
            generate_fn=backend.generate
            if backend
            else partial(generate, runtime=runtime),
            generation_kwargs_fn=generation_kwargs,
            validate_request_fn=validate_request,
            backend_name="DeepSeek V4.1 batched"
            if backend
            else "DeepSeek V4.1 DSpark K4",
            **batch_options,
        )
        if backend is not None:
            app.router.on_shutdown.append(backend.close)
    except BaseException:
        if backend is not None:
            backend.close()
        raise
    print()
    host_display = "localhost" if host == "0.0.0.0" else host

    def _print_ready() -> None:
        mode = f"batched mode, {lanes} lanes" if backend else "DSpark K4 serial mode"
        print(f"  Ready: http://{host_display}:{port}/v1  ({mode})")
        print(f"  Docs:  http://{host_display}:{port}/docs")
        print()

    from rapid_mlx._uvicorn import run_uvicorn

    try:
        run_uvicorn(
            app,
            host=host,
            port=port,
            log_level=uvicorn_log_level,
            timeout_keep_alive=30,
            on_server_accepting=_print_ready,
            port_explicit=port_explicit,
        )
    finally:
        if backend is not None:
            backend.close()
