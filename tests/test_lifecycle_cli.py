# SPDX-License-Identifier: Apache-2.0
"""CLI forwarding tests for lifecycle configuration flags."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def restore_server_globals():
    """Restore mutated server module globals between lifecycle tests."""
    import vllm_mlx.server as srv

    sentinel = object()
    global_names = (
        "_engine",
        "_model_name",
        "_model_path",
        "_default_model_key",
        "_default_max_tokens",
        "_default_timeout",
        "_default_temperature",
        "_default_top_p",
        "_force_mllm_model",
        "_auto_unload_idle_seconds",
        "_lazy_load_model",
        "_residency_manager",
        "_lifecycle_task",
        "_lifespan_active",
        "_mcp_manager",
        "_mcp_executor",
        "_embedding_engine",
        "_embedding_model_locked",
        "_api_key",
        "_auth_warning_logged",
        "_rate_limiter",
        "_reasoning_parser",
        "_enable_auto_tool_choice",
        "_tool_call_parser",
        "_tool_parser_instance",
        "_idle_unload_enabled",
    )
    snapshot = {name: getattr(srv, name, sentinel) for name in global_names}

    yield

    leaked_task = getattr(srv, "_lifecycle_task", None)
    original_task = snapshot["_lifecycle_task"]
    if (
        leaked_task is not sentinel
        and leaked_task is not None
        and leaked_task is not original_task
        and not leaked_task.done()
    ):
        leaked_task.cancel()

    for name, value in snapshot.items():
        if value is sentinel:
            if hasattr(srv, name):
                delattr(srv, name)
        else:
            setattr(srv, name, value)

    # _idle_unload_enabled is a lazily-created asyncio.Event bound to whatever
    # event loop was running when _get_idle_unload_event() was first called.
    # Reset to None so the next test gets a fresh Event on its own loop.
    srv._idle_unload_enabled = None


class TestLifecycleCli:
    """Lock in the first lifecycle CLI surface area."""

    def test_main_parses_auto_unload_idle_seconds_flag(self, monkeypatch):
        """The top-level CLI should accept an idle-unload timeout knob."""
        import vllm_mlx.cli as cli

        captured = {}

        def fake_serve_command(args):
            captured["auto_unload_idle_seconds"] = args.auto_unload_idle_seconds

        monkeypatch.setattr(cli, "serve_command", fake_serve_command)
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "vllm-mlx",
                "serve",
                "mlx-community/Qwen3-0.6B-8bit",
                "--auto-unload-idle-seconds",
                "300",
            ],
        )

        cli.main()

        assert captured["auto_unload_idle_seconds"] == 300

    def test_main_parses_lazy_load_model_flag(self, monkeypatch):
        """The top-level CLI should accept lazy lifecycle startup."""
        import vllm_mlx.cli as cli

        captured = {}

        def fake_serve_command(args):
            captured["lazy_load_model"] = args.lazy_load_model

        monkeypatch.setattr(cli, "serve_command", fake_serve_command)
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "vllm-mlx",
                "serve",
                "mlx-community/Qwen3-0.6B-8bit",
                "--lazy-load-model",
            ],
        )

        cli.main()

        assert captured["lazy_load_model"] is True

    def test_main_defaults_lazy_load_model_to_false(self, monkeypatch):
        """Serve startup should stay eager unless the user explicitly opts in."""
        import vllm_mlx.cli as cli

        captured = {}

        def fake_serve_command(args):
            captured["lazy_load_model"] = args.lazy_load_model

        monkeypatch.setattr(cli, "serve_command", fake_serve_command)
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "vllm-mlx",
                "serve",
                "mlx-community/Qwen3-0.6B-8bit",
            ],
        )

        cli.main()

        assert captured["lazy_load_model"] is False

    def test_main_parses_mllm_draft_flags(self, monkeypatch):
        """The top-level CLI should accept MLLM assistant drafter knobs."""
        import vllm_mlx.cli as cli

        captured = {}

        def fake_serve_command(args):
            captured["mllm_draft_model"] = args.mllm_draft_model
            captured["mllm_draft_kind"] = args.mllm_draft_kind
            captured["mllm_draft_block_size"] = args.mllm_draft_block_size

        monkeypatch.setattr(cli, "serve_command", fake_serve_command)
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "vllm-mlx",
                "serve",
                "google/gemma-4-E2B-it",
                "--mllm",
                "--mllm-draft-model",
                "google/gemma-4-E2B-it-assistant",
                "--mllm-draft-kind",
                "mtp",
                "--mllm-draft-block-size",
                "4",
            ],
        )

        cli.main()

        assert captured == {
            "mllm_draft_model": "google/gemma-4-E2B-it-assistant",
            "mllm_draft_kind": "mtp",
            "mllm_draft_block_size": 4,
        }

    def test_main_rejects_nonpositive_mllm_draft_block_size(self, monkeypatch, capsys):
        """Drafter block size must be positive at parse time."""
        import vllm_mlx.cli as cli

        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "vllm-mlx",
                "serve",
                "google/gemma-4-E2B-it",
                "--mllm",
                "--mllm-draft-model",
                "assistant",
                "--mllm-draft-kind",
                "mtp",
                "--mllm-draft-block-size",
                "0",
            ],
        )

        with pytest.raises(SystemExit):
            cli.main()

        assert "--mllm-draft-block-size" in capsys.readouterr().err

    def test_serve_command_rejects_mllm_draft_without_mllm(self, capsys):
        """Drafter flags should not be silently ignored on text-only models."""
        import vllm_mlx.cli as cli

        args = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            models_config=None,
            served_model_name=None,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            gpu_memory_utilization=0.90,
            max_tokens=32768,
            max_request_tokens=32768,
            mllm_draft_model="assistant",
            mllm_draft_kind="mtp",
            mllm_draft_block_size=4,
            continuous_batching=False,
            auto_unload_idle_seconds=0.0,
            lazy_load_model=False,
            mllm=False,
        )

        with pytest.raises(SystemExit):
            cli.serve_command(args)

        assert "--mllm-draft-model requires --mllm" in capsys.readouterr().out

    def test_serve_command_rejects_mllm_draft_with_models_config(self, capsys):
        """Registry mode must not accept drafter flags that it does not load."""
        import vllm_mlx.cli as cli

        args = SimpleNamespace(
            model=None,
            models_config="/tmp/models.yaml",
            served_model_name=None,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            gpu_memory_utilization=0.90,
            max_tokens=32768,
            max_request_tokens=32768,
            mllm_draft_model="assistant",
            mllm_draft_kind="mtp",
            mllm_draft_block_size=4,
            continuous_batching=False,
            auto_unload_idle_seconds=0.0,
            lazy_load_model=False,
            mllm=True,
        )

        with pytest.raises(SystemExit):
            cli.serve_command(args)

        assert "--mllm-draft-model cannot be used with --models-config" in (
            capsys.readouterr().out
        )

    def test_serve_command_wires_auto_unload_idle_seconds_into_load_model(
        self, monkeypatch
    ):
        """serve_command should pass lifecycle config into model loading."""
        import uvicorn

        import vllm_mlx.cli as cli
        import vllm_mlx.server as srv
        import vllm_mlx.utils.download as download_mod

        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(srv, "load_model", fake_load_model)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            download_mod,
            "ensure_model_downloaded",
            lambda *args, **kwargs: "/tmp/model",
        )

        args = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            host="127.0.0.1",
            port=8000,
            max_num_seqs=256,
            prefill_batch_size=8,
            completion_batch_size=32,
            enable_prefix_cache=True,
            disable_prefix_cache=False,
            prefix_cache_size=100,
            cache_memory_mb=None,
            cache_memory_percent=0.20,
            no_memory_aware_cache=False,
            kv_cache_quantization=False,
            kv_cache_quantization_bits=8,
            kv_cache_quantization_group_size=64,
            kv_cache_min_quantize_tokens=256,
            stream_interval=7,
            max_tokens=32768,
            continuous_batching=False,
            use_paged_cache=False,
            paged_cache_block_size=64,
            max_cache_blocks=1000,
            chunked_prefill_tokens=0,
            enable_mtp=False,
            mtp_num_draft_tokens=1,
            mtp_optimistic=False,
            prefill_step_size=2048,
            specprefill=False,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
            specprefill_draft_model=None,
            mcp_config=None,
            api_key=None,
            rate_limit=0,
            timeout=300.0,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            reasoning_parser=None,
            mllm=False,
            default_temperature=None,
            default_top_p=None,
            default_top_k=None,
            default_min_p=None,
            default_presence_penalty=None,
            default_repetition_penalty=None,
            default_chat_template_kwargs=None,
            served_model_name=None,
            embedding_model=None,
            gpu_memory_utilization=0.90,
            enable_metrics=False,
            download_timeout=120,
            download_retries=3,
            mllm_prefill_step_size=None,
            lazy_load_model=True,
            auto_unload_idle_seconds=300,
            models_config=None,
            default_thinking_token_budget=None,
            max_kv_size=0,
        )

        cli.serve_command(args)

        assert captured["kwargs"]["stream_interval"] == 1
        assert captured["kwargs"]["auto_unload_idle_seconds"] == 300
        assert captured["kwargs"]["lazy_load_model"] is True

    def test_serve_command_preserves_mtp_scheduler_config_with_residency(
        self, monkeypatch
    ):
        """serve_command should keep batching/MTP config intact alongside residency."""
        import uvicorn

        import vllm_mlx.cli as cli
        import vllm_mlx.server as srv

        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        class FakeSchedulerConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        monkeypatch.setattr(srv, "load_model", fake_load_model)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
        monkeypatch.setitem(
            sys.modules,
            "vllm_mlx.scheduler",
            SimpleNamespace(SchedulerConfig=FakeSchedulerConfig),
        )

        args = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            host="127.0.0.1",
            port=8000,
            max_num_seqs=256,
            prefill_batch_size=8,
            completion_batch_size=32,
            enable_prefix_cache=True,
            disable_prefix_cache=False,
            prefix_cache_size=100,
            cache_memory_mb=None,
            cache_memory_percent=0.20,
            no_memory_aware_cache=False,
            kv_cache_quantization=False,
            kv_cache_quantization_bits=8,
            kv_cache_quantization_group_size=64,
            kv_cache_min_quantize_tokens=256,
            stream_interval=7,
            max_tokens=32768,
            continuous_batching=True,
            use_paged_cache=False,
            paged_cache_block_size=64,
            max_cache_blocks=1000,
            chunked_prefill_tokens=0,
            enable_mtp=True,
            mtp_num_draft_tokens=4,
            mtp_optimistic=True,
            prefill_step_size=2048,
            specprefill=False,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
            specprefill_draft_model=None,
            mcp_config=None,
            api_key=None,
            rate_limit=0,
            timeout=300.0,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            reasoning_parser=None,
            mllm=False,
            default_temperature=None,
            default_top_p=None,
            default_top_k=None,
            default_min_p=None,
            default_presence_penalty=None,
            default_repetition_penalty=None,
            default_chat_template_kwargs=None,
            served_model_name=None,
            embedding_model=None,
            gpu_memory_utilization=0.90,
            enable_metrics=False,
            download_timeout=120,
            download_retries=3,
            mllm_prefill_step_size=0,
            lazy_load_model=True,
            auto_unload_idle_seconds=300,
            models_config=None,
            default_thinking_token_budget=None,
            max_kv_size=0,
        )

        cli.serve_command(args)

        scheduler_config = captured["kwargs"]["scheduler_config"]
        assert captured["kwargs"]["use_batching"] is True
        assert captured["kwargs"]["stream_interval"] == 7
        assert captured["kwargs"]["auto_unload_idle_seconds"] == 300
        assert captured["kwargs"]["lazy_load_model"] is True
        assert scheduler_config.enable_mtp is True
        assert scheduler_config.mtp_num_draft_tokens == 4
        assert scheduler_config.mtp_optimistic is True

    def test_server_main_preserves_use_batching_with_residency_flags(self, monkeypatch):
        """server.main should forward use_batching and residency knobs together."""
        import vllm_mlx.server as srv

        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(srv, "load_model", fake_load_model)
        monkeypatch.setattr(srv, "load_embedding_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(srv.uvicorn, "run", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vllm_mlx.server",
                "--model",
                "mlx-community/Qwen3-0.6B-8bit",
                "--continuous-batching",
                "--auto-unload-idle-seconds",
                "300",
                "--lazy-load-model",
            ],
        )

        srv.main()

        assert captured["kwargs"]["use_batching"] is True
        assert captured["kwargs"]["auto_unload_idle_seconds"] == 300.0
        assert captured["kwargs"]["lazy_load_model"] is True

    def test_serve_command_describes_lazy_startup_without_claiming_model_is_loaded(
        self, monkeypatch, capsys
    ):
        """Lazy startup output should mention the first request, not claim eager load."""
        import uvicorn

        import vllm_mlx.cli as cli
        import vllm_mlx.server as srv

        monkeypatch.setattr(srv, "load_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

        args = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            host="127.0.0.1",
            port=8000,
            max_num_seqs=256,
            prefill_batch_size=8,
            completion_batch_size=32,
            enable_prefix_cache=True,
            disable_prefix_cache=False,
            prefix_cache_size=100,
            cache_memory_mb=None,
            cache_memory_percent=0.20,
            no_memory_aware_cache=False,
            kv_cache_quantization=False,
            kv_cache_quantization_bits=8,
            kv_cache_quantization_group_size=64,
            kv_cache_min_quantize_tokens=256,
            stream_interval=1,
            max_tokens=32768,
            continuous_batching=False,
            use_paged_cache=False,
            paged_cache_block_size=64,
            max_cache_blocks=1000,
            chunked_prefill_tokens=0,
            enable_mtp=False,
            mtp_num_draft_tokens=1,
            mtp_optimistic=False,
            prefill_step_size=2048,
            specprefill=False,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
            specprefill_draft_model=None,
            mcp_config=None,
            api_key=None,
            rate_limit=0,
            timeout=300.0,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            reasoning_parser=None,
            mllm=False,
            default_temperature=None,
            default_top_p=None,
            default_top_k=None,
            default_min_p=None,
            default_presence_penalty=None,
            default_repetition_penalty=None,
            default_chat_template_kwargs=None,
            served_model_name=None,
            embedding_model=None,
            gpu_memory_utilization=0.90,
            enable_metrics=False,
            download_timeout=120,
            download_retries=3,
            mllm_prefill_step_size=None,
            lazy_load_model=True,
            auto_unload_idle_seconds=0.0,
            models_config=None,
            default_thinking_token_budget=None,
            max_kv_size=0,
        )

        cli.serve_command(args)
        out = capsys.readouterr().out

        assert "Loading model: mlx-community/Qwen3-0.6B-8bit" not in out
        assert "first request" in out
