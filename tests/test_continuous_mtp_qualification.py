# SPDX-License-Identifier: Apache-2.0
"""Per-artifact qualification for continuous self-MTP."""

import json
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


class _StopServeError(Exception):
    """Stop the real serve path immediately after scheduler construction."""


def _parsed_serve_args(*argv: str):
    """Return the real parser namespace without entering the serve runtime."""
    from vllm_mlx import cli

    captured = {}

    def _capture(args) -> None:
        captured["args"] = args

    with (
        mock.patch.object(cli, "serve_command", _capture),
        mock.patch.object(sys, "argv", ["rapid-mlx", "serve", *argv]),
    ):
        cli.main()
    return captured["args"]


@contextmanager
def _serve_cache_policy_context(cli, *, stop_at_scheduler: bool):
    """Patch only external boundaries before the cache-policy serve block."""
    patches = [
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        mock.patch.object(
            cli, "_gather_kv_cache_dtype_inputs", lambda *a, **k: ({}, None)
        ),
        mock.patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        mock.patch(
            "vllm_mlx.utils.tokenizer.load_model_with_fallback",
            return_value=(object(), object()),
        ),
        mock.patch(
            "vllm_mlx.server.configure_cors_from_env",
            return_value=[],
        ),
        mock.patch(
            "vllm_mlx.server.configure_trusted_hosts",
            return_value=None,
        ),
        mock.patch(
            "vllm_mlx.middleware.request_logging.install_request_logging_middleware",
            return_value=None,
        ),
        mock.patch.object(sys.stdin, "isatty", return_value=False),
    ]
    if stop_at_scheduler:
        patches.append(
            mock.patch.object(
                sys.modules["vllm_mlx.scheduler"],
                "SchedulerConfig",
                side_effect=_StopServeError,
            )
        )
    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        yield


def _args(model: str, payload: str | None, *, force: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        speculative_config=payload,
        enable_ddtree=False,
        enable_dflash=False,
        enable_mtp=False,
        no_spec_decode=False,
        spec_decode="none",
        dflash_drafter_path="",
        mtp_num_draft_tokens=1,
        mtp_optimistic=False,
        mtp_sidecar=None,
        mtp_max_k=None,
        mtp_disable_auto_k=False,
        force_spec_decode=force,
        suffix_decoding=False,
        suffix_max_draft=None,
        suffix_max_suffix_len=None,
        suffix_min_confidence=None,
        suffix_min_draft_len=None,
    )


@pytest.mark.parametrize(
    ("alias", "tier"),
    [
        ("qwen3.5-4b-4bit", "verified"),
        ("qwen3.5-9b-4bit", "verified"),
        ("qwen3.6-27b-4bit", "verified"),
        ("qwen3.8-27b-4bit", "verified"),
        ("qwen3.5-9b-8bit", "unknown"),
    ],
)
def test_catalog_records_only_exact_measured_artifacts(alias: str, tier: str) -> None:
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile(alias)
    assert profile is not None
    assert profile.mtp_continuous_batching_tier == tier


def test_alias_qualification_fails_closed_when_registry_raises(monkeypatch) -> None:
    from vllm_mlx import cli, model_aliases

    def _raise(_model: str):
        raise RuntimeError("broken alias registry")

    monkeypatch.setattr(model_aliases, "resolve_profile", _raise)

    assert cli._alias_continuous_mtp_tier("qwen3.5-9b-4bit") == "unknown"


def test_serve_rejects_continuous_mtp_cache_conflict_before_scheduler(
    scheduler_config_stub, capsys
) -> None:
    from vllm_mlx import cli

    args = _parsed_serve_args(
        "qwen3.5-9b-4bit",
        "--kv-cache-turboquant",
        "k8v4",
    )
    with (
        _serve_cache_policy_context(cli, stop_at_scheduler=False),
        pytest.raises(SystemExit) as excinfo,
    ):
        cli.serve_command(args)

    assert excinfo.value.code == 2
    assert "continuous MTP requires an unquantized BF16 KV cache" in (
        capsys.readouterr().out
    )


def test_reasoning_continuous_mtp_logs_bf16_cache_policy(
    scheduler_config_stub, caplog
) -> None:
    import logging

    from vllm_mlx import cli

    args = _parsed_serve_args("qwen3.5-9b-4bit", "--reasoning")
    caplog.set_level(logging.INFO, logger="vllm_mlx.cli")
    with (
        _serve_cache_policy_context(cli, stop_at_scheduler=True),
        pytest.raises(_StopServeError),
    ):
        cli.serve_command(args)

    assert "Continuous MTP cache policy: keeping BF16 KV cache" in caplog.text


def test_verified_tier_can_request_continuous_mtp_without_force() -> None:
    from vllm_mlx import cli

    args = _args(
        "qwen3.5-9b-4bit",
        '{"method":"mtp","continuous_batching":true}',
    )
    cli._normalize_speculative_config_or_exit(args)

    assert args.mtp_continuous_batching is True
    assert args.mtp_continuous_batching_tier == "verified"
    assert args.mtp_sidecar == "mlx-community/Qwen3.5-9B-MTP-4bit"


def test_unknown_alias_explicit_opt_in_fails_closed(capsys) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args(
        "qwen3.5-9b-8bit",
        '{"method":"mtp","continuous_batching":true}',
    )
    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    assert args.mtp_continuous_batching_tier == "unknown"
    assert "has not completed continuous-MTP qualification" in capsys.readouterr().err


def test_blocked_alias_explicit_opt_in_fails_closed(monkeypatch, capsys) -> None:
    from vllm_mlx import cli

    monkeypatch.setattr(cli, "_alias_continuous_mtp_tier", lambda _model: "blocked")

    args = _args(
        "qwen3.5-4b-4bit",
        '{"method":"mtp","continuous_batching":true}',
    )

    with pytest.raises(SystemExit) as excinfo:
        cli._normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    assert args.mtp_continuous_batching_tier == "blocked"
    assert "failed continuous-MTP qualification" in capsys.readouterr().err


def test_force_override_keeps_unqualified_artifact_experimental(monkeypatch) -> None:
    from vllm_mlx import cli

    monkeypatch.setattr(cli, "_alias_continuous_mtp_tier", lambda _model: "blocked")

    args = _args(
        "qwen3.5-4b-4bit",
        '{"method":"mtp","continuous_batching":true}',
        force=True,
    )
    cli._normalize_speculative_config_or_exit(args)

    assert args.mtp_continuous_batching is True
    assert args.mtp_continuous_batching_tier == "blocked"


@pytest.mark.parametrize(
    "alias",
    [
        "qwen3.5-4b-4bit",
        "qwen3.5-9b-4bit",
        "qwen3.6-27b-4bit",
        "qwen3.8-27b-4bit",
    ],
)
def test_verified_alias_defaults_to_continuous_when_mtp_is_selected(
    alias: str,
) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args(alias, '{"method":"mtp"}')
    _normalize_speculative_config_or_exit(args)

    assert args.mtp_continuous_batching is True
    assert args.mtp_continuous_batching_tier == "verified"


def test_verified_alias_explicit_false_keeps_ordinary_mtp() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args(
        "qwen3.5-9b-4bit",
        '{"method":"mtp","continuous_batching":false}',
    )
    _normalize_speculative_config_or_exit(args)

    assert args.mtp_continuous_batching is False
    assert args.mtp_continuous_batching_tier == "verified"


def test_legacy_enable_mtp_uses_the_same_verified_default() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args("qwen3.5-9b-4bit", None)
    args.enable_mtp = True
    _normalize_speculative_config_or_exit(args)

    assert args.spec_decode == "mtp"
    assert args.enable_mtp is True
    assert args.mtp_continuous_batching is True
    assert args.mtp_continuous_batching_tier == "verified"
    assert args._speculative_config.method == "mtp"


@pytest.mark.parametrize(
    "alias",
    [
        "qwen3.5-4b-4bit",
        "qwen3.5-9b-4bit",
        "qwen3.6-27b-4bit",
        "qwen3.8-27b-4bit",
    ],
)
def test_verified_alias_defaults_mtp_on_without_any_speculative_flag(
    alias: str,
) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args(alias, None)
    _normalize_speculative_config_or_exit(args)

    assert args.spec_decode == "mtp"
    assert args.mtp_continuous_batching is True
    assert args.mtp_continuous_batching_tier == "verified"
    assert args._speculative_config.method == "mtp"


def test_no_spec_decode_remains_an_explicit_default_off_override() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args("qwen3.5-9b-4bit", None)
    args.no_spec_decode = True
    _normalize_speculative_config_or_exit(args)

    assert args.spec_decode == "none"
    assert args.mtp_continuous_batching is False
    assert args._speculative_config is None


def test_unknown_alias_defaults_to_ordinary_mtp() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args("qwen3.5-9b-8bit", '{"method":"mtp"}')
    _normalize_speculative_config_or_exit(args)

    assert args.mtp_continuous_batching is False
    assert args.mtp_continuous_batching_tier == "unknown"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"kv_cache_turboquant": "k8v4"},
            "--kv-cache-turboquant k8v4 is incompatible",
        ),
        (
            {"kv_cache_quantization": True},
            "--kv-cache-quantization is incompatible",
        ),
        (
            {"kv_cache_dtype": "int8"},
            "--kv-cache-dtype int8 is incompatible",
        ),
    ],
)
def test_continuous_mtp_rejects_explicit_quantized_cache(
    overrides: dict[str, object], message: str
) -> None:
    from vllm_mlx.cli import continuous_mtp_cache_conflict

    args = SimpleNamespace(
        mtp_continuous_batching=True,
        kv_cache_turboquant=None,
        kv_cache_quantization=False,
        kv_cache_dtype="bf16",
    )
    for name, value in overrides.items():
        setattr(args, name, value)

    assert message in (continuous_mtp_cache_conflict(args) or "")


def test_continuous_mtp_accepts_bf16_and_explicit_turboquant_off() -> None:
    from vllm_mlx.cli import continuous_mtp_cache_conflict

    args = SimpleNamespace(
        mtp_continuous_batching=True,
        kv_cache_turboquant="none",
        kv_cache_quantization=False,
        kv_cache_dtype="bf16",
    )

    assert continuous_mtp_cache_conflict(args) is None


def test_ordinary_mtp_preserves_existing_cache_defaults() -> None:
    from vllm_mlx.cli import continuous_mtp_cache_conflict

    args = SimpleNamespace(
        mtp_continuous_batching=False,
        kv_cache_turboquant="k8v4",
        kv_cache_quantization=False,
        kv_cache_dtype="int4",
    )

    assert continuous_mtp_cache_conflict(args) is None


def test_continuous_mtp_suppresses_alias_turboquant_auto_default() -> None:
    from vllm_mlx.cli import _resolve_turboquant_with_mtp_policy

    args = SimpleNamespace(
        mtp_continuous_batching=True,
        kv_cache_turboquant=None,
        kv_cache_quantization=False,
    )
    detected = SimpleNamespace(turboquant_tier="k8v4_verified")

    assert (
        _resolve_turboquant_with_mtp_policy(
            args,
            model_name="qwen3.5-9b-4bit",
            _detected_config=detected,
        )
        is None
    )


def test_ordinary_mtp_keeps_alias_turboquant_auto_default(
    scheduler_config_stub,
) -> None:
    from vllm_mlx.cli import _resolve_turboquant_with_mtp_policy

    args = SimpleNamespace(
        mtp_continuous_batching=False,
        kv_cache_turboquant=None,
        kv_cache_quantization=False,
    )
    detected = SimpleNamespace(turboquant_tier="k8v4_verified")

    assert (
        _resolve_turboquant_with_mtp_policy(
            args,
            model_name="qwen3.5-9b-4bit",
            _detected_config=detected,
        )
        == "k8v4"
    )


@pytest.mark.parametrize("tier", ["verified", "blocked"])
def test_non_unknown_tier_requires_an_mtp_target(tier: str) -> None:
    from vllm_mlx.model_aliases import _coerce

    with pytest.raises(ValueError, match="requires supports_native_mtp"):
        _coerce(
            "bad",
            {
                "hf_path": "example/model",
                "mtp_continuous_batching_tier": tier,
            },
        )


def test_invalid_tier_fails_alias_registry_validation() -> None:
    from vllm_mlx.model_aliases import _coerce

    with pytest.raises(ValueError, match="must be one of"):
        _coerce(
            "bad",
            {
                "hf_path": "example/model",
                "supports_native_mtp": True,
                "mtp_speculative_tokens": 1,
                "mtp_continuous_batching_tier": "probably",
            },
        )


@pytest.mark.parametrize("tier", [[], {}, True, None])
def test_non_string_tier_fails_alias_registry_validation(tier: object) -> None:
    from vllm_mlx.model_aliases import _coerce

    with pytest.raises(ValueError, match="must be a string"):
        _coerce(
            "bad",
            {
                "hf_path": "example/model",
                "supports_native_mtp": True,
                "mtp_speculative_tokens": 1,
                "mtp_continuous_batching_tier": tier,
            },
        )


def test_qualification_benchmark_dry_run_is_network_free() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "bench" / "bench_continuous_mtp_server.py"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--label",
            "candidate",
            "--model",
            "example/model",
            "--runs",
            "2",
            "--concurrency",
            "3",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["planned_requests"] == 6
    assert len(set(payload["lane_prompt_sha256"].values())) == 3


def test_qualification_prompts_are_lane_specific_and_stable() -> None:
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1] / "bench" / "bench_continuous_mtp_server.py"
    )
    spec = importlib.util.spec_from_file_location("continuous_mtp_bench", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    prompts = [module._prompt_for_lane(lane) for lane in range(4)]
    assert len(set(prompts)) == 4
    assert prompts == [module._prompt_for_lane(lane) for lane in range(4)]
