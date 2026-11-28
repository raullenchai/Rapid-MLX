# SPDX-License-Identifier: Apache-2.0
"""Engine-start causes from small local checkpoint layouts."""

from __future__ import annotations

import json

import pytest
from jinja2.exceptions import TemplateSyntaxError

from rapid_mlx.model_load_errors import IncompatibleWeights, TokenizerLoadFailed
from rapid_mlx.telemetry.model_events import serve_error_class


def _model(tmp_path, *, model_type="qwen3", quantization=None):
    path = tmp_path / "local-model"
    path.mkdir()
    config = {"model_type": model_type}
    if quantization is not None:
        config["quantization"] = quantization
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.mark.parametrize(
    ("case", "failure", "model_type", "quantization", "expected", "cause", "step"),
    [
        (
            "tokenizer",
            TokenizerLoadFailed("Tokenizer loading failed: no tokenizer files"),
            "qwen3",
            None,
            "tokenizer_load_failed",
            "no tokenizer files",
            "tokenizer.json",
        ),
        (
            "template",
            ValueError(
                "Cannot use apply_chat_template because tokenizer.chat_template is not set"
            ),
            "qwen3",
            None,
            "invalid_config",
            "chat template",
            "tokenizer_config.json",
        ),
        (
            "architecture",
            ValueError("Model type future_arch not supported."),
            "future_arch",
            None,
            "unsupported_architecture",
            "future_arch",
            "qwen3-0.6b-4bit",
        ),
        (
            "vlm_config",
            TypeError("TextConfig.__init__() missing required arguments"),
            "qwen3_vl",
            None,
            "invalid_config",
            "text_config",
            "config.json",
        ),
        (
            "vlm_processor",
            OSError(
                "Can't load image processor for this model: missing preprocessor_config.json"
            ),
            "qwen3_vl",
            None,
            "local_path_missing",
            "preprocessor_config.json",
            "preprocessor_config.json",
        ),
        (
            "quantization",
            IncompatibleWeights(
                "Model weights are incompatible: Expected shape (32, 4) "
                "but received shape (32, 32) for parameter model.layers.0.self_attn.q_proj.weight"
            ),
            "qwen3",
            {"group_size": 32, "bits": 4, "mode": "mxfp4"},
            "quantization_mismatch",
            "MXFP4",
            "matching MLX MXFP4 weights",
        ),
    ],
)
def test_local_engine_start_diagnosis(
    tmp_path, case, failure, model_type, quantization, expected, cause, step
):
    model = _model(tmp_path, model_type=model_type, quantization=quantization)
    assert serve_error_class(failure, model_ref=str(model)) == expected, case

    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    lines = startup_error_lines(failure, str(model))
    assert lines is not None
    assert cause in lines[0]
    assert step in lines[1]
    assert "\n" not in lines[0] + lines[1]


def test_memory_failure_uses_existing_class_and_remedy(tmp_path, monkeypatch):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 16.0)
    model = _model(tmp_path)
    failure = MemoryError()
    assert serve_error_class(failure, model_ref=str(model)) == "insufficient_memory"
    assert startup_error_lines(failure, str(model)) == (
        "The model ran out of memory while loading weights.",
        "Free memory, or choose a smaller model with `rapid-mlx models`.",
    )


@pytest.mark.parametrize(
    ("ram_gb", "failed_ref", "expected_alias"),
    [
        (16.0, "qwen3.8-27b-4bit", "qwen3.5-4b-4bit"),
        (18.0, "qwen3.8-27b-4bit", "qwen3.5-9b-4bit"),
        (48.0, "deepseek-v41-flash-1017", "qwen3.8-27b-4bit"),
        (16.0, "qwen3.5-4b-4bit", "lfm2.5-1b-4bit"),
        (24.0, "qwen3.5-4b-4bit", "lfm2.5-1b-4bit"),
        (32.0, "qwen3.5-4b-4bit", "lfm2.5-1b-4bit"),
        (48.0, "qwen3.8-27b-4bit", "qwen3.5-4b-4bit"),
        (48.0, "bonsai-27b-2bit", "qwen3.5-4b-4bit"),
    ],
)
def test_memory_failure_names_ram_tier_recommendation(
    monkeypatch, ram_gb, failed_ref, expected_alias
):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines
    from rapid_mlx.recommendations import (
        _catalog_footprint_gb,
        memory_tier_recommendation,
        recommendation_footprint_gb,
    )

    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: ram_gb)
    recommended = memory_tier_recommendation(ram_gb, exclude=failed_ref)
    assert recommended == expected_alias
    failed_fp = _catalog_footprint_gb(failed_ref)
    rec_fp = recommendation_footprint_gb(recommended)
    assert failed_fp is not None and rec_fp is not None and rec_fp < failed_fp

    lines = startup_error_lines(MemoryError(), failed_ref)
    assert lines is not None
    assert f"rapid-mlx serve {expected_alias}" in lines[1]


@pytest.mark.parametrize(
    ("ram_gb", "failed_ref"),
    [
        (24.0, "unknown-org/custom-model"),
        (32.0, "/tmp/unmeasured-local-checkpoint"),
        (16.0, "lfm2.5-1b-4bit"),
        (24.0, "qwen3-0.6b-4bit"),
    ],
)
def test_memory_failure_falls_back_to_generic_when_no_smaller_candidate(
    monkeypatch, ram_gb, failed_ref
):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines
    from rapid_mlx.recommendations import memory_tier_recommendation

    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: ram_gb)
    assert memory_tier_recommendation(ram_gb, exclude=failed_ref) is None
    lines = startup_error_lines(MemoryError(), failed_ref)
    assert lines == (
        "The model ran out of memory while loading weights.",
        "Free memory, or choose a smaller model with `rapid-mlx models`.",
    )


def test_pre_download_memory_info_informs_and_never_blocks(monkeypatch, capsys):
    from rapid_mlx.byom import preflight

    gib = 1 << 30
    monkeypatch.setattr(preflight, "_FOOTPRINT_NOTICES_SHOWN", set())
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 16 * gib)
    printed = preflight.inform_if_footprint_exceeds_memory("qwen3.8-27b-4bit")
    assert printed is True
    # Preflight and the pull path both call this; the second call is silent.
    assert preflight.inform_if_footprint_exceeds_memory("qwen3.8-27b-4bit") is False
    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert "qwen3.8-27b-4bit" in err_lines[0]
    assert "20 GB" in err_lines[0]
    assert "16 GB" in err_lines[0]
    assert "qwen3.5-4b-4bit" in err_lines[0]
    assert "continuing download" in err_lines[0]

    # Fitting model stays silent.
    assert preflight.inform_if_footprint_exceeds_memory("qwen3.5-4b-4bit") is False
    assert capsys.readouterr().err == ""

    # Never names a model larger than the one being downloaded.
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 1 * gib)
    printed_small = preflight.inform_if_footprint_exceeds_memory(
        "unknown-org/tiny-model", estimated_bytes=int(1.5 * gib)
    )
    assert printed_small is True
    small_err = capsys.readouterr().err
    assert "unknown-org/tiny-model" in small_err
    assert "rapid-mlx serve" not in small_err
    assert "continuing download" in small_err


def test_pre_download_memory_info_fails_soft_without_ram_or_on_stream_error(
    monkeypatch,
):
    from rapid_mlx.byom import preflight

    gib = 1 << 30
    monkeypatch.setattr(preflight, "_FOOTPRINT_NOTICES_SHOWN", set())
    for unknown_ram in (0, None):
        monkeypatch.setattr(preflight, "physical_ram_bytes", lambda r=unknown_ram: r)
        assert (
            preflight.inform_if_footprint_exceeds_memory(
                "qwen3.8-27b-4bit", estimated_bytes=20 * gib
            )
            is False
        )

    class _BrokenStream:
        def write(self, _data: str) -> int:
            raise OSError("broken pipe")

    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 16 * gib)
    assert (
        preflight.inform_if_footprint_exceeds_memory(
            "qwen3.8-27b-4bit",
            estimated_bytes=20 * gib,
            stream=_BrokenStream(),
        )
        is False
    )

    # Unlisted model falls back to model_sizes.size_bytes + serve overhead.
    from io import StringIO

    from rapid_mlx import model_sizes

    monkeypatch.setattr(preflight, "_FOOTPRINT_NOTICES_SHOWN", set())
    monkeypatch.setattr(model_sizes, "size_bytes", lambda _ref: 20 * gib)
    buf = StringIO()
    assert (
        preflight.inform_if_footprint_exceeds_memory("owner/unlisted-large", stream=buf)
        is True
    )
    assert "owner/unlisted-large" in buf.getvalue()
    assert "continuing download" in buf.getvalue()


def test_catalog_footprint_handles_import_and_size_failures(monkeypatch):
    import sys
    from types import SimpleNamespace

    from rapid_mlx import model_aliases, model_sizes
    from rapid_mlx.recommendations import _catalog_footprint_gb

    fake_profile = SimpleNamespace(hf_path="owner/unmeasured", resident_memory_gb=None)
    monkeypatch.setattr(model_aliases, "catalog_identity", lambda _ref: None)
    monkeypatch.setattr(model_aliases, "resolve_profile", lambda _ref: fake_profile)
    monkeypatch.setattr(
        model_sizes,
        "size_bytes",
        lambda _repo: (_ for _ in ()).throw(RuntimeError("size lookup failed")),
    )
    assert _catalog_footprint_gb("owner/unmeasured") is None

    monkeypatch.setattr(model_sizes, "size_bytes", lambda _repo: None)
    assert _catalog_footprint_gb("owner/unmeasured") is None

    monkeypatch.setitem(sys.modules, "rapid_mlx.model_aliases", None)
    assert _catalog_footprint_gb("qwen3.5-4b-4bit") is None


def test_memory_tier_recommendation_handles_small_ram_and_alias_errors(monkeypatch):
    import sys

    from rapid_mlx import recommendations
    from rapid_mlx.recommendations import (
        Recommendation,
        RecommendationTier,
        memory_tier_recommendation,
    )

    with monkeypatch.context() as ctx:
        ctx.setitem(sys.modules, "rapid_mlx.model_aliases", None)
        assert (
            memory_tier_recommendation(
                16.0, exclude="custom/model", max_footprint_gb=10.0
            )
            == "qwen3.5-4b-4bit"
        )

    # Sub-tier RAM: baseline qualifies at 4 GiB, nothing qualifies at 1 GiB.
    assert memory_tier_recommendation(ram_gb=4.0) == "lfm2.5-1b-4bit"
    assert memory_tier_recommendation(ram_gb=1.0) is None

    # Sub-tier RAM fallback to another tier-0 pick when the baseline exceeds RAM.
    synthetic_tier = (
        RecommendationTier(
            floor_gb=8,
            picks=(
                Recommendation("smart", "tiny-pick", 1.5, 60, None, ()),
                Recommendation("fast", "larger-baseline", 3.5, 50, None, ()),
            ),
        ),
    )
    monkeypatch.setattr(
        recommendations,
        "load_recommendation_tiers",
        lambda **_kw: synthetic_tier,
    )
    assert memory_tier_recommendation(ram_gb=2.5) == "tiny-pick"


@pytest.mark.parametrize("wrapped", [False, True])
def test_metal_preflight_failure_uses_existing_class_and_remedy(
    tmp_path, monkeypatch, wrapped
):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines
    from rapid_mlx.memory_budget import MetalPreflightError, format_preflight_error

    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 16.0)
    model = _model(tmp_path)
    failure = MetalPreflightError(
        format_preflight_error(
            required_bytes=20_000_000_000,
            active_bytes=18_000_000_000,
            min_kv_bytes=2_000_000_000,
            cap_bytes=16_000_000_000,
            utilization=0.8,
            device_budget_bytes=20_000_000_000,
        )
    )
    if wrapped:
        try:
            raise RuntimeError("engine failed") from failure
        except RuntimeError as error:
            failure = error
    assert serve_error_class(failure, model_ref=str(model)) == "insufficient_memory"
    assert startup_error_lines(failure, str(model)) == (
        "The model ran out of memory while loading weights.",
        "Free memory, or choose a smaller model with `rapid-mlx models`.",
    )


@pytest.mark.parametrize(
    "message",
    [
        "[METAL] Command buffer execution failed: Insufficient Memory "
        "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)",
        "[metal::malloc] Attempting to allocate 21474836480 bytes which is greater "
        "than the maximum allowed buffer size of 17179869184 bytes.",
        "[malloc] Unable to allocate 1073741824 bytes.",
    ],
)
def test_metal_oom_uses_existing_class_and_remedy(tmp_path, monkeypatch, message):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 18.0)
    model = _model(tmp_path)
    failure = RuntimeError(message)
    assert serve_error_class(failure, model_ref=str(model)) == "insufficient_memory"
    assert startup_error_lines(failure, str(model)) == (
        "The model ran out of memory while loading weights.",
        "Free memory, or choose a smaller model with `rapid-mlx models`.",
    )


def test_unrelated_error_is_not_relabeled(tmp_path):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    model = _model(tmp_path, model_type="qwen3_vl")
    failure = RuntimeError("Unexpected scheduler bug")
    assert serve_error_class(failure, model_ref=str(model)) == "other"
    assert startup_error_lines(failure, str(model)) is None


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("plugin could not allocate tokenizer ID 7"),
        AttributeError("Engine object has no attribute processor"),
        IncompatibleWeights(
            "Expected shape (32,) but received shape (64,) for parameter model.norm.weight"
        ),
    ],
)
def test_diagnostic_does_not_infer_from_unrelated_words(tmp_path, failure):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    model = _model(
        tmp_path,
        model_type="qwen3_vl",
        quantization={"group_size": 32, "bits": 4, "mode": "mxfp4"},
    )
    assert startup_error_lines(failure, str(model)) is None


def test_template_syntax_error_has_file_remedy(tmp_path):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    model = _model(tmp_path)
    failure = TemplateSyntaxError("Expected an expression", 1)
    assert serve_error_class(failure, model_ref=str(model)) == "invalid_config"
    assert "chat_template.jinja" in startup_error_lines(failure, str(model))[1]


def test_typed_file_cause_keeps_availability_class_and_no_template_hint(tmp_path):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    model = _model(tmp_path)
    failure = ValueError("Cannot use apply_chat_template because no template is set")
    failure.__cause__ = FileNotFoundError(2, "missing", str(model / "tokenizer.json"))
    assert serve_error_class(failure, model_ref=str(model)) == "local_path_missing"
    assert startup_error_lines(failure, str(model)) is None


def test_untrusted_model_type_is_not_printed(tmp_path):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    model = _model(tmp_path, model_type="bad\nNext step: run untrusted code")
    failure = ValueError("Model type bad not supported.")
    assert startup_error_lines(failure, str(model)) is None


def test_local_diagnosis_fails_closed_on_unreadable_input(tmp_path):
    from rapid_mlx.local_serve_diagnostics import local_serve_diagnostic

    assert local_serve_diagnostic(RuntimeError("unknown"), None) is None
    assert (
        local_serve_diagnostic(RuntimeError("unknown"), str(tmp_path / "absent"))
        is None
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    assert local_serve_diagnostic(RuntimeError("unknown"), str(empty)) is None


def test_local_diagnosis_handles_exception_cycles_and_hostile_text(tmp_path):
    from rapid_mlx.local_serve_diagnostics import local_serve_diagnostic

    model = _model(tmp_path)
    cycle = RuntimeError("unknown")
    cycle.__cause__ = cycle
    assert local_serve_diagnostic(cycle, str(model)) is None

    class HostileError(RuntimeError):
        def __str__(self):
            raise KeyboardInterrupt

    assert local_serve_diagnostic(HostileError(), str(model)) is None


@pytest.mark.parametrize("mode", ["config", "local", "hub_debug"])
async def test_engine_start_prints_two_lines_and_preserves_failure(
    tmp_path, monkeypatch, capsys, caplog, mode
):
    from fastapi import FastAPI
    from uvicorn import Config
    from uvicorn.lifespan.on import LifespanOn

    from rapid_mlx import _signal_observability, server
    from rapid_mlx.model_load_errors import typed_weight_boundary
    from rapid_mlx.routes import agents
    from rapid_mlx.telemetry import server_start, track

    model = _model(tmp_path, model_type="qwen3_vl")
    failure = TypeError("TextConfig.__init__() missing required arguments")

    ref = "owner/checkpoint" if mode == "hub_debug" else str(model)
    if mode != "config":
        failure = ValueError(
            "Expected shape (8, 4) but received shape (8, 1) for parameter embed.weight"
            if mode == "hub_debug"
            else "Received 1 parameters not in model: \nforeign.weight."
        )

    class FailingEngine:
        _loaded = False

        async def start(self):
            if mode != "config":
                with typed_weight_boundary():
                    raise failure
            raise failure

    monkeypatch.setattr(server, "_engine", FailingEngine())
    monkeypatch.setattr(server, "_model_alias", ref)
    monkeypatch.setattr(server, "_primary_model_lifecycle", None)
    monkeypatch.setattr(server, "_primary_lazy_load", False)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 0)
    monkeypatch.setattr(server, "_gc_control", False)
    monkeypatch.setattr(agents, "start_agent_service_lifecycle", lambda: None)
    monkeypatch.setattr(
        _signal_observability, "install_signal_observability", lambda: None
    )
    stages = []
    emitted = []
    monkeypatch.setattr(
        server_start, "failed", lambda stage, **_kw: stages.append(stage)
    )
    monkeypatch.setattr(
        track, "would_accept", lambda event, props: emitted.append(props)
    )

    debug = mode == "hub_debug"
    caplog.set_level("DEBUG" if debug else "INFO", logger="uvicorn.error")
    lifespan = LifespanOn(Config(FastAPI(lifespan=server.lifespan), log_config=None))
    await lifespan.startup()
    assert lifespan.should_exit
    assert stages == ["engine_start"]
    (event,) = emitted
    assert event["failure_stage"] == "engine_start"
    assert ("Traceback (most recent call last)" in caplog.text) is debug
    if mode != "config":
        assert event["error_class"] == "incompatible_weights"
        assert "failure_subtype" not in event
        assert capsys.readouterr().err.splitlines() == [
            "  Cause: The checkpoint's tensor names or shapes do not match the model Rapid-MLX loaded.",
            "  Next step: Use a compatible MLX conversion of the same model, or run `rapid-mlx models` to find a supported build.",
        ]
        return
    assert capsys.readouterr().err.splitlines() == [
        "  Cause: config.json is missing required VLM text_config fields.",
        "  Next step: Copy a complete config.json from the matching VLM checkpoint.",
    ]


def test_memory_tier_recommendation_returns_none_when_ram_unknown(monkeypatch):
    from rapid_mlx import recommendations

    monkeypatch.setattr(recommendations, "physical_ram_gb", lambda: 0.0)
    assert recommendations.memory_tier_recommendation() is None
    assert recommendations.memory_tier_recommendation(ram_gb=0) is None
    assert (
        recommendations.memory_tier_recommendation(ram_gb=-1, exclude="qwen3.5-9b-4bit")
        is None
    )
