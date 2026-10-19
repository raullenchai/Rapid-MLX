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


def test_memory_failure_uses_existing_class_and_remedy(tmp_path):
    from rapid_mlx.local_serve_diagnostics import startup_error_lines

    model = _model(tmp_path)
    failure = MemoryError()
    assert serve_error_class(failure, model_ref=str(model)) == "insufficient_memory"
    assert startup_error_lines(failure, str(model)) == (
        "The model ran out of memory while loading weights.",
        "Free memory, then run `rapid-mlx serve qwen3-0.6b-4bit` for a smaller model.",
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


@pytest.mark.asyncio
async def test_engine_start_prints_two_lines_and_preserves_failure(
    tmp_path, monkeypatch, capsys
):
    from fastapi import FastAPI

    from rapid_mlx import _signal_observability, server
    from rapid_mlx.routes import agents
    from rapid_mlx.telemetry import model_events, server_start

    model = _model(tmp_path, model_type="qwen3_vl")
    failure = TypeError("TextConfig.__init__() missing required arguments")

    class FailingEngine:
        _loaded = False

        async def start(self):
            raise failure

    monkeypatch.setattr(server, "_engine", FailingEngine())
    monkeypatch.setattr(server, "_model_alias", str(model))
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
    monkeypatch.setattr(server_start, "failed", stages.append)
    monkeypatch.setattr(
        model_events,
        "emit_model_serve_failed",
        lambda exc, **kwargs: emitted.append((exc, kwargs)),
    )

    with pytest.raises(TypeError) as caught:
        await anext(server.lifespan(FastAPI()))

    assert caught.value is failure
    assert stages == ["engine_start"]
    assert emitted[0][0] is failure
    assert emitted[0][1]["failure_stage"] == "engine_start"
    assert capsys.readouterr().err.splitlines() == [
        "  Cause: config.json is missing required VLM text_config fields.",
        "  Next step: Copy a complete config.json from the matching VLM checkpoint.",
    ]
