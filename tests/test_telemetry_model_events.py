# SPDX-License-Identifier: Apache-2.0
"""Model-event invariants and loopback wire contract."""

from __future__ import annotations

import errno
import http.client
import json
import sys
import threading
import urllib.error
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
import requests

import rapid_mlx
from rapid_mlx.model_load_errors import (
    IncompatibleWeights,
    InvalidModelConfig,
    QuantizationMismatch,
    TokenizerLoadFailed,
    load_mlx_lm_checked,
    load_model_checked,
    load_tokenizer_checked,
    load_weights_checked,
    quantize_checked,
    typed_quantization_boundary,
    typed_weight_boundary,
    validate_model_config_file,
)
from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing
from rapid_mlx.telemetry import (
    consent_runtime,
    envelope,
    model_events,
    model_id,
    posthog_sender,
    state,
    store,
)
from rapid_mlx.telemetry import track as track_module
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts

STAMP = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
INSTALL_ID = "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
SESSION_ID = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
ITEM_ID = uuid.UUID("12345678-1234-5678-9234-567812345678")
FACTS = PlatformFacts(
    os="darwin",
    os_version="25.3",
    arch="arm64",
    chip="m3-ultra",
    memory_gb=64,
    python_version="3.11",
)


@pytest.fixture(autouse=True)
def isolated_model_events(monkeypatch, tmp_path):
    for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    monkeypatch.setattr(track_module.common_props, "read_platform_facts", lambda: FACTS)
    monkeypatch.setattr(state, "get_or_create_client_id", lambda: INSTALL_ID)
    monkeypatch.setattr(state, "session_id", lambda: SESSION_ID)
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(posthog_sender.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(
        model_events,
        "_submit_model_served",
        lambda callback: (callback(), True)[1],
    )
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: "7-29"
    )
    track_module._reset_for_tests()
    model_events._reset_for_tests()
    posthog_sender._reset_for_tests()
    state.set_cli_kill_switch(False)
    yield
    posthog_sender._reset_for_tests()
    track_module._reset_for_tests()
    model_events._reset_for_tests()
    state.set_cli_kill_switch(False)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (None, "unknown"),
        (-1, "unknown"),
        (0, "lt_1gb"),
        (1024**3, "1_2gb"),
        (64 * 1024**3, "64gb_plus"),
    ],
)
def test_size_bucket_is_closed_and_half_open(size, expected):
    assert model_events.size_bucket(size) == expected


def test_model_served_submission_failure_is_contained(monkeypatch):
    monkeypatch.setattr(
        model_events,
        "_submit_model_served",
        lambda _callback: (_ for _ in ()).throw(RuntimeError("submit failed")),
    )
    assert model_events.emit_model_served(None, "sdxl-base", False) is False


def test_pull_error_classes_are_type_based():
    from huggingface_hub.errors import HfHubHTTPError
    from huggingface_hub.utils import (
        GatedRepoError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    response = httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co/org/model")
    )
    server_error = httpx.Response(
        503, request=httpx.Request("GET", "https://huggingface.co/org/model")
    )
    assert (
        model_events.pull_error_class(RepositoryNotFoundError("x", response=response))
        == "not_found"
    )
    assert (
        model_events.pull_error_class(GatedRepoError("x", response=response)) == "gated"
    )
    assert model_events.pull_error_class(OSError(errno.ENOSPC, "x")) == "disk_full"
    assert model_events.pull_error_class(urllib.error.URLError("x")) == "network"
    assert model_events.pull_error_class(TimeoutError()) == "network"
    assert (
        model_events.pull_error_class(HfHubHTTPError("x", response=server_error))
        == "other"
    )
    for exc in (
        LocalEntryNotFoundError("no cached snapshot"),
        requests.ConnectionError("private detail"),
        requests.ConnectTimeout("private detail"),
        requests.ReadTimeout("private detail"),
        httpx.ConnectError("private detail"),
    ):
        assert model_events.pull_error_class(exc) == "network"
    wrapped = RuntimeError("outer private detail")
    wrapped.__cause__ = requests.ConnectionError("inner private detail")
    assert model_events.pull_error_class(wrapped) == "network"
    contextual = RuntimeError("outer private detail")
    contextual.__context__ = requests.ReadTimeout("inner private detail")
    assert model_events.pull_error_class(contextual) == "other"
    cyclic = RuntimeError("cycle")
    cyclic.__cause__ = cyclic
    assert model_events.pull_error_class(cyclic) == "other"
    assert model_events.pull_error_class(ValueError("x")) == "other"


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(401, "gated"), (404, "not_found"), (500, "other")],
)
def test_pull_error_class_classifies_urllib_http_errors(status_code, expected):
    failure = urllib.error.HTTPError(
        "https://huggingface.co/org/model", status_code, "private", {}, None
    )

    assert model_events.pull_error_class(failure) == expected


def test_pull_error_class_treats_httpx_read_error_as_network():
    assert model_events.pull_error_class(httpx.ReadError("private detail")) == "network"


def test_pull_error_class_chain_regression():
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    response = httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co/org/model")
    )
    outer = RuntimeError("loader wrapper")
    outer.__context__ = RepositoryNotFoundError("context", response=response)
    middle = RuntimeError("explicit wrapper")
    outer.__cause__ = middle
    middle.__cause__ = GatedRepoError("cause", response=response)

    # Only the explicit cause chain participates; the stale context is ignored.
    assert model_events.pull_error_class(outer) == "gated"


def test_pull_error_event_never_sends_exception_text(monkeypatch):
    calls = []
    monkeypatch.setattr(
        track_module,
        "track",
        lambda event, props: calls.append((event, props)),
    )
    model_events.emit_model_pull_failed(
        requests.ConnectionError("token=secret-hostname"), model_ref="tmax-9b"
    )
    assert calls[0][1]["error_class"] == "network"
    assert "secret" not in json.dumps(calls)


def test_model_type_uses_only_profile_modality(monkeypatch):
    profiles = iter(
        (
            SimpleNamespace(modality="text", supports_image_input=False),
            SimpleNamespace(modality="text", supports_image_input=True),
            SimpleNamespace(modality="image-gen", supports_image_input=False),
            SimpleNamespace(modality="rogue", supports_image_input=False),
            None,
        )
    )
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.resolve_profile", lambda _name: next(profiles)
    )
    assert model_events.model_type("a") == "llm"
    assert model_events.model_type("b") == "vlm"
    assert model_events.model_type("c") == "image-gen"
    assert model_events.model_type("d") == "other"
    assert model_events.model_type("e") == "other"
    assert model_events.model_type(None) == "other"


@pytest.mark.parametrize(
    ("model", "expected"),
    [("kokoro", "audio"), ("cogvideox-fun-5b-q4", "video-gen")],
)
def test_model_type_reaches_registered_non_text_modalities(model, expected):
    assert model_events.model_type(model) == expected


def test_model_served_preserves_image_alias_modality(monkeypatch):
    calls = []
    monkeypatch.setattr(store, "note_model_served", lambda _model: 1)
    monkeypatch.setattr(
        track_module,
        "track",
        lambda event, props, **kwargs: calls.append((event, props, kwargs)),
    )

    model_events.emit_model_served(object(), "sdxl-base", False)

    assert calls[0][0] == "model_served"
    assert calls[0][1]["model_type"] == "image-gen"


def test_model_pulled_registry_requires_infallible_model_type():
    registry = json.loads(
        (Path(rapid_mlx.__file__).parent / "telemetry" / "events.json").read_text()
    )
    props = registry["events"]["model_pulled"]["props"]
    assert props["model_type"]["required"] is True
    assert props["size_bucket"]["required"] is False


def test_model_type_fails_closed_on_bad_profile(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.resolve_profile",
        lambda _name: (_ for _ in ()).throw(ValueError("bad registry")),
    )
    assert model_events.model_type("catalog-entry") == "other"


def _serve_exception(error_class):
    typed = {
        "invalid_config": InvalidModelConfig,
        "tokenizer_load_failed": TokenizerLoadFailed,
        "incompatible_weights": IncompatibleWeights,
        "quantization_mismatch": QuantizationMismatch,
    }
    if error_class in typed:
        return typed[error_class]("typed loader failure")
    if error_class == "insufficient_memory":
        return MemoryError()
    if error_class == "download_failed":
        return FileNotFoundError("model-00001-of-00002.safetensors")
    if error_class == "unsupported_architecture":
        return ValueError("Model type future_arch not supported.")
    if error_class == "corrupt_weights":
        return RuntimeError("size mismatch for shard")
    return RuntimeError("unclassified")


def _chain_serve_exception(inner, shape):
    if shape == "bare":
        return inner
    if shape == "cause":
        outer = RuntimeError("loader wrapper")
        outer.__cause__ = inner
        return outer
    outer = RuntimeError("loader wrapper")
    middle = RuntimeError("second loader wrapper")
    outer.__cause__ = middle
    middle.__cause__ = inner
    return outer


@pytest.mark.parametrize(
    "error_class",
    [
        "insufficient_memory",
        "download_failed",
        "unsupported_architecture",
        "corrupt_weights",
        "invalid_config",
        "tokenizer_load_failed",
        "incompatible_weights",
        "quantization_mismatch",
        "other",
    ],
)
@pytest.mark.parametrize("shape", ["bare", "cause", "two_levels_deep"])
def test_serve_error_classes_across_exception_chain(error_class, shape):
    exc = _chain_serve_exception(_serve_exception(error_class), shape)

    assert model_events.serve_error_class(exc) == error_class


def test_serve_error_class_ignores_implicit_context():
    try:
        raise FileNotFoundError("optional tokenizer probe")
    except FileNotFoundError:
        try:
            raise ValueError("malformed tokenizer config")
        except ValueError as terminal:
            exc = terminal

    assert exc.__suppress_context__ is False
    assert model_events.serve_error_class(exc) == "other"


@pytest.mark.parametrize(
    "error_class",
    [
        "insufficient_memory",
        "download_failed",
        "unsupported_architecture",
        "corrupt_weights",
        "invalid_config",
        "tokenizer_load_failed",
        "incompatible_weights",
        "quantization_mismatch",
        "other",
    ],
)
def test_serve_error_class_terminates_on_cycles(error_class):
    outer = RuntimeError("loader wrapper")
    inner = _serve_exception(error_class)
    outer.__cause__ = inner
    inner.__cause__ = outer

    assert model_events.serve_error_class(outer) == error_class


@pytest.mark.parametrize(
    ("outer_class", "inner_class"),
    [
        ("insufficient_memory", "corrupt_weights"),
        ("download_failed", "insufficient_memory"),
        ("unsupported_architecture", "download_failed"),
        ("corrupt_weights", "unsupported_architecture"),
    ],
)
def test_serve_error_class_outermost_match_wins(outer_class, inner_class):
    outer = _serve_exception(outer_class)
    middle = RuntimeError("second loader wrapper")
    outer.__cause__ = middle
    middle.__cause__ = _serve_exception(inner_class)

    assert model_events.serve_error_class(outer) == outer_class


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_serve_error_class_base_exceptions_are_other(exc):
    assert model_events.serve_error_class(exc) == "other"


def test_serve_error_class_handles_hostile_exception_text():
    class HostileError(Exception):
        def __str__(self):
            raise KeyboardInterrupt

    assert model_events.serve_error_class(HostileError()) == "other"


def test_serve_error_class_stops_at_chain_bound():
    outer = RuntimeError("loader wrapper 0")
    current = outer
    for index in range(40):
        cause = RuntimeError(f"loader wrapper {index + 1}")
        current.__cause__ = cause
        current = cause
    current.__cause__ = MemoryError()

    assert model_events.serve_error_class(outer) == "other"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            ModuleNotFoundError(
                "No module named 'mlx_lm.models.future_arch'",
                name="mlx_lm.models.future_arch",
            ),
            "unsupported_architecture",
        ),
        (
            ModuleNotFoundError("No module named 'mlx_lm.models.future_arch'"),
            "unsupported_architecture",
        ),
        (ModuleNotFoundError("No module named 'optional_accelerator'"), "other"),
        (RuntimeError("corrupt safetensor header"), "corrupt_weights"),
    ],
)
def test_serve_error_class_preserves_existing_variants(exc, expected):
    assert model_events.serve_error_class(exc) == expected


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            ValueError(
                "The checkpoint you are trying to load has model type `future_arch` "
                "but Transformers does not recognize this architecture."
            ),
            "unsupported_architecture",
        ),
        (
            RuntimeError("scoped_pymalloc(): could not allocate 4096 bytes of memory!"),
            "insufficient_memory",
        ),
    ],
)
def test_serve_error_class_recognizes_engine_start_wording(exc, expected):
    assert model_events.serve_error_class(exc) == expected


def test_typed_quantization_beats_memory_wording():
    typed = QuantizationMismatch(
        "[quantized_matmul] out of memory while checking uint32 weights"
    )
    outer = RuntimeError("scoped_pymalloc(): could not allocate 4096 bytes")
    outer.__cause__ = typed

    assert model_events.serve_error_class(outer) == "quantization_mismatch"


def test_could_not_allocate_is_not_a_generic_oom_marker():
    exc = RuntimeError("plugin could not allocate tokenizer ID 7")

    assert model_events.serve_error_class(exc) == "other"


def test_invalid_config_boundary_is_classified(tmp_path, monkeypatch):
    from rapid_mlx.utils import tokenizer

    model_dir = tmp_path / "bad-config"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":', encoding="utf-8")
    monkeypatch.setattr(tokenizer, "_resolve_subfolder_checkpoint", lambda value: value)
    monkeypatch.setattr(tokenizer, "_local_snapshot_if_cached", lambda value: value)
    monkeypatch.setattr(tokenizer, "_resolve_model_path", lambda _value: None)

    with pytest.raises(InvalidModelConfig) as raised:
        tokenizer.load_model_with_fallback(str(model_dir))

    assert isinstance(raised.value.__cause__, json.JSONDecodeError)
    assert model_events.serve_error_class(raised.value) == "invalid_config"


def test_config_boundary_handles_non_model_paths_and_invalid_shapes(tmp_path):
    assert validate_model_config_file(tmp_path / "remote-repo-id") is None
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert validate_model_config_file(empty_dir) is None

    config_path = empty_dir / "config.json"
    config_path.write_text("[]", encoding="utf-8")
    with pytest.raises(InvalidModelConfig, match="top-level value"):
        validate_model_config_file(empty_dir)

    config_path.write_text(json.dumps({"model_type": ""}), encoding="utf-8")
    with pytest.raises(InvalidModelConfig, match="non-empty string"):
        validate_model_config_file(empty_dir)

    config_path.write_text(json.dumps({"model_file": ""}), encoding="utf-8")
    with pytest.raises(InvalidModelConfig, match="model_file must be"):
        validate_model_config_file(empty_dir)


def test_config_boundary_preserves_existing_typed_failure(tmp_path, monkeypatch):
    model_dir = tmp_path / "typed-config"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    failure = InvalidModelConfig("already classified")
    monkeypatch.setattr(json, "load", lambda _file: (_ for _ in ()).throw(failure))

    with pytest.raises(InvalidModelConfig) as raised:
        validate_model_config_file(model_dir)

    assert raised.value is failure


def test_tokenizer_load_boundary_is_classified():
    def load_invalid_tokenizer():
        raise ValueError("tokenizer.json has an invalid model section")

    with pytest.raises(TokenizerLoadFailed) as raised:
        load_tokenizer_checked(load_invalid_tokenizer)

    assert isinstance(raised.value.__cause__, ValueError)
    assert model_events.serve_error_class(raised.value) == "tokenizer_load_failed"


def test_tokenizer_wrapper_classifies_local_file_failure():
    missing = FileNotFoundError("tokenizer.json")

    def load_missing_tokenizer():
        raise missing

    with pytest.raises(TokenizerLoadFailed) as raised:
        load_tokenizer_checked(load_missing_tokenizer)

    assert raised.value.__cause__ is missing
    assert model_events.serve_error_class(raised.value) == "tokenizer_load_failed"


@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("missing runtime"),
        OptionalRuntimeMissing(
            extra="audio",
            install_hint="pip install rapid-mlx[audio]",
            detail="audio runtime is missing",
            status="absent",
        ),
    ],
)
def test_tokenizer_wrapper_preserves_runtime_availability_failures(failure):
    def fail():
        raise failure

    with pytest.raises(type(failure)) as raised:
        load_tokenizer_checked(fail)

    assert raised.value is failure


def test_tokenizer_wrapper_preserves_remote_hub_failure():
    from huggingface_hub.utils import RepositoryNotFoundError

    response = httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co/org/missing")
    )
    failure = RepositoryNotFoundError("missing", response=response)

    with pytest.raises(RepositoryNotFoundError) as raised:
        load_tokenizer_checked(lambda: (_ for _ in ()).throw(failure))

    assert raised.value is failure


@pytest.mark.parametrize(
    "boundary,failure",
    [
        (typed_weight_boundary, IncompatibleWeights("already typed")),
        (typed_quantization_boundary, QuantizationMismatch("already typed")),
    ],
)
def test_typed_model_boundaries_preserve_existing_failure(boundary, failure):
    with pytest.raises(type(failure)) as raised, boundary():
        raise failure

    assert raised.value is failure


def test_weight_load_boundary_is_classified():
    class ShapeCheckingModel:
        def load_weights(self, weights, *, strict):
            assert strict is True
            shape = dict(weights)["model.embed_tokens.weight"]["shape"]
            if shape != (32, 16):
                raise ValueError(f"expected shape (32, 16), got {shape}")

    bad_weights = {"model.embed_tokens.weight": {"shape": (31, 16)}}
    model = ShapeCheckingModel()

    with pytest.raises(IncompatibleWeights) as raised:
        load_weights_checked(model, bad_weights, strict=True)

    assert isinstance(raised.value.__cause__, ValueError)
    assert model_events.serve_error_class(raised.value) == "incompatible_weights"
    load_weights_checked(
        model,
        {"model.embed_tokens.weight": {"shape": (32, 16)}},
        strict=True,
    )


def test_quantization_boundary_is_classified():
    def apply_quantization(config):
        if (config["bits"], config["group_size"], config["dtype"]) != (
            4,
            64,
            "uint32",
        ):
            raise ValueError("quantized weight metadata does not match the model")

    with pytest.raises(QuantizationMismatch) as raised:
        quantize_checked(
            apply_quantization,
            {"bits": 8, "group_size": 128, "dtype": "bfloat16"},
        )

    assert isinstance(raised.value.__cause__, ValueError)
    assert model_events.serve_error_class(raised.value) == "quantization_mismatch"
    assert (
        quantize_checked(
            lambda config: config["bits"],
            {"bits": 4, "group_size": 64, "dtype": "uint32"},
        )
        == 4
    )


def test_generic_model_loader_types_quantization_boundary(tmp_path, monkeypatch):
    model_dir = tmp_path / "bad-quantization"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "synthetic"}), encoding="utf-8"
    )

    nn = ModuleType("mlx.nn")

    class Module:
        def load_weights(self, *_args, **_kwargs):
            return None

    def quantize():
        raise ValueError("group_size does not divide the weight shape")

    nn.Module = Module
    nn.quantize = quantize
    mlx = ModuleType("mlx")
    mlx.nn = nn
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)

    def loader(_model_path):
        nn.quantize()

    with pytest.raises(QuantizationMismatch) as raised:
        load_model_checked(loader, model_dir)

    assert model_events.serve_error_class(raised.value) == "quantization_mismatch"


def test_generic_model_loader_preserves_unclassified_value_error(tmp_path):
    model_dir = tmp_path / "unsupported-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "future_arch"}), encoding="utf-8"
    )

    def loader(_model_path):
        raise ValueError("Model type future_arch not supported.")

    with pytest.raises(ValueError, match="not supported") as raised:
        load_model_checked(loader, model_dir)

    assert model_events.serve_error_class(raised.value) == "unsupported_architecture"


def test_generic_eager_loader_separates_tokenizer_boundary(tmp_path, monkeypatch):
    model_dir = tmp_path / "generic-loader"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "synthetic"}), encoding="utf-8"
    )
    model = object()
    tokenizer = object()
    utils = ModuleType("mlx_lm.utils")
    utils._download = lambda _name: (_ for _ in ()).throw(
        AssertionError("local checkpoints must not be sent to the Hub downloader")
    )
    utils.load_model = lambda _path, **_kwargs: (model, {"eos_token_id": [1, 2]})

    def load_tokenizer(_path, _config, *, eos_token_ids):
        assert eos_token_ids == [1, 2]
        return tokenizer

    utils.load_tokenizer = load_tokenizer
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)

    assert load_mlx_lm_checked(str(model_dir), {"legacy": False}) == (
        model,
        tokenizer,
    )


def test_generic_eager_loader_normalizes_missing_tokenizer_config(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "generic-loader"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "synthetic"}), encoding="utf-8"
    )
    model = object()
    tokenizer = object()
    utils = ModuleType("mlx_lm.utils")
    utils._download = lambda _name: model_dir
    utils.load_model = lambda _path, **_kwargs: (model, {})

    def load_tokenizer(_path, config, *, eos_token_ids):
        assert config == {}
        assert eos_token_ids is None
        return tokenizer

    utils.load_tokenizer = load_tokenizer
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)

    assert load_mlx_lm_checked(str(model_dir)) == (model, tokenizer)


def _prepare_generic_tokenizer_dispatch(monkeypatch):
    from rapid_mlx.utils import tokenizer

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    gemma = ModuleType("rapid_mlx.models.gemma4_text")
    gemma.gemma4_load_plan = lambda _name: (None, False)
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.gemma4_text", gemma)
    monkeypatch.setattr(tokenizer, "_register_vendored_archs", lambda: None)
    monkeypatch.setattr(tokenizer, "_needs_tokenizer_fallback", lambda _name: False)
    monkeypatch.setattr(tokenizer, "_is_vendored_arch_model", lambda _name: False)
    monkeypatch.setattr(
        tokenizer, "_neutralize_unbundled_template_types", lambda _name, cfg: cfg
    )
    return tokenizer


def test_generic_tokenizer_dispatch_uses_typed_eager_loader(monkeypatch):
    tokenizer = _prepare_generic_tokenizer_dispatch(monkeypatch)
    model = object()
    loaded_tokenizer = SimpleNamespace(chat_template="template")
    monkeypatch.setattr(
        tokenizer,
        "load_mlx_lm_checked",
        lambda *_args, **_kwargs: (model, loaded_tokenizer),
    )
    monkeypatch.setattr(tokenizer, "_try_inject_mtp_post_load", lambda *_args: None)
    monkeypatch.setattr(
        tokenizer, "augment_eos_token_ids_from_generation_config", lambda *_args: None
    )
    monkeypatch.setattr(tokenizer, "repair_byte_level_decoder", lambda *_args: None)

    assert tokenizer._load_model_with_fallback_impl("org/model", {}) == (
        model,
        loaded_tokenizer,
    )


@pytest.mark.parametrize(
    ("failure", "fallback_name"),
    [
        (
            TokenizerLoadFailed("tokenizer failed"),
            "tokenizer",
        ),
        (
            IncompatibleWeights("weights failed"),
            "weights",
        ),
    ],
)
def test_generic_tokenizer_dispatch_preserves_existing_fallbacks(
    monkeypatch, failure, fallback_name
):
    tokenizer = _prepare_generic_tokenizer_dispatch(monkeypatch)
    cause = ValueError(
        "Tokenizer class is unavailable"
        if fallback_name == "tokenizer"
        else "Missing parameters in model"
    )
    failure.__cause__ = cause

    def fail_load(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(tokenizer, "load_mlx_lm_checked", fail_load)
    monkeypatch.setattr(
        tokenizer,
        "_load_with_tokenizer_fallback",
        lambda *_args, **_kwargs: ("fallback-model", "fallback-tokenizer"),
    )
    monkeypatch.setattr(
        tokenizer,
        "_load_strict_false",
        lambda *_args, **_kwargs: ("loose-model", "loose-tokenizer"),
    )

    expected = (
        ("fallback-model", "fallback-tokenizer")
        if fallback_name == "tokenizer"
        else ("loose-model", "loose-tokenizer")
    )
    assert tokenizer._load_model_with_fallback_impl("org/model", {}) == expected


def test_strict_false_loader_uses_typed_boundaries(tmp_path, monkeypatch):
    from rapid_mlx.utils import tokenizer

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "synthetic"}), encoding="utf-8"
    )
    model = object()
    loaded_tokenizer = object()
    utils = ModuleType("mlx_lm.utils")
    utils.load_model = lambda _path, **_kwargs: (model, {"eos_token_id": 7})
    utils.load_tokenizer = lambda *_args, **_kwargs: loaded_tokenizer
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)
    monkeypatch.setattr(tokenizer, "_try_inject_mtp", lambda *_args: None)
    monkeypatch.setattr(tokenizer, "_apply_chat_template_sidecar", lambda *_args: None)
    monkeypatch.setattr(
        tokenizer, "augment_eos_token_ids_from_generation_config", lambda *_args: None
    )
    monkeypatch.setattr(tokenizer, "repair_byte_level_decoder", lambda *_args: None)

    assert tokenizer._load_strict_false(str(tmp_path), {}) == (
        model,
        loaded_tokenizer,
    )


def test_raw_tokenizer_fallback_uses_typed_boundaries(tmp_path, monkeypatch):
    from rapid_mlx.utils import tokenizer

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "synthetic"}), encoding="utf-8"
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    model = object()
    loaded_tokenizer = SimpleNamespace(chat_template=None)
    utils = ModuleType("mlx_lm.utils")
    utils.load_model = lambda _path, **_kwargs: (model, {})
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)
    fp8 = ModuleType("rapid_mlx.fp8_repack")
    fp8.is_fp8_block_checkpoint = lambda _path: False
    fp8.load_fp8_model_online = lambda _path: None
    monkeypatch.setitem(sys.modules, "rapid_mlx.fp8_repack", fp8)
    tokenizers = ModuleType("tokenizers")
    tokenizers.Tokenizer = SimpleNamespace(from_file=lambda _path: object())
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    transformers = ModuleType("transformers")
    tokenizer_kwargs = []

    def build_tokenizer(**kwargs):
        tokenizer_kwargs.append(kwargs)
        return loaded_tokenizer

    transformers.PreTrainedTokenizerFast = build_tokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(tokenizer, "_register_vendored_archs", lambda: None)
    monkeypatch.setattr(
        tokenizer,
        "_deepseek_v4_quantization_override",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(tokenizer, "_uses_rapid_owned_runtime", lambda _path: False)
    monkeypatch.setattr(tokenizer, "_apply_chat_template_sidecar", lambda *_args: False)
    monkeypatch.setattr(tokenizer, "_needs_tokenizer_fallback", lambda _name: False)
    monkeypatch.setattr(tokenizer, "repair_byte_level_decoder", lambda *_args: None)
    monkeypatch.setattr(
        tokenizer, "augment_eos_token_ids_from_generation_config", lambda *_args: None
    )

    assert tokenizer._load_with_tokenizer_fallback(str(tmp_path)) == (
        model,
        loaded_tokenizer,
    )

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": {"content": "<eos>"},
                "unk_token": "<unknown>",
                "pad_token": "<padding>",
                "chat_template": "{{ messages }}",
            }
        ),
        encoding="utf-8",
    )
    assert tokenizer._load_with_tokenizer_fallback(str(tmp_path)) == (
        model,
        loaded_tokenizer,
    )
    assert tokenizer_kwargs[-1] == {
        "tokenizer_object": tokenizer_kwargs[-1]["tokenizer_object"],
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unknown>",
        "pad_token": "<padding>",
    }
    assert loaded_tokenizer.chat_template == "{{ messages }}"

    (tmp_path / "tokenizer_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(TokenizerLoadFailed, match="must contain an object"):
        tokenizer._load_with_tokenizer_fallback(str(tmp_path))

    (tmp_path / "tokenizer_config.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(TokenizerLoadFailed) as raised:
        tokenizer._load_with_tokenizer_fallback(str(tmp_path))
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)


def test_serve_download_error_class():
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(
        500, request=httpx.Request("GET", "https://huggingface.co/org/model")
    )
    assert (
        model_events.serve_error_class(HfHubHTTPError("x", response=response))
        == "download_failed"
    )


def test_pull_failed_includes_optional_size_bucket(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        track_module,
        "track",
        lambda event, props: calls.append((event, props)),
    )
    model_events.emit_model_pull_failed(TimeoutError(), size_bytes=1024**3)
    assert calls == [
        (
            "model_pull_failed",
            {"error_class": "network", "size_bucket": "1_2gb"},
        )
    ]


def test_model_pulled_omits_unknown_snapshot_size(monkeypatch):
    calls = []
    monkeypatch.setattr(
        track_module,
        "track",
        lambda event, props: calls.append((event, props)),
    )

    model_events.emit_model_pulled("tmax-9b", "hf", None)

    assert calls == [
        (
            "model_pulled",
            {
                "model": "tmax-9b",
                "model_type": "llm",
                "source": "hf",
            },
        )
    ]


def test_model_served_is_only_note_site_and_maps_zero_to_none(monkeypatch):
    calls: list[tuple[str, dict[str, object], int | None]] = []
    alias_or_path = "/Users/secret/acme-internal-ft"
    monkeypatch.setattr(model_id, "engine_telemetry_id", lambda _engine: "<local>")
    monkeypatch.setattr(
        model_events,
        "model_type",
        lambda name: "llm" if name == alias_or_path else "other",
    )
    noted_models = []

    def note_model_served(model):
        noted_models.append(model)
        return 0

    monkeypatch.setattr(store, "note_model_served", note_model_served)
    monkeypatch.setattr(
        track_module,
        "track",
        lambda event, props, *, nth_model_served=None: calls.append(
            (event, props, nth_model_served)
        ),
    )
    model_events.emit_model_served(object(), alias_or_path, True)
    assert noted_models == ["<local>"]
    assert calls == [
        (
            "model_served",
            {
                "model": "<local>",
                "model_type": "llm",
                "auto_selected": True,
                "quant": "unknown",
            },
            None,
        )
    ]


@pytest.mark.parametrize("official", [False, True])
def test_model_served_ineligible_process_never_creates_store(
    monkeypatch, tmp_path, official
):
    monkeypatch.setattr(
        track_module.build_gate,
        "official_build",
        (lambda: STAMP) if official else (lambda: None),
    )
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: not official)

    model_events.emit_model_served(None, "tmax-9b", False)

    assert not store.db_path().exists()


def test_served_quant_prefers_resolved_hf_path(monkeypatch):
    calls = []
    monkeypatch.setattr(model_id, "engine_telemetry_id", lambda _engine: "tmax-9b")
    monkeypatch.setattr(store, "note_model_served", lambda _model: 1)
    monkeypatch.setattr(
        track_module,
        "track",
        lambda event, props, **kwargs: calls.append((event, props, kwargs)),
    )

    model_events.emit_model_served(object(), "tmax-9b", False)

    assert calls[0][1]["quant"] == "4bit"


def test_served_quant_falls_back_to_alias_when_profile_resolution_fails(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.resolve_profile",
        lambda _name: (_ for _ in ()).throw(RuntimeError("catalog unavailable")),
    )
    assert model_events._quant_for_ref("qwen3.5-4b-4bit") == "4bit"


@pytest.mark.parametrize("auto_selected", [False, True])
def test_auto_selected_is_not_hardcoded_on_success(monkeypatch, auto_selected):
    calls = []
    monkeypatch.setattr(model_id, "engine_telemetry_id", lambda _engine: "<custom>")
    monkeypatch.setattr(store, "note_model_served", lambda _model: 1)
    monkeypatch.setattr(
        track_module,
        "track",
        lambda _event, props, **_kwargs: calls.append(props),
    )
    model_events.emit_model_served(object(), "unknown", auto_selected)
    assert calls[0]["auto_selected"] is auto_selected


@pytest.mark.parametrize("auto_selected", [False, True])
def test_auto_selected_is_not_hardcoded_on_failure(monkeypatch, auto_selected):
    calls = []
    monkeypatch.setattr(
        track_module, "track", lambda _event, props: calls.append(props)
    )
    model_events.emit_model_serve_failed(
        RuntimeError("load"), alias_or_path="unknown", auto_selected=auto_selected
    )
    assert calls[0]["auto_selected"] is auto_selected
    model_events._reset_for_tests()


def test_failure_uses_only_privacy_reduced_model_on_wire(monkeypatch, tmp_path):
    hostile = str(tmp_path / "alice-secret" / "weights")
    calls = []
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(props))
    model_events.emit_model_serve_failed(RuntimeError("load"), alias_or_path=hostile)
    assert calls[0]["model"] == "<local>"
    assert hostile not in repr(calls)


def test_failure_prefers_engine_telemetry_identity(monkeypatch):
    calls = []
    engine = object()
    monkeypatch.setattr(model_id, "engine_telemetry_id", lambda value: "tmax-9b")
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(props))
    model_events.emit_model_serve_failed(RuntimeError("load"), engine=engine)
    assert calls == [{"error_class": "other", "model": "tmax-9b"}]


def test_optional_runtime_failure_class_and_extra_are_structured(monkeypatch):
    from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing

    calls = []
    failure = OptionalRuntimeMissing(
        extra="vision",
        install_hint="pip install 'rapid-mlx[vision]'",
        detail="private diagnostic detail",
        status="broken",
    )
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(props))

    model_events.emit_model_serve_failed(failure)

    assert model_events.serve_error_class(failure) == "missing_extra"
    assert calls == [{"error_class": "missing_extra", "extra": "vision"}]
    assert "private diagnostic detail" not in repr(calls)


def test_wrapped_optional_runtime_failure_preserves_class_and_extra(monkeypatch):
    from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing

    calls = []
    missing = OptionalRuntimeMissing(
        extra="vision",
        install_hint="pip install 'rapid-mlx[vision]'",
        detail="private diagnostic detail",
        status="absent",
    )
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = missing
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(props))

    model_events.emit_model_serve_failed(wrapped)

    assert model_events.serve_error_class(wrapped) == "missing_extra"
    assert calls == [{"error_class": "missing_extra", "extra": "vision"}]
    assert "private diagnostic detail" not in repr(calls)


def test_optional_runtime_lookup_tolerates_hostile_cause_access():
    class HostileCauseError(RuntimeError):
        def __getattribute__(self, name):
            if name == "__cause__":
                raise KeyboardInterrupt
            return super().__getattribute__(name)

    assert model_events.find_optional_runtime_missing(HostileCauseError()) is None


def test_failure_loses_race_after_payload_build_without_emitting(monkeypatch):
    calls = []

    def claim_during_build(_value):
        model_events._serve_failure_claimed = True
        return "other"

    monkeypatch.setattr(model_events, "model_type", claim_during_build)
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(props))
    model_events.emit_model_serve_failed(RuntimeError("load"), alias_or_path="unknown")
    assert calls == []


def test_pull_source_is_argument_driven_and_closed(monkeypatch):
    calls = []
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "hf")
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(props))
    model_events.emit_model_pulled("qwen3.5-4b-4bit", "mirror", 1)
    model_events.emit_model_pulled("qwen3.5-4b-4bit", "environment", 1)
    assert [props["source"] for props in calls] == ["mirror"]


class _UnprintableError(Exception):
    def __str__(self):
        raise RuntimeError("string conversion exploded")


def test_emitters_never_raise_and_failed_latch_is_not_burned(monkeypatch):
    calls = []
    monkeypatch.setattr(
        track_module, "track", lambda event, props, **kw: calls.append(event)
    )
    monkeypatch.setattr(
        model_id,
        "telemetry_model_id",
        lambda value: (
            (_ for _ in ()).throw(_UnprintableError())
            if value == "poison"
            else "<custom>"
        ),
    )
    monkeypatch.setattr(
        model_id,
        "engine_telemetry_id",
        lambda engine: (
            (_ for _ in ()).throw(_UnprintableError())
            if engine == "poison"
            else "<custom>"
        ),
    )

    model_events.emit_model_pulled("poison", "hf", 1)
    model_events.emit_model_pull_failed(RuntimeError("x"), model_ref="poison")
    model_events.emit_model_served("poison", "unknown", False)
    model_events.emit_model_serve_failed(_UnprintableError(), alias_or_path="unknown")
    model_events.emit_model_serve_failed(RuntimeError("valid"), alias_or_path="unknown")

    assert calls == ["model_serve_failed"]


def test_serve_failure_latch_claims_before_building(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        model_events,
        "serve_error_class",
        lambda _exc: calls.append("classify") or "other",
    )
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(event))
    model_events.emit_model_serve_failed(RuntimeError("first"))
    model_events.emit_model_serve_failed(RuntimeError("second"))
    assert calls == ["classify", "model_serve_failed"]


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(self.rfile.read(length))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, _format: str, *_args: object) -> None:
        pass


def test_all_four_model_events_reach_exact_loopback_json(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        posthog_sender.POSTHOG_URL_ENV,
        f"http://127.0.0.1:{server.server_port}/batch/",
    )

    def loopback_post(_url: str, body: bytes, timeout: float) -> int:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=timeout
        )
        try:
            connection.request(
                "POST",
                "/batch/",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            return connection.getresponse().status
        finally:
            connection.close()

    sender = posthog_sender.PostHogSender(
        post=loopback_post, gate=lambda: STAMP, allowed=lambda: True
    )
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    monkeypatch.setattr(envelope.uuid, "uuid4", lambda: ITEM_ID)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 21, 12, 34, 56, tzinfo=timezone.utc)

    monkeypatch.setattr(envelope, "datetime", _FixedDatetime)
    monkeypatch.setattr(
        model_id, "engine_telemetry_id", lambda _engine: "qwen3.5-4b-4bit"
    )
    monkeypatch.setattr(
        model_events,
        "model_type",
        lambda name: "llm" if name == "qwen3.5-4b-4bit" else "other",
    )
    monkeypatch.setattr(store, "note_model_served", lambda _model: 2)

    model_events.emit_model_pulled("qwen3.5-4b-4bit", "mirror", 3 * 1024**3)
    model_events.emit_model_pull_failed(
        TimeoutError(), model_ref="qwen3.5-4b-4bit", source="hf"
    )
    model_events.emit_model_served(object(), "qwen3.5-4b-4bit", True)
    hostile_path = "/Users/alice/private-checkout/weights"
    model_events.emit_model_serve_failed(MemoryError(), alias_or_path=hostile_path)
    sender.flush()
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()

    bodies = server.bodies  # type: ignore[attr-defined]
    events = [item for body in bodies for item in json.loads(body)["batch"]]
    common = {
        "app_version": "0.15.1",
        "surface": "cli",
        "os": "darwin",
        "os_version": "25.3",
        "arch": "arm64",
        "chip": "m3-ultra",
        "memory_gb": 64,
        "python_version": "3.11",
        "install_id": INSTALL_ID,
        "session_id": SESSION_ID,
        "channel": "stable",
        "days_since_first_run_bucket": "7-29",
        "$geoip_disable": True,
        "$process_person_profile": False,
    }
    expected_props = [
        {
            **common,
            "model": "qwen3.5-4b-4bit",
            "model_type": "llm",
            "source": "mirror",
            "size_bucket": "2_4gb",
        },
        {
            **common,
            "model": "qwen3.5-4b-4bit",
            "model_type": "llm",
            "source": "hf",
            "error_class": "network",
        },
        {
            **common,
            "nth_model_served": 2,
            "model": "qwen3.5-4b-4bit",
            "model_type": "llm",
            "auto_selected": True,
            "quant": "4bit",
        },
        {
            **common,
            "model": "<local>",
            "model_type": "other",
            "auto_selected": False,
            "quant": "unknown",
            "error_class": "insufficient_memory",
        },
    ]
    assert events == [
        {
            "uuid": str(ITEM_ID),
            "event": name,
            "distinct_id": INSTALL_ID,
            "timestamp": "2026-09-21T12:34:56Z",
            "properties": props,
        }
        for name, props in zip(
            (
                "model_pulled",
                "model_pull_failed",
                "model_served",
                "model_serve_failed",
            ),
            expected_props,
            strict=True,
        )
    ]
    assert hostile_path not in repr(events)
