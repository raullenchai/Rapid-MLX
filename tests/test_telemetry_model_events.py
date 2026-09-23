# SPDX-License-Identifier: Apache-2.0
"""Model-event invariants and loopback wire contract."""

from __future__ import annotations

import errno
import http.client
import json
import threading
import urllib.error
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import requests

import rapid_mlx
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
