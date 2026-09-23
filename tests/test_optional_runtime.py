# SPDX-License-Identifier: Apache-2.0
"""Optional serving lanes converge on one actionable telemetry failure."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rapid_mlx import cli, server
from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing
from rapid_mlx.telemetry import model_events, registry, server_start


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(self.rfile.read(length))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


def _run_real_missing_extra_dispatch(
    tmp_path: Path,
    *,
    lane: str,
    model: str,
    status: str = "absent",
    standalone: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[SimpleNamespace]]:
    home = tmp_path / "home"
    telemetry_dir = home / ".rapid-mlx"
    telemetry_dir.mkdir(parents=True)
    (telemetry_dir / "telemetry-consent.yaml").write_text(
        "consent: true\nprompted_version: 0.15.1\nnotice_revision_seen: 1\n",
        encoding="utf-8",
    )
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        """
import importlib.util
import os

from rapid_mlx import cli
from rapid_mlx.telemetry import build_gate
from rapid_mlx.telemetry.build_gate import ReleaseStamp

build_gate.official_build = lambda: ReleaseStamp(
    channel="rc", posthog_key="phc_" + "a" * 32
)
cli._port_preflight_or_die = lambda *_args, **_kwargs: None
cli._check_alias_min_memory = lambda *_args, **_kwargs: None
cli._check_disk_space = lambda *_args, **_kwargs: None
cli._check_memory_capacity = lambda *_args, **_kwargs: None
cli._ensure_model_downloaded = lambda *_args, **_kwargs: None

lane = os.environ["RAPID_MLX_TEST_EXTRA_LANE"]
status = os.environ.get("RAPID_MLX_TEST_EXTRA_STATUS", "absent")
if lane in {"vision", "vision-present", "bonsai"}:
    from rapid_mlx.models import mllm

    runtime_status = {
        "absent": mllm.VisionRuntimeStatus.ABSENT,
        "broken": mllm.VisionRuntimeStatus.BROKEN,
        "incompatible": mllm.VisionRuntimeStatus.INCOMPATIBLE,
        "present": mllm.VisionRuntimeStatus.OK,
    }[status]
    mllm.vision_runtime_status = lambda: (runtime_status, "test detail")
if lane == "video":
    from rapid_mlx.runtime import video_lane

    video_lane._default_video_runtime_requirements = lambda _model: [
        "the `rapid-mlx[video]` Python extra"
    ]
    video_lane._resolve_ffmpeg = lambda: "/usr/bin/ffmpeg"
if lane == "image":
    from rapid_mlx.runtime import image_lane
    from rapid_mlx import _download_gate

    image_lane.image_runtime_issue = lambda _model: (
        "image generation requires the `rapid-mlx[image]` Python extra "
        "(`pip install 'rapid-mlx[image]'`)."
    )
    _download_gate.mflux_missing_weights = lambda _model: []
if lane == "audio":
    real_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name: (
        None if name == "mlx_audio" else real_find_spec(name)
    )
if lane == "vision-present":
    from rapid_mlx.telemetry import posthog_sender

    def stop_after_optional_guards(*_args, **_kwargs):
        posthog_sender.get_sender().flush(5.0)
        os._exit(0)

    cli._validate_v41_product_spec_flags = stop_after_optional_guards
""".lstrip(),
        encoding="utf-8",
    )

    sink = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    sink.bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()
    env = dict(
        os.environ,
        HOME=str(home),
        USER="rc",
        PYTHONPATH=os.pathsep.join((str(hooks), str(Path.cwd()))),
        RAPID_MLX_POSTHOG_URL=f"http://127.0.0.1:{sink.server_port}/batch/",
        RAPID_MLX_DISABLE_VERSION_CHECK="1",
        RAPID_MLX_TEST_EXTRA_LANE=lane,
        RAPID_MLX_TEST_EXTRA_STATUS=status,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    for name in (
        "CI",
        "GITHUB_ACTIONS",
        "RAPID_MLX_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        env.pop(name, None)
    if lane == "bonsai":
        model_dir = tmp_path / model
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"model_type":"prism_hadamard_qwen35"}',
            encoding="utf-8",
        )
    command = (
        [sys.executable, "-m", "rapid_mlx.server"]
        if standalone
        else [str(Path(sys.executable).with_name("rapid-mlx"))]
    )
    command.extend(
        ["--model", model, "--port", "0"]
        if standalone
        else ["serve", model, "--port", "0"]
    )
    try:
        proc = subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        sink.shutdown()
        thread.join(timeout=2.0)
        sink.server_close()
    events = [
        SimpleNamespace(
            event=item["event"],
            props=SimpleNamespace(**item["properties"]),
        )
        for body in sink.bodies  # type: ignore[attr-defined]
        for item in json.loads(body)["batch"]
    ]
    return proc, events


def _expected_install_hint_line(extra: str) -> str:
    if extra == "vision":
        return (
            f"    {shlex.quote(sys.executable)} -m pip install "
            "--upgrade --force-reinstall 'rapid-mlx[vision]'"
        )
    if extra == "audio":
        return "Install with: pip install 'rapid-mlx[audio]'"
    if extra == "image":
        return (
            "  Error: image generation requires the `rapid-mlx[image]` Python "
            "extra (`pip install 'rapid-mlx[image]'`)."
        )
    return f"pip install 'rapid-mlx[{extra}]'"


def _assert_actionable_failure_contract(
    stderr: str, *, extra: str, marker_reason: str
) -> tuple[str, str]:
    expected_marker = f"RAPID-MLX-STARTUP-FAILURE: {marker_reason} extra={extra}"
    markers = [
        line
        for line in stderr.splitlines()
        if line.startswith("RAPID-MLX-STARTUP-FAILURE:")
    ]
    assert markers == [expected_marker]

    expected_hint = _expected_install_hint_line(extra)
    assert stderr.splitlines().count(expected_hint) == 1
    return expected_marker, expected_hint


def _normalized_failure_text(stderr: str) -> str:
    lines = stderr.splitlines()
    marker_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("RAPID-MLX-STARTUP-FAILURE:")
    )
    log_prefixes = ("rapid-mlx: anonymous usage reporting", "INFO:", "WARNING:")
    last_log_index = max(
        (
            index
            for index, line in enumerate(lines[:marker_index])
            if line.startswith(log_prefixes)
        ),
        default=-1,
    )
    return "\n".join(lines[last_log_index + 1 : marker_index]).strip()


def _contracted_failure_events(
    events: list[SimpleNamespace],
) -> list[tuple[object, ...]]:
    return [
        (
            event.event,
            getattr(event.props, "state", None),
            getattr(event.props, "failure_stage", None),
            getattr(event.props, "error_class", None),
            getattr(event.props, "extra", None),
        )
        for event in events
        if event.event in {"server_start_state", "model_serve_failed"}
        and (
            event.event == "model_serve_failed"
            or getattr(event.props, "state", None) == "failed"
        )
    ]


@pytest.fixture(autouse=True)
def _reset_one_shot_state():
    server_start._reset_for_tests()
    model_events._reset_for_tests()
    yield
    server_start._reset_for_tests()
    model_events._reset_for_tests()


def _capture(monkeypatch):
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.posthog_sender.install_atexit", lambda: None
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda event, props, **_kwargs: events.append((event, dict(props))),
    )
    return events


def _lane_failure(monkeypatch, lane: str) -> tuple[OptionalRuntimeMissing, str]:
    if lane == "vision":
        from rapid_mlx.models import mllm

        monkeypatch.setattr(
            mllm,
            "vision_runtime_status",
            lambda: (mllm.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
        )
        call = lambda: mllm.require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
        model = "ui-tars-1.5-7b-4bit"
    elif lane == "video":
        from rapid_mlx.runtime import video_lane

        version = namedtuple("Version", "major minor")
        monkeypatch.setattr(video_lane.sys, "version_info", version(3, 11))
        monkeypatch.setattr(video_lane, "_is_ltx25_name", lambda _name: False)
        monkeypatch.setattr(video_lane, "_is_cogvideox_name", lambda _name: False)
        monkeypatch.setattr(
            video_lane,
            "_default_video_runtime_requirements",
            lambda _name: ["the `rapid-mlx[video]` Python extra"],
        )
        monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "ffmpeg")
        call = lambda: video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
        model = "wan2.2-ti2v-5b-q8"
    elif lane == "image":
        from rapid_mlx.runtime import image_lane

        monkeypatch.setattr(
            image_lane,
            "image_runtime_issue",
            lambda _name: (
                "image generation requires the `rapid-mlx[image]` Python extra "
                "(`pip install 'rapid-mlx[image]'`)."
            ),
        )
        call = lambda: image_lane.require_image_runtime_or_exit("flux-schnell")
        model = "flux-schnell"
    else:
        import importlib.util

        from rapid_mlx.audio import probe

        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
        call = lambda: probe.require_audio_or_exit("kokoro")
        model = "kokoro"

    with pytest.raises(OptionalRuntimeMissing) as caught:
        call()
    return caught.value, model


@pytest.mark.parametrize("lane", ["vision", "video", "audio", "image"])
def test_missing_lane_is_one_actionable_telemetered_failure(
    monkeypatch, capsys, lane: str
) -> None:
    failure, model = _lane_failure(monkeypatch, lane)
    events = _capture(monkeypatch)
    server_start.attempted(model, load_policy="eager")

    with pytest.raises(SystemExit) as caught:
        cli._handle_optional_runtime_missing(failure, alias_or_path=model)

    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert failure.install_hint in stderr
    assert f"RAPID-MLX-STARTUP-FAILURE: {failure.marker_reason} extra={lane}" in stderr
    terminal = [
        props
        for name, props in events
        if name == "server_start_state" and props["state"] != "attempted"
    ]
    failures = [props for name, props in events if name == "model_serve_failed"]
    assert terminal == [
        {
            "state": "failed",
            "model_type": model_events.model_type(model),
            "load_policy": "eager",
            "failure_stage": "preflight",
        }
    ]
    assert len(failures) == 1
    assert failures[0]["error_class"] == "missing_extra"
    assert failures[0]["extra"] == lane
    assert "detail" not in failures[0]
    assert registry.validate("model_serve_failed", failures[0]) == failures[0]


@pytest.mark.parametrize(
    ("lane", "model", "status", "marker_reason"),
    [
        ("vision", "ui-tars-1.5-7b-4bit", "absent", "runtime_extra_missing"),
        ("vision", "ui-tars-1.5-7b-4bit", "broken", "runtime_broken"),
        (
            "vision",
            "ui-tars-1.5-7b-4bit",
            "incompatible",
            "runtime_incompatible",
        ),
        ("video", "wan2.2-ti2v-5b-q8", "absent", "runtime_extra_missing"),
        ("image", "flux-schnell", "absent", "runtime_extra_missing"),
        ("audio", "kokoro", "absent", "runtime_extra_missing"),
    ],
)
def test_real_dispatch_posts_one_actionable_failure_to_loopback(
    tmp_path, lane: str, model: str, status: str, marker_reason: str
) -> None:
    proc, events = _run_real_missing_extra_dispatch(
        tmp_path,
        lane=lane,
        model=model,
        status=status,
    )

    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    _assert_actionable_failure_contract(
        proc.stderr,
        extra=lane,
        marker_reason=marker_reason,
    )
    assert (
        len(
            [
                event
                for event in events
                if event.event == "server_start_state"
                and event.props.state == "failed"
                and event.props.failure_stage == "preflight"
            ]
        )
        == 1
    )
    matching_failures = [
        event
        for event in events
        if event.event == "model_serve_failed"
        and event.props.error_class == "missing_extra"
        and event.props.extra == lane
    ]
    assert len(matching_failures) == 1
    assert not hasattr(matching_failures[0].props, "detail")


def test_standalone_bonsai_dispatch_uses_same_handler_and_loopback_sink(
    tmp_path,
) -> None:
    standalone_proc, standalone_events = _run_real_missing_extra_dispatch(
        tmp_path / "standalone",
        lane="bonsai",
        model="bonsai2-27b-2bit",
        standalone=True,
    )
    cli_proc, cli_events = _run_real_missing_extra_dispatch(
        tmp_path / "cli",
        lane="bonsai",
        model="bonsai2-27b-2bit",
    )

    for proc in (standalone_proc, cli_proc):
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr
        _assert_actionable_failure_contract(
            proc.stderr,
            extra="vision",
            marker_reason="runtime_extra_missing",
        )

    expected_events = [
        ("server_start_state", "failed", "preflight", None, None),
        ("model_serve_failed", None, None, "missing_extra", "vision"),
    ]
    standalone_contract = _contracted_failure_events(standalone_events)
    cli_contract = _contracted_failure_events(cli_events)
    assert standalone_contract == expected_events
    assert cli_contract == standalone_contract
    assert _normalized_failure_text(cli_proc.stderr) == _normalized_failure_text(
        standalone_proc.stderr
    )


def test_actionable_failure_contract_rejects_corrupt_marker() -> None:
    expected_hint = _expected_install_hint_line("vision")
    stderr = (
        f"missing vision runtime\n{expected_hint}\n"
        "RAPID-MLX-STARTUP-FAILURE: "
        "runtime_extra_missing extra=vision-corrupt\n"
    )

    with pytest.raises(AssertionError):
        _assert_actionable_failure_contract(
            stderr,
            extra="vision",
            marker_reason="runtime_extra_missing",
        )


def test_standalone_failure_guard_routes_optional_runtime_with_context(
    monkeypatch,
) -> None:
    failure = OptionalRuntimeMissing(
        extra="vision",
        install_hint="pip install 'rapid-mlx[vision]'",
        detail="missing vision",
        status="absent",
    )
    engine = object()
    calls = []
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_standalone_start_model", "bonsai2-27b-2bit")

    def handle(exc, **kwargs):
        calls.append((exc, kwargs))
        raise SystemExit(2)

    monkeypatch.setattr(server, "handle_optional_runtime_missing", handle)

    @server._capture_start_failures
    def fail():
        raise failure

    with pytest.raises(SystemExit, match="2"):
        fail()

    assert calls == [
        (
            failure,
            {
                "engine": engine,
                "alias_or_path": "bonsai2-27b-2bit",
                "auto_selected": False,
            },
        )
    ]


def test_standalone_main_records_model_before_startup(monkeypatch) -> None:
    from rapid_mlx.telemetry import consent_runtime

    class StopStartup(BaseException):
        pass

    parsed = SimpleNamespace(model="bonsai2-27b-2bit", lazy_load=False)
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda _self: parsed)
    monkeypatch.setattr(
        consent_runtime,
        "startup",
        lambda **_kwargs: (_ for _ in ()).throw(StopStartup()),
    )

    with pytest.raises(StopStartup):
        server.main()

    assert server._standalone_start_model == "bonsai2-27b-2bit"


def test_real_dispatch_with_present_vision_extra_emits_no_failure(tmp_path) -> None:
    proc, events = _run_real_missing_extra_dispatch(
        tmp_path,
        lane="vision-present",
        model="ui-tars-1.5-7b-4bit",
        status="present",
    )

    assert proc.returncode == 0
    markers = [
        line
        for line in proc.stderr.splitlines()
        if line.startswith("RAPID-MLX-STARTUP-FAILURE:")
    ]
    assert markers == []
    assert "rapid-mlx[vision]" not in proc.stderr
    assert (
        len(
            [
                event
                for event in events
                if event.event == "server_start_state" and event.props.state == "failed"
            ]
        )
        == 0
    )
    assert len([event for event in events if event.event == "model_serve_failed"]) == 0


def test_bonsai_engine_preflight_is_missing_vision_failure(monkeypatch, capsys) -> None:
    from rapid_mlx import model_aliases, model_metadata
    from rapid_mlx.models import mllm

    real_resolve_profile = model_aliases.resolve_profile
    monkeypatch.setattr(model_aliases, "resolve_model", lambda model: model)
    monkeypatch.setattr(model_aliases, "resolve_profile", lambda _model: None)
    monkeypatch.setattr(
        server, "_prefetch_routing_metadata", lambda _model: "/cached/bonsai"
    )
    monkeypatch.setattr(
        model_metadata,
        "read_model_metadata",
        lambda _path: SimpleNamespace(
            snapshot_dir=None,
            config={"model_type": "prism_hadamard_qwen35"},
        ),
    )
    monkeypatch.setattr(
        model_metadata, "checkpoint_has_multimodal_weights", lambda *_a: False
    )
    monkeypatch.setattr(model_metadata, "config_indicates_multimodal", lambda _c: False)
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )

    with pytest.raises(OptionalRuntimeMissing) as caught:
        server._preflight_vision_runtime("bonsai2-27b-2bit")
    monkeypatch.setattr(model_aliases, "resolve_profile", real_resolve_profile)

    events = _capture(monkeypatch)
    server_start.attempted("bonsai2-27b-2bit", load_policy="eager")
    with pytest.raises(SystemExit) as exited:
        cli._handle_optional_runtime_missing(
            caught.value, alias_or_path="bonsai2-27b-2bit"
        )

    assert exited.value.code == 2
    assert "rapid-mlx[vision]" in capsys.readouterr().err
    assert [props for name, props in events if name == "model_serve_failed"] == [
        {
            "error_class": "missing_extra",
            "extra": "vision",
            "model": "bonsai2-27b-2bit",
            "model_type": "vlm",
            "auto_selected": False,
            "quant": "2bit",
        }
    ]
    assert [
        props
        for name, props in events
        if name == "server_start_state" and props["state"] == "failed"
    ][0]["failure_stage"] == "preflight"


def test_present_extras_emit_no_failure(monkeypatch) -> None:
    from rapid_mlx.audio import probe
    from rapid_mlx.models import mllm
    from rapid_mlx.runtime import image_lane, video_lane

    events = _capture(monkeypatch)
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.OK, None),
    )
    monkeypatch.setattr(image_lane, "image_runtime_issue", lambda _name: None)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())

    mllm.require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
    monkeypatch.setattr(video_lane.sys, "version_info", sys.version_info)
    monkeypatch.setattr(video_lane, "_is_ltx25_name", lambda _name: False)
    monkeypatch.setattr(video_lane, "_is_cogvideox_name", lambda _name: False)
    monkeypatch.setattr(video_lane, "_default_video_runtime_requirements", lambda _: [])
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "ffmpeg")
    video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
    image_lane.require_image_runtime_or_exit("flux-schnell")
    probe.require_audio_or_exit("kokoro")

    assert events == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [("BROKEN", "broken"), ("INCOMPATIBLE", "incompatible")],
)
def test_engine_vision_preflight_preserves_runtime_status(
    monkeypatch, status: str, expected: str
) -> None:
    from rapid_mlx.models import mllm

    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (getattr(mllm.VisionRuntimeStatus, status), "detail"),
    )

    with pytest.raises(OptionalRuntimeMissing) as caught:
        mllm._require_mlx_vlm("model")

    assert caught.value.status == expected


def test_model_load_optional_failure_passes_through(monkeypatch, tmp_path) -> None:
    from rapid_mlx import server as server_module

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"llama"}')
    args = cli.build_parser().parse_args(["serve", str(model_dir), "--no-mllm"])
    failure = OptionalRuntimeMissing(
        extra="vision",
        install_hint="pip install 'rapid-mlx[vision]'",
        detail="missing vision",
        status="absent",
    )
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_alias_min_memory", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_kw: None)
    monkeypatch.setattr(server_module, "configure_cors_from_env", lambda *_a: [])
    monkeypatch.setattr(server_module, "configure_trusted_hosts", lambda *_a: None)
    monkeypatch.setattr(
        server_module, "configure_model_residency", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        server_module,
        "load_model",
        lambda *_a, **_kw: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        "rapid_mlx.middleware.request_logging.install_request_logging_middleware",
        lambda *_a: None,
    )

    with pytest.raises(OptionalRuntimeMissing) as caught:
        cli.serve_command(args)

    assert caught.value is failure


def test_main_routes_optional_failure_to_single_handler(
    monkeypatch, tmp_path, capsys
) -> None:
    from rapid_mlx.telemetry import consent_runtime

    model = tmp_path / "model"
    model.mkdir()
    failure = OptionalRuntimeMissing(
        extra="image",
        install_hint="pip install 'rapid-mlx[image]'",
        detail="missing image",
        status="absent",
    )
    events = _capture(monkeypatch)
    monkeypatch.setattr(consent_runtime, "startup", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_start_v2_lifecycle", lambda _command: None)
    monkeypatch.setattr(
        cli, "serve_command", lambda _args: (_ for _ in ()).throw(failure)
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(model)])

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "RAPID-MLX-STARTUP-FAILURE: runtime_extra_missing extra=image" in (
        capsys.readouterr().err
    )
    assert len([name for name, _props in events if name == "model_serve_failed"]) == 1


@pytest.mark.asyncio
async def test_lifespan_optional_failure_reuses_cli_handler(monkeypatch) -> None:
    failure = OptionalRuntimeMissing(
        extra="audio",
        install_hint="pip install 'rapid-mlx[audio]'",
        detail="missing audio",
        status="absent",
    )
    lifecycle = SimpleNamespace(ensure_loaded=AsyncMock(side_effect=failure))
    engine = SimpleNamespace(_loaded=False)
    calls = []

    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_primary_model_lifecycle", lifecycle)
    monkeypatch.setattr(server, "_primary_lazy_load", False)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 0.0)
    monkeypatch.setattr(server, "_model_alias", "kokoro")
    monkeypatch.setattr(server, "_model_path", "/unused")
    monkeypatch.setattr(server, "_telemetry_auto_selected", False)

    def handle(exc, **kwargs):
        calls.append((exc, kwargs))
        raise SystemExit(2)

    monkeypatch.setattr(cli, "_handle_optional_runtime_missing", handle)
    lifespan = server.lifespan(server.app)

    with pytest.raises(SystemExit, match="2"):
        await lifespan.__anext__()

    assert calls == [
        (
            failure,
            {
                "engine": engine,
                "alias_or_path": "kokoro",
                "auto_selected": False,
            },
        )
    ]
