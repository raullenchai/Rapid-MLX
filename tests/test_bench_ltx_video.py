# SPDX-License-Identifier: Apache-2.0
"""Tests for the dev-only LTX benchmark summarization helpers."""

import io
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PATH = Path(__file__).parents[1] / "bench" / "bench_ltx_video.py"
SPEC = spec_from_file_location("bench_ltx_video", PATH)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_swap_parser_handles_megabytes(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.subprocess,
        "check_output",
        lambda *args, **kwargs: "total = 2048.00M  used = 12.50M  free = 2035.50M",
    )
    assert MODULE._swap_used_bytes() == round(12.5 * 1024**2)


def test_duration_rounds_to_nearest_ltx_frame_shape() -> None:
    assert MODULE._frames_for_seconds(5, 24) == 121
    assert MODULE._frames_for_seconds(10, 24) == 241


def test_main_dispatches_worker_and_propagates_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_ltx_video.py",
            "--worker",
            "--runtime",
            "ltx25",
            "--model",
            "/model",
            "--frames",
            "9",
        ],
    )
    sentinel = object()
    monkeypatch.setattr(MODULE, "_worker", lambda args: sentinel)
    assert MODULE.main() is sentinel


def test_legacy_text_lines_produce_no_events() -> None:
    """Progress text is prompt-injectable: it never becomes an event."""
    events = []
    lines = []
    MODULE._handle_line(
        "[Loading transformer (weights.safetensors)] done in 0.4s\n",
        "stderr",
        MODULE.time.monotonic_ns(),
        events,
        lines,
        b"",
        [0],
    )
    assert events == []
    MODULE._handle_line(
        "STAGE:1:STEP:1:9:\n",
        "stderr",
        MODULE.time.monotonic_ns(),
        events,
        lines,
        b"",
        [0],
    )
    assert events == []


def test_unauthenticated_event_lines_are_ignored() -> None:
    """An echoed prompt cannot forge worker events without the HMAC key."""
    events = []
    lines = []
    forged = MODULE.EVENT_PREFIX + '{"kind": "complete", "time_ns": 1}'
    MODULE._handle_line(
        forged,
        "stdout",
        MODULE.time.monotonic_ns(),
        events,
        lines,
        b"test-key",
        [0],
    )
    assert events == []
    assert any("complete" in line for line in lines)  # logged, not trusted

    # A properly signed event is accepted.
    import hmac
    import json as jsonlib

    body = {"kind": "worker_ready", "time_ns": 1}
    payload = jsonlib.dumps(body, sort_keys=True)
    sig = hmac.new(b"test-key", payload.encode(), "sha256").hexdigest()
    envelope = MODULE.EVENT_PREFIX + jsonlib.dumps(
        {"sig": sig, "payload": body}, sort_keys=True
    )
    MODULE._handle_line(
        envelope,
        "stdout",
        MODULE.time.monotonic_ns(),
        events,
        lines,
        b"test-key",
        [0],
    )
    assert [event["kind"] for event in events] == ["worker_ready"]

    # No key at all: fail closed.
    MODULE._handle_line(
        envelope, "stdout", MODULE.time.monotonic_ns(), events, lines, b"", [0]
    )
    assert [event["kind"] for event in events] == ["worker_ready"]


def test_markdown_excludes_failed_and_swap_limited_runs_from_medians() -> None:
    document = {
        "machine": {
            "hostname": "mzr",
            "hardware_model": "Mac16,11",
            "architecture": "arm64",
            "memory_bytes": 48 * 1024**3,
            "macos_version": "26.5.1",
            "macos_build": "25F80",
        },
        "config": {
            "model": "example/ltx",
            "model_identity": "org/model@abc123",
            "runtime": "mlx23",
            "runtime_revision": "0.1.36",
            "prompt": "fixed prompt",
            "width": 768,
            "height": 512,
            "frames": 121,
            "fps": 24,
            "seed": 42,
            "sample_interval_s": 1,
        },
        "results": [
            {
                "status": "completed",
                "cold_start_to_first_step_s": 10.0,
                "cold_start_to_weights_ready_s": 9.5,
                "step_median_s": 3.0,
                "stage_step_median_s": {"1": 2.5, "2": 7.5},
                "total_s": 20.0,
                "peak_tree_rss_bytes": 2 * 1024**3,
                "mlx_peak_bytes": 1024**3,
                "swap_growth_bytes": 0,
                "swap_limited": False,
            },
            {
                "status": "failed",
                "cold_start_to_first_step_s": 999.0,
                "cold_start_to_weights_ready_s": 998.0,
                "step_median_s": 999.0,
                "stage_step_median_s": {},
                "total_s": 999.0,
                "peak_tree_rss_bytes": 3 * 1024**3,
                "mlx_peak_bytes": None,
                "swap_growth_bytes": 0,
                "swap_limited": False,
            },
            {
                "status": "completed",
                "cold_start_to_first_step_s": 777.0,
                "cold_start_to_weights_ready_s": 776.0,
                "step_median_s": 777.0,
                "stage_step_median_s": {"1": 777.0, "2": 777.0},
                "total_s": 777.0,
                "peak_tree_rss_bytes": 4 * 1024**3,
                "mlx_peak_bytes": 2 * 1024**3,
                "swap_growth_bytes": 512 * 1024**2,
                "swap_limited": True,
            },
        ],
    }
    rendered = MODULE._markdown(document)
    assert "10.00 s" in rendered
    assert (
        "Median cold start to first diffusion step" in rendered
        and "10.00 s" in rendered
    )
    assert "abc123" in rendered
    assert "48.00 GiB" in rendered
    assert "3.00 s" in rendered
    assert "20.00 s" in rendered
    assert "Median stage 1 diffusion step: 2.50 s" in rendered
    assert "Median stage 2 diffusion step: 7.50 s" in rendered
    assert "999.00 s" not in rendered
    assert "777.00 s" not in rendered
    assert "4.00 GiB" in rendered
    assert "0.50 GiB" in rendered


def _main_exits_with(argv: list[str], message: str) -> None:
    monkeypatch_argv = argv
    saved = sys.argv
    sys.argv = monkeypatch_argv
    try:
        try:
            MODULE.main()
        except SystemExit as exc:
            assert exc.code == message
            return
        raise AssertionError(f"expected SystemExit({message!r})")
    finally:
        sys.argv = saved


def test_rejects_vacuous_zero_run_sweep() -> None:
    _main_exits_with(
        [
            "bench_ltx_video.py",
            "--runtime",
            "mlx23",
            "--model",
            "m",
            "--frames",
            "9",
            "--runs",
            "0",
        ],
        "--runs must be at least 1",
    )


def test_rejects_nonpositive_size_dimensions() -> None:
    for size in ("0x0", "0x512", "768x0"):
        _main_exits_with(
            [
                "bench_ltx_video.py",
                "--runtime",
                "mlx23",
                "--model",
                "m",
                "--frames",
                "9",
                "--size",
                size,
                "--runs",
                "1",
            ],
            "--size dimensions must be positive",
        )


def test_rejects_nonpositive_fps_and_interval() -> None:
    _main_exits_with(
        [
            "bench_ltx_video.py",
            "--runtime",
            "mlx23",
            "--model",
            "m",
            "--seconds",
            "5",
            "--fps",
            "0",
            "--runs",
            "1",
        ],
        "--fps must be at least 1",
    )
    _main_exits_with(
        [
            "bench_ltx_video.py",
            "--runtime",
            "mlx23",
            "--model",
            "m",
            "--frames",
            "9",
            "--sample-interval",
            "0",
            "--runs",
            "1",
        ],
        "--sample-interval must be positive",
    )


def test_rejects_negative_cooldown() -> None:
    _main_exits_with(
        [
            "bench_ltx_video.py",
            "--runtime",
            "mlx23",
            "--model",
            "m",
            "--frames",
            "9",
            "--cooldown",
            "-1",
            "--runs",
            "1",
        ],
        "--cooldown must not be negative",
    )


def test_measured_bar_preserves_tqdm_object_api(monkeypatch) -> None:
    emitted = []

    class FakeBar:
        iterable = ["a", "b"]

        def __init__(self) -> None:
            self.total = 3
            self.items = iter(["a", "b"])
            self.updates: list[int] = []
            self.entered = False
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.items)

        def update(self, amount):
            self.updates.append(amount)

        def close(self):
            self.closed = True

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *exc_info):
            self.closed = True
            return False

    fake = FakeBar()
    monkeypatch.setattr(
        MODULE,
        "_emit",
        lambda kind, **fields: emitted.append({"kind": kind, **fields}),
    )
    bar = MODULE._MeasuredBar(fake, 7, "diffuse", MODULE._emit)

    assert list(bar) == ["a", "b"]
    assert [event["step"] for event in emitted] == [1, 2, 2]
    assert emitted[-1]["kind"] == "stage_end"
    assert all(event["stage"] == 7 and event["total"] == 3 for event in emitted)

    bar.update(1)
    assert fake.updates == [1]
    with bar as entered:
        assert entered is bar and fake.entered
    assert fake.closed


def test_stage_durations_same_stage_adjacent_starts_only() -> None:
    step_events = [
        {"stage": 1, "observed_elapsed_s": 0.0},
        {"stage": 1, "observed_elapsed_s": 10.0},
        {"stage": 2, "observed_elapsed_s": 30.0},
    ]
    complete = {"observed_elapsed_s": 90.0}
    # Same-stage adjacent starts plus the stage_end boundary for the final
    # step; without a boundary event the final step is omitted.
    assert MODULE._stage_duration_samples(step_events, {}) == {"1": [10.0]}
    stage_ends = {2: {"observed_elapsed_s": 45.0}}
    assert MODULE._stage_duration_samples(step_events, stage_ends) == {
        "1": [10.0],
        "2": [15.0],
    }

    # Every stage's final step gets its own stage_end duration.
    multi = [
        {"stage": 1, "observed_elapsed_s": 0.0},
        {"stage": 1, "observed_elapsed_s": 8.0},
        {"stage": 2, "observed_elapsed_s": 12.0},
        {"stage": 2, "observed_elapsed_s": 20.0},
    ]
    multi_ends = {1: {"observed_elapsed_s": 11.0}, 2: {"observed_elapsed_s": 26.0}}
    assert MODULE._stage_duration_samples(multi, multi_ends) == {
        "1": [8.0, 3.0],
        "2": [8.0, 6.0],
    }


def test_stat_fingerprint_hashes_single_files(tmp_path) -> None:
    """A module file path must produce a real fingerprint, not an empty hash."""
    module = tmp_path / "runtime.py"
    module.write_text("print('v1')")
    before = MODULE._stat_fingerprint(module)
    assert (
        before != MODULE._stat_fingerprint(tmp_path / "missing.py") if False else before
    )
    module.write_text("print('v2')")
    after = MODULE._stat_fingerprint(module)
    assert before != after


def test_source_stat_fingerprint_detects_submodule_mutation(tmp_path) -> None:
    """A change to a non-__init__ runtime file must change the fingerprint."""
    pkg = tmp_path / "rapid_mlx"
    (pkg / "runtime").mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1")
    (pkg / "runtime" / "video_lane.py").write_text("y = 1")
    (pkg / "runtime" / "__pycache__").mkdir()
    (pkg / "runtime" / "__pycache__" / "x.pyc").write_bytes(b"cache")
    before = MODULE._source_stat_fingerprint(pkg)
    (pkg / "runtime" / "video_lane.py").write_text("y = 2")
    after = MODULE._source_stat_fingerprint(pkg)
    assert before != after
    # Cache-only changes are ignored.
    (pkg / "runtime" / "__pycache__" / "x.pyc").write_bytes(b"changed")
    assert MODULE._source_stat_fingerprint(pkg) == after


def test_stage_durations_cross_stage_pairs_excluded() -> None:
    compute = MODULE._stage_duration_samples
    step_events = [
        {"stage": 1, "observed_elapsed_s": 0.0},
        {"stage": 2, "observed_elapsed_s": 5.0},
    ]
    assert compute(step_events, {}) == {}
    assert compute(
        [
            {"stage": 1, "observed_elapsed_s": 0.0},
            {"stage": 1, "observed_elapsed_s": 4.0},
            {"stage": 2, "observed_elapsed_s": 9.0},
        ],
        {},
    ) == {"1": [4.0]}


def test_run_once_observations_precede_start_and_terminate_orphan(
    monkeypatch, tmp_path
) -> None:
    order: list[str] = []
    clock = {"t": 0.0}

    def fake_monotonic_ns():
        clock["t"] += 1.0
        order.append("clock")
        return int(clock["t"] * 1e9)

    class FakeProc:
        pid = 4242

        def __init__(self) -> None:
            self.returncode = None  # still running after the sample loop
            self.wait_calls: list[float | None] = []
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                raise MODULE.subprocess.TimeoutExpired("cmd", timeout)
            self.returncode = -9
            return 0

    monkeypatch.setattr(MODULE.time, "monotonic_ns", fake_monotonic_ns)
    monkeypatch.setattr(
        MODULE,
        "_swap_used_bytes",
        lambda timeout=3.0: (order.append("swap"), 0)[1],
    )
    monkeypatch.setattr(
        MODULE, "_thermal_observation", lambda: (order.append("thermal"), None)[1]
    )
    holder: dict[str, object] = {}

    def fake_popen(*args, **kwargs):
        proc = FakeProc()
        holder["proc"] = proc
        order.append("popen")
        return proc

    monkeypatch.setattr(MODULE.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(MODULE, "_tree_rss", lambda pid: 0)

    class FakeThread:
        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            order.append("thread_join")

    monkeypatch.setattr(MODULE.threading, "Thread", lambda *a, **k: FakeThread())
    killed = []

    def fake_killpg(pid, sig):
        if sig == 0:
            # Existence probe: the group vanishes once reaped.
            if holder["proc"].returncode is not None:
                raise ProcessLookupError
            return
        order.append("killpg")
        if holder["proc"].returncode is not None:
            raise ProcessLookupError
        killed.append((pid, sig))

    monkeypatch.setattr(MODULE.os, "killpg", fake_killpg)
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: None)

    args = MODULE._parser().parse_args(
        ["--runtime", "mlx23", "--model", "m", "--frames", "9", "--runs", "1"]
    )
    args.worker = False
    args.output_dir = str(tmp_path)
    args.runtime = "mlx23"
    args.width, args.height = 768, 512
    args.invocation = "test-invocation"
    digest_dir = tmp_path / "fake-snapshot"
    digest_dir.mkdir(exist_ok=True)
    (digest_dir / "weights.safetensors").write_bytes(b"fake-weights")
    args.resolved_model = str(digest_dir)
    args.model_stat_fingerprint = MODULE._stat_fingerprint(digest_dir)
    args.event_key = "fixture-key"
    args.runtime_stat_fingerprint = None
    result = MODULE._run_once(args, 1)

    # Pre-launch probes precede Popen, and start_ns sits between them.
    assert order[:4] == ["swap", "thermal", "clock", "popen"], order
    # Orphan cleanup: SIGTERM, then SIGKILL after the wait timeout, all
    # before any reader join.
    assert killed == [(4242, MODULE.signal.SIGTERM), (4242, MODULE.signal.SIGKILL)]
    assert order.index("killpg") < order.index("thread_join")
    assert result["returncode"] == -9


def test_run_deadline_terminates_wedged_worker(monkeypatch, tmp_path) -> None:
    killed: list[tuple[int, object]] = []

    class FakeProc:
        pid = 99

        def __init__(self) -> None:
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()
            self.wait_calls: list[float | None] = []
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                raise MODULE.subprocess.TimeoutExpired("cmd", timeout)
            self.returncode = -9
            return 0

    proc = FakeProc()
    clock = {"t": 0}

    def fake_monotonic_ns():
        clock["t"] += int(1e9)  # advance one virtual second per call
        return clock["t"]

    monkeypatch.setattr(MODULE.time, "monotonic_ns", fake_monotonic_ns)
    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(MODULE, "_swap_used_bytes", lambda timeout=3.0: 0)
    monkeypatch.setattr(MODULE, "_thermal_observation", lambda: None)
    monkeypatch.setattr(MODULE, "_tree_rss", lambda pid: 0)

    def fake_killpg(pid, sig):
        if sig == 0:
            if proc.returncode is not None:
                raise ProcessLookupError
            return
        if proc.returncode is not None:
            raise ProcessLookupError
        killed.append(sig)

    monkeypatch.setattr(MODULE.os, "killpg", fake_killpg)
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: None)

    class FakeThread:
        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(MODULE.threading, "Thread", lambda *a, **k: FakeThread())

    args = MODULE._parser().parse_args(
        [
            "--runtime",
            "mlx23",
            "--model",
            "m",
            "--frames",
            "9",
            "--runs",
            "1",
            "--deadline",
            "0.0000001",
        ]
    )
    args.worker = False
    args.output_dir = str(tmp_path)
    args.runtime = "mlx23"
    args.width, args.height = 768, 512
    args.invocation = "test-invocation"
    digest_dir = tmp_path / "fake-snapshot"
    digest_dir.mkdir(exist_ok=True)
    (digest_dir / "weights.safetensors").write_bytes(b"fake-weights")
    args.resolved_model = str(digest_dir)
    args.model_stat_fingerprint = MODULE._stat_fingerprint(digest_dir)
    args.event_key = "fixture-key"
    args.runtime_stat_fingerprint = None
    result = MODULE._run_once(args, 1)

    assert result["status"] == "failed"
    assert result["deadline_exceeded"] is True
    assert killed == [MODULE.signal.SIGTERM, MODULE.signal.SIGKILL]


def test_model_snapshot_pins_hf_layouts_and_rejects_bare_dirs(
    monkeypatch, tmp_path
) -> None:
    # HF snapshots/<sha> layout pins to the directory basename.
    hf_dir = tmp_path / "snapshots" / ("a" * 40)
    hf_dir.mkdir(parents=True)
    (hf_dir / "weights.safetensors").write_bytes(b"weights")
    identity, snapshot = MODULE._resolve_model_snapshot(str(hf_dir))
    assert snapshot == str(hf_dir)
    assert "+sha256:" in identity
    assert identity.startswith(f"{str(hf_dir.resolve())}@{'a' * 40}")
    # Rewriting a weight in place changes the identity.
    (hf_dir / "weights.safetensors").write_bytes(b"tampered")
    identity2, _ = MODULE._resolve_model_snapshot(str(hf_dir))
    assert identity2 != identity

    # refs/main carries the revision; the resolved snapshots/<sha> directory
    # is what the worker receives.
    checkout = tmp_path / "models--org--m"
    snapshot_dir = checkout / "snapshots" / ("b" * 40)
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}")
    (checkout / "refs").mkdir(parents=True)
    (checkout / "refs" / "main").write_text("b" * 40 + "\n")
    identity, snapshot = MODULE._resolve_model_snapshot(str(checkout))
    expected = str(snapshot_dir.resolve())
    assert snapshot == expected
    assert "+sha256:" in identity
    assert identity.startswith(expected + "@" + "b" * 40)

    # A refs/main without the matching snapshot directory falls back to a
    # digest-only identity instead of reporting a revision that isn't there.
    dangling = tmp_path / "models--org--dangling"
    (dangling / "refs").mkdir(parents=True)
    (dangling / "refs" / "main").write_text("c" * 40 + "\n")
    (dangling / "weights.safetensors").write_bytes(b"weights")
    identity, snapshot = MODULE._resolve_model_snapshot(str(dangling))
    assert snapshot == str(dangling.resolve())
    assert identity.startswith(f"{snapshot}@sha256:")
    assert "+" + "c" * 40 not in identity

    # A bare directory is pinned by content digest, as the README promises.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "weights.safetensors").write_bytes(b"0" * 128)
    identity, snapshot = MODULE._resolve_model_snapshot(str(bare))
    assert snapshot == str(bare.resolve())
    assert identity.startswith(f"{snapshot}@sha256:")

    # A plain file is rejected as a snapshot.
    try:
        MODULE._resolve_model_snapshot(str(PATH))
    except SystemExit as exc:
        assert "snapshot directory" in str(exc.code)
    else:
        raise AssertionError("expected SystemExit for a file path")

    downloads: list[tuple[str, str]] = []
    full_sha = "d" * 40

    class FakeApi:
        def model_info(self, repo):
            assert repo == "org/model"
            return type("Info", (), {"sha": full_sha})()

    hf_snapshot = tmp_path / "hf-cache" / "org--model"
    hf_snapshot.mkdir(parents=True)
    (hf_snapshot / "weights.safetensors").write_bytes(b"remote-weights")

    def fake_snapshot_download(repo_id, revision):
        downloads.append((repo_id, revision))
        return str(hf_snapshot)

    class FakeHub:
        HfApi = FakeApi
        snapshot_download = staticmethod(fake_snapshot_download)

    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = FakeHub()
    try:
        identity, snapshot = MODULE._resolve_model_snapshot("org/model")
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved
    assert identity.startswith(f"org/model@{full_sha}+sha256:")
    assert snapshot == str(hf_snapshot)
    assert downloads == [("org/model", full_sha)]

    # An abbreviated SHA cannot pin a commit: fail closed.
    class ShortApi:
        def model_info(self, repo):
            return type("Info", (), {"sha": "abc123"})()

    sys.modules["huggingface_hub"] = type(
        "Hub", (), {"HfApi": ShortApi, "snapshot_download": None}
    )()
    try:
        MODULE._resolve_model_snapshot("org/model")
    except SystemExit as exc:
        assert "not an immutable commit SHA" in str(exc.code)
    else:
        raise AssertionError("abbreviated SHAs must be rejected")
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved


def test_main_rejects_zero_runs_before_model_resolution(monkeypatch) -> None:
    """Parameter validation must not touch the network or the registry."""

    def boom(*args, **kwargs):
        raise AssertionError("model identity must not be resolved on bad input")

    monkeypatch.setattr(MODULE, "_resolve_model_snapshot", boom)
    saved = sys.argv
    sys.argv = [
        "bench_ltx_video.py",
        "--runtime",
        "mlx23",
        "--model",
        "m",
        "--frames",
        "9",
        "--runs",
        "0",
    ]
    try:
        try:
            MODULE.main()
        except SystemExit as exc:
            assert exc.code == "--runs must be at least 1"
            return
        raise AssertionError("expected SystemExit")
    finally:
        sys.argv = saved


def test_run_once_reports_structured_launch_failure(monkeypatch, tmp_path) -> None:
    def failing_popen(*args, **kwargs):
        raise OSError("interpreter not found")

    monkeypatch.setattr(MODULE.subprocess, "Popen", failing_popen)
    monkeypatch.setattr(MODULE, "_swap_used_bytes", lambda timeout=3.0: 0)
    monkeypatch.setattr(MODULE, "_thermal_observation", lambda: None)

    args = MODULE._parser().parse_args(
        ["--runtime", "mlx23", "--model", "m", "--frames", "9", "--runs", "1"]
    )
    args.worker = False
    args.output_dir = str(tmp_path)
    args.runtime = "mlx23"
    args.width, args.height = 768, 512
    args.invocation = "test-invocation"
    digest_dir = tmp_path / "fake-snapshot"
    digest_dir.mkdir(exist_ok=True)
    (digest_dir / "weights.safetensors").write_bytes(b"fake-weights")
    args.resolved_model = str(digest_dir)
    args.model_stat_fingerprint = MODULE._stat_fingerprint(digest_dir)
    args.event_key = "fixture-key"
    args.runtime_stat_fingerprint = None
    result = MODULE._run_once(args, 1)

    assert result["status"] == "failed"
    assert "interpreter not found" in result["launch_error"]


def test_measured_bar_passes_through_manual_and_delegates_updates() -> None:
    class ManualBar:
        iterable = None
        total = 3

        def update(self, n=1):
            return n

    class IterableBar:
        iterable = ["x"]
        total = 1

        def __iter__(self):
            return iter(self.iterable)

        def update(self, n=1):
            return n

    events = []
    bar = MODULE._MeasuredBar(
        IterableBar(),
        1,
        "diffuse",
        lambda kind, **fields: events.append({"kind": kind, **fields}),
    )
    bar.update(3)  # Pure delegation: no fabricated step event.
    assert events == []
    next(bar)
    assert [event["kind"] for event in events] == ["step_start"]

    # The factory measures only diffusion-sampler bars.
    manual = ManualBar()
    assert MODULE._wrap_progress_bar(manual, 1, "Denoising", events.append) is manual
    wrapped_iter = MODULE._wrap_progress_bar(
        IterableBar(),
        1,
        "Denoising",
        lambda kind, **fields: events.append({"kind": kind, **fields}),
    )
    assert isinstance(wrapped_iter, MODULE._MeasuredBar)
    # Unrelated iterable bars (downloads, loading, encode) pass through.
    unrelated = IterableBar()
    assert (
        MODULE._wrap_progress_bar(unrelated, 1, "Loading weights", events.append)
        is unrelated
    )


def test_stage_transition_recorded_separately_from_step_duration() -> None:
    step_events = [
        {"stage": 1, "observed_elapsed_s": 0.0},
        {"stage": 2, "observed_elapsed_s": 7.0},
        {"stage": 2, "observed_elapsed_s": 9.0},
    ]
    MODULE.annotate_step_boundaries(step_events)
    assert step_events[0]["stage_transition_s"] == 7.0
    assert "duration_to_next_step_s" not in step_events[0]
    assert step_events[1]["duration_to_next_step_s"] == 2.0
