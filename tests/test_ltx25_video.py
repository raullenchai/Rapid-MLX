"""LTX-2.5 external MLX runtime integration tests."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.runtime import video_lane
from vllm_mlx.runtime.video_lane import VideoEngine, VideoRuntimeError
from vllm_mlx.video import ltx25


def test_ltx25_alias_routes_to_video_lane() -> None:
    profile = resolve_profile("ltx-2.5-mlx-q8")
    assert profile is not None
    assert profile.hf_path == "MrMofer/ltx-2.5-mlx-q8"
    assert profile.modality == "video-gen"
    assert profile.min_memory_gb == 24
    assert ltx25.is_ltx25_model("org/my-ltx25-experiment") is False


def test_ltx25_capabilities_match_distilled_controls() -> None:
    from vllm_mlx.routes.video import _video_capabilities

    capabilities = _video_capabilities(
        SimpleNamespace(model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5")
    )
    assert capabilities["family"] == "ltx-2.5"
    assert capabilities["limits"]["size"]["width"]["multiple_of"] == 32
    assert capabilities["limits"]["workload"]["dimension_rounding"] == "ceil_to_32"
    assert capabilities["controls"]["guidance_scale"] is None
    assert capabilities["controls"]["negative_prompt"] is False
    assert capabilities["controls"]["conditioning_strength"] == {
        "minimum": 0.0,
        "maximum": 1.0,
    }


def test_ltx25_direct_engine_rejects_conditioning_without_image() -> None:
    engine = VideoEngine.__new__(VideoEngine)
    engine._ltx25_engine = object()

    with pytest.raises(VideoRuntimeError, match="requires an input image"):
        engine.generate(
            prompt="fox",
            output_path=Path("unused.mp4"),
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
            conditioning_strength=0.5,
        )


def test_ltx25_runtime_preflight_fails_before_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: None)
    monkeypatch.setattr(
        video_lane, "_resolve_ffmpeg", lambda: "/opt/homebrew/bin/ffmpeg"
    )
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: "/usr/bin/uv")

    with pytest.raises(SystemExit, match="2"):
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = capsys.readouterr().err
    assert ltx25.LTX25_RUNTIME_COMMIT in error
    assert "video generation guide" in error


def test_ltx25_runtime_preflight_requires_uv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: "/runtime/ltx-2-mlx")
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: None)

    with pytest.raises(SystemExit, match="2"):
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = capsys.readouterr().err
    assert "uv (`brew install uv`)" in error
    # The runtime itself resolved, so the clone/checkout walkthrough is noise.
    assert "git clone" not in error


def test_ltx25_missing_runtime_prints_setup_walkthrough(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: None)
    monkeypatch.setattr(
        video_lane, "_resolve_ffmpeg", lambda: "/opt/homebrew/bin/ffmpeg"
    )
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: "/usr/bin/uv")

    with pytest.raises(SystemExit, match="2"):
        # A path-qualified name passes _is_ltx25_name; the walkthrough must
        # not interpolate this user-controlled string into shell commands.
        video_lane.require_video_runtime_or_exit("$(uname)/ltx-2.5-mlx-q8")

    error = capsys.readouterr().err
    assert "docs/guides/video-generation.md" in error
    # Conditional clone (gated on a real Git checkout, not a bare
    # directory) + unconditional fetch: the same block repairs an
    # existing checkout pinned to a stale revision (plain clone would fail).
    assert (
        f"[ -d ltx-2-mlx/.git ] || git clone --branch ltx25 "
        f"{ltx25.LTX25_RUNTIME_REPOSITORY}" in error
    )
    assert "git -C ltx-2-mlx fetch --quiet origin" in error
    assert f"git -C ltx-2-mlx checkout {ltx25.LTX25_RUNTIME_COMMIT}" in error
    assert "uv sync --project ltx-2-mlx" in error
    # Canonical alias only — the raw model_name must not appear in the
    # copy-pastable command.
    assert (
        'RAPID_MLX_LTX25_RUNTIME="$PWD/ltx-2-mlx/.venv/bin/ltx-2-mlx" '
        "rapid-mlx serve ltx-2.5-mlx-q8" in error
    )
    assert "$(uname)" not in error


def test_ltx25_provisioning_failure_surfaces_cause_not_clone_steps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: "/runtime/ltx-2-mlx")

    def boom(executable: str) -> None:
        # Mirrors the message prepare_ltx25_runtime() actually raises for a
        # failed `uv sync` (see test_ltx25_provisioning_error_includes_uv_
        # failure_detail, which pins the production construction).
        raise ltx25.LTX25BackendError(
            "The pinned LTX-2.5 runtime could not be provisioned: "
            "`uv sync --frozen` failed with exit code 2 "
            "(error: lockfile out of date)"
        )

    monkeypatch.setattr(ltx25, "prepare_ltx25_runtime", boom)
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: "/usr/bin/uv")

    with pytest.raises(SystemExit, match="2"):
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = capsys.readouterr().err
    assert "a provisioned pinned LTX-2.5 runtime" in error
    # The underlying failure reason is the actionable part.
    assert "`uv sync --frozen` failed with exit code 2" in error
    assert "lockfile out of date" in error
    assert "docs/guides/video-generation.md" in error
    # The checkout already resolved — re-cloning is not the fix.
    assert "git clone" not in error


def test_ltx25_provisioning_error_includes_uv_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", None)
    monkeypatch.setattr(ltx25.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda repo, dest: None)

    def failing_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            2, ["uv", "sync"], stderr="error: lockfile out of date\n"
        )

    monkeypatch.setattr(ltx25.subprocess, "run", failing_run)

    with pytest.raises(ltx25.LTX25BackendError) as excinfo:
        ltx25.prepare_ltx25_runtime("/checkout/.venv/bin/ltx-2-mlx")

    message = str(excinfo.value)
    assert "could not be provisioned" in message
    assert "`uv sync --frozen` failed with exit code 2" in message
    assert "lockfile out of date" in message


def test_ltx25_provisioning_detail_sanitizes_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control sequences and index credentials in uv output must not reach
    the terminal (escape injection / credential exposure)."""
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", None)
    monkeypatch.setattr(ltx25.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda repo, dest: None)

    def failing_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            1,
            ["uv", "sync"],
            stderr=(
                "\x1b[31merror\x1b[0m: failed to fetch "
                "https://build:s3cret@index.example/simple/foo\n"
            ),
        )

    monkeypatch.setattr(ltx25.subprocess, "run", failing_run)

    with pytest.raises(ltx25.LTX25BackendError) as excinfo:
        ltx25.prepare_ltx25_runtime("/checkout/.venv/bin/ltx-2-mlx")

    message = str(excinfo.value)
    assert "\x1b" not in message
    assert "s3cret" not in message
    assert "https://***@index.example/simple/foo" in message


def test_ltx25_sanitize_redacts_percent_encoded_userinfo() -> None:
    from vllm_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "failed to fetch https://build%40corp:s3cret@index.example/simple/"
    )
    assert "s3cret" not in out
    assert "build%40corp" not in out
    assert "https://***@index.example/simple/" in out


def test_ltx25_sanitize_redacts_query_tokens_and_bearer() -> None:
    from vllm_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "fetch https://index.example/simple?token=s3cret&x=1 failed; "
        "api_key=abc123 Authorization: Bearer eyJhbGci.xyz"
    )
    assert "s3cret" not in out
    assert "abc123" not in out
    assert "eyJhbGci" not in out
    assert "https://index.example/simple?token=***" in out


def test_ltx25_sanitize_redacts_signed_url_params() -> None:
    """Signed-URL auth params don't end in the classic credential
    suffixes (codex on #2166): AWS presigned ``X-Amz-Signature``, Azure
    SAS ``sig``/``sas``, generic ``auth``/``jwt`` must all redact, since
    uv stderr can echo the full index URL query string."""
    from vllm_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "GET https://bucket.s3.example/wheel.whl"
        "?X-Amz-Credential=AKIA%2F20260820&X-Amz-Signature=deadbeefcafe"
        "&X-Amz-Expires=300 and https://acct.blob.example/pkg?sv=2024"
        "&sig=Zm9vYmFy%3D and auth=topsecret9 jwt=eyJ0eXAi.abc"
    )
    assert "deadbeefcafe" not in out
    assert "Zm9vYmFy" not in out
    assert "topsecret9" not in out
    assert "eyJ0eXAi" not in out
    assert "X-Amz-Signature=***" in out
    assert "sig=***" in out
    # Non-credential params survive so the diagnostic stays useful.
    assert "X-Amz-Expires=300" in out


def test_ltx25_sanitize_redacts_quoted_credential_values() -> None:
    from vllm_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "config error: token=\"quoted-s3cret\" password='single-s3cret' left"
    )
    assert "quoted-s3cret" not in out
    assert "single-s3cret" not in out
    assert "left" in out


def test_ltx25_sanitize_redacts_prefixed_credential_names() -> None:
    from vllm_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "access_token=at-s3cret client_secret=cs-s3cret "
        "HF_TOKEN=hf-s3cret AWS_SECRET_ACCESS_KEY=aws-s3cret api-key: ak-s3cret"
    )
    for leak in ("at-s3cret", "cs-s3cret", "hf-s3cret", "aws-s3cret", "ak-s3cret"):
        assert leak not in out
    assert "access_token=***" in out
    assert "client_secret=***" in out


def test_ltx25_sanitize_redacts_basic_auth_header() -> None:
    from vllm_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "request failed; Authorization: Basic dXNlcjpwYXNz and "
        "authorization: Digest response-s3cret"
    )
    assert "dXNlcjpwYXNz" not in out
    assert "response-s3cret" not in out
    assert "Authorization: Basic ***" in out


def test_ltx25_oserror_detail_is_sanitized_and_bounded() -> None:
    from vllm_mlx.video.ltx25 import _provisioning_failure_detail

    exc = OSError(
        "\x1b[31mdisk full\x1b[0m at https://user:tok3n@mirror.example/x " + "p" * 500
    )
    detail = _provisioning_failure_detail(exc)
    assert "\x1b" not in detail
    assert "tok3n" not in detail
    assert len(detail) <= 301


def test_ltx25_stderr_tail_is_bounded() -> None:
    """Only a bounded tail of uv stderr is ever read back into memory."""
    import tempfile

    with tempfile.TemporaryFile() as f:
        f.write(b"x" * 100_000 + b"\nfinal line\n")
        tail = ltx25._stderr_tail(f)

    assert len(tail) <= 4096
    assert "final line" in tail


def test_ltx25_provisioning_timeout_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", None)
    monkeypatch.setattr(ltx25.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda repo, dest: None)

    def timing_out_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["uv", "sync"], 1800)

    monkeypatch.setattr(ltx25.subprocess, "run", timing_out_run)

    with pytest.raises(ltx25.LTX25BackendError) as excinfo:
        ltx25.prepare_ltx25_runtime("/checkout/.venv/bin/ltx-2-mlx")

    assert "timed out after 1800s" in str(excinfo.value)


def test_ltx25_runtime_override_must_be_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "ltx-2-mlx"
    runtime.write_text("#!/bin/sh\n")
    monkeypatch.setenv("RAPID_MLX_LTX25_RUNTIME", str(runtime))
    monkeypatch.setattr(
        ltx25, "_runtime_revision", lambda _: ltx25.LTX25_RUNTIME_COMMIT
    )
    monkeypatch.setattr(
        ltx25.shutil,
        "which",
        lambda _: pytest.fail(
            "an invalid explicit override must not fall back to PATH"
        ),
    )

    assert ltx25.resolve_ltx25_runtime() is None

    runtime.chmod(0o755)
    assert ltx25.resolve_ltx25_runtime() == str(runtime)


def test_ltx25_runtime_rejects_unpinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    monkeypatch.setenv("RAPID_MLX_LTX25_RUNTIME", str(runtime))
    monkeypatch.setattr(ltx25, "_runtime_revision", lambda _: "wrong-revision")

    assert ltx25.resolve_ltx25_runtime() is None


def test_ltx25_materialization_ignores_checkout_and_replace_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "runtime"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    tracked = repository / "tracked.py"
    tracked.write_text("safe = True\n")
    (repository / "uv.lock").write_text("version = 1\n")
    subprocess.run(
        ["git", "-C", str(repository), "add", "tracked.py", "uv.lock"], check=True
    )
    tree = subprocess.run(
        ["git", "-C", str(repository), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(repository), "commit-tree", tree, "-m", "fixture"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "update-ref", "HEAD", commit], check=True
    )
    tracked.write_text("raise RuntimeError('replaced')\n")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.py"], check=True)
    replacement_tree = subprocess.run(
        ["git", "-C", str(repository), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    replacement_commit = subprocess.run(
        ["git", "-C", str(repository), "commit-tree", replacement_tree, "-m", "evil"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "replace", commit, replacement_commit],
        check=True,
    )
    (repository / "untracked.py").write_text("raise RuntimeError('unsafe')\n")
    destination = tmp_path / "snapshot"
    destination.mkdir()

    monkeypatch.setattr(ltx25, "LTX25_RUNTIME_COMMIT", commit)
    ltx25._materialize_runtime(repository, destination)

    assert (destination / "tracked.py").read_text() == "safe = True\n"
    assert not (destination / "untracked.py").exists()


def test_ltx25_materialization_wraps_malformed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = subprocess.CompletedProcess([], 0, stdout=b"not a tar", stderr=b"")
    monkeypatch.setattr(ltx25.subprocess, "run", lambda *args, **kwargs: malformed)

    with pytest.raises(ltx25.LTX25BackendError, match="could not be materialized"):
        ltx25._materialize_runtime(tmp_path, tmp_path / "snapshot")


def test_ltx25_runtime_is_provisioned_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "checkout" / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", None)
    monkeypatch.setattr(ltx25.shutil, "which", lambda _: "/trusted/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda *args: None)

    def provision(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        workspace = Path(command[command.index("--project") + 1])
        interpreter = workspace / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("#!/bin/sh\n")
        interpreter.chmod(0o755)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ltx25.subprocess, "run", provision)

    first = ltx25.prepare_ltx25_runtime(str(runtime))
    second = ltx25.prepare_ltx25_runtime(str(runtime))

    assert first == second
    assert len(calls) == 1
    assert calls[0][0][:3] == ["/trusted/uv", "sync", "--frozen"]


def test_ltx25_generation_uses_cache_without_rechecking_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ltx25, "_prepared_ltx25_runtime", lambda: tmp_path / "runtime-cache"
    )
    monkeypatch.setattr(
        ltx25,
        "resolve_ltx25_runtime",
        lambda: pytest.fail("a prepared runtime must not recheck the checkout"),
    )

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            self.command = command

        def communicate(self, *, input: str, timeout: int) -> None:
            Path(self.command[self.command.index("--output") + 1]).write_bytes(b"mp4")

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8").generate(
        prompt="fox",
        output_path=tmp_path / "result.mp4",
        width=704,
        height=480,
        num_frames=97,
        fps=24,
        seed=7,
        image=None,
    )


def test_serve_routes_ltx25_model_to_specific_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx import cli

    class PreflightReachedError(RuntimeError):
        pass

    def stop_at_preflight(model_name: str) -> None:
        assert model_name == "ltx-2.5-mlx-q8"
        raise PreflightReachedError

    monkeypatch.setattr(video_lane, "require_video_runtime_or_exit", stop_at_preflight)
    args = SimpleNamespace(model="ltx-2.5-mlx-q8", max_tokens=None, watchdog_ppid=None)
    with pytest.raises(PreflightReachedError):
        cli.serve_command(args)


def test_ltx25_engine_invokes_pinned_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.mp4"
    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    calls: list[tuple[list[str], dict]] = []
    communicates: list[tuple[str, int]] = []
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    runtime_python = runtime.with_name("python")
    runtime_python.write_text("#!/bin/sh\n")
    runtime_python.chmod(0o755)
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: str(runtime))
    runtime_cache = tmp_path / "runtime-cache"
    monkeypatch.setattr(ltx25, "prepare_ltx25_runtime", lambda _: runtime_cache)

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            self.command = command
            calls.append((command, kwargs))

        def communicate(self, *, input: str, timeout: int) -> tuple[str, str]:
            communicates.append((input, timeout))
            generated = Path(self.command[self.command.index("--output") + 1])
            generated.write_bytes(b"mp4-with-audio")
            return "", ""

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8").generate(
        prompt="a fox",
        output_path=output,
        width=704,
        height=480,
        num_frames=97,
        fps=24,
        seed=7,
        image=image,
        conditioning_strength=0.6,
    )

    command, run_kwargs = calls[0]
    assert command[:2] == [str(runtime_cache / ".venv/bin/python"), "-c"]
    assert command[2] == ltx25._STDIN_PROMPT_RUNNER
    assert command[3:5] == ["generate", "--model"]
    generated = Path(command[command.index("--output") + 1])
    assert generated != output
    assert generated.parent.parent == output.parent
    assert output.read_bytes() == b"mp4-with-audio"
    assert "--distilled" in command
    assert "--low-ram" in command
    assert "a fox" not in command
    assert command[command.index("--image") + 1 :] == [str(image), "0", "0.6"]
    assert run_kwargs == {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "text": True,
        "start_new_session": True,
    }
    assert communicates == [("a fox", 7200)]
    assert output.read_bytes() == b"mp4-with-audio"


def test_ltx25_engine_reports_subprocess_failure_without_leaking_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    runtime.with_name("python").write_text("#!/bin/sh\n")
    runtime.with_name("python").chmod(0o755)
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: str(runtime))
    monkeypatch.setattr(
        ltx25, "prepare_ltx25_runtime", lambda _: tmp_path / "runtime-cache"
    )

    class Process:
        returncode = 1

        def __init__(self, command: list[str], **kwargs) -> None:
            pass

        def communicate(self, *, input: str, timeout: int) -> tuple[str, str]:
            return "private-output-sentinel", "private-error-sentinel"

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    terminated = []
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: terminated.append(candidate)),
    )
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    with pytest.raises(ltx25.LTX25BackendError) as exc:
        engine.generate(
            prompt="a fox",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )
    assert str(exc.value) == (
        "LTX-2.5 runtime exited with code 1; runtime output is not retained "
        "because it may contain request data."
    )
    assert "private-output-sentinel" not in str(exc.value)
    assert "private-error-sentinel" not in str(exc.value)
    assert len(terminated) == 1


def test_ltx25_zero_exit_cannot_reuse_stale_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.mp4"
    output.write_bytes(b"stale-video")
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: str(runtime))
    monkeypatch.setattr(
        ltx25, "prepare_ltx25_runtime", lambda _: tmp_path / "runtime-cache"
    )

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            pass

        def communicate(self, *, input: str, timeout: int) -> tuple[str, str]:
            return "", ""

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    terminated = []
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: terminated.append(candidate)),
    )

    with pytest.raises(ltx25.LTX25BackendError, match="without an MP4 output"):
        ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8").generate(
            prompt="a fox",
            output_path=output,
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )

    assert output.read_bytes() == b"stale-video"
    assert list(tmp_path.glob(".result.mp4.*")) == []
    assert len(terminated) == 1


def test_ltx25_timeout_terminates_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    runtime.with_name("python").write_text("#!/bin/sh\n")
    runtime.with_name("python").chmod(0o755)
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: str(runtime))
    monkeypatch.setattr(
        ltx25, "prepare_ltx25_runtime", lambda _: tmp_path / "runtime-cache"
    )
    terminated = []

    class Process:
        returncode = None

        def __init__(self, command: list[str], **kwargs) -> None:
            pass

        def communicate(self, *, input: str, timeout: int) -> None:
            raise subprocess.TimeoutExpired("ltx-2-mlx", timeout)

    process = Process([], stdin=subprocess.PIPE, text=True)
    monkeypatch.setattr(ltx25.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: terminated.append(candidate)),
    )
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")

    with pytest.raises(ltx25.LTX25BackendError, match="time limit"):
        engine.generate(
            prompt="a fox",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )

    assert terminated == [process]
    assert engine._process is None


def test_ltx25_termination_does_not_reuse_exited_leader_pgid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = []

    class Process:
        pid = 123

        def poll(self) -> int:
            return 1

        def wait(self, *, timeout: int) -> None:
            pytest.fail("an exited group leader must not be waited on again")

    monkeypatch.setattr(ltx25.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    ltx25.LTX25VideoEngine._terminate_process(Process())  # type: ignore[arg-type]

    assert signals == []


def test_ltx25_termination_escalates_while_leader_is_unreaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = []

    class Process:
        pid = 123
        waits = 0

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: int) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("ltx", timeout)
            return -signal.SIGKILL

    process = Process()
    monkeypatch.setattr(ltx25.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    ltx25.LTX25VideoEngine._terminate_process(process)  # type: ignore[arg-type]

    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
    assert process.waits == 2


def test_ltx25_unexpected_communication_error_terminates_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: str(runtime))
    monkeypatch.setattr(
        ltx25, "prepare_ltx25_runtime", lambda _: tmp_path / "runtime-cache"
    )
    terminated = []

    class Process:
        returncode = None

        def communicate(self, *, input: str, timeout: int) -> None:
            raise UnicodeEncodeError("utf-8", "bad surrogate \ud800", 14, 15, "invalid")

        def poll(self) -> None:
            return None

    process = Process()
    monkeypatch.setattr(ltx25.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: terminated.append(candidate)),
    )
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")

    with pytest.raises(ltx25.LTX25BackendError, match="isolated runtime"):
        engine.generate(
            prompt="bad surrogate \ud800",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )

    assert terminated == [process]
    assert engine._process is None


def test_ltx25_invalid_timeout_does_not_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    runtime.with_name("python").write_text("#!/bin/sh\n")
    runtime.with_name("python").chmod(0o755)
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: str(runtime))
    monkeypatch.setattr(ltx25.shutil, "which", lambda _: "/trusted/uv")
    monkeypatch.setenv("RAPID_MLX_LTX25_TIMEOUT_SEC", "invalid")
    monkeypatch.setattr(
        ltx25.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("invalid config must fail before spawn"),
    )

    with pytest.raises(ltx25.LTX25BackendError, match="integer"):
        ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8").generate(
            prompt="a fox",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )


def test_video_engine_selects_ltx25_and_preserves_generated_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def generate(self, **kwargs) -> None:
        captured.update(kwargs)
        kwargs["output_path"].write_bytes(b"mp4-with-audio")

    monkeypatch.setattr(ltx25.LTX25VideoEngine, "generate", generate)
    monkeypatch.setattr(
        VideoEngine,
        "_crop_generated_output",
        lambda *args, **kwargs: captured.update(crop=kwargs),
    )
    engine = VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    output = tmp_path / "result.mp4"
    engine.generate(
        prompt="a fox",
        output_path=output,
        width=704,
        height=512,
        num_frames=97,
        fps=24,
        seed=7,
        image=None,
    )

    assert engine.video_family == "ltx-2.5"
    assert output.read_bytes() == b"mp4-with-audio"
    assert captured["crop"]["family"] == "LTX-2.5"


def test_ltx25_real_crop_preserves_audio_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffmpeg = video_lane._resolve_ffmpeg()
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the real media assertion")
    output = tmp_path / "with-audio.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=0.2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    VideoEngine._crop_generated_output(
        output_path=output,
        width=64,
        height=64,
        output_width=32,
        output_height=32,
        family="LTX-2.5",
    )
    streams = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert streams.stdout.strip() == "audio"


def test_ltx25_distilled_rejects_cfg_controls() -> None:
    engine = VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    with pytest.raises(VideoRuntimeError, match="does not support"):
        engine.generate(
            prompt="a fox",
            output_path=Path("result.mp4"),
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
            guidance_scale=4.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported",
    [{"guidance_scale": 4.0}, {"negative_prompt": "blurry"}],
)
async def test_ltx25_route_rejects_unsupported_cfg_before_queueing(
    monkeypatch: pytest.MonkeyPatch, unsupported: dict[str, object]
) -> None:
    from vllm_mlx.routes import video

    engine = SimpleNamespace(
        model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5"
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)
    monkeypatch.setattr(video, "_accepting_jobs", True)
    before = set(video._jobs)

    with pytest.raises(HTTPException, match="does not support") as exc:
        await video.create_video(
            prompt="a fox",
            model="ltx-2.5-mlx-q8",
            seconds="1",
            size="704x512",
            seed=7,
            input_reference=None,
            **unsupported,
        )

    assert exc.value.status_code == 400
    assert set(video._jobs) == before
