"""CLI Hub recovery with a mocked downloader; no model or network required."""

import argparse
import errno
import http.client
from types import SimpleNamespace

import httpx
import pytest

from rapid_mlx import cli
from tests.pull_integrity_helpers import (  # noqa: F401
    hub_integrity_metadata,
    snapshot_tree,
)

pytestmark = pytest.mark.usefixtures("ample_pull_disk_space")


def test_pull_disk_full_has_retry_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _repo: tmp_path)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")),
    )
    with pytest.raises(SystemExit) as raised:
        cli._pull_repository(
            argparse.Namespace(model="owner/repo", bits=None, format=None),
            emit_lifecycle_event=False,
        )
    assert raised.value.code == 1
    stderr = capsys.readouterr().err
    assert "not enough disk space" in stderr
    assert "Retry: rapid-mlx pull owner/repo" in stderr


def test_runtime_asset_retry_preserves_parent_pull(monkeypatch):
    from rapid_mlx._download_gate import SD35_REPO

    seen = []
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)

    def pull(args, **_kwargs):
        seen.append((args.model, args._pull_retry_command))

    monkeypatch.setattr(cli, "_pull_repository", pull)
    cli.pull_command(argparse.Namespace(model=SD35_REPO, bits=None, format=None))
    assert len(seen) > 1
    assert all(command == f"rapid-mlx pull {SD35_REPO}" for _, command in seen)


def _hub_error(error_type, status):
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://huggingface.co/x")
    )
    return error_type("secret response body", response=response)


def test_retry_reuses_same_cache_and_stops_after_success(monkeypatch, capsys):
    import requests

    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    cache = {"finished": False}
    calls = []

    def transfer():
        calls.append(cache["finished"])
        if not cache["finished"]:
            cache["finished"] = True
            raise requests.ConnectionError("private URL")
        return "snapshot"

    assert cli._retry_hub_download(transfer, "owner/repo") == "snapshot"
    assert calls == [False, True]
    assert sleeps == [1]
    assert "private URL" not in capsys.readouterr().err


@pytest.mark.parametrize("suppress_context", [False, True])
def test_contextual_timeout_is_not_retried(monkeypatch, capsys, suppress_context):
    calls = []
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    def transfer():
        calls.append(1)
        try:
            raise TimeoutError("private timeout")
        except TimeoutError:
            if suppress_context:
                raise RuntimeError("wrapper") from None
            raise RuntimeError("wrapper")

    with pytest.raises(RuntimeError, match="wrapper") as raised:
        cli._retry_hub_download(transfer, "owner/repo")
    assert not cli._is_transient_hub_error(raised.value)
    assert calls == [1]
    assert sleeps == []
    assert "retrying" not in capsys.readouterr().err


def test_retry_is_bounded_for_network_and_server_errors(monkeypatch):
    import requests
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    wrapped = LocalEntryNotFoundError("uncached")
    wrapped.__cause__ = httpx.ConnectError("private URL")

    monkeypatch.setattr("time.sleep", lambda _: None)
    for failure in (
        requests.ConnectionError("dropped"),
        _hub_error(HfHubHTTPError, 503),
        wrapped,
        httpx.RemoteProtocolError("peer closed with secret URL"),
        httpx.ProxyError("proxy rejected secret URL"),
    ):
        calls = []

        def transfer(failure=failure, calls=calls):
            calls.append(1)
            raise failure

        with pytest.raises(type(failure)):
            cli._retry_hub_download(transfer, "owner/repo")
        assert len(calls) == 3


def test_permanent_hub_errors_are_not_retried(monkeypatch):
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
    )

    monkeypatch.setattr("time.sleep", lambda _: pytest.fail("unexpected backoff"))
    for failure in (
        _hub_error(GatedRepoError, 403),
        _hub_error(RepositoryNotFoundError, 404),
        _hub_error(HfHubHTTPError, 401),
        OfflineModeIsEnabled("offline"),
    ):
        calls = []

        def transfer(failure=failure, calls=calls):
            calls.append(1)
            raise failure

        with pytest.raises(type(failure)):
            cli._retry_hub_download(transfer, "owner/repo")
        assert len(calls) == 1


def test_pull_retries_mocked_snapshot_and_keeps_revision(
    hub_online_env, monkeypatch, tmp_path
):
    import requests

    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _repo: tmp_path)
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs.copy()))
        if len(calls) == 1:
            raise requests.ConnectionError("dropped")
        return str(snapshot_tree(tmp_path))

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    cli._pull_repository(
        argparse.Namespace(model="owner/repo", bits=None, format=None),
        revision_override="a" * 40,
    )
    assert calls == [
        ("owner/repo", {"revision": "a" * 40}),
        ("owner/repo", {"revision": "a" * 40}),
    ]


def test_gated_and_misspelled_repo_guidance(monkeypatch):
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    gated = cli.render_hub_error(_hub_error(GatedRepoError, 403), "owner/private")
    assert "https://huggingface.co/owner/private" in gated
    assert "hf auth login" in gated
    assert "HF_TOKEN" in gated

    monkeypatch.setattr(
        "rapid_mlx.model_aliases.list_builtin_aliases",
        lambda: {"qwen3-0.6b-4bit": "mlx-community/Qwen3-0.6B-4bit"},
    )
    missing = cli.render_hub_error(
        _hub_error(RepositoryNotFoundError, 404),
        "mlx-community/Qwen3-0.6B-4bi",
    )
    assert "qwen3-0.6b-4bit" in missing
    assert "namespace/repository-name" in missing
    assert "secret response body" not in missing

    protocol = cli.render_hub_error(
        httpx.RemoteProtocolError("peer closed with secret URL"), "owner/model"
    )
    assert "could not reach Hugging Face" in protocol
    assert "secret URL" not in protocol

    proxy = cli.render_hub_error(
        httpx.ProxyError("403 secret proxy URL"), "owner/model"
    )
    assert "could not reach Hugging Face" in proxy
    assert "secret proxy URL" not in proxy

    server_error = cli.render_hub_error(
        _hub_error(HfHubHTTPError, 503),
        "owner/model",
    )
    assert "temporarily unavailable" in server_error

    monkeypatch.setattr("rapid_mlx.model_aliases.suggest_similar", lambda _name: [])
    monkeypatch.setattr("rapid_mlx.model_aliases.list_builtin_aliases", lambda: {})
    assert "rapid-mlx models" in cli._hub_not_found_hint("owner/unknown")


def test_proxy_rejection_is_a_network_failure():
    from rapid_mlx.telemetry.model_events import pull_error_class

    assert pull_error_class(httpx.ProxyError("403 Forbidden")) == "network"


@pytest.mark.parametrize("emit_lifecycle_event", [False, True])
def test_mirror_only_integrity_failure_has_safe_cli_guidance(
    monkeypatch, tmp_path, capsys, emit_lifecycle_event
):
    from rapid_mlx._mirror import MirrorOnlyPullError

    events = []
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_pull_failed",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _repo: tmp_path)
    monkeypatch.setattr(
        cli,
        "_try_mirror_prefetch",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            MirrorOnlyPullError("private file path and digest")
        ),
    )
    with pytest.raises(SystemExit) as raised:
        cli._pull_repository(
            argparse.Namespace(model="owner/repo", bits=None, format=None),
            emit_lifecycle_event=emit_lifecycle_event,
        )
    assert raised.value.code == 1
    assert len(events) == int(emit_lifecycle_event)
    stderr = capsys.readouterr().err
    assert "cannot verify every file" in stderr
    assert "Retry: rapid-mlx pull owner/repo" in stderr
    assert "private file path and digest" not in stderr


@pytest.mark.parametrize(
    ("status", "expected"),
    [(403, "hf auth login"), (404, "Closest catalog aliases")],
)
def test_pull_exits_with_actionable_permanent_error(
    monkeypatch, capsys, tmp_path, status, expected
):
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    repo = "mlx-community/Qwen3-0.6B-4bi"
    failure_type = GatedRepoError if status == 403 else RepositoryNotFoundError
    failure = _hub_error(failure_type, status)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _repo: tmp_path)
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.list_builtin_aliases",
        lambda: {"qwen3-0.6b-4bit": "mlx-community/Qwen3-0.6B-4bit"},
    )
    calls = []

    def download(*_args, **_kwargs):
        calls.append(1)
        raise failure

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(SystemExit) as caught:
        cli._pull_repository(
            argparse.Namespace(model=repo, bits=None, format=None),
            emit_lifecycle_event=False,
        )
    assert caught.value.code == 1
    assert len(calls) == 1
    output = capsys.readouterr()
    assert expected in output.out + output.err
    assert "secret response body" not in output.out + output.err
    assert "Downloaded" not in output.out + output.err


def test_serve_prefetch_retries_snapshot_with_pinned_revision(monkeypatch):
    import requests

    repo = "owner/model"
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_args, **_kwargs: SimpleNamespace(
            sha="a" * 40, siblings=[]
        ),
    )
    monkeypatch.setattr("rapid_mlx._download_gate.pin_main_ref", lambda *_a: None)
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs.copy()))
        if len(calls) == 1:
            raise requests.ConnectionError("dropped")
        return "/tmp/fake-snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    cli._ensure_model_downloaded(repo)
    assert calls == [
        (repo, {"revision": "a" * 40}),
        (repo, {"revision": "a" * 40}),
    ]


def test_pull_protocol_disconnect_has_bounded_retry_and_safe_guidance(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _repo: tmp_path)
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = []

    def download(*_args, **_kwargs):
        calls.append(1)
        raise httpx.RemoteProtocolError("peer closed with private URL")

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(SystemExit) as caught:
        cli._pull_repository(
            argparse.Namespace(model="owner/repo", bits=None, format=None),
            emit_lifecycle_event=False,
        )
    assert caught.value.code == 1
    assert len(calls) == 3
    stderr = capsys.readouterr().err
    assert "could not reach Hugging Face" in stderr
    assert "private URL" not in stderr


def test_pull_variant_metadata_retries_before_transfer(
    hub_online_env, monkeypatch, tmp_path
):
    probe_calls = []

    def resolve_variant(*_args):
        probe_calls.append(1)
        if len(probe_calls) == 1:
            raise httpx.ConnectError("temporary")
        return ["4bit/*"]

    monkeypatch.setattr(cli, "_resolve_variant_allow_patterns", resolve_variant)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _repo: tmp_path)
    monkeypatch.setattr("time.sleep", lambda _: None)
    snapshot_calls = []

    def download(_repo, **kwargs):
        snapshot_calls.append(kwargs)
        return str(snapshot_tree(tmp_path, filename="4bit/model.safetensors"))

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    cli._pull_repository(
        argparse.Namespace(model="owner/repo", bits="4", format=None),
        emit_lifecycle_event=False,
    )
    assert len(probe_calls) == 2
    assert snapshot_calls == [{"allow_patterns": ["4bit/*"]}]


def test_transient_mirror_failure_falls_back_to_hub(
    hub_online_env, monkeypatch, tmp_path, capsys
):
    def mirror(*_args, **_kwargs):
        raise httpx.RemoteProtocolError("mirror peer closed with secret URL")

    monkeypatch.setattr(cli, "_try_mirror_prefetch", mirror)
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _repo: tmp_path)
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, **kwargs: (
            calls.append((repo, kwargs)) or str(snapshot_tree(tmp_path))
        ),
    )
    cli._pull_repository(
        argparse.Namespace(model="owner/repo", bits=None, format=None),
        emit_lifecycle_event=False,
    )
    assert calls == [("owner/repo", {})]
    output = capsys.readouterr().out
    assert "trying Hugging Face" in output
    assert "secret URL" not in output


def test_serve_transient_mirror_failure_falls_back_to_hub(monkeypatch, capsys):
    def mirror(*_args, **_kwargs):
        raise httpx.RemoteProtocolError("mirror closed with secret URL")

    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", mirror)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_args, **_kwargs: SimpleNamespace(
            sha="a" * 40, siblings=[]
        ),
    )
    monkeypatch.setattr("rapid_mlx._download_gate.pin_main_ref", lambda *_a: None)
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, **kwargs: calls.append((repo, kwargs)) or "/tmp/fake-snapshot",
    )
    cli._ensure_model_downloaded("owner/repo")
    assert calls == [("owner/repo", {"revision": "a" * 40})]
    output = capsys.readouterr().out
    assert "trying Hugging Face" in output
    assert "secret URL" not in output


def test_variant_metadata_error_is_guided_without_hiding_unknown_errors(
    monkeypatch, capsys
):
    failure = httpx.RemoteProtocolError("secret URL")

    def fail(*_args):
        raise failure

    monkeypatch.setattr(cli, "_resolve_variant_allow_patterns", fail)
    monkeypatch.setattr("time.sleep", lambda _: None)
    args = argparse.Namespace(model="owner/repo", bits="4", format=None)
    with pytest.raises(SystemExit) as caught:
        cli._pull_repository(args, emit_lifecycle_event=False)
    assert caught.value.code == 1
    assert "could not reach Hugging Face" in capsys.readouterr().err

    unknown = RuntimeError("implementation bug")

    def fail_unknown(*_args):
        raise unknown

    monkeypatch.setattr(cli, "_resolve_variant_allow_patterns", fail_unknown)
    with pytest.raises(RuntimeError) as caught_unknown:
        cli._pull_repository(args, emit_lifecycle_event=False)
    assert caught_unknown.value is unknown


@pytest.mark.parametrize("status", [429, 503])
def test_shared_error_renderer_does_not_claim_unobserved_attempts(status):
    from huggingface_hub.errors import HfHubHTTPError

    # Metadata callers may render a single failure without running the retry helper.
    rendered = cli.render_hub_error(_hub_error(HfHubHTTPError, status), "owner/model")
    assert "temporarily unavailable" in rendered
    assert "completed cached files will be reused" in rendered
    assert "after three attempts" not in rendered


def test_verified_hub_interruption_does_not_start_snapshot_download(
    monkeypatch, capsys
):
    from rapid_mlx._mirror import ResumableDownloadError

    def transfer(*args, **kwargs):
        raise ResumableDownloadError("resume retained bytes") from httpx.ReadError(
            "private URL"
        )

    monkeypatch.setattr(cli, "_try_mirror_prefetch", transfer)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **k: pytest.fail("must not start an unrelated Hub blob"),
    )
    with pytest.raises(SystemExit) as caught:
        cli._pull_repository(
            argparse.Namespace(model="owner/repo", bits=None, format=None),
            emit_lifecycle_event=False,
        )
    assert caught.value.code == 1
    output = capsys.readouterr().err
    assert "could not reach Hugging Face" in output
    assert "private URL" not in output


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionResetError("private"),
        http.client.IncompleteRead(b"private"),
    ],
)
def test_wrapped_stream_failure_has_network_guidance(failure):
    from rapid_mlx._mirror import ResumableDownloadError

    error = ResumableDownloadError("interrupted")
    error.__cause__ = failure
    assert "could not reach Hugging Face" in cli.render_hub_error(error, "owner/repo")


def test_verified_prefetch_retries_shared_downloader(monkeypatch):
    from rapid_mlx import _mirror

    sleeps = []
    calls = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    def download(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise _mirror.ResumableDownloadError("interrupted") from httpx.ReadError(
                "private"
            )
        return True

    monkeypatch.setattr(_mirror, "download_with_mirror_fallback", download)
    assert cli._try_mirror_prefetch("owner/repo")
    assert len(calls) == 2
    assert sleeps == [1]


@pytest.mark.parametrize(
    "status", [429, 500, 501, 502, 503, 504, 599, 401, 403, 404, 400]
)
def test_direct_hub_http_errors_retry_and_render_safely(status, tmp_path, monkeypatch):
    import urllib.error

    from rapid_mlx import _mirror

    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    metadata = SimpleNamespace(location="https://cdn.example.test/private")
    monkeypatch.setattr(
        "huggingface_hub.file_download.get_hf_file_metadata", lambda *a, **k: metadata
    )
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise urllib.error.HTTPError(
            metadata.location, status, "private body", {}, None
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)

    def transfer():
        return _mirror._hf_resumable_one(
            "owner/repo",
            "model.safetensors",
            "a" * 40,
            tmp_path / "target",
            100,
            "a" * 64,
            None,
            tmp_path / "sidecar",
            tmp_path,
            None,
        )

    if status in (401, 403):
        # Anonymous access denied: hand off to the credential-aware Hub client.
        assert cli._retry_hub_download(transfer, "owner/repo") is False
        assert len(calls) == 1
        assert sleeps == []
        return

    with pytest.raises(_mirror.ResumableDownloadError) as caught:
        cli._retry_hub_download(transfer, "owner/repo")
    retryable = status == 429 or 500 <= status < 600
    assert len(calls) == (3 if retryable else 1)
    assert sleeps == ([1, 2] if retryable else [])
    rendered = cli.render_hub_error(caught.value, "owner/repo")
    if retryable:
        assert "Hugging Face is temporarily unavailable" in rendered
    elif status == 404:
        assert "not found" in rendered
    else:
        assert rendered is None
    assert "private" not in (rendered or "")


def test_r2_transient_errors_retry_with_backoff_and_hf_resumes_partial(
    tmp_path, monkeypatch
):
    import hashlib
    import io
    import urllib.error

    from rapid_mlx import _mirror

    body = b"0123456789" * 10
    sha = hashlib.sha256(body).hexdigest()
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    class _FakeResp:
        def __init__(self, status: int, data: bytes, headers: dict[str, str]):
            self.status = status
            self._buf = io.BytesIO(data)
            self.headers = headers

        def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)

        def read1(self, n: int = -1) -> bytes:
            return self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    # 1. Cold R2 transient 503 + URLError retry in place with bounded backoff.
    r2_attempts = 0

    def fake_r2_urlopen(req, timeout=None):
        nonlocal r2_attempts
        r2_attempts += 1
        if r2_attempts == 1:
            raise urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", {}, None
            )
        if r2_attempts == 2:
            raise urllib.error.URLError("temporary dns failure")
        return _FakeResp(200, body, {"Content-Length": str(len(body))})

    monkeypatch.setattr("urllib.request.urlopen", fake_r2_urlopen)
    target = tmp_path / "r2" / "snapshots" / ("a" * 40) / "model.safetensors"
    ok, reason = _mirror._download_one_from_r2(
        "https://models.rapidmlx.com/owner/repo/model.safetensors",
        target,
        len(body),
        expected_sha256=sha,
        sidecar_dir=tmp_path / "r2" / ".rapid-mlx-mirror",
        sidecar_key="model.safetensors",
        repo_root=tmp_path / "r2",
        enforce_floor=False,
        max_retries=2,
    )
    assert (ok, reason) == (True, "")
    assert target.read_bytes() == body
    assert r2_attempts == 3
    assert sleeps == [1.0, 2.0]

    # 2. Mid-stream disconnects resume partial .part bytes via Range across retries.
    sleeps.clear()
    seen_ranges: list[str | None] = []
    hf_attempts = 0
    location = "https://cdn.example.test/model.safetensors"
    monkeypatch.setattr(
        "huggingface_hub.file_download.get_hf_file_metadata",
        lambda *_a, **_kw: SimpleNamespace(location=location),
    )

    def fake_hf_urlopen(req, timeout=None):
        nonlocal hf_attempts
        hf_attempts += 1
        seen_ranges.append(req.get_header("Range"))
        if hf_attempts == 1:
            return _FakeResp(200, body[:40], {"Content-Length": str(len(body))})
        if hf_attempts == 2:
            assert req.get_header("Range") == "bytes=40-"
            return _FakeResp(
                206,
                body[40:75],
                {
                    "Content-Length": str(len(body) - 40),
                    "Content-Range": f"bytes 40-{len(body) - 1}/{len(body)}",
                },
            )
        assert req.get_header("Range") == "bytes=75-"
        return _FakeResp(
            206,
            body[75:],
            {
                "Content-Length": str(len(body) - 75),
                "Content-Range": f"bytes 75-{len(body) - 1}/{len(body)}",
            },
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_hf_urlopen)
    hf_root = tmp_path / "hf"
    hf_target = hf_root / "snapshots" / ("a" * 40) / "model.safetensors"
    assert (
        cli._retry_hub_download(
            lambda: _mirror._hf_resumable_one(
                "owner/repo",
                "model.safetensors",
                "a" * 40,
                hf_target,
                len(body),
                sha,
                None,
                hf_root / ".rapid-mlx-mirror",
                hf_root,
                None,
            ),
            "owner/repo",
        )
        is True
    )
    assert hf_target.read_bytes() == body
    assert seen_ranges == [None, "bytes=40-", "bytes=75-"]
    assert sleeps == [1, 2]


def test_r2_permanent_404_does_not_retry(tmp_path, monkeypatch):
    import urllib.error

    from rapid_mlx import _mirror

    sleeps: list[float] = []
    monkeypatch.setattr(_mirror.time, "sleep", sleeps.append)
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, _reason = _mirror._download_one_from_r2(
        "https://models.rapidmlx.com/owner/repo/missing.bin",
        tmp_path / "missing.bin",
        10,
        sidecar_dir=tmp_path / "sidecar",
        sidecar_key="missing.bin",
        enforce_floor=False,
        max_retries=2,
    )
    assert ok is False
    assert len(calls) == 1
    assert sleeps == []


def test_model_downloads_retries_transient_errors_with_bounded_backoff(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    from rapid_mlx import model_downloads

    sleeps: list[float] = []
    monkeypatch.setattr(model_downloads.time, "sleep", sleeps.append)

    snap_calls = 0

    def fake_snap(repo_id, **kwargs):
        nonlocal snap_calls
        snap_calls += 1
        if snap_calls < 3:
            raise httpx.ReadTimeout("slow")
        return "/cache/snap"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snap)
    assert model_downloads.snapshot_download("owner/repo") == "/cache/snap"
    assert snap_calls == 3
    assert sleeps == [1.0, 2.0]

    sleeps.clear()
    file_calls = 0

    def fake_file(repo_id, filename, **kwargs):
        nonlocal file_calls
        file_calls += 1
        if file_calls == 1:
            raise _hub_error(HfHubHTTPError, 502)
        return "/cache/file"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_file)
    assert model_downloads.hf_hub_download("owner/repo", "config.json") == "/cache/file"
    assert file_calls == 2
    assert sleeps == [1.0]

    # Permanent 403 must not retry.
    sleeps.clear()
    gated_calls = 0

    def fake_gated(*_a, **_kw):
        nonlocal gated_calls
        gated_calls += 1
        raise _hub_error(GatedRepoError, 403)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_gated)
    with pytest.raises(GatedRepoError):
        model_downloads.snapshot_download("owner/gated")
    assert gated_calls == 1
    assert sleeps == []


def test_final_hub_error_messages_include_next_steps(monkeypatch, capsys):
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
    )

    from rapid_mlx._mirror import ResumableDownloadError

    offline = cli.render_hub_error(OfflineModeIsEnabled("offline"), "owner/model")
    assert "HF_HUB_OFFLINE" in offline
    assert "TRANSFORMERS_OFFLINE" in offline
    assert "rapid-mlx pull owner/model" in offline

    net = cli.render_hub_error(httpx.ConnectError("down"), "owner/model")
    assert "HF_HUB_OFFLINE" in net
    assert "rapid-mlx pull owner/model" in net

    unavailable = cli.render_hub_error(
        _hub_error(HfHubHTTPError, 503),
        "owner/model",
        retry_command="rapid-mlx pull owner/model --bits 4",
    )
    assert "rapid-mlx pull owner/model --bits 4" in unavailable

    gated = cli.render_hub_error(_hub_error(GatedRepoError, 403), "owner/gated")
    assert "hf auth login" in gated and "HF_TOKEN" in gated

    monkeypatch.setattr(
        "rapid_mlx.model_aliases.list_builtin_aliases",
        lambda: {"qwen3.5-4b-4bit": "mlx-community/Qwen3.5-4B-MLX-4bit"},
    )
    missing = cli.render_hub_error(
        _hub_error(RepositoryNotFoundError, 404),
        "mlx-community/Qwen3.5-4B-MLX-4bi",
    )
    assert "qwen3.5-4b-4bit" in missing

    # Auto-pull in serve/chat prints an actionable retry command on fatal resumable error.
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli,
        "_try_mirror_prefetch",
        lambda *_a, **_kw: (_ for _ in ()).throw(ResumableDownloadError("interrupted")),
    )
    with pytest.raises(SystemExit) as caught:
        cli._ensure_target_model_downloaded("owner/model")
    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert "None" not in err
    assert "Retry: rapid-mlx pull owner/model" in err
