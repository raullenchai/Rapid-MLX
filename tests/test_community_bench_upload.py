# SPDX-License-Identifier: Apache-2.0
"""Transport layer for community benchmark submissions.

mlx-free on purpose: ``upload.py`` is stdlib-only and these tests never
import MLX, so they run on the Linux CI runner rather than being silently
skipped there (the failure mode #1236 shipped a defect through).
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from vllm_mlx.community_bench import upload


def test_install_id_is_stable_across_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    first = upload.install_id()
    assert len(first) == 12
    assert all(c in "0123456789abcdef" for c in first)
    assert upload.install_id() == first, "the id must persist, not be re-minted"


def test_install_id_is_not_derived_from_hardware(tmp_path, monkeypatch) -> None:
    """Two fresh installs must not collide.

    oMLX derives its owner_hash from IOPlatformUUID, which makes every public
    row traceable to a machine. Ours is random per install; the test that
    proves it is that a second install on the SAME machine gets a different
    id.
    """
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path / "a"))
    a = upload.install_id()
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path / "b"))
    b = upload.install_id()
    assert a != b


def test_install_id_survives_an_unwritable_home(tmp_path, monkeypatch) -> None:
    """An unwritable config dir must not block a submission."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    monkeypatch.setenv("RAPID_MLX_HOME", str(blocker))
    got = upload.install_id()
    assert len(got) == 12


def test_run_group_ids_are_distinct() -> None:
    assert upload.new_run_group() != upload.new_run_group()


def test_board_url_env_override(monkeypatch) -> None:
    monkeypatch.delenv(upload.BOARD_URL_ENV, raising=False)
    assert upload.board_url() == upload.DEFAULT_BOARD_URL
    monkeypatch.setenv(upload.BOARD_URL_ENV, "http://127.0.0.1:8787/api/benchmarks")
    assert upload.board_url() == "http://127.0.0.1:8787/api/benchmarks"


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_post_returns_decoded_response(monkeypatch) -> None:
    seen = {}

    def fake_open(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        seen["ct"] = req.headers.get("Content-type")
        return _Resp(b'{"ok":true,"submission_id":"abcdef012345"}')

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_open)
    got = upload.post_submission({"submission_id": "abcdef012345"}, url="https://x/api")
    assert got["ok"] is True
    assert seen["url"] == "https://x/api"
    assert seen["body"]["submission_id"] == "abcdef012345"
    assert seen["ct"] == "application/json"


def test_4xx_is_not_retried(monkeypatch) -> None:
    """A rejected payload will be rejected identically on every retry.

    Retrying a 400 just replays the same failure and, on a 429, actively
    fights the response.
    """
    calls = []

    def fake_open(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 400, "Bad", {}, None)

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_open)
    with pytest.raises(upload.SubmitError) as exc:
        upload.post_submission({"a": 1}, url="https://x/api")
    assert calls == [1], "a 4xx must be attempted exactly once"
    assert "400" in str(exc.value)


def test_5xx_is_retried_then_surfaces(monkeypatch) -> None:
    calls = []

    def fake_open(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 503, "Down", {}, None)

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_open)
    with pytest.raises(upload.SubmitError):
        upload.post_submission({"a": 1}, url="https://x/api")
    assert len(calls) == 3, "transient server errors get the full retry budget"


def test_transient_transport_error_then_success(monkeypatch) -> None:
    calls = []

    def fake_open(req, timeout=None):
        calls.append(1)
        if len(calls) < 2:
            raise urllib.error.URLError("connection reset")
        return _Resp(b'{"ok":true}')

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_open)
    assert upload.post_submission({"a": 1}, url="https://x/api")["ok"] is True
    assert len(calls) == 2


def test_unreachable_board_mentions_the_local_copy(monkeypatch) -> None:
    """The error text is the contributor's recovery instruction."""

    def fake_open(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_open)
    with pytest.raises(upload.SubmitError) as exc:
        upload.post_submission({"a": 1}, url="https://x/api")
    assert "saved" in str(exc.value).lower()


def test_consent_text_describes_what_actually_happens() -> None:
    """A consent prompt that misdescribes the action is worse than none.

    The original text promised ``git fetch`` / fork creation / ``gh pr
    create``. Those were accurate for the pull-request flow and became a lie
    the moment submission turned into an HTTPS POST — this test fails if the
    prompt ever drifts back.
    """
    import io

    from vllm_mlx.community_bench.submission import _ask_consent

    out = io.StringIO()
    _ask_consent(
        {"hardware": {"chip": "Apple M2 Pro", "ram_gb": 32}, "model": {"alias": "x"}},
        stdin=io.StringIO("n\n"),
        stdout=out,
    )
    text = out.getvalue()
    for stale in ("gh pr create", "git push", "git fetch", "fork"):
        assert stale not in text, f"consent text still promises {stale!r}"
    assert "POST" in text
    assert "PUBLIC" in text
    assert "bench-install-id" in text, "the reset path must be discoverable"
