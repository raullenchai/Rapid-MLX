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


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    """Retries sleep; tests should not."""
    monkeypatch.setattr(upload.time, "sleep", lambda _s: None)


def test_install_id_is_stable_across_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    first = upload.install_id()
    assert len(first) == 12
    assert all(c in "0123456789abcdef" for c in first)
    assert upload.install_id() == first, "the id must persist, not be re-minted"


def test_install_id_comes_from_the_csprng_not_the_machine(
    tmp_path, monkeypatch
) -> None:
    """The privacy claim is about *derivation*, so test derivation.

    An earlier version of this test only checked that two different home
    directories produced different ids — which an implementation hashing
    IOPlatformUUID together with the home path would also satisfy. Controlling
    the randomness and asserting the id IS that randomness is what actually
    pins the claim.

    oMLX derives its owner_hash from IOPlatformUUID, which makes every public
    row traceable to a specific machine. Ours must not.
    """
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    monkeypatch.setattr(upload.secrets, "token_hex", lambda n: "0" * (2 * n))
    assert upload.peek_install_id() == "0" * 12

    # And nothing machine-identifying is consulted along the way.
    import subprocess

    def _forbidden(*a, **k):  # pragma: no cover - only runs on regression
        raise AssertionError(f"install id must not shell out: {a!r}")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "check_output", _forbidden)
    assert upload.peek_install_id() == "0" * 12


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
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

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


def test_builder_output_validates_against_the_repo_schema() -> None:
    """The payload we generate must satisfy the schema we ship.

    Codex round 1 on #1403 caught this: the PR added ``config.spec_decode``
    and ``run_group`` while ``schema.json`` still declared
    ``additionalProperties: false`` and ``schema_version`` in ``[1, 2]`` — so
    the repository's own GHA validator would have rejected every payload the
    new CLI produced. Unit tests on the builder alone cannot see that; only
    checking the builder against the shipped schema can.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import jsonschema

    from vllm_mlx.community_bench.submission import build_submission_payload

    class _Stat:
        def to_schema_dict(self):
            return {
                "decode_tps": {"median": 61.4, "min": 61.2, "max": 61.5, "stddev": 0.1},
                "prefill_tps": {
                    "median": 300.6,
                    "min": 295.4,
                    "max": 302.8,
                    "stddev": 2.4,
                },
                "ttft_ms": {
                    "median": 1779.8,
                    "min": 1766.9,
                    "max": 1811.3,
                    "stddev": 14.7,
                },
                # Exactly 5, because the suite locks the round count and the
                # schema enforces it — a fixture with fewer would pass a unit
                # test while being unrepresentable on the wire.
                "rounds_raw": [
                    {"decode_tps": 61.2 + i, "prefill_tps": 295.4, "ttft_ms": 1811.3}
                    for i in range(5)
                ],
            }

    class _Bench:
        sampling = "greedy"
        prompt_hash = "abc123def4567890"
        peak_ram_mb = 4014
        short = _Stat()
        long = _Stat()

    from vllm_mlx.community_bench.hardware import Hardware, Software

    hw = Hardware(chip="Apple M2 Pro", ram_gb=32, cpu_cores=10, gpu_cores=16)
    sw = Software(macos="26.5.2", rapid_mlx="0.11.9", mlx="0.31.2", python="3.12.13")

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1] / "community-benchmarks" / "schema.json"
        ).read_text()
    )

    for spec, group in [
        (None, None),  # v3 baseline == v2 bytes
        ({"method": "mtp", "num_speculative_tokens": 3}, "abcdefabcdef"),
    ]:
        payload = build_submission_payload(
            hardware=hw,
            software=sw,
            alias="qwen3.5-4b-4bit",
            hf_path="mlx-community/Qwen3.5-4B-MLX-4bit",
            bench=_Bench(),
            notes=None,
            now=datetime(2026, 8, 2, tzinfo=timezone.utc),
            spec_decode=spec,
            run_group=group,
        )
        jsonschema.validate(payload, schema)

    # A baseline v3 payload must not smuggle in an empty spec_decode key —
    # that is what keeps it byte-comparable with the v1/v2 corpus.
    baseline = build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-4b-4bit",
        hf_path="mlx-community/Qwen3.5-4B-MLX-4bit",
        bench=_Bench(),
        notes=None,
        now=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    assert "spec_decode" not in baseline["config"]
    assert "run_group" not in baseline


def test_schema_still_couples_tier_to_its_result_block_at_v3() -> None:
    """Bumping the version enum must not switch the tier conditionals off.

    Codex round 2 on #1403: the smoke/harness presence rules were gated on
    ``schema_version == 2``. Adding 3 to the enum silently disabled them, so a
    v3 row could claim ``tier="harness"`` while carrying no harness_result at
    all — the exact "we never ran it but the board says it failed" ambiguity
    the tier coupling exists to prevent.
    """
    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1] / "community-benchmarks" / "schema.json"
        ).read_text()
    )
    base = {
        "schema_version": 3,
        "submission_id": "abcdef012345",
        "submitted_at": "2026-08-02T10:00:00+00:00",
        "hardware": {"chip": "Apple M2 Pro", "ram_gb": 32, "cpu_cores": 10},
        "software": {
            "macos": "26.5.2",
            "rapid_mlx": "0.11.9",
            "mlx": "0.31.2",
            "python": "3.12.13",
        },
        "model": {
            "alias": "qwen3.5-4b-4bit",
            "hf_path": "mlx-community/Qwen3.5-4B-MLX-4bit",
        },
        "config": {
            "rounds": 5,
            "warmup_rounds": 1,
            "sampling": "greedy",
            "buckets_spec": {
                "short": {"prompt_tokens": 512, "max_tokens": 128},
                "long": {"prompt_tokens": 2048, "max_tokens": 512},
            },
            "prompt_hash": "abc123def4567890",
        },
        "buckets": {
            k: {
                "decode_tps": {"median": 60.0, "min": 59.0, "max": 61.0, "stddev": 0.5},
                "prefill_tps": {
                    "median": 300.0,
                    "min": 295.0,
                    "max": 305.0,
                    "stddev": 2.0,
                },
                "ttft_ms": {
                    "median": 1700.0,
                    "min": 1690.0,
                    "max": 1710.0,
                    "stddev": 5.0,
                },
                "rounds_raw": [
                    {"decode_tps": 60.0, "prefill_tps": 300.0, "ttft_ms": 1700.0}
                ]
                * 5,
            }
            for k in ("short", "long")
        },
    }
    jsonschema.validate(base, schema)  # speed-only v3 is fine

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**base, "tier": "harness"}, schema)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**base, "tier": "smoke"}, schema)


def test_an_older_declared_version_cannot_smuggle_newer_fields() -> None:
    """schema_version has to mean something to the aggregator."""
    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1] / "community-benchmarks" / "schema.json"
        ).read_text()
    )
    minimal = {
        "schema_version": 2,
        "submission_id": "abcdef012345",
        "submitted_at": "2026-08-02T10:00:00+00:00",
        "hardware": {"chip": "Apple M2 Pro", "ram_gb": 32, "cpu_cores": 10},
        "software": {
            "macos": "26.5.2",
            "rapid_mlx": "0.11.9",
            "mlx": "0.31.2",
            "python": "3.12.13",
        },
        "model": {
            "alias": "qwen3.5-4b-4bit",
            "hf_path": "mlx-community/Qwen3.5-4B-MLX-4bit",
        },
        "config": {
            "rounds": 5,
            "warmup_rounds": 1,
            "sampling": "greedy",
            "buckets_spec": {
                "short": {"prompt_tokens": 512, "max_tokens": 128},
                "long": {"prompt_tokens": 2048, "max_tokens": 512},
            },
            "prompt_hash": "abc123def4567890",
        },
        "buckets": {
            k: {
                "decode_tps": {"median": 60.0, "min": 59.0, "max": 61.0, "stddev": 0.5},
                "prefill_tps": {
                    "median": 300.0,
                    "min": 295.0,
                    "max": 305.0,
                    "stddev": 2.0,
                },
                "ttft_ms": {
                    "median": 1700.0,
                    "min": 1690.0,
                    "max": 1710.0,
                    "stddev": 5.0,
                },
                "rounds_raw": [
                    {"decode_tps": 60.0, "prefill_tps": 300.0, "ttft_ms": 1700.0}
                ]
                * 5,
            }
            for k in ("short", "long")
        },
    }
    jsonschema.validate(minimal, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**minimal, "run_group": "abcdefabcdef"}, schema)


def test_a_2xx_that_is_not_an_acceptance_is_an_error(monkeypatch) -> None:
    """A 200 is the transport talking, not the board accepting."""

    def fake_open(req, timeout=None):
        return _Resp(b'{"ok": false, "error": "rejected"}')

    monkeypatch.setattr(upload.urllib.request, "urlopen", fake_open)
    with pytest.raises(upload.SubmitError) as exc:
        upload.post_submission({"a": 1}, url="https://x/api")
    assert "rejected" in str(exc.value)
    assert "NOT submitted" in str(exc.value)


def test_a_corrupted_id_file_does_not_crash_the_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    path = tmp_path / "bench-install-id"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe\x00 not utf-8 at all")
    got = upload.install_id()  # UnicodeDecodeError is a ValueError, not OSError
    assert len(got) == 12


def test_consent_names_the_actual_destination(tmp_path, monkeypatch) -> None:
    """A consent screen that names the wrong host is worse than none.

    ``RAPID_MLX_BENCH_BOARD_URL`` can point the submission anywhere. The
    prompt used to hardcode rapidmlx.com, so an overridden target would have
    been disclosed as going somewhere it was not.
    """
    import io

    from vllm_mlx.community_bench import submission as sub

    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    monkeypatch.setenv(upload.BOARD_URL_ENV, "https://elsewhere.example/api")
    monkeypatch.setattr(
        "vllm_mlx.community_bench.upload.post_submission",
        lambda payload, **k: {"ok": True},
    )
    out = io.StringIO()
    sub.submit_interactive(
        {
            "submission_id": "abcdef012345",
            "hardware": {"chip": "Apple M2 Pro"},
            "model": {"alias": "x"},
        },
        tmp_path,
        stdin=io.StringIO("n\n"),
        stdout=out,
    )
    text = out.getvalue()
    assert "elsewhere.example" in text, "consent must name where the data is going"
    assert "rapidmlx.com" not in text, "and must not name a host it is not going to"


def test_a_malformed_id_file_is_replaced_not_re_minted_forever(
    tmp_path, monkeypatch
) -> None:
    """A corrupted id file must not cost the install its identity.

    With O_EXCL alone the create always raises FileExistsError, the garbage is
    never replaced, and every call returns a fresh unpersisted id — so the
    per-install cap resets on every submission and the contributor's board
    name changes every run.
    """
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    path = tmp_path / "bench-install-id"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("obviously not a hex id\n")

    first = upload.install_id()
    assert len(first) == 12
    assert upload.install_id() == first, "identity must stabilise after one repair"
    assert path.read_text().strip() == first
    assert not list(tmp_path.glob("*.tmp")), "no temp file may be left behind"


def test_a_plain_http_override_is_refused(monkeypatch) -> None:
    """The consent text promises HTTPS; the resolver has to mean it."""
    monkeypatch.setenv(upload.BOARD_URL_ENV, "http://evil.example/api")
    with pytest.raises(upload.SubmitError) as exc:
        upload.board_url()
    assert "clear text" in str(exc.value)


def test_loopback_http_is_allowed_for_local_development(monkeypatch) -> None:
    monkeypatch.setenv(upload.BOARD_URL_ENV, "http://127.0.0.1:8787/api/benchmarks")
    assert upload.board_url() == "http://127.0.0.1:8787/api/benchmarks"
    monkeypatch.setenv(upload.BOARD_URL_ENV, "https://staging.example/api")
    assert upload.board_url() == "https://staging.example/api"


def test_peeking_an_id_never_touches_the_disk(tmp_path, monkeypatch) -> None:
    """Declining consent must leave no trace, including no id file.

    The consent screen says "nothing has been written yet". Minting the
    identifier eagerly made that false for anyone who typed n.
    """
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    got = upload.peek_install_id()
    assert len(got) == 12
    assert not (tmp_path / "bench-install-id").exists()
    assert not list(tmp_path.glob("*"))


def test_committing_adopts_a_concurrent_winner(tmp_path, monkeypatch) -> None:
    """One machine must never report as two installs."""
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    winner = upload.commit_install_id("aaaaaaaaaaaa")
    assert winner == "aaaaaaaaaaaa"
    # A second process that peeked before the winner landed offers its own
    # candidate and must adopt what is already on disk.
    assert upload.commit_install_id("bbbbbbbbbbbb") == "aaaaaaaaaaaa"
    assert not list(tmp_path.glob("*.tmp")), "no temp file may be left behind"


def test_declining_consent_writes_nothing_at_all(tmp_path, monkeypatch) -> None:
    import io

    from vllm_mlx.community_bench import submission as sub

    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    sent = []
    monkeypatch.setattr(
        "vllm_mlx.community_bench.upload.post_submission",
        lambda *a, **k: sent.append(1) or {"ok": True},
    )
    rc = sub.submit_interactive(
        {
            "schema_version": 3,
            "submission_id": "abcdef012345",
            "hardware": {"chip": "Apple M2 Pro"},
            "model": {"alias": "x"},
        },
        tmp_path,
        stdin=io.StringIO("n\n"),
        stdout=io.StringIO(),
    )
    assert rc == 0
    assert sent == []
    assert not list(tmp_path.rglob("*")), "declining must leave the disk untouched"


def test_an_invalid_board_url_is_a_controlled_error(tmp_path, monkeypatch) -> None:
    """Not a traceback: the documented contract is a non-zero return."""
    import io

    from vllm_mlx.community_bench import submission as sub

    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    monkeypatch.setenv(upload.BOARD_URL_ENV, "http://evil.example/api")
    out = io.StringIO()
    rc = sub.submit_interactive(
        {
            "schema_version": 3,
            "submission_id": "abcdef012345",
            "hardware": {"chip": "Apple M2 Pro"},
            "model": {"alias": "x"},
        },
        tmp_path,
        stdin=io.StringIO("y\n"),
        stdout=out,
    )
    assert rc == 2
    assert "clear text" in out.getvalue()


def test_transport_errors_do_not_claim_a_local_copy_exists(monkeypatch) -> None:
    """Only the caller knows whether archiving worked."""

    def boom(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(upload.urllib.request, "urlopen", boom)
    with pytest.raises(upload.SubmitError) as exc:
        upload.post_submission({"a": 1}, url="https://x/api")
    assert "saved" not in str(exc.value).lower()
