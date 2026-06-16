# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the community-bench submission pipeline.

Scope: every layer that doesn't require loading an MLX model:

- ``vllm_mlx.community_bench.hardware`` — the allowlist guard,
  ``is_apple_silicon`` gate, version probes.
- ``vllm_mlx.community_bench.runner`` — pure helpers (``_stat``,
  ``_build_synthetic_prompt``, ``_prompt_hash``,
  ``_make_sampling_params_factory``, ``standardized_config_dict``).
- ``vllm_mlx.community_bench.submission`` — payload builder, slugs,
  consent prompt, filename, manual-fallback printing,
  ``submit_interactive`` end-to-end with monkeypatched git/gh.
- ``community-benchmarks/scripts/validate.py`` — every failure mode
  on synthetic JSON.

The aggregator + website are intentionally deferred to a follow-up
PR (see ``community-benchmarks/README.md``), so this PR has no
aggregator tests.

The real end-to-end bench (load model → run rounds → submit) is not
unit-testable without spinning up MLX; it's covered manually before
PR-merge.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "community-benchmarks" / "scripts"
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"
ALIASES_PATH = REPO_ROOT / "vllm_mlx" / "aliases.json"


# ---------------------------------------------------------------------------
# hardware.py
# ---------------------------------------------------------------------------


def test_run_rejects_disallowed_binary() -> None:
    """``_run`` must refuse anything not on ``_PERMITTED_BINARIES``.

    The allowlist is the bedrock of the privacy contract — bypass it
    and the module no longer guarantees what it claims to. We test the
    guard directly so a refactor that quietly inlines the subprocess
    call still trips this check.
    """
    from vllm_mlx.community_bench import hardware

    with pytest.raises(RuntimeError, match="disallowed binary"):
        hardware._run(["/bin/ls", "/"], timeout=1.0)


def test_run_rejects_empty_argv() -> None:
    """Empty cmd[] used to raise IndexError on ``cmd[0]``; the contract
    promises RuntimeError on every bad input. (Codex PR #582 round-7
    BLOCKING.)"""
    from vllm_mlx.community_bench import hardware

    with pytest.raises(RuntimeError, match="disallowed binary"):
        hardware._run([], timeout=1.0)


def test_run_executes_allowlisted_binary(tmp_path: Path) -> None:
    """A known-good allowlisted binary returns its stripped stdout."""
    from vllm_mlx.community_bench import hardware

    if sys.platform != "darwin":
        pytest.skip("sw_vers only exists on macOS")
    # ``sw_vers -productName`` returns "macOS" on every supported macOS;
    # we only check the call succeeds and returns a non-empty string.
    out = hardware._run(["/usr/bin/sw_vers", "-productName"], timeout=2.0)
    assert out  # non-empty


def test_is_apple_silicon_matches_platform() -> None:
    """Sanity check the gate matches the actual host."""
    from vllm_mlx.community_bench import hardware

    expected = sys.platform == "darwin" and os.uname().machine == "arm64"
    assert hardware.is_apple_silicon() is expected


def test_collect_refuses_non_apple_silicon(monkeypatch) -> None:
    """Calling ``collect()`` off Apple Silicon must raise."""
    from vllm_mlx.community_bench import hardware

    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    with pytest.raises(RuntimeError, match="Apple-Silicon-only"):
        hardware.collect()


def test_rapid_mlx_version_resolves() -> None:
    """The probe should at least return a string (real version or 'unknown')."""
    from vllm_mlx.community_bench import hardware

    v = hardware._rapid_mlx_version()
    assert isinstance(v, str) and v


# ---------------------------------------------------------------------------
# runner.py — pure helpers
# ---------------------------------------------------------------------------


def test_stat_single_value() -> None:
    """``_stat`` with one sample uses pstdev (0), not raising sample stdev."""
    from vllm_mlx.community_bench.runner import _stat

    assert _stat([5.0]) == {"median": 5.0, "min": 5.0, "max": 5.0, "stddev": 0.0}


def test_stat_multi_value() -> None:
    from vllm_mlx.community_bench.runner import _stat

    s = _stat([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["median"] == 3.0
    assert s["min"] == 1.0
    assert s["max"] == 5.0
    assert s["stddev"] > 0.0  # pstdev of [1..5] is non-zero


def test_synthetic_prompt_deterministic() -> None:
    """Same seed + same tokenizer ⇒ same prompt tokens.

    The aggregator's ability to re-compute ``prompt_hash`` for tampering
    detection rests on this property. We use a stub tokenizer (decoded
    string is just a join of stringified ids) to keep the test free of
    model weights.
    """
    from vllm_mlx.community_bench import runner

    class _StubTokenizer:
        vocab_size = 32_000

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    t = _StubTokenizer()
    text_a, ids_a = runner._build_synthetic_prompt(t, 100, seed=42)
    text_b, ids_b = runner._build_synthetic_prompt(t, 100, seed=42)
    assert ids_a == ids_b
    assert text_a == text_b


def test_synthetic_prompt_seed_varies() -> None:
    """Different seed ⇒ different prompt (probabilistically certain for n=100)."""
    from vllm_mlx.community_bench import runner

    class _StubTokenizer:
        vocab_size = 32_000

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    _, ids_a = runner._build_synthetic_prompt(_StubTokenizer(), 100, seed=42)
    _, ids_b = runner._build_synthetic_prompt(_StubTokenizer(), 100, seed=43)
    assert ids_a != ids_b


def test_synthetic_prompt_rejects_tiny_vocab() -> None:
    """A pathologically small vocab raises rather than silently producing
    a degenerate prompt."""
    from vllm_mlx.community_bench import runner

    class _TinyTok:
        vocab_size = 50

        def decode(self, ids):
            return ""

    with pytest.raises(RuntimeError, match="vocab too small"):
        runner._build_synthetic_prompt(_TinyTok(), 100, seed=1)


def test_prompt_hash_stable() -> None:
    from vllm_mlx.community_bench.runner import _prompt_hash

    h = _prompt_hash([1, 2, 3], [4, 5, 6])
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)
    # Stability across reorder of the call args: hash([1,2],[3,4]) ≠ hash([3,4],[1,2])
    assert _prompt_hash([1, 2], [3, 4]) != _prompt_hash([3, 4], [1, 2])


def test_make_sampling_params_factory() -> None:
    from vllm_mlx.community_bench import runner

    greedy = runner._make_sampling_params_factory("greedy")
    sampled = runner._make_sampling_params_factory("sampled")

    g = greedy(128)
    s = sampled(128)
    assert g.max_tokens == 128 and g.temperature == 0.0 and g.top_p == 1.0
    assert s.max_tokens == 128 and s.temperature == 0.7 and s.top_p == 0.9

    with pytest.raises(ValueError):
        runner._make_sampling_params_factory("bogus")


def test_bench_sampling_always_ignores_eos() -> None:
    """Regression guard for #567 (dineshdb).

    The standardized bench's ``tg128`` / ``tg512`` contract requires every
    round to generate exactly ``max_tokens`` decode steps for cross-machine
    comparability. If the synthetic random-token prompt nudges the model
    toward EOS — qwen3.5-9b-4bit on the locked ``0xC0FFEE`` seed does this
    — the round aborts early and the guard in ``_run_one_round`` (correctly)
    rejects the result.

    The fix is to set ``ignore_eos=True`` on the sampling params so the
    engine suppresses the model's own EOS / chat-terminator tokens. This
    test pins that contract at the factory boundary: BOTH sampling modes
    MUST set ``ignore_eos=True``. If a future refactor drops the flag,
    the bench fails again on the same models that hit it the first time.
    """
    from vllm_mlx.community_bench import runner

    for sampling_mode in ("greedy", "sampled"):
        factory = runner._make_sampling_params_factory(sampling_mode)
        for max_tokens in (128, 512):
            params = factory(max_tokens)
            assert params.ignore_eos is True, (
                f"_make_sampling_params_factory({sampling_mode!r})({max_tokens}) "
                f"must set ignore_eos=True — without it the bench fails on any "
                f"model whose response to the 0xC0FFEE synthetic prompt is to "
                f"emit EOS early (see issue #567 for the original report)."
            )


def test_scheduler_honours_ignore_eos() -> None:
    """The other half of the #567 fix: the scheduler must actually act
    on ``sampling_params.ignore_eos``. The factory above can set the flag
    all it wants — if the scheduler still merges the model's EOS / chat-
    terminator tokens into the BatchGenerator's ``stop_tokens``, the bench
    aborts at token 88 anyway.

    Exercise the **real** production helper that ``_create_batch_generator``
    calls — a local stand-in would still pass if the production branch
    were deleted. Deleting the ``ignore_eos`` branch from
    ``_assemble_stop_tokens`` now fails this test.
    """
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import _assemble_stop_tokens

    model_eos_tokens = {2, 151645, 151643}  # arbitrary stand-in EOS ids

    # Case A: ignore_eos=True, no extras → empty
    p = SamplingParams(max_tokens=512, ignore_eos=True)
    assert _assemble_stop_tokens(p, model_eos_tokens) == set()

    # Case B: ignore_eos=True + caller stop ids → only the caller's
    p = SamplingParams(max_tokens=512, ignore_eos=True, stop_token_ids=[99])
    assert _assemble_stop_tokens(p, model_eos_tokens) == {99}

    # Case C: ignore_eos=False (default) → model EOS preserved
    p = SamplingParams(max_tokens=512)
    assert _assemble_stop_tokens(p, model_eos_tokens) == model_eos_tokens

    # Case D: default + caller stop ids → union
    p = SamplingParams(max_tokens=512, stop_token_ids=[99])
    assert _assemble_stop_tokens(p, model_eos_tokens) == model_eos_tokens | {99}

    # Case E: caller-supplied model-stop set must not be mutated. The
    # previous in-place ``.update()`` on the ``_get_stop_tokens()`` return
    # value would have poisoned the cached set on subsequent requests if
    # the helper failed to copy.
    immutable_model = frozenset({1, 2, 3})
    p = SamplingParams(max_tokens=512, stop_token_ids=[99])
    result = _assemble_stop_tokens(p, immutable_model)
    assert result == {1, 2, 3, 99}
    assert immutable_model == frozenset({1, 2, 3})  # unchanged


def test_standardized_config_dict_matches_schema_consts() -> None:
    """The hardcoded constants in ``config`` must equal the schema's
    ``const`` values — schema validation depends on this."""
    from vllm_mlx.community_bench.runner import standardized_config_dict

    cfg = standardized_config_dict("greedy", "deadbeefcafebabe")
    schema = json.loads(SCHEMA_PATH.read_text())
    schema_cfg = schema["properties"]["config"]["properties"]
    assert cfg["rounds"] == schema_cfg["rounds"]["const"]
    assert cfg["warmup_rounds"] == schema_cfg["warmup_rounds"]["const"]
    assert (
        cfg["buckets_spec"]["short"]["prompt_tokens"]
        == schema_cfg["buckets_spec"]["properties"]["short"]["properties"][
            "prompt_tokens"
        ]["const"]
    )
    assert (
        cfg["buckets_spec"]["long"]["max_tokens"]
        == schema_cfg["buckets_spec"]["properties"]["long"]["properties"]["max_tokens"][
            "const"
        ]
    )


# ---------------------------------------------------------------------------
# submission.py
# ---------------------------------------------------------------------------


def _stub_bench_result(sampling: str = "greedy"):
    """Build a ``BenchResult`` with plausible numbers for payload tests."""
    from vllm_mlx.community_bench.runner import (
        BenchResult,
        BucketResult,
        RoundResult,
    )

    rounds = [
        RoundResult(decode_tps=42.0, prefill_tps=500.0, ttft_ms=120.0) for _ in range(5)
    ]
    return BenchResult(
        short=BucketResult(rounds_raw=rounds),
        long=BucketResult(rounds_raw=rounds),
        peak_ram_mb=8192,
        prompt_hash="deadbeefcafebabe",
        sampling=sampling,
    )


def _stub_hw_sw():
    from vllm_mlx.community_bench.hardware import Hardware, Software

    hw = Hardware(chip="Apple M4 Pro", ram_gb=24, cpu_cores=12, gpu_cores=20)
    sw = Software(macos="26.5.1", rapid_mlx="0.7.6", mlx="0.31.2", python="3.12.13")
    return hw, sw


def test_build_payload_matches_schema() -> None:
    """The payload built from real-shaped inputs must validate."""
    jsonschema = pytest.importorskip("jsonschema")
    from vllm_mlx.community_bench.submission import build_submission_payload

    hw, sw = _stub_hw_sw()
    payload = build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        bench=_stub_bench_result(),
        notes="unit test",
        now=datetime(2026, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
    )
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=payload, schema=schema)


def test_build_payload_omits_optional_fields_when_none() -> None:
    from vllm_mlx.community_bench.runner import (
        BenchResult,
        BucketResult,
        RoundResult,
    )
    from vllm_mlx.community_bench.submission import build_submission_payload

    rounds = [RoundResult(40, 500, 100) for _ in range(5)]
    bench = BenchResult(
        short=BucketResult(rounds_raw=rounds),
        long=BucketResult(rounds_raw=rounds),
        peak_ram_mb=None,  # probe failed
        prompt_hash="0123456789abcdef",
        sampling="greedy",
    )
    hw, sw = _stub_hw_sw()
    payload = build_submission_payload(
        hw,
        sw,
        "qwen3.5-9b-4bit",
        "mlx-community/Qwen3.5-9B-4bit",
        bench,
        notes=None,
        now=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )
    assert "notes" not in payload
    assert "peak_ram_mb" not in payload


def test_slugify() -> None:
    from vllm_mlx.community_bench.submission import _slugify

    assert _slugify("Apple M3 Ultra") == "apple-m3-ultra"
    assert _slugify("Qwen3.5-9B-4bit") == "qwen3-5-9b-4bit"
    assert _slugify("__weird---name__") == "weird-name"


def test_submission_filename_shape() -> None:
    """Filename must match the regex the validator enforces."""
    import re

    from vllm_mlx.community_bench.submission import _submission_filename

    payload = {
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "hardware": {"chip": "Apple M4 Pro"},
        "model": {"alias": "qwen3.5-9b-4bit"},
        "submission_id": "abcdef012345",
    }
    name = _submission_filename(payload)
    assert re.match(r"^[0-9]{8}-[a-z0-9-]+-[a-z0-9.-]+-[0-9a-f]{12}\.json$", name)
    assert name.startswith("20260615-apple-m4-pro-")


def test_ask_consent_yes() -> None:
    from vllm_mlx.community_bench.submission import _ask_consent

    stdin = io.StringIO("y\n")
    stdout = io.StringIO()
    assert _ask_consent({"key": "val"}, stdin=stdin, stdout=stdout) is True
    assert "[y]" in stdout.getvalue()
    # Round-7 BLOCKING: consent text must disclose BOTH network ops,
    # not just `gh pr create`. A user who just wanted to "open a PR"
    # should know git push runs first.
    text = stdout.getvalue()
    assert "git push" in text
    assert "gh pr create" in text


def test_ask_consent_default_no() -> None:
    """Empty input (just Enter) cancels — defaults are safe."""
    from vllm_mlx.community_bench.submission import _ask_consent

    stdin = io.StringIO("\n")
    stdout = io.StringIO()
    assert _ask_consent({"k": "v"}, stdin=stdin, stdout=stdout) is False


def test_ask_consent_eof_is_no() -> None:
    """Piped non-interactive stdin must NOT count as consent.

    Running ``rapid-mlx bench --submit < /dev/null`` in CI should never
    fire a PR off — explicit opt-in only.
    """
    from vllm_mlx.community_bench.submission import _ask_consent

    stdin = io.StringIO("")
    stdout = io.StringIO()
    assert _ask_consent({}, stdin=stdin, stdout=stdout) is False


def test_ask_consent_anything_other_than_y_is_no() -> None:
    from vllm_mlx.community_bench.submission import _ask_consent

    for ans in ["n\n", "no\n", "Yes please\n", "definitely\n"]:
        # "Yes please" is interesting: ``Yes`` would pass but ``Yes please``
        # shouldn't — we strip+lower then compare against {"y","yes"}. The
        # whole stripped string is what's compared, so "yes please" != "yes".
        stdin = io.StringIO(ans)
        stdout = io.StringIO()
        assert _ask_consent({}, stdin=stdin, stdout=stdout) is False, ans


def test_submit_interactive_user_cancels(tmp_path: Path) -> None:
    """A 'no' answer must not write the JSON file or touch git."""
    from vllm_mlx.community_bench.submission import submit_interactive

    # Real `git init` so `git rev-parse --show-toplevel` succeeds.
    # We also wire `origin` to the canonical Rapid-MLX URL so the
    # remote-verification guard (codex round-3 BLOCKING) accepts it.
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://github.com/raullenchai/Rapid-MLX.git",
        ],
        check=True,
        capture_output=True,
    )
    payload = {
        "schema_version": 1,
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "hardware": {"chip": "Apple M4 Pro"},
        "model": {"alias": "qwen3.5-9b-4bit"},
    }
    rc = submit_interactive(
        payload, tmp_path, stdin=io.StringIO("n\n"), stdout=io.StringIO()
    )
    assert rc == 0
    assert not (tmp_path / "community-benchmarks").exists()


def test_submit_interactive_requires_git_repo(tmp_path: Path) -> None:
    """Non-repo paths should return rc=2 (configuration error)."""
    from vllm_mlx.community_bench.submission import submit_interactive

    payload = {"submission_id": "abcdef012345"}
    rc = submit_interactive(
        payload, tmp_path, stdin=io.StringIO("y\n"), stdout=io.StringIO()
    )
    assert rc == 2


def test_submit_interactive_rejects_wrong_repo(tmp_path: Path) -> None:
    """Regression: a git repo whose origin is not raullenchai/Rapid-MLX
    must be rejected with rc=2. (Codex PR #582 round-3 BLOCKING.)
    """
    from vllm_mlx.community_bench.submission import submit_interactive

    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://github.com/some-other/repo.git",
        ],
        check=True,
        capture_output=True,
    )
    payload = {"submission_id": "abcdef012345"}
    out = io.StringIO()
    rc = submit_interactive(payload, tmp_path, stdin=io.StringIO("y\n"), stdout=out)
    assert rc == 2
    text = out.getvalue()
    assert "raullenchai/Rapid-MLX" in text


def test_find_upstream_remote_accepts_ssh_and_https(
    tmp_path: Path, monkeypatch
) -> None:
    """``_find_upstream_remote`` must accept all git URL forms with or
    without the trailing ``.git`` suffix and on either ``origin`` or a
    fork-style ``upstream`` remote."""
    from vllm_mlx.community_bench.submission import _find_upstream_remote

    # Numeric-prefixed dirs avoid case-insensitive filesystem
    # collisions — macOS APFS folds case by default, so two URL
    # variants that differ only in case would otherwise share a
    # directory and the second mkdir() would fail.
    forms = [
        "https://github.com/raullenchai/Rapid-MLX",
        "https://github.com/raullenchai/Rapid-MLX.git",
        "git@github.com:raullenchai/Rapid-MLX",
        "git@github.com:raullenchai/Rapid-MLX.git",
        # Different capitalisation should still match (GitHub URLs are
        # case-insensitive).
        "https://github.com/RaullenChai/RAPID-mlx",
    ]
    for i, url in enumerate(forms):
        d = tmp_path / f"repo_{i}"
        d.mkdir()
        subprocess.run(["git", "init", "-q", str(d)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(d), "remote", "add", "origin", url],
            check=True,
            capture_output=True,
        )
        assert _find_upstream_remote(d) == "origin", f"should accept {url!r}"


def test_find_upstream_remote_accepts_fork_with_upstream(
    tmp_path: Path,
) -> None:
    """Standard community fork: origin is the user's fork, upstream is
    raullenchai/Rapid-MLX. Both must be recognized so the contributor
    can submit. (Codex PR #582 round-6 BLOCKING.)"""
    from vllm_mlx.community_bench.submission import (
        _find_upstream_remote,
        _origin_is_safe_github,
    )

    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://github.com/some-contributor/Rapid-MLX.git",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "upstream",
            "https://github.com/raullenchai/Rapid-MLX.git",
        ],
        check=True,
        capture_output=True,
    )
    assert _find_upstream_remote(tmp_path) == "upstream"
    origin_ok, origin_owner = _origin_is_safe_github(tmp_path)
    assert origin_ok is True
    assert origin_owner == "some-contributor"


def test_origin_is_safe_github_rejects_malicious_pushurl(tmp_path) -> None:
    """A repo with origin=github.com (fetch) but pushurl=evil.com must
    fail the gate — ``git push origin`` honours pushurl, so a fetch-URL-
    only check would push the payload to the attacker. (Codex PR #582
    round-7 BLOCKING.)"""
    from vllm_mlx.community_bench.submission import _origin_is_safe_github

    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://github.com/raullenchai/Rapid-MLX.git",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "set-url",
            "--push",
            "origin",
            "https://evil.example.com/raullenchai/Rapid-MLX.git",
        ],
        check=True,
        capture_output=True,
    )
    ok, owner = _origin_is_safe_github(tmp_path)
    assert ok is False
    assert owner is None


def test_make_pr_via_gh_branches_from_upstream_and_uses_owner_head(
    tmp_path, monkeypatch
) -> None:
    """The auto-PR sequence must (a) fetch upstream/main and branch from
    FETCH_HEAD (not local HEAD) so the contributor's other work isn't
    swept in, and (b) pass ``--head <owner>:<branch>`` so gh finds the
    branch on the fork. (Codex PR #582 round-7 BLOCKING.)"""
    from vllm_mlx.community_bench import submission as sub_mod

    captured: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check, cwd):
        captured.append(cmd)

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr(sub_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(sub_mod.shutil, "which", lambda name: "/usr/local/bin/gh")
    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {"alias": "qwen", "hf_path": "x/y"},
        "hardware": {"chip": "Apple M4 Pro", "ram_gb": 24},
        "software": {"rapid_mlx": "0.7.6", "mlx": "0.31.2"},
        "config": {"sampling": "greedy"},
        "buckets": {
            "short": {"decode_tps": {"median": 40.0}},
            "long": {"decode_tps": {"median": 40.0}},
        },
        "notes": None,
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    sub_mod._make_pr_via_gh(
        tmp_path,
        sub_path,
        payload,
        stdout=io.StringIO(),
        origin_owner="some-contributor",
        upstream_remote="upstream",
    )
    # First command must fetch upstream's main.
    assert captured[0][:5] == ["git", "-C", str(tmp_path), "fetch", "--quiet"]
    assert captured[0][5] == "upstream"
    # Checkout must branch FROM FETCH_HEAD, not the local HEAD.
    checkout = captured[1]
    assert checkout[-1] == "FETCH_HEAD", (
        f"checkout must branch from FETCH_HEAD, got cmd: {checkout}"
    )
    # gh pr create must use `--head <owner>:<branch>`.
    pr_create = captured[-1]
    head_idx = pr_create.index("--head")
    assert pr_create[head_idx + 1] == "some-contributor:community-bench/abcdef012345"


def test_find_upstream_remote_rejects_evil_github_lookalike(
    tmp_path: Path,
) -> None:
    """``endswith('github.com/raullenchai/rapid-mlx')`` accepts
    ``evilgithub.com/raullenchai/rapid-mlx``; our parser must not.
    (Codex PR #582 round-6 BLOCKING — URL spoofing.)"""
    from vllm_mlx.community_bench.submission import _find_upstream_remote

    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://evilgithub.com/raullenchai/Rapid-MLX.git",
        ],
        check=True,
        capture_output=True,
    )
    assert _find_upstream_remote(tmp_path) is None


def test_submit_interactive_writes_file_then_falls_back_on_dirty_tree(
    tmp_path: Path, monkeypatch
) -> None:
    """When git is dirty, the file IS written but no PR is opened.

    Privacy contract: the user can always recover the file and finish
    the PR by hand — we never block them on automation working.
    """
    from vllm_mlx.community_bench import submission as sub_mod

    # Real `git init` so `git rev-parse --show-toplevel` succeeds.
    # We also wire `origin` to the canonical Rapid-MLX URL so the
    # remote-verification guard (codex round-3 BLOCKING) accepts it.
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://github.com/raullenchai/Rapid-MLX.git",
        ],
        check=True,
        capture_output=True,
    )
    # Force ``_git_is_clean`` to report dirty without invoking real git.
    monkeypatch.setattr(sub_mod, "_git_is_clean", lambda repo: False)

    payload = {
        "schema_version": 1,
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "hardware": {"chip": "Apple M4 Pro", "ram_gb": 24},
        "model": {"alias": "qwen3.5-9b-4bit", "hf_path": "x/y"},
        "buckets": {
            "short": {"decode_tps": {"median": 1.0}},
            "long": {"decode_tps": {"median": 1.0}},
        },
        "config": {"sampling": "greedy"},
        "software": {"rapid_mlx": "0.7.6", "mlx": "0.31.2"},
    }

    stdout = io.StringIO()
    rc = sub_mod.submit_interactive(
        payload, tmp_path, stdin=io.StringIO("y\n"), stdout=stdout
    )
    assert rc == 0
    # ``_slugify`` collapses '.' to '-' so the alias slug is "qwen3-5-9b-4bit",
    # not the literal alias key.
    expected_file = (
        tmp_path
        / "community-benchmarks"
        / "submissions"
        / "20260615-apple-m4-pro-qwen3-5-9b-4bit-abcdef012345.json"
    )
    assert expected_file.exists()
    text = stdout.getvalue()
    assert "Thank you" in text
    assert "git checkout -b community-bench/abcdef012345" in text


def test_submit_interactive_clean_tree_reaches_pr_step(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: ``_git_is_clean`` must be sampled BEFORE the new file
    is written, otherwise the newly-untracked submission file makes
    every clean checkout look dirty and the auto-PR path is never
    reachable. (Codex PR #582 BLOCKING #2.)

    We stub ``_git_is_clean`` to read the actual filesystem state at
    the time it's called: returns True iff the submissions/ dir is
    still empty when probed. With the bug present, the writer has
    already deposited the file before the check, so the stub returns
    False and ``_make_pr_via_gh`` is skipped. With the fix, the check
    runs first and ``_make_pr_via_gh`` is invoked.
    """
    from vllm_mlx.community_bench import submission as sub_mod

    # Real `git init` so `git rev-parse --show-toplevel` succeeds.
    # We also wire `origin` to the canonical Rapid-MLX URL so the
    # remote-verification guard (codex round-3 BLOCKING) accepts it.
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://github.com/raullenchai/Rapid-MLX.git",
        ],
        check=True,
        capture_output=True,
    )
    submissions_dir = tmp_path / "community-benchmarks" / "submissions"
    pr_invoked: list[bool] = []

    def fake_clean(repo):
        return not submissions_dir.exists() or not any(submissions_dir.iterdir())

    def fake_pr(repo, path, payload, *, stdout, origin_owner, upstream_remote):
        pr_invoked.append(True)
        # ``_make_pr_via_gh`` now returns (success, completed_steps);
        # round-5 state-aware fallback needs the set on failure.
        return True, {
            "fetch_base",
            "checkout",
            "stage",
            "commit",
            "push",
            "pr_create",
        }

    monkeypatch.setattr(sub_mod, "_git_is_clean", fake_clean)
    monkeypatch.setattr(sub_mod, "_make_pr_via_gh", fake_pr)

    payload = {
        "schema_version": 1,
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "hardware": {"chip": "Apple M4 Pro", "ram_gb": 24},
        "model": {"alias": "qwen3.5-9b-4bit", "hf_path": "x/y"},
        "buckets": {
            "short": {"decode_tps": {"median": 1.0}},
            "long": {"decode_tps": {"median": 1.0}},
        },
        "config": {"sampling": "greedy"},
        "software": {"rapid_mlx": "0.7.6", "mlx": "0.31.2"},
    }
    rc = sub_mod.submit_interactive(
        payload, tmp_path, stdin=io.StringIO("y\n"), stdout=io.StringIO()
    )
    assert rc == 0
    assert pr_invoked == [True], (
        "auto-PR path was not reached on a clean tree — likely the "
        "_git_is_clean ordering regression has returned"
    )


# ---------------------------------------------------------------------------
# validate.py — schema + sanity gate
# ---------------------------------------------------------------------------


def _good_payload() -> dict:
    """A payload that passes every check, used as the baseline for
    mutation tests below.

    Summary stats are computed from ``rounds_raw`` so the new
    summary-matches-raw check (Codex PR #582 round-2 BLOCKING) passes.
    Hand-edited mutation tests below override individual fields to
    trigger specific failures.
    """
    import statistics

    aliases = json.loads(ALIASES_PATH.read_text())
    alias = next(iter(aliases))
    # Five distinct values so min ≠ max ≠ median ≠ stddev — exercises
    # the recomputation path properly. Means: decode=42, prefill=500,
    # ttft=100.
    rounds = [
        {
            "decode_tps": 40.0 + i,
            "prefill_tps": 498.0 + i,
            "ttft_ms": 98.0 + i,
        }
        for i in range(5)
    ]

    def _summary(values: list[float]) -> dict:
        return {
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "stddev": float(statistics.pstdev(values)),
        }

    bucket = {
        "decode_tps": _summary([r["decode_tps"] for r in rounds]),
        "prefill_tps": _summary([r["prefill_tps"] for r in rounds]),
        "ttft_ms": _summary([r["ttft_ms"] for r in rounds]),
        "rounds_raw": rounds,
    }
    return {
        "schema_version": 1,
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "hardware": {
            "chip": "Apple M4 Pro",
            "ram_gb": 24,
            "cpu_cores": 12,
            "gpu_cores": 20,
        },
        "software": {
            "macos": "26.5.1",
            "rapid_mlx": "0.7.6",
            "mlx": "0.31.2",
            "python": "3.12.13",
        },
        "model": {
            "alias": alias,
            "hf_path": aliases[alias]["hf_path"],
        },
        "config": {
            "rounds": 5,
            "warmup_rounds": 1,
            "sampling": "greedy",
            "buckets_spec": {
                "short": {"prompt_tokens": 512, "max_tokens": 128},
                "long": {"prompt_tokens": 2048, "max_tokens": 512},
            },
            "prompt_hash": "deadbeefcafebabe",
        },
        "buckets": {"short": bucket, "long": bucket},
    }


def _run_validate(*paths: Path) -> tuple[int, str]:
    """Run validate.py in a subprocess so it sees the real argv path."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "validate.py")]
    cmd.extend(str(p) for p in paths)
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return r.returncode, r.stdout + r.stderr


def _write_submission(tmp_path: Path, payload: dict, name: str | None = None) -> Path:
    """Write a payload to ``tmp_path/community-benchmarks/submissions/<name>``.

    Note: validate.py resolves the submissions dir relative to its own
    file location (``REPO_ROOT/community-benchmarks/submissions``). So
    these tests write to the REAL submissions dir under ``tmp_path``
    only when ``tmp_path`` is the real repo (it isn't). Instead, we
    pass synthetic files into the REAL submissions dir for end-to-end
    runs — see ``_run_validate_against_repo`` below.
    """
    sub_dir = tmp_path / "community-benchmarks" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    fname = name or "20260615-apple-m4-pro-qwen3.5-9b-4bit-abcdef012345.json"
    path = sub_dir / fname
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_to_real_submissions(payload: dict, name: str | None = None) -> Path:
    """Write a synthetic payload into the REAL submissions dir, returning
    a path the test must clean up afterwards.

    Tests use this to validate against the actual validate.py's
    ``_check_path_in_submissions`` guard, which resolves the path
    relative to the real repo. Cleanup is the test's responsibility.
    """
    sub_dir = REPO_ROOT / "community-benchmarks" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    fname = name or "20260615-apple-m4-pro-qwen3.5-9b-4bit-abcdef012345.json"
    path = sub_dir / fname
    path.write_text(json.dumps(payload, indent=2))
    return path


@pytest.fixture
def cleanup_real_submissions():
    """Track and remove any files written to the real submissions dir."""
    created: list[Path] = []
    yield created
    for p in created:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def test_validate_accepts_good_payload(cleanup_real_submissions) -> None:
    """A perfectly-formed payload must produce rc=0."""
    pytest.importorskip("jsonschema")
    path = _write_to_real_submissions(_good_payload())
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 0, out
    assert "OK" in out


def test_validate_rejects_bad_schema(cleanup_real_submissions) -> None:
    """Missing required field ⇒ schema failure."""
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    del bad["schema_version"]
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "schema" in out


def test_validate_rejects_wrong_const(cleanup_real_submissions) -> None:
    """``rounds=7`` violates ``const: 5``."""
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    bad["config"]["rounds"] = 7
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "rounds" in out or "schema" in out


def test_validate_rejects_unknown_alias(cleanup_real_submissions) -> None:
    bad = _good_payload()
    bad["model"]["alias"] = "definitely-not-a-real-alias"
    bad["model"]["hf_path"] = "x/y"
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "whitelist" in out.lower() or "alias" in out


def test_validate_rejects_mismatched_hf_path(cleanup_real_submissions) -> None:
    """Right alias key, wrong hf_path — possible silent retargeting."""
    bad = _good_payload()
    bad["model"]["hf_path"] = "evil/path"
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "hf_path" in out


def test_validate_rejects_implausible_decode_tps(cleanup_real_submissions) -> None:
    """A decode_tps of 9000 trips the sanity ceiling."""
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    bad["buckets"]["short"]["decode_tps"]["median"] = 9000.0
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    # schema's range cap is 5000, so this also fails schema validation
    # before we get to the sanity check — both are valid outcomes.
    assert rc == 1


def test_validate_rejects_bad_filename(cleanup_real_submissions) -> None:
    pytest.importorskip("jsonschema")
    # Wrong shape: no date prefix
    path = _write_to_real_submissions(_good_payload(), name="bogus.json")
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "filename" in out


def test_validate_rejects_zero_decode_tps(cleanup_real_submissions) -> None:
    """Regression: ``decode_tps=0`` must fail. Round-2 used a sanity
    check on the summary median; round-7 hardened the schema to
    ``exclusiveMinimum: 0`` on per-round throughput so the rejection
    fires at the schema layer even if the sanity check were skipped.
    Either error path is acceptable; the test only insists the file
    is rejected and the offending field is named.
    """
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    # Zero every round AND every summary stat.
    for r in bad["buckets"]["short"]["rounds_raw"]:
        r["decode_tps"] = 0.0
    bad["buckets"]["short"]["decode_tps"] = {
        "median": 0.0,
        "min": 0.0,
        "max": 0.0,
        "stddev": 0.0,
    }
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "decode_tps" in out


def test_validate_rejects_non_apple_chip(cleanup_real_submissions) -> None:
    """Regression: hand-edited submission with non-Apple chip must fail.

    The CLI gate is bypassable by anyone willing to edit JSON, so the
    validator boundary has to re-enforce the Apple-Silicon-only
    contract. (Codex PR #582 round-3 BLOCKING.)
    """
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    bad["hardware"]["chip"] = "Intel Xeon E5-2670"
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "Apple" in out and "chip" in out


def test_validate_rejects_duplicate_submission_id(
    cleanup_real_submissions,
) -> None:
    """Regression: a submission whose ``submission_id`` already exists
    in submissions/ must be rejected. (Codex PR #582 round-3 BLOCKING.)

    Setup: drop a "first" file with one id, then a "second" file with
    the SAME id but a different filename. Validate the second — it
    should fail.
    """
    pytest.importorskip("jsonschema")
    first = _good_payload()
    first["submission_id"] = "aaaaaaaaaaaa"
    second = _good_payload()
    second["submission_id"] = "aaaaaaaaaaaa"  # collision

    first_path = _write_to_real_submissions(
        first, name="20260101-apple-m4-pro-qwen3-5-9b-4bit-aaaaaaaaaaaa.json"
    )
    cleanup_real_submissions.append(first_path)
    second_path = _write_to_real_submissions(
        second, name="20260102-apple-m4-pro-qwen3-5-9b-4bit-aaaaaaaaaaaa.json"
    )
    cleanup_real_submissions.append(second_path)

    rc, out = _run_validate(second_path)
    assert rc == 1
    assert "submission_id" in out and "already exists" in out


def test_validate_rejects_bad_datetime_format(cleanup_real_submissions) -> None:
    """Regression: schema ``format: date-time`` must actually be enforced.

    jsonschema's ``validate()`` ignored ``format:`` until we wired in
    a Draft202012Validator with FORMAT_CHECKER. (Codex PR #582 round-3
    NIT.)
    """
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    bad["submitted_at"] = "definitely not a timestamp"
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1


@pytest.mark.parametrize(
    "bogus_value",
    [
        # Date-only — fromisoformat accepts this, but schema promises
        # date-time. Without the regex gate the validator silently passed
        # these as valid timestamps. (Codex PR #602 round-1 BLOCKING.)
        "2026-06-16",
        # Naïve timestamp — has T-separator and time component but no
        # timezone offset. fromisoformat returns a tz-less datetime;
        # schema requires Z or ±HH:MM per RFC 3339.
        "2026-06-16T12:00:00",
        "2026-06-16T12:00:00.123456",
        # Time-only — fromisoformat happily parses HH:MM:SS but it's
        # not a valid date-time.
        "12:00:00Z",
        # Empty string.
        "",
        # Wrong separator.
        "2026/06/16T12:00:00Z",
        # Space separator — ISO 8601 permits it as a readability
        # alternative but RFC 3339 §5.6 only admits T/t. Schema
        # promises RFC 3339, so this must reject.
        # (Codex PR #602 round-2 BLOCKING.)
        "2026-06-16 12:00:00Z",
    ],
)
def test_validate_rejects_non_rfc3339_datetimes(
    cleanup_real_submissions, bogus_value
) -> None:
    """Regression: the ``date-time`` checker must enforce full RFC 3339,
    not the loose subset accepted by ``datetime.fromisoformat`` alone.

    Codex's PR #602 round-1 BLOCKING: date-only strings and naïve
    timestamps without offset slipped through the first version of the
    fix because ``fromisoformat`` accepts them. The current implementation
    requires a regex match for the full ``YYYY-MM-DDThh:mm:ss(.frac)?
    (Z|±hh:mm)`` shape AND a non-None tzinfo on the parsed object.
    """
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    bad["submitted_at"] = bogus_value
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1, (
        f"validator must reject submitted_at={bogus_value!r} per RFC 3339; "
        f"got rc=0 (passed) with out={out!r}"
    )


@pytest.mark.parametrize(
    "good_value",
    [
        # Canonical CLI output: Z suffix.
        "2026-06-16T12:00:00Z",
        # With fractional seconds.
        "2026-06-16T12:00:00.123456Z",
        # Numeric UTC offset.
        "2026-06-16T12:00:00+00:00",
        # Non-UTC numeric offset.
        "2026-06-16T08:00:00-04:00",
        # Lowercase t separator is also valid RFC 3339.
        "2026-06-16t12:00:00Z",
        # Lowercase z offset is also valid RFC 3339 §5.6.
        # (Codex PR #602 round-3 BLOCKING.)
        "2026-06-16T12:00:00z",
        "2026-06-16t12:00:00z",
    ],
)
def test_validate_accepts_valid_rfc3339_datetimes(
    cleanup_real_submissions, good_value
) -> None:
    """Counterpart: the tightened checker must not over-reject legitimate
    RFC 3339 forms — Z suffix, fractional seconds, numeric offsets,
    lowercase ``t``."""
    pytest.importorskip("jsonschema")
    good = _good_payload()
    good["submitted_at"] = good_value
    path = _write_to_real_submissions(good)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 0, (
        f"validator wrongly rejected submitted_at={good_value!r}; out={out!r}"
    )


def test_validate_rejects_summary_mismatch_with_rounds(
    cleanup_real_submissions,
) -> None:
    """Regression: summary stats must be derivable from ``rounds_raw``
    (Codex PR #582 round-2 BLOCKING).

    Without this check, a malicious contributor could ship a 200 tok/s
    median alongside a plausible 40 tok/s ``rounds_raw`` and the
    aggregator (which trusts ``median``) would publish the lie.
    """
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    # Median is 42 (true), but we claim 99 — large enough delta to
    # exceed the 1e-3 relative tolerance.
    bad["buckets"]["short"]["decode_tps"]["median"] = 99.0
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "rounds_raw" in out and "median" in out


def test_validate_rejects_nan_in_payload(cleanup_real_submissions) -> None:
    """Regression: NaN/Infinity in any numeric field must fail.

    json.loads accepts these by default and every comparison against
    NaN evaluates False, so the sanity bounds were bypassable. (Codex
    PR #582 round-7 BLOCKING.)
    """
    pytest.importorskip("jsonschema")
    payload = _good_payload()
    path = _write_to_real_submissions(payload)
    cleanup_real_submissions.append(path)
    # Write a NaN by hand — json.dump won't emit it for a Python float
    # with allow_nan=False, but a hand-edited PR JSON could carry one.
    raw = path.read_text().replace('"decode_tps": 40.0', '"decode_tps": NaN', 1)
    path.write_text(raw)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "non-finite" in out.lower() or "nan" in out.lower()


def test_validate_rejects_symlink(cleanup_real_submissions, tmp_path) -> None:
    """Symlinks under submissions/ are rejected — they could collapse
    the dedup check and inflate a single contributor's row count.
    (Codex PR #582 round-7 BLOCKING.)
    """
    pytest.importorskip("jsonschema")
    target_payload = _good_payload()
    target_path = _write_to_real_submissions(target_payload)
    cleanup_real_submissions.append(target_path)

    sub_dir = REPO_ROOT / "community-benchmarks" / "submissions"
    link_path = sub_dir / "20260615-apple-m4-pro-qwen3-5-9b-4bit-bbbbbbbbbbbb.json"
    link_path.symlink_to(target_path.name)  # relative within submissions/
    cleanup_real_submissions.append(link_path)

    rc, out = _run_validate(link_path)
    assert rc == 1
    assert "symlink" in out.lower()


def test_validate_rejects_zero_in_rounds_raw(cleanup_real_submissions) -> None:
    """A zero (or negative) value in any rounds_raw row must fail, even
    when the summary median lands in the realistic band. (Codex PR
    #582 round-7 BLOCKING.)
    """
    pytest.importorskip("jsonschema")
    bad = _good_payload()
    # Park 4 normal rows and one zero; the median is still legit.
    rounds = bad["buckets"]["short"]["rounds_raw"]
    rounds[2]["decode_tps"] = 0.0
    # Re-derive summary so summary-vs-raw recompute passes.
    import statistics

    values = [r["decode_tps"] for r in rounds]
    bad["buckets"]["short"]["decode_tps"] = {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "stddev": float(statistics.pstdev(values)),
    }
    path = _write_to_real_submissions(bad)
    cleanup_real_submissions.append(path)
    rc, out = _run_validate(path)
    assert rc == 1
    assert "rounds_raw" in out and "decode_tps" in out


def test_validate_rejects_empty_hf_path_in_alias_entry(
    cleanup_real_submissions, monkeypatch, tmp_path
) -> None:
    """If aliases.json carries an empty hf_path for an alias, the
    submission MUST be rejected — not waved through. (Codex PR #582
    round-7 BLOCKING.)

    We patch aliases.json at the path the validate.py subprocess
    resolves, run the validator against a payload that names a real
    alias, and verify it raises.
    """
    pytest.importorskip("jsonschema")
    payload = _good_payload()
    path = _write_to_real_submissions(payload)
    cleanup_real_submissions.append(path)

    # Snapshot the real aliases.json and restore it via try/finally —
    # the cleanup fixture only knows how to delete temp submission
    # files, not how to restore a checked-in JSON.
    real_aliases = ALIASES_PATH.read_text()
    aliases_dict = json.loads(real_aliases)
    target_alias = payload["model"]["alias"]
    aliases_dict[target_alias] = {}  # missing hf_path
    ALIASES_PATH.write_text(json.dumps(aliases_dict, indent=2))
    try:
        rc, out = _run_validate(path)
        assert rc == 1
        assert "hf_path" in out
    finally:
        ALIASES_PATH.write_text(real_aliases)


def test_submission_make_pr_uses_repo_cwd(tmp_path, monkeypatch) -> None:
    """Regression: ``gh pr create`` must run with ``cwd=repo`` so the
    PR lands in the right checkout, not in whatever directory the
    user happened to be in when they ran ``rapid-mlx bench --submit``.
    (Codex PR #582 round-2 BLOCKING.)
    """
    from vllm_mlx.community_bench import submission as sub_mod

    calls: list[dict] = []

    def fake_run(cmd, capture_output, text, check, cwd):
        calls.append({"cmd": cmd, "cwd": cwd})

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr(sub_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(sub_mod.shutil, "which", lambda name: "/usr/local/bin/gh")

    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {"alias": "x", "hf_path": "y/z"},
        "hardware": {"chip": "Apple M4 Pro", "ram_gb": 24},
        "software": {"rapid_mlx": "0.7.6", "mlx": "0.31.2"},
        "config": {"sampling": "greedy"},
        "buckets": {
            "short": {"decode_tps": {"median": 40.0}},
            "long": {"decode_tps": {"median": 40.0}},
        },
        "notes": None,
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    sub_mod._make_pr_via_gh(
        tmp_path,
        sub_path,
        payload,
        stdout=io.StringIO(),
        origin_owner="raullenchai",
        upstream_remote="origin",
    )

    assert calls, "no subprocess calls captured"
    # Every step must run inside the resolved repo dir.
    for c in calls:
        assert c["cwd"] == str(tmp_path), (
            f"step {c['cmd'][0]!r} ran with cwd={c['cwd']!r}, "
            f"expected {str(tmp_path)!r}"
        )


def test_state_aware_fallback_skips_already_completed_steps(tmp_path, capsys) -> None:
    """Regression: when ``_make_pr_via_gh`` bails after ``push`` (e.g.
    ``gh pr create`` failed), the manual-fallback instructions must
    NOT tell the user to ``git checkout -b <branch>`` because the
    branch already exists and is already pushed. (Codex PR #582
    round-5 BLOCKING.)
    """
    from vllm_mlx.community_bench import submission as sub_mod

    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {"alias": "x", "hf_path": "y/z"},
        "hardware": {"chip": "Apple M4 Pro", "ram_gb": 24},
        "software": {"rapid_mlx": "0.7.6", "mlx": "0.31.2"},
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    completed = {"checkout", "stage", "commit", "push"}  # all but pr_create
    out = io.StringIO()
    sub_mod._print_manual_fallback(
        tmp_path, sub_path, payload, stdout=out, completed=completed
    )
    text = out.getvalue()
    # Must NOT instruct to recreate the branch — that would fail
    # because it's already pushed.
    assert "git checkout -b" not in text
    assert "git add" not in text
    assert "git commit" not in text
    assert "git push" not in text
    # Must still show the only remaining step.
    assert "gh pr create" in text
    # Should explain what already happened.
    assert "Already completed" in text


def test_manual_fallback_without_gh_points_at_web_ui(tmp_path, monkeypatch) -> None:
    """Regression: when ``gh`` is not on PATH (the common newcomer case),
    the fallback must not tell the user to run ``gh pr create`` — it
    has to point them at the GitHub compare page and at the
    paste-to-issue escape hatch so they can finish the submission
    without installing extra tooling. (Subagent-friction report, #4.)
    """
    from vllm_mlx.community_bench import submission as sub_mod

    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {"alias": "qwen3.5-9b-4bit", "hf_path": "y/z"},
        "hardware": {"chip": "Apple M3 Ultra", "ram_gb": 64},
        "software": {"rapid_mlx": "0.7.13", "mlx": "0.31.2"},
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    monkeypatch.setattr(sub_mod.shutil, "which", lambda _: None)
    # tmp_path has no git remote — _origin_is_safe_github returns
    # (False, None) so the owner-less compare URL is used.
    out = io.StringIO()
    sub_mod._print_manual_fallback(tmp_path, sub_path, payload, stdout=out)
    text = out.getvalue()

    # gh isn't installed — must NOT recommend gh pr create.
    assert "gh pr create" not in text
    # Must surface the compare-page URL with the branch ref filled in.
    branch = f"community-bench/{payload['submission_id']}"
    assert (
        f"https://github.com/{sub_mod.UPSTREAM_REPO_FOR_GH}"
        f"/compare/main...{branch}?expand=1"
    ) in text
    # Must also offer the paste-to-issue escape hatch so users with no
    # git fluency at all still have a way to land their numbers.
    assert (f"https://github.com/{sub_mod.UPSTREAM_REPO_FOR_GH}/issues/new") in text
    # Title (alias + chip) round-trips through urlencode — verify both
    # appear unmodified after decoding.
    import urllib.parse as _up

    issue_line = next(
        line for line in text.splitlines() if "issues/new" in line
    ).strip()
    query = issue_line.split("?", 1)[1]
    parsed = dict(_up.parse_qsl(query))
    assert parsed["title"] == "community-bench: qwen3.5-9b-4bit on Apple M3 Ultra"


def test_manual_fallback_without_gh_uses_fork_owner_when_origin_is_fork(
    tmp_path, monkeypatch
) -> None:
    """Regression: when origin is a contributor's fork (the normal
    contribution path), the compare URL must use ``main...<owner>:<branch>``
    so GitHub can find the head branch on the fork — bare ``main...<branch>``
    only resolves on the upstream repo. (Codex PR #600 round-1 BLOCKING.)
    """
    from vllm_mlx.community_bench import submission as sub_mod

    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {"alias": "qwen3.5-9b-4bit", "hf_path": "y/z"},
        "hardware": {"chip": "Apple M3 Ultra", "ram_gb": 64},
        "software": {"rapid_mlx": "0.7.14", "mlx": "0.31.2"},
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    monkeypatch.setattr(sub_mod.shutil, "which", lambda _: None)
    # Mock origin → contributor's fork on github.com.
    monkeypatch.setattr(
        sub_mod,
        "_origin_is_safe_github",
        lambda _repo: (True, "some-contributor"),
    )
    out = io.StringIO()
    sub_mod._print_manual_fallback(tmp_path, sub_path, payload, stdout=out)
    text = out.getvalue()

    branch = f"community-bench/{payload['submission_id']}"
    # Cross-fork compare URL: head is owner-prefixed.
    assert (f"compare/main...some-contributor:{branch}?expand=1") in text
    # Must NOT print the bare same-repo form (would 404 for the user).
    assert f"compare/main...{branch}?expand=1" not in text


def test_manual_fallback_without_gh_skips_owner_when_origin_is_upstream(
    tmp_path, monkeypatch
) -> None:
    """Counterpart to the fork case: when origin owner equals the
    upstream owner (maintainer running locally), the URL must NOT
    include the ``upstream:`` prefix — that would be redundant and
    GitHub's compare page redirects it anyway.
    """
    from vllm_mlx.community_bench import submission as sub_mod

    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {"alias": "x", "hf_path": "y/z"},
        "hardware": {"chip": "Apple M4 Pro", "ram_gb": 24},
        "software": {"rapid_mlx": "0.7.14", "mlx": "0.31.2"},
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    upstream_owner = sub_mod.UPSTREAM_REPO_FOR_GH.split("/", 1)[0]
    monkeypatch.setattr(sub_mod.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        sub_mod,
        "_origin_is_safe_github",
        lambda _repo: (True, upstream_owner),
    )
    out = io.StringIO()
    sub_mod._print_manual_fallback(tmp_path, sub_path, payload, stdout=out)
    text = out.getvalue()

    branch = f"community-bench/{payload['submission_id']}"
    assert f"compare/main...{branch}?expand=1" in text
    assert f"compare/main...{upstream_owner}:" not in text


def test_manual_fallback_compare_url_quotes_owner_and_branch(
    tmp_path, monkeypatch
) -> None:
    """Regression: the ``main...<owner>:<branch>`` compare URL must
    URL-quote both halves independently — any owner or branch ref
    carrying ``#``, ``?``, ``%`` or other URL-reserved chars would
    otherwise truncate or rewrite the URL silently. Branch is
    constructed by us (``community-bench/<hex>``) and owner is
    validated upstream, but pinning the quoting at this layer guards
    against future loosening. (Codex PR #600 round-2 BLOCKING.)
    """
    from vllm_mlx.community_bench import submission as sub_mod

    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {"alias": "x", "hf_path": "y/z"},
        "hardware": {"chip": "Apple M3 Ultra", "ram_gb": 64},
        "software": {"rapid_mlx": "0.7.14", "mlx": "0.31.2"},
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    monkeypatch.setattr(sub_mod.shutil, "which", lambda _: None)
    # Deliberately pathological owner with URL-reserved chars — a real
    # github.com username can't carry these, but we want the quoting
    # layer to be load-bearing regardless.
    monkeypatch.setattr(
        sub_mod,
        "_origin_is_safe_github",
        lambda _repo: (True, "weird?owner#name%"),
    )
    out = io.StringIO()
    sub_mod._print_manual_fallback(tmp_path, sub_path, payload, stdout=out)
    text = out.getvalue()

    # The pathological chars must appear percent-encoded — not raw —
    # so the URL parses as a single path segment, not as a malformed
    # query/fragment split.
    compare_line = next(
        line for line in text.splitlines() if "/compare/main..." in line
    ).strip()
    assert "weird?owner#name%" not in compare_line
    assert "weird%3Fowner%23name%25" in compare_line
    # The ``:`` between owner and branch stays literal (GitHub's
    # syntax requirement).
    assert ":community-bench/" in compare_line


def test_manual_fallback_url_encodes_alias_special_chars(tmp_path, monkeypatch) -> None:
    """Regression: aliases or chip names containing ``&``, ``#``, ``/``,
    ``%``, or spaces must round-trip cleanly through the issue-new
    URL. Bare ``.replace(' ', '%20')`` produced malformed URLs for
    realistic aliases like ``qwen3.6/27b``. (Codex PR #600 round-1 NIT.)
    """
    from vllm_mlx.community_bench import submission as sub_mod

    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {
            "alias": "qwen3.6/27b & friends #beta 50%",
            "hf_path": "y/z",
        },
        "hardware": {"chip": "Apple M3 Ultra", "ram_gb": 64},
        "software": {"rapid_mlx": "0.7.14", "mlx": "0.31.2"},
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    monkeypatch.setattr(sub_mod.shutil, "which", lambda _: None)
    out = io.StringIO()
    sub_mod._print_manual_fallback(tmp_path, sub_path, payload, stdout=out)
    text = out.getvalue()

    # Decode the title param back and assert it round-trips losslessly.
    import urllib.parse as _up

    issue_line = next(
        line for line in text.splitlines() if "issues/new" in line
    ).strip()
    query = issue_line.split("?", 1)[1]
    parsed = dict(_up.parse_qsl(query))
    assert (
        parsed["title"]
        == "community-bench: qwen3.6/27b & friends #beta 50% on Apple M3 Ultra"
    )


def test_manual_fallback_with_gh_keeps_gh_command(tmp_path, monkeypatch) -> None:
    """Counterpart: when ``gh`` IS installed, the fallback should keep
    surfacing ``gh pr create`` (the original behavior — used when a
    later git step failed mid-sequence). Pins that we don't drop the
    gh path while fixing the gh-missing one.
    """
    from vllm_mlx.community_bench import submission as sub_mod

    payload = {
        "submission_id": "abcdef012345",
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "model": {"alias": "x", "hf_path": "y/z"},
        "hardware": {"chip": "Apple M4 Pro", "ram_gb": 24},
        "software": {"rapid_mlx": "0.7.13", "mlx": "0.31.2"},
    }
    sub_path = tmp_path / "submission.json"
    sub_path.write_text("{}")

    monkeypatch.setattr(sub_mod.shutil, "which", lambda name: "/opt/homebrew/bin/gh")
    out = io.StringIO()
    sub_mod._print_manual_fallback(tmp_path, sub_path, payload, stdout=out)
    text = out.getvalue()

    assert "gh pr create" in text
    # Web-UI fallback shouldn't appear when gh is available.
    assert "compare/main..." not in text
    assert "/issues/new" not in text


def test_decode_tps_formula_uses_n_minus_one(monkeypatch) -> None:
    """Regression: ``decode_tps`` must be computed as ``(N-1)/window``,
    not ``N/window`` — the inter-token window measures N-1 gaps for
    N tokens. (Codex PR #582 round-5 BLOCKING.) Verify the field
    value matches vLLM TPOT semantics on a synthetic round.
    """
    import asyncio

    from vllm_mlx.community_bench import runner

    class _FakeOutput:
        def __init__(
            self,
            new_token_ids: list[int],
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            output_token_ids: list[int] | None = None,
            finished: bool = False,
        ):
            self.new_token_ids = new_token_ids
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.output_token_ids = output_token_ids or []
            self.finished = finished

    class _FakeEngine:
        async def add_request(self, prompt, params):
            return "rid"

        async def stream_outputs(self, rid, timeout):
            # First call yields token 1 (sets t_first_token).
            yield _FakeOutput(new_token_ids=[1])
            # Five more tokens. Total N = 6.
            for i in range(5):
                yield _FakeOutput(new_token_ids=[i + 2])
            yield _FakeOutput(
                new_token_ids=[],
                prompt_tokens=512,
                completion_tokens=6,
                output_token_ids=list(range(1, 7)),
                finished=True,
            )

    # Pin perf_counter so the decode window has a known value.
    # ``_run_one_round`` calls perf_counter exactly three times:
    #   1. t_start at function entry
    #   2. t_first_token on the first chunk carrying new_token_ids
    #   3. t_end after the loop exits
    # We don't need per-chunk timestamps; the formula only uses the
    # window between calls 2 and 3.
    times = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(times))

    async def _run():
        from vllm_mlx.request import SamplingParams

        return await runner._run_one_round(
            _FakeEngine(),
            "synthetic prompt",
            SamplingParams(max_tokens=6, temperature=0.0, top_p=1.0, top_k=0),
            target_prompt_tokens=512,
            expected_completion_tokens=6,
        )

    result = asyncio.run(_run())
    # t_first_token=1.0, t_end=2.0 ⇒ decode_window = 1.0s.
    # N=6 completion_tokens. (N-1)/window = 5/1 = 5.0 tok/s.
    # The OLD (buggy) formula would give 6.0 tok/s; assert we get 5.0.
    assert abs(result.decode_tps - 5.0) < 1e-9, (
        f"decode_tps={result.decode_tps}; expected 5.0 = (6-1)/1.0"
    )


def test_run_one_round_rejects_eos_early_stop(monkeypatch) -> None:
    """Regression: a round that produces FEWER tokens than the bucket's
    ``max_tokens`` (e.g. because the model emitted EOS early) must
    fail loudly rather than silently publish a non-comparable number.
    (Codex PR #582 round-6 BLOCKING.)"""
    import asyncio

    from vllm_mlx.community_bench import runner

    class _FakeOutput:
        def __init__(
            self,
            new_token_ids: list[int],
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            output_token_ids: list[int] | None = None,
            finished: bool = False,
        ):
            self.new_token_ids = new_token_ids
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.output_token_ids = output_token_ids or []
            self.finished = finished

    class _FakeEngine:
        async def add_request(self, prompt, params):
            return "rid"

        async def stream_outputs(self, rid, timeout):
            yield _FakeOutput(new_token_ids=[1])
            for i in range(2):
                yield _FakeOutput(new_token_ids=[i + 2])
            # Only 3 tokens emitted; bucket asked for 6.
            yield _FakeOutput(
                new_token_ids=[],
                prompt_tokens=512,
                completion_tokens=3,
                output_token_ids=[1, 2, 3],
                finished=True,
            )

    times = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(times))

    async def _run():
        from vllm_mlx.request import SamplingParams

        return await runner._run_one_round(
            _FakeEngine(),
            "synthetic prompt",
            SamplingParams(max_tokens=6, temperature=0.0, top_p=1.0, top_k=0),
            target_prompt_tokens=512,
            expected_completion_tokens=6,
        )

    with pytest.raises(RuntimeError, match="generated 3 tokens"):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# validate.py path-in-submissions check (GHA trust-gate regression)
# ---------------------------------------------------------------------------


def test_validate_path_check_works_from_relocated_validator(
    cleanup_real_submissions, tmp_path: Path
) -> None:
    """The GHA workflow copies validate.py to ``/tmp/base-validator/`` so the
    "trusted base" pass can run a frozen-in-time validator against the PR's
    added files. With ``SUBMISSIONS_DIR`` previously computed from the
    validator's own ``REPO_ROOT``, the relocated copy thought *every*
    submission file was "not inside community-benchmarks/submissions/" —
    blocking every community PR. (See PR #585/#586 validation failures
    after v0.7.9 release.)

    Regression: copy validate.py to a temp location, invoke it against a
    submission file in the REAL repo. The check must accept it because
    the file's *own* ancestry is ``community-benchmarks/submissions/``.
    """
    pytest.importorskip("jsonschema")
    payload = _good_payload()
    path = _write_to_real_submissions(payload)
    cleanup_real_submissions.append(path)

    # Mirror the GHA setup: copy just the validator script to /tmp-style
    # location. SCHEMA_PATH and ALIASES_PATH still resolve via the
    # relocated REPO_ROOT, so we need to seed them too — that's what the
    # workflow does with ``git show "$BASE:..."``.
    relocated = tmp_path / "community-benchmarks" / "scripts"
    relocated.mkdir(parents=True)
    (tmp_path / "community-benchmarks").mkdir(exist_ok=True)
    (tmp_path / "vllm_mlx").mkdir(parents=True, exist_ok=True)
    (relocated / "validate.py").write_bytes((SCRIPTS_DIR / "validate.py").read_bytes())
    (tmp_path / "community-benchmarks" / "schema.json").write_bytes(
        SCHEMA_PATH.read_bytes()
    )
    (tmp_path / "vllm_mlx" / "aliases.json").write_bytes(ALIASES_PATH.read_bytes())

    r = subprocess.run(
        [sys.executable, str(relocated / "validate.py"), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"relocated validator rejected a valid submission: {out}"
    assert "is not inside community-benchmarks/submissions/" not in out


def test_relocated_validator_catches_duplicate_id(
    cleanup_real_submissions, tmp_path: Path
) -> None:
    """The duplicate-``submission_id`` gate must still fire even when the
    validator is relocated (GHA trust-gate setup).

    The first fix relied on the validator's location-derived
    ``SUBMISSIONS_DIR`` to enumerate the corpus when scanning for
    duplicates. Under relocation that directory is empty, so
    ``_load_submission_id_index`` returned an empty index and a
    malicious PR could ship a duplicate ``submission_id`` past the
    trusted base pass. (Codex PR #587 BLOCKING.) The follow-up derives
    the corpus from the target file's parent — which the structural
    path check already validated as the real submissions/ folder — so
    the duplicate check fires regardless of where the validator script
    itself lives.
    """
    pytest.importorskip("jsonschema")

    # Plant a duplicate-id payload on disk in the REAL submissions dir.
    # ``submission_id`` is the dedup key; both files share it.
    payload_a = _good_payload()
    sid = payload_a["submission_id"]
    payload_b = _good_payload()
    payload_b["submission_id"] = sid  # same id, different filename hex below

    name_a = "20260615-apple-m4-pro-qwen3.5-9b-4bit-abcdef012345.json"
    name_b = "20260615-apple-m4-pro-qwen3.5-9b-4bit-fedcba543210.json"
    path_a = _write_to_real_submissions(payload_a, name=name_a)
    path_b = _write_to_real_submissions(payload_b, name=name_b)
    cleanup_real_submissions.extend([path_a, path_b])

    # Relocate just the validator + its data deps to a tmp tree —
    # mirroring the GHA workflow.
    relocated = tmp_path / "community-benchmarks" / "scripts"
    relocated.mkdir(parents=True)
    (tmp_path / "vllm_mlx").mkdir(parents=True, exist_ok=True)
    (relocated / "validate.py").write_bytes((SCRIPTS_DIR / "validate.py").read_bytes())
    (tmp_path / "community-benchmarks" / "schema.json").write_bytes(
        SCHEMA_PATH.read_bytes()
    )
    (tmp_path / "vllm_mlx" / "aliases.json").write_bytes(ALIASES_PATH.read_bytes())

    # Invoke the RELOCATED validator on file B. Pre-fix this would have
    # silently passed because /tmp/.../submissions/ is empty (no
    # duplicates visible to the index). Post-fix it must fail because
    # the corpus is derived from path_b's own parent (the real dir
    # containing both files).
    r = subprocess.run(
        [sys.executable, str(relocated / "validate.py"), str(path_b)],
        capture_output=True,
        text=True,
        check=False,
    )
    out = r.stdout + r.stderr
    assert r.returncode != 0, f"relocated validator missed duplicate id: {out}"
    assert "duplicate" in out.lower() or "submission_id" in out.lower()


# ---------------------------------------------------------------------------
# _run_submit_flow alias-key gate
# ---------------------------------------------------------------------------


def test_submit_flow_guard_consults_original_alias(monkeypatch, capsys) -> None:
    """The CLI dispatcher mutates ``args.model`` to the resolved HF path
    *before* calling ``_run_submit_flow``, then stashes the user-typed
    alias on ``args._original_alias``. The whitelist gate must read that
    stash, not ``args.model`` — otherwise every alias submission fails
    with a spurious "not the canonical alias key" error even when the
    contributor did type the canonical alias.

    Regression test for the v0.7.7 bench --submit failure on
    llama3-1b-4bit (the resolved HF path
    ``mlx-community/Llama-3.2-1B-Instruct-4bit`` tripped the ``"/" in
    args.model`` branch).
    """
    from argparse import Namespace

    from vllm_mlx import cli as cli_mod
    from vllm_mlx.community_bench import hardware as hw

    # Pretend we're on Apple Silicon so the gate above this one passes.
    # ``is_apple_silicon`` is imported inside _run_submit_flow, so we
    # patch the source module.
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)

    args = Namespace(
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        _original_alias="llama3-1b-4bit",
        notes=None,
        sampled=False,
        repo_root=None,
        force_disk_check=False,
    )

    # We expect the flow to get past the guard and fail later — capture
    # whatever happens by intercepting the next stage. The simplest probe
    # is to swap ``_check_disk_space`` with a sentinel that raises a
    # known marker; if the guard fires first we never see the marker.
    marker = RuntimeError("__past_guard__")

    def _sentinel(*_a, **_kw):
        raise marker

    monkeypatch.setattr(cli_mod, "_check_disk_space", _sentinel)

    with pytest.raises(RuntimeError, match="__past_guard__"):
        cli_mod._run_submit_flow(args)

    out = capsys.readouterr().out
    # Sanity: the bogus "not the canonical alias key" branch should NOT
    # have printed.
    assert "not the resolved HF path" not in out
    assert "Run `rapid-mlx models`" not in out


def test_submit_flow_guard_rejects_hf_path_when_no_original_alias(
    monkeypatch, capsys
) -> None:
    """When the user passes an HF path directly (no alias resolution
    happens, so ``_original_alias`` is unset), the guard must still
    reject it — that's the whole point of requiring canonical alias
    keys."""
    from argparse import Namespace

    from vllm_mlx import cli as cli_mod
    from vllm_mlx.community_bench import hardware as hw

    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)

    args = Namespace(
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        notes=None,
        sampled=False,
        repo_root=None,
        force_disk_check=False,
    )

    rc = cli_mod._run_submit_flow(args)
    assert rc == 2
    out = capsys.readouterr().out
    assert "requires the canonical alias key" in out
    assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in out
