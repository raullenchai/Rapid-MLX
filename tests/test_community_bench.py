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
- ``community-benchmarks/scripts/aggregate.py`` — grouping + percentile
  output on synthetic submissions.

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
        == schema_cfg["buckets_spec"]["properties"]["long"]["properties"][
            "max_tokens"
        ]["const"]
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
        RoundResult(decode_tps=42.0, prefill_tps=500.0, ttft_ms=120.0)
        for _ in range(5)
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
        hw, sw, "qwen3.5-9b-4bit", "mlx-community/Qwen3.5-9B-4bit",
        bench, notes=None,
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
    assert re.match(
        r"^[0-9]{8}-[a-z0-9-]+-[a-z0-9.-]+-[0-9a-f]{12}\.json$", name
    )
    assert name.startswith("20260615-apple-m4-pro-")


def test_ask_consent_yes() -> None:
    from vllm_mlx.community_bench.submission import _ask_consent

    stdin = io.StringIO("y\n")
    stdout = io.StringIO()
    assert _ask_consent({"key": "val"}, stdin=stdin, stdout=stdout) is True
    assert "Press [y]" in stdout.getvalue()


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

    (tmp_path / ".git").mkdir()
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


def test_submit_interactive_writes_file_then_falls_back_on_dirty_tree(
    tmp_path: Path, monkeypatch
) -> None:
    """When git is dirty, the file IS written but no PR is opened.

    Privacy contract: the user can always recover the file and finish
    the PR by hand — we never block them on automation working.
    """
    from vllm_mlx.community_bench import submission as sub_mod

    (tmp_path / ".git").mkdir()
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


# ---------------------------------------------------------------------------
# validate.py — schema + sanity gate
# ---------------------------------------------------------------------------


def _good_payload() -> dict:
    """A payload that passes every check, used as the baseline for
    mutation tests below."""
    aliases = json.loads(ALIASES_PATH.read_text())
    alias = next(iter(aliases))
    rounds = [{"decode_tps": 42.0, "prefill_tps": 500.0, "ttft_ms": 100.0}] * 5
    bucket = {
        "decode_tps": {"median": 42.0, "min": 41.0, "max": 43.0, "stddev": 1.0},
        "prefill_tps": {"median": 500.0, "min": 490.0, "max": 510.0, "stddev": 5.0},
        "ttft_ms": {"median": 100.0, "min": 95.0, "max": 105.0, "stddev": 3.0},
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


# ---------------------------------------------------------------------------
# aggregate.py
# ---------------------------------------------------------------------------


def _run_aggregate() -> tuple[int, str]:
    """Run aggregate.py in a subprocess."""
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "aggregate.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.returncode, r.stdout + r.stderr


def test_aggregate_percentile_single_value() -> None:
    """One sample ⇒ every percentile is that sample."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import aggregate as agg
    finally:
        sys.path.pop(0)
    assert agg._percentile([5.0], 25.0) == 5.0
    assert agg._percentile([5.0], 75.0) == 5.0


def test_aggregate_percentile_linear() -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import aggregate as agg
    finally:
        sys.path.pop(0)
    # p50 of [1,2,3,4,5] is 3.0 — direct hit on index 2
    assert agg._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0) == 3.0
    # p25 of [1,2,3,4,5] is 2.0 — direct hit on index 1
    assert agg._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 25.0) == 2.0


def test_aggregate_agg_empty_returns_zeros() -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import aggregate as agg
    finally:
        sys.path.pop(0)
    r = agg._agg([])
    assert r["count"] == 0
    assert r["median"] == 0.0


def test_aggregate_groups_by_chip_and_alias(tmp_path: Path) -> None:
    """Two submissions with same key collapse to one row; sample_count=2."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import aggregate as agg
    finally:
        sys.path.pop(0)

    def mk_payload(decode_tps: float) -> dict:
        p = _good_payload()
        p["buckets"]["short"]["decode_tps"]["median"] = decode_tps
        p["buckets"]["long"]["decode_tps"]["median"] = decode_tps
        return p

    rows = agg._aggregate([mk_payload(40.0), mk_payload(50.0)])
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_count"] == 2
    assert row["buckets"]["short"]["decode_tps"]["count"] == 2
    assert row["buckets"]["short"]["decode_tps"]["median"] == 45.0


def test_aggregate_skips_unsupported_schema_version(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import aggregate as agg
    finally:
        sys.path.pop(0)

    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir()
    good = _good_payload()
    bad = _good_payload()
    bad["schema_version"] = 999
    (submissions_dir / "good.json").write_text(json.dumps(good))
    (submissions_dir / "bad.json").write_text(json.dumps(bad))
    loaded = agg._load_all(submissions_dir.glob("*.json"))
    assert len(loaded) == 1
    assert loaded[0]["schema_version"] == 1
