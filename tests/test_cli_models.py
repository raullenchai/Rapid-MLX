# SPDX-License-Identifier: Apache-2.0
"""Tests for `rapid-mlx models` — pins the per-alias profile rendering."""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

from vllm_mlx import cli
from vllm_mlx.model_aliases import list_profiles


def _capture_models_output() -> str:
    """Run `models_command` with stdout captured. Patches the version-check
    helper so the test stays hermetic (no PyPI lookup at test time)."""
    buf = io.StringIO()
    with (
        patch.object(sys, "stdout", buf),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.models_command(None)
    return buf.getvalue()


def test_models_command_lists_all_aliases():
    """Every alias in aliases.json must appear in the table."""
    out = _capture_models_output()
    profiles = list_profiles()
    assert len(profiles) >= 20, "expected 20+ aliases (per project goal)"
    for alias in profiles:
        assert alias in out, f"alias {alias!r} missing from `rapid-mlx models` output"
    assert f"({len(profiles)} aliases)" in out


def test_models_command_shows_capability_columns():
    """Capability columns (Tools, Reasoning, Spec-Decode, Suffix Tier) appear."""
    out = _capture_models_output()
    for header in ("Tools", "Reasoning", "Spec-Decode", "Suffix Tier"):
        assert header in out, f"column header {header!r} missing"


def test_models_command_renders_hybrid_marker_for_qwen35_moe():
    """Hybrid MoE models (e.g. qwen3.5-35b-4bit, the A3B variant) must
    show '✗ hybrid' + tier 'n/a'.

    r6-A R6-C1: the test was previously written against the DENSE
    ``qwen3.5-4b-4bit`` alias, which we've now flipped to non-hybrid (it
    was the metal::malloc wedge surface). The CLI column contract is the
    same — only the surface alias changes — so pivot to the A3B MoE
    Qwen3.5 variant which still legitimately wears the hybrid marker.

    The point of the column is to spare users an `info` round-trip when
    deciding whether spec-decode/suffix-decode will help. Trust the gate.
    """
    out = _capture_models_output()
    profiles = list_profiles()
    qwen35_moe = profiles.get("qwen3.5-35b-4bit")
    assert qwen35_moe is not None, "qwen3.5-35b-4bit alias missing — fixture drift"
    assert qwen35_moe.is_hybrid, (
        "qwen3.5-35b-4bit (A3B MoE) should remain is_hybrid=True"
    )

    matches = [line for line in out.splitlines() if "qwen3.5-35b-4bit " in line]
    assert matches, "no row found for qwen3.5-35b-4bit"
    row = matches[0]
    assert "✗ hybrid" in row, f"expected '✗ hybrid' marker in row: {row!r}"
    assert "n/a" in row, f"expected suffix tier 'n/a' in row: {row!r}"


def test_models_command_renders_parser_for_hermes3_8b():
    """Non-hybrid model with parser + benched tier renders both columns.

    Two contracts: (1) the tool parser cell shows the value from the
    alias registry, (2) the suffix-tier cell shows the tier currently
    recorded in ``aliases.json``. Reading the expected tier from the
    registry (not hardcoding it) means a future bench re-sweep that
    reclassifies hermes3-8b-4bit doesn't break this test, while a *display*
    regression (tier dropped from the row entirely) still does.
    """
    out = _capture_models_output()
    matches = [line for line in out.splitlines() if "hermes3-8b-4bit " in line]
    assert matches, "no row found for hermes3-8b-4bit"
    row = matches[0]
    profile = list_profiles().get("hermes3-8b-4bit")
    assert profile is not None, "hermes3-8b-4bit alias missing — fixture drift"
    assert (profile.tool_call_parser or "") in row, (
        f"expected tool parser {profile.tool_call_parser!r} in row: {row!r}"
    )
    assert profile.suffix_decoding_tier in row, (
        f"expected suffix tier {profile.suffix_decoding_tier!r} in row: {row!r}"
    )


def test_models_command_renders_em_dash_for_unset_parsers():
    """An alias without tool_call_parser / reasoning_parser should show '—'."""
    out = _capture_models_output()
    profiles = list_profiles()
    # Find any alias that has neither parser populated.
    candidates = [
        a
        for a, p in profiles.items()
        if not p.tool_call_parser and not p.reasoning_parser
    ]
    if not candidates:
        # Schema may have tightened — skip cleanly.
        import pytest

        pytest.skip("no aliases without parsers (schema may have tightened)")
    alias = candidates[0]
    matches = [line for line in out.splitlines() if f"{alias} " in line]
    assert matches, f"no row found for {alias}"
    row = matches[0]
    # Both placeholders should appear.
    assert row.count("—") >= 2, f"expected two em-dashes in row: {row!r}"


def test_models_command_mentions_chat_pull_serve_in_tip():
    """The footer should advertise the four canonical actions."""
    out = _capture_models_output()
    for cmd in ("info", "pull", "chat", "serve"):
        assert f"rapid-mlx {cmd}" in out, (
            f"footer tip missing 'rapid-mlx {cmd}' suggestion"
        )


def test_models_command_subparser_smoke():
    """`rapid-mlx models --help` exits 0 (subparser still wired)."""
    import pytest

    with (
        patch.object(sys, "argv", ["rapid-mlx", "models", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


# ----------------------------------------------------------------------
# D2 — --cached / ls view
# ----------------------------------------------------------------------


def test_ls_subcommand_registered():
    """``rapid-mlx ls --help`` exits 0 (top-level alias is wired)."""
    import pytest

    with (
        patch.object(sys, "argv", ["rapid-mlx", "ls", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


def test_ls_routes_to_models_with_cached(monkeypatch):
    """``rapid-mlx ls`` invokes the cached view via ``models_command``
    with ``args.cached = True``. We patch ``models_command`` to capture
    the args namespace before the body runs."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "ls"]),
        patch.object(cli, "models_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    assert captured[0].cached is True


def test_models_cached_flag_routes_to_cached_view(monkeypatch, capsys):
    """``models --cached`` must call the cached-view path (not the
    full alias table). We assert via the printed header difference."""
    # No cached models will be found in a fresh tmp HF cache.
    monkeypatch.setenv("HF_HOME", "/nonexistent_path_for_this_test_xyz")
    # huggingface_hub.constants.HF_HUB_CACHE is evaluated at module load,
    # so we instead patch the helper directly.
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])

    with (
        patch.object(sys, "argv", ["rapid-mlx", "models", "--cached"]),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.main()
    out = capsys.readouterr().out
    assert "No models cached yet" in out
    # And the full alias table header is absent.
    assert "Available models" not in out


def test_models_default_view_unchanged(monkeypatch, capsys):
    """Bare ``rapid-mlx models`` still prints the capability table —
    --cached is opt-in. Backward-compat contract."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "models"]),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.main()
    out = capsys.readouterr().out
    assert "Available models" in out
    assert "Tools" in out and "Reasoning" in out


def test_cached_view_renders_alias_for_known_repo(tmp_path, monkeypatch, capsys):
    """A cached HF repo whose path matches an alias should render under
    the alias name (e.g. ``qwen3.5-4b-4bit``), not the raw HF path."""
    from vllm_mlx.model_aliases import list_profiles

    profiles = list_profiles()
    # Pick any alias for the test; we'll synthesize a fake cache entry
    # at its hf_path.
    alias = next(iter(profiles))
    hf_path = profiles[alias].hf_path

    monkeypatch.setattr(
        cli, "_scan_hf_cache_models", lambda: [(hf_path, 1024 * 1024 * 100, 0.0)]
    )
    cli._print_cached_models()
    out = capsys.readouterr().out
    assert alias in out, f"expected alias {alias!r} in cached view"
    assert hf_path[:40] in out, "expected HF path in cached view"


def test_cached_view_renders_unmapped_for_unknown_repo(monkeypatch, capsys):
    """A cached HF repo with no alias entry must show ``(unmapped)``."""
    monkeypatch.setattr(
        cli,
        "_scan_hf_cache_models",
        lambda: [("some/totally-unmapped-repo", 1024, 0.0)],
    )
    cli._print_cached_models()
    out = capsys.readouterr().out
    assert "(unmapped)" in out
    assert "totally-unmapped-repo" in out


def test_format_bytes_unit_selection():
    """``_format_bytes`` picks the largest unit where value >= 1.

    Suffixes are IEC base-1024 (KiB/MiB/GiB) — aligned with
    ``_format_size`` in ``vllm_mlx._download_gate`` so the same byte
    count is rendered identically by ``ls --cached`` and the B2 prompt
    (DeepSeek round-3 NIT #4)."""
    assert cli._format_bytes(0) == "0 B"
    assert cli._format_bytes(512) == "512 B"
    assert cli._format_bytes(2048) == "2.0 KiB"
    assert cli._format_bytes(5 * 1024 * 1024) == "5.0 MiB"
    assert cli._format_bytes(int(2.5 * 1024**3)) == "2.5 GiB"


def test_scan_hf_cache_models_filters_to_models_only(tmp_path, monkeypatch):
    """Only ``models--*`` directories should show in the listing — not
    ``datasets--*`` or ``spaces--*``."""
    cache_root = tmp_path / "hub"
    cache_root.mkdir()
    (cache_root / "models--mlx-community--FakeModel").mkdir()
    (cache_root / "models--mlx-community--FakeModel" / "blob1").write_bytes(b"x" * 128)
    (cache_root / "datasets--squad").mkdir()
    (cache_root / "datasets--squad" / "data").write_bytes(b"y" * 999)
    (cache_root / "spaces--gradio--demo").mkdir()

    # Patch the constants lookup inside _scan_hf_cache_models.
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(cache_root), raising=False
    )
    rows = cli._scan_hf_cache_models()
    repos = [r[0] for r in rows]
    assert "mlx-community/FakeModel" in repos
    assert all("squad" not in r for r in repos)
    assert all("demo" not in r for r in repos)
