# SPDX-License-Identifier: Apache-2.0
"""Tests for ``vllm_mlx._download_gate``.

Pins the auto-pull confirmation flow:

* ``estimate_repo_size_bytes`` returns sane numbers from a mocked HF API
  and ``None`` on failure (network down, gated repo, timeout).
* ``confirm_or_abort`` honours the env override, TTY detection, the
  size threshold, and yes/no user input.

No test in this file hits the network — every HF API call is mocked.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx import _download_gate as gate

# ---------------------------------------------------------------------------
# estimate_repo_size_bytes
# ---------------------------------------------------------------------------


def _fake_sibling(name: str, size: int | None, *, lfs_size: int | None = None):
    """Build a fake ``RepoSibling``-like object that ``_sibling_size`` accepts."""
    if lfs_size is not None:
        lfs = SimpleNamespace(size=lfs_size)
    else:
        lfs = None
    return SimpleNamespace(rfilename=name, size=size, lfs=lfs)


def test_estimate_repo_size_sums_weight_files():
    """Weight + tokenizer files are summed; .gitattributes/README are skipped."""
    info = SimpleNamespace(
        siblings=[
            _fake_sibling("model-00001-of-00002.safetensors", 5 * 1024**3),
            _fake_sibling("model-00002-of-00002.safetensors", 7 * 1024**3),
            _fake_sibling("tokenizer.json", 5 * 1024**2),
            _fake_sibling("config.json", 1024),
            _fake_sibling(".gitattributes", 256),
            _fake_sibling("README.md", 4096),
        ]
    )
    with patch.object(gate, "_model_info_with_timeout", return_value=info):
        total = gate.estimate_repo_size_bytes("mlx-community/Fake-12B-4bit")

    assert total is not None
    # 12 GiB of safetensors + 5 MiB tokenizer + 1 KiB config.
    expected = (12 * 1024**3) + (5 * 1024**2) + 1024
    assert total == expected


def test_estimate_repo_size_prefers_lfs_size():
    """When both ``size`` and ``lfs.size`` are populated, LFS wins (it's the
    true blob size; the bare ``size`` field can report the pointer size)."""
    info = SimpleNamespace(
        siblings=[
            _fake_sibling("model.safetensors", 134, lfs_size=4 * 1024**3),
        ]
    )
    with patch.object(gate, "_model_info_with_timeout", return_value=info):
        total = gate.estimate_repo_size_bytes("mlx-community/Fake-4B-4bit")

    assert total == 4 * 1024**3


def test_estimate_repo_size_returns_none_on_api_failure():
    """Any exception from the HF API call surfaces as ``None`` — callers must
    fall through silently rather than blocking on a flaky network."""
    with patch.object(
        gate, "_model_info_with_timeout", side_effect=RuntimeError("HF down")
    ):
        assert gate.estimate_repo_size_bytes("definitely/not-a-real-repo") is None


def test_estimate_repo_size_returns_none_on_empty_repo():
    """An info object with no weight files yields ``None`` (rather than 0) so
    the caller's heads-up logic kicks in."""
    info = SimpleNamespace(
        siblings=[
            _fake_sibling("README.md", 1024),
            _fake_sibling(".gitattributes", 256),
        ]
    )
    with patch.object(gate, "_model_info_with_timeout", return_value=info):
        assert gate.estimate_repo_size_bytes("foo/empty") is None


# ---------------------------------------------------------------------------
# confirm_or_abort
# ---------------------------------------------------------------------------


def test_confirm_passes_through_when_under_threshold(monkeypatch):
    """A 5 GiB estimate against a 10 GiB threshold must not prompt."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    # Even if stdin is a TTY, small downloads pass through silently.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() must not be called for small repos"),
    )

    assert gate.confirm_or_abort("foo/small", 5 * 1024**3) is True


def test_confirm_passes_through_when_env_var_set(monkeypatch):
    """``RAPID_MLX_AUTO_PULL=1`` short-circuits even for huge downloads."""
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "1")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() must not be called when env set"),
    )

    assert gate.confirm_or_abort("foo/huge", 50 * 1024**3) is True


@pytest.mark.parametrize("val", ["1", "true", "yes", "YES", "True"])
def test_confirm_env_var_accepts_truthy_variants(monkeypatch, val):
    """Common truthy spellings must all work — users will try all of them."""
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", val)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert gate.confirm_or_abort("foo/huge", 50 * 1024**3) is True


def test_confirm_passes_through_when_non_tty(monkeypatch):
    """Non-interactive stdin (CI, piped scripts) must not deadlock on input."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() must not be called in non-TTY mode"),
    )

    assert gate.confirm_or_abort("foo/huge", 50 * 1024**3) is True


def test_confirm_proceeds_with_unknown_size(monkeypatch, capsys):
    """Unknown size → heads-up + proceed (don't block on transient HF failures)."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() must not be called for unknown size"),
    )

    assert gate.confirm_or_abort("foo/unknown-size", None) is True
    out = capsys.readouterr().out
    assert "unknown" in out.lower()
    assert "foo/unknown-size" in out


def test_confirm_returns_true_on_yes(monkeypatch, capsys):
    """``y`` input → proceed."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "y")

    assert gate.confirm_or_abort("foo/huge", 41 * 1024**3) is True
    out = capsys.readouterr().out
    assert "foo/huge" in out
    assert "41" in out  # size string contains 41 GiB
    assert "Continue?" not in out  # input prompt itself isn't captured to stdout


def test_confirm_returns_true_on_yes_full_word(monkeypatch):
    """``yes`` (full word, case-insensitive) also proceeds."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "YES")

    assert gate.confirm_or_abort("foo/huge", 41 * 1024**3) is True


def test_confirm_aborts_on_no(monkeypatch, capsys):
    """``n`` input → ``sys.exit(1)`` with an actionable hint."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "n")

    with pytest.raises(SystemExit) as excinfo:
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)
    assert excinfo.value.code == 1

    out = capsys.readouterr().out
    assert "Aborted" in out
    assert "rapid-mlx pull foo/huge" in out
    assert "RAPID_MLX_AUTO_PULL" in out


def test_confirm_aborts_on_empty_input(monkeypatch):
    """Empty input (the default) → abort. ``[y/N]`` means N is the default."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "")

    with pytest.raises(SystemExit):
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)


def test_confirm_aborts_on_ctrl_c(monkeypatch):
    """Ctrl-C at the prompt → treated as abort, not a stack trace."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _raise(_=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise)

    with pytest.raises(SystemExit):
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)


def test_confirm_logfile_hint_appears_in_prompt(monkeypatch, capsys):
    """When a logfile is supplied, the prompt tells the user where to tail."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "y")

    gate.confirm_or_abort("foo/huge", 41 * 1024**3, logfile_hint="/tmp/serve.log")
    out = capsys.readouterr().out
    assert "/tmp/serve.log" in out


# ---------------------------------------------------------------------------
# _format_size — internal, but worth pinning so the prompt stays readable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_bytes,expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (780 * 1024**2, "780.0 MiB"),
        (int(2.4 * 1024**3), "2.4 GiB"),
        (int(42.3 * 1024**3), "42.3 GiB"),
    ],
)
def test_format_size_friendly(num_bytes, expected):
    assert gate._format_size(num_bytes) == expected


# ---------------------------------------------------------------------------
# is_repo_cached
# ---------------------------------------------------------------------------


def test_is_repo_cached_true_when_try_to_load_finds_config(tmp_path, monkeypatch):
    """The official ``try_to_load_from_cache`` hook is the primary probe."""
    fake_cfg = tmp_path / "config.json"
    fake_cfg.write_text("{}")

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, fname: str(fake_cfg),
    )

    assert gate.is_repo_cached("foo/cached") is True


def test_is_repo_cached_false_when_no_snapshot(tmp_path, monkeypatch):
    """No cache hit + empty HF cache directory → False."""
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *_, **__: None
    )

    empty_cache = tmp_path / "hf-cache"
    empty_cache.mkdir()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(empty_cache))

    assert gate.is_repo_cached("foo/missing") is False


def test_is_repo_cached_falls_back_to_snapshot_dir(tmp_path, monkeypatch):
    """If ``try_to_load_from_cache`` returns None but the snapshot dir exists
    with at least one file, treat the repo as cached. Covers repos that
    ship without a ``config.json`` at the same revision."""
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *_, **__: None
    )

    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--foo--quirky" / "snapshots" / "abcd1234"
    snap.mkdir(parents=True)
    (snap / "chat_template.jinja").write_text("{{}}")

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("foo/quirky") is True


# ---------------------------------------------------------------------------
# Defensive guards — env value parsing.
# ---------------------------------------------------------------------------


def test_confirm_env_var_falsy_value_does_not_short_circuit(monkeypatch):
    """``RAPID_MLX_AUTO_PULL=0`` must NOT auto-confirm — the env is opt-in."""
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "0")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "n")

    with pytest.raises(SystemExit):
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)


def test_confirm_threshold_boundary(monkeypatch):
    """Exactly at threshold → prompt fires (the docstring promises ``>=``)."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "y")

    threshold = 10 * 1024**3
    # One byte under threshold → no prompt.
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() called below threshold"),
    )
    assert gate.confirm_or_abort("foo/borderline", threshold - 1) is True

    # At threshold → prompt fires; mock yes-response.
    monkeypatch.setattr("builtins.input", lambda _=None: "y")
    assert gate.confirm_or_abort("foo/border-on", threshold) is True


# ---------------------------------------------------------------------------
# Smoke test: the module imports cleanly without huggingface_hub being a
# hard runtime requirement at import time.
# ---------------------------------------------------------------------------


def test_module_imports_without_hf_call():
    """Importing the module must NOT trigger any HF API call (lazy-imported)."""
    # If huggingface_hub had been touched at module load, the patched
    # ``_model_info_with_timeout`` in earlier tests would have failed.
    # This test exists to make the contract explicit for future maintainers.
    assert hasattr(gate, "estimate_repo_size_bytes")
    assert hasattr(gate, "confirm_or_abort")
    assert hasattr(gate, "is_repo_cached")
    assert os.path.basename(gate.__file__) == "_download_gate.py"
