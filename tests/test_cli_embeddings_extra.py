# SPDX-License-Identifier: Apache-2.0
"""R11-G / H-08 — ``rapid-mlx serve --embedding-model`` install guard.

H-08 (Bo r11 carry from R8-H3): the ``--embedding-model`` flag was
advertised in ``rapid-mlx serve --help`` but ``mlx_embeddings`` lives
behind the ``[embeddings]`` extra. A user on a clean PyPI install ran
``rapid-mlx serve <chat> --embedding-model X`` and got an opaque
``ModuleNotFoundError: mlx_embeddings`` traceback deep inside
``EmbeddingEngine.load``. The fix probes for the extra at flag-parse
time and exits cleanly with an actionable install hint on stderr.

This file pins the helper that owns the CLI side of the guard
(``_load_embedding_model_or_exit`` in :mod:`vllm_mlx.cli`) — it MUST
short-circuit via ``require_mlx_embeddings_or_exit`` BEFORE touching
the alias registry or the loader. The broader H-08+H-09+H-13 net
lives in :mod:`tests.test_embeddings_extra_guard`; this file exists
because the task spec named ``tests/test_cli_embeddings_extra.py``
explicitly and ``vllm_mlx/cli.py`` is the natural grep destination
for the CLI surface.

Coordinate with r11-K (deferred #258): if/when audio-mode serve
learns to honour ``--embedding-model``, the test added in this file
becomes the parity pin for the audio-mode boot path too. The
``_load_embedding_model_or_exit`` helper is the single source of
truth — audio-mode integration should route through it rather than
duplicate the guard logic. See the helper's docstring in
``vllm_mlx/cli.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_load_embedding_helper_exits_2_when_extra_missing(monkeypatch, capsys):
    """``_load_embedding_model_or_exit`` MUST short-circuit via
    ``require_mlx_embeddings_or_exit`` with ``sys.exit(2)`` and the
    install hint on stderr when ``mlx_embeddings`` isn't importable.

    The loader callable MUST NOT be reached — pre-fix the helper
    called ``load_fn`` first and the ``ModuleNotFoundError`` came out
    of ``mlx_embeddings.load`` deep in the trace. The probe is the
    structural fix.
    """
    from vllm_mlx.cli import _load_embedding_model_or_exit

    # Force the probe to report mlx_embeddings missing — this is the
    # bug condition. ``mlx_embeddings_available`` is the lazy probe
    # that the helper consults.
    monkeypatch.setattr("vllm_mlx.embedding.mlx_embeddings_available", lambda: False)

    def _fake_loader(*_args, **_kwargs):  # pragma: no cover — must not run
        raise AssertionError(
            "loader was invoked despite missing [embeddings] extra — "
            "the H-08 short-circuit is gone"
        )

    args = SimpleNamespace(embedding_model="mlx-community/Qwen3-Embedding-0.6B-4bit")
    with pytest.raises(SystemExit) as exc_info:
        _load_embedding_model_or_exit(args, _fake_loader)

    # Exit code 2 — argparse's conventional usage-error code. NOT 1
    # (that's reserved for loader failures further down the helper).
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    # The hint must name the flag, the extra group, and the canonical
    # pip-install command so the user can copy-paste the fix.
    assert "--embedding-model" in err
    assert "[embeddings]" in err
    assert "pip install 'rapid-mlx[embeddings]'" in err


def test_load_embedding_helper_proceeds_when_extra_installed(monkeypatch):
    """When the extra IS installed the probe returns True and the
    helper reaches the loader (the success-path smoke check).

    Pin the call shape too: ``load_fn`` is invoked with the resolved
    embedding-model name as a positional arg and ``lock=True`` kw —
    the boot path always locks the engine so the H-09 route guard
    has a non-None ``embedding_model_locked`` to consult.
    """
    from vllm_mlx.cli import _load_embedding_model_or_exit

    monkeypatch.setattr("vllm_mlx.embedding.mlx_embeddings_available", lambda: True)

    captured: dict = {}

    def _fake_loader(name, *, lock):
        captured["name"] = name
        captured["lock"] = lock

    args = SimpleNamespace(embedding_model="mlx-community/some-embed-model")
    _load_embedding_model_or_exit(args, _fake_loader)

    assert captured == {
        "name": "mlx-community/some-embed-model",
        "lock": True,
    }


def test_install_hint_string_is_canonical():
    """``EMBEDDINGS_EXTRA_INSTALL_HINT`` is the canonical install
    string — both the CLI probe (H-08) AND the route guard (H-09)
    consume it. Pin the exact string so a future refactor that
    accidentally drops the single-quotes (``rapid-mlx[embeddings]``
    bare would be parsed as a shell glob in zsh) is caught.
    """
    from vllm_mlx.embedding import EMBEDDINGS_EXTRA_INSTALL_HINT

    assert EMBEDDINGS_EXTRA_INSTALL_HINT == (
        "Install with: pip install 'rapid-mlx[embeddings]'"
    )


def test_pyproject_declares_embeddings_extra():
    """R8-H3 closure: ``[embeddings]`` MUST be declared as an optional
    dep group in ``pyproject.toml`` (mirrors ``[audio]`` /
    ``[vision]``). Without this, the install hint above would point
    at a non-existent extra and the user could never recover.
    """
    import sys
    from pathlib import Path

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover — 3.10 floor
        import tomli as tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    extras = data["project"]["optional-dependencies"]

    assert "embeddings" in extras, (
        "pyproject.toml is missing the [embeddings] optional-deps group. "
        "The H-08 install hint points at it — without the group declared, "
        "the hint is a dead-end."
    )
    deps = extras["embeddings"]
    assert any(d.startswith("mlx-embeddings") for d in deps), (
        f"[embeddings] extra must include mlx-embeddings; got {deps!r}"
    )


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
