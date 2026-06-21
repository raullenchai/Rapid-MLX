# SPDX-License-Identifier: Apache-2.0
"""H-08 + H-09 regression tests — embeddings flag/route guards.

Two bugs, shared embedding surface, one PR:

* **H-08**: ``--embedding-model`` was advertised in the CLI help but
  ``mlx_embeddings`` lives behind the ``[embeddings]`` extra, so the
  base install crashed with a raw ``ModuleNotFoundError`` traceback as
  soon as a user passed the flag. Probe at flag-parse time, exit 2
  with an install hint on stderr.
* **H-09**: ``/v1/embeddings`` without ``--embedding-model`` silently
  loaded the *chat* model name through ``mlx_embeddings.load()`` and
  returned its pooled hidden states as if they were real embeddings —
  silent-wrong vectors that get stuffed into vector stores. Route
  entry now 400s with the canonical envelope + the same install hint.

The probe is a lazy ``import mlx_embeddings`` inside
``vllm_mlx.embedding.mlx_embeddings_available`` — the base install must
never see a top-level import of ``mlx_embeddings`` (that's the bug we
were trying to avoid). Pin that here too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# H-08 — CLI probe
# ---------------------------------------------------------------------------


class TestEmbeddingsExtraProbe:
    """``require_mlx_embeddings_or_exit`` is the systemic fix for H-08.

    When ``mlx_embeddings`` isn't importable, the CLI ``serve`` path
    must exit 2 with the install hint on stderr instead of letting the
    ``ModuleNotFoundError`` bubble out of the engine load path.
    """

    def test_probe_returns_true_when_installed(self):
        """Sanity: when the extra IS installed (the CI default for the
        embeddings suite), the probe returns True and the CLI is free
        to load. Skip cleanly if the extra isn't installed in this
        environment so the test suite stays portable."""
        pytest.importorskip("mlx_embeddings")
        from vllm_mlx.embedding import mlx_embeddings_available

        assert mlx_embeddings_available() is True

    def test_probe_returns_false_when_missing(self, monkeypatch):
        """When ``mlx_embeddings`` is not importable the probe returns
        False — the caller then surfaces the install hint. Simulated
        by poisoning ``sys.modules`` with a sentinel that raises on
        attribute access, which is how ``importlib`` reports an
        unimportable module without us having to actually uninstall
        the package in CI."""

        # Drop any cached module and forbid re-import.
        monkeypatch.delitem(sys.modules, "mlx_embeddings", raising=False)

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def _fake_import(name, *args, **kwargs):
            if name == "mlx_embeddings" or name.startswith("mlx_embeddings."):
                raise ImportError("simulated: extras not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)
        from vllm_mlx.embedding import mlx_embeddings_available

        assert mlx_embeddings_available() is False

    def test_require_or_exit_bails_with_install_hint(self, monkeypatch, capsys):
        """The CLI helper prints an actionable install hint to stderr
        and exits 2 — argparse's conventional usage-error code. The
        message must name the ``[embeddings]`` extra and the
        ``rapid-mlx`` install command verbatim so the user can copy-
        paste the fix."""
        monkeypatch.delitem(sys.modules, "mlx_embeddings", raising=False)

        real_import = __import__

        def _fake_import(name, *args, **kwargs):
            if name == "mlx_embeddings" or name.startswith("mlx_embeddings."):
                raise ImportError("simulated: extras not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)
        from vllm_mlx.embedding import require_mlx_embeddings_or_exit

        with pytest.raises(SystemExit) as exc:
            require_mlx_embeddings_or_exit()
        assert exc.value.code == 2

        err = capsys.readouterr().err
        assert "--embedding-model" in err
        assert "[embeddings]" in err
        assert "pip install 'rapid-mlx[embeddings]'" in err

    def test_require_or_exit_noop_when_installed(self):
        """Sanity: when the extra IS installed the CLI helper returns
        silently — no stderr, no exit. Pinned so a future refactor that
        accidentally turns the probe into ``not is_available`` is
        caught immediately."""
        pytest.importorskip("mlx_embeddings")
        from vllm_mlx.embedding import require_mlx_embeddings_or_exit

        # Returns None without raising.
        assert require_mlx_embeddings_or_exit() is None

    def test_mlx_embeddings_not_imported_at_module_top_level(self):
        """The whole point of H-08: ``mlx_embeddings`` must NOT be
        imported at module top level by any rapid-mlx source file.
        Source-grep the package — if a future refactor moves a
        top-level ``import mlx_embeddings`` into ``embedding.py`` (or
        anywhere else), the base install starts crashing on import.

        Allow ``import mlx_embeddings`` ONLY when it appears indented
        (inside a function / method) — that's the lazy form. Top-level
        bare ``import mlx_embeddings`` / ``from mlx_embeddings import``
        is the failure mode."""
        pkg_root = Path(__file__).resolve().parents[1] / "vllm_mlx"
        offenders = []
        for path in pkg_root.rglob("*.py"):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.lstrip()
                if stripped != line:
                    # Indented — inside a function/method, that's the
                    # lazy-import form we want.
                    continue
                if (
                    stripped.startswith("import mlx_embeddings")
                    or stripped.startswith("from mlx_embeddings")
                ):
                    offenders.append(f"{path.relative_to(pkg_root)}:{lineno}: {line}")
        assert not offenders, (
            "Top-level ``import mlx_embeddings`` found — H-08 regression: "
            "the base install (no [embeddings] extra) will crash on import. "
            "Move the import inside the function that needs it.\n"
            + "\n".join(offenders)
        )


# ---------------------------------------------------------------------------
# H-09 — /v1/embeddings route guard
# ---------------------------------------------------------------------------


def _build_embed_app(monkeypatch, engine, *, embedding_model_locked):
    """Mount the embeddings router with a stubbed engine and the
    chosen lock state. Mirrors the helper in
    ``tests/test_embeddings_timeout_admission.py`` but exposes the
    lock as a parameter so each test can pick the success vs guard
    path explicitly. Installs the OpenAI-shaped exception handlers
    so the 400 envelope matches the production wire shape (the same
    wrappers ``server.app`` mounts)."""
    from vllm_mlx.config import get_config
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import embeddings as emb_route

    app = FastAPI()
    app.include_router(emb_route.router)
    install_exception_handlers(app)

    cfg = get_config()
    saved = {
        "embedding_engine": cfg.embedding_engine,
        "embedding_model_locked": cfg.embedding_model_locked,
        "api_key": cfg.api_key,
    }
    cfg.embedding_engine = engine
    cfg.embedding_model_locked = embedding_model_locked
    cfg.api_key = None

    # Also override the server globals so the route's "bridge" branch
    # (cfg→server fallback) doesn't reset our locked value.
    import vllm_mlx.server as srv

    saved_srv = {
        "_embedding_engine": srv._embedding_engine,
        "_embedding_model_locked": srv._embedding_model_locked,
    }
    srv._embedding_engine = engine
    srv._embedding_model_locked = embedding_model_locked

    monkeypatch.setattr(
        "vllm_mlx.server.load_embedding_model",
        lambda *_a, **_kw: None,
        raising=False,
    )

    def _restore():
        for k, v in saved.items():
            setattr(cfg, k, v)
        for k, v in saved_srv.items():
            setattr(srv, k, v)

    return TestClient(app), _restore


class TestEmbeddingsRouteGuard:
    """H-09: ``POST /v1/embeddings`` must 400 when no embedding model
    is configured. Previously the route silently re-used the chat
    model as if it were an embedding model — vectors shaped right
    (1024-dim for Qwen3-0.6B) but semantically meaningless."""

    def test_unconfigured_server_returns_400(self, monkeypatch):
        engine = MagicMock()
        client, restore = _build_embed_app(
            monkeypatch, engine, embedding_model_locked=None
        )
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "qwen3-0.6b-8bit", "input": "hello"},
            )
        finally:
            restore()

        assert r.status_code == 400, r.text
        body = r.json()
        # Canonical OpenAI-shaped envelope.
        assert "error" in body
        msg = body["error"]["message"]
        assert "embeddings model not configured" in msg
        # Install hint is the actionable bit — must be in the envelope.
        assert "pip install 'rapid-mlx[embeddings]'" in msg
        # The engine must NOT have been touched — guard fires before
        # any model load (no silent chat-model fallback).
        engine.embed.assert_not_called()
        engine.embed_tokens.assert_not_called()

    def test_configured_server_returns_200(self, monkeypatch):
        """Sanity: when an embedding model IS configured, the route
        returns 200 with the engine's vectors. Smoke test only — value
        equality is exercised by the dedicated engine tests."""
        engine = MagicMock()
        engine.model_name = "stub-embed"
        engine.embed.return_value = [[0.5, 0.5, 0.5, 0.5]]
        engine.count_tokens.return_value = 3
        client, restore = _build_embed_app(
            monkeypatch, engine, embedding_model_locked="stub-embed"
        )
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "stub-embed", "input": "hello"},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["data"][0]["embedding"] == [0.5, 0.5, 0.5, 0.5]
        engine.embed.assert_called_once()


# ---------------------------------------------------------------------------
# H-09 — /v1/models capability advertisement
# ---------------------------------------------------------------------------


class TestModelsListEmbeddingCapability:
    """``/v1/models`` must not lie about which entries are embedding-
    capable. When no embedding model is configured, NO entry carries
    ``"embedding"`` in ``capabilities`` — the route guard already
    400s ``/v1/embeddings``, so advertising the chat model as
    embedding-capable would mislead the desktop client into thinking
    the path works."""

    def _mount_models_app(self, monkeypatch, *, embedding_model_locked):
        from vllm_mlx.config import get_config
        from vllm_mlx.routes import models as models_route

        app = FastAPI()
        app.include_router(models_route.router)

        cfg = get_config()
        saved = {
            k: getattr(cfg, k, None)
            for k in (
                "model_name",
                "model_alias",
                "model_registry",
                "embedding_model_locked",
                "api_key",
            )
        }
        cfg.model_name = "mlx-community/Qwen3-0.6B-8bit"
        cfg.model_alias = "qwen3-0.6b-8bit"
        cfg.model_registry = None
        cfg.embedding_model_locked = embedding_model_locked
        cfg.api_key = None

        import vllm_mlx.server as srv

        saved_srv = {"_embedding_model_locked": srv._embedding_model_locked}
        srv._embedding_model_locked = embedding_model_locked

        def _restore():
            for k, v in saved.items():
                setattr(cfg, k, v)
            for k, v in saved_srv.items():
                setattr(srv, k, v)

        return TestClient(app), _restore

    def test_no_embedding_model_no_capability_tag(self, monkeypatch):
        """When no embedding model is configured, no entry in
        ``/v1/models`` carries ``"embedding"`` in its capabilities.
        Don't lie."""
        client, restore = self._mount_models_app(
            monkeypatch, embedding_model_locked=None
        )
        try:
            r = client.get("/v1/models")
        finally:
            restore()
        assert r.status_code == 200
        body = r.json()
        for entry in body["data"]:
            caps = entry.get("capabilities", [])
            assert "embedding" not in caps, (
                f"Model {entry['id']} advertises embedding capability "
                "but no embedding model is configured. /v1/embeddings would 400."
            )

    def test_configured_embedding_model_carries_capability(self, monkeypatch):
        """When an embedding model IS configured, exactly that id
        carries the ``"embedding"`` capability tag — and it appears in
        the listing alongside the chat model."""
        embed_id = "mlx-community/all-MiniLM-L6-v2-4bit"
        client, restore = self._mount_models_app(
            monkeypatch, embedding_model_locked=embed_id
        )
        try:
            r = client.get("/v1/models")
        finally:
            restore()
        assert r.status_code == 200
        body = r.json()

        ids = [e["id"] for e in body["data"]]
        assert embed_id in ids, (
            f"Configured embedding model {embed_id} missing from /v1/models"
        )

        for entry in body["data"]:
            caps = entry.get("capabilities", [])
            if entry["id"] == embed_id:
                assert "embedding" in caps
            else:
                assert "embedding" not in caps

    def test_retrieve_embedding_model_by_id(self, monkeypatch):
        """Per-id retrieval also surfaces the configured embedding
        model — desktop hydrates per-model state from this path.
        Uses a slash-free id so FastAPI's path matcher accepts the
        ``/v1/models/{id}`` route as-is (HF paths with slashes work in
        production behind a URL-encoder; the routing semantics under
        test here are about lookup, not about path serialization)."""
        embed_id = "all-minilm-embed"
        client, restore = self._mount_models_app(
            monkeypatch, embedding_model_locked=embed_id
        )
        try:
            r = client.get(f"/v1/models/{embed_id}")
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"] == embed_id
        assert "embedding" in body.get("capabilities", [])
