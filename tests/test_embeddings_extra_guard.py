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
        by patching ``importlib.util.find_spec`` to report the
        top-level package as missing, which is exactly the signal the
        probe's new ``find_spec``-based implementation reads (codex
        nit closure: don't mask broken-installed-package errors
        behind the missing-extra hint)."""
        import importlib.util as _ilu

        real_find_spec = _ilu.find_spec

        def _fake_find_spec(name, *args, **kwargs):
            if name == "mlx_embeddings":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr("importlib.util.find_spec", _fake_find_spec)
        from vllm_mlx.embedding import mlx_embeddings_available

        assert mlx_embeddings_available() is False

    def test_require_or_exit_bails_with_install_hint(self, monkeypatch, capsys):
        """The CLI helper prints an actionable install hint to stderr
        and exits 2 — argparse's conventional usage-error code. The
        message must name the ``[embeddings]`` extra and the
        ``rapid-mlx`` install command verbatim so the user can copy-
        paste the fix."""
        import importlib.util as _ilu

        real_find_spec = _ilu.find_spec

        def _fake_find_spec(name, *args, **kwargs):
            if name == "mlx_embeddings":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr("importlib.util.find_spec", _fake_find_spec)
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

    def test_guard_fires_before_banner_in_cli_serve(self):
        """F-H08-INCOMPLETE follow-up: the ``[embeddings]`` extra check
        in ``cli.py::serve_command`` must run BEFORE the model-download
        prefetch and the startup banner. Pre-fix the probe lived deep
        in ``serve_command`` so the operator saw the alias-resolved log
        line, the "🐆 Rapid-MLX" banner, the feature list, AND the
        Model id BEFORE the error and ``sys.exit(2)`` — Diego reported
        this as a warning-and-fall-through because the banner masked
        the actual exit.

        Source-pin the ordering: in ``cli.py::serve_command``, the
        first ``require_mlx_embeddings_or_exit`` reference must appear
        BEFORE the first ``_ensure_model_downloaded`` reference AND
        BEFORE the first ``"🐆 Rapid-MLX"`` banner string.
        """
        cli_file = Path(__file__).resolve().parents[1] / "vllm_mlx" / "cli.py"
        source = cli_file.read_text()
        # Locate serve_command body — between ``def serve_command`` and
        # the next top-level ``def`` so we only scan the relevant function.
        start = source.index("def serve_command(")
        # Look for the next ``\ndef `` (top-level function) after start.
        end = source.find("\ndef ", start + 1)
        body = source[start : end if end != -1 else len(source)]

        # The guard must appear in serve_command at all. Look for the
        # actual CALL form ``require_mlx_embeddings_or_exit()`` so
        # docstring/comment mentions of the helper name don't confuse
        # the ordering check.
        idx_require = body.find("require_mlx_embeddings_or_exit()")
        assert idx_require != -1, (
            "serve_command does not call require_mlx_embeddings_or_exit() — "
            "the H-08 guard is gone. Restore the probe at the top of "
            "serve_command."
        )

        # Ordering vs the model download. Match the CALL form so a
        # comment that mentions ``_ensure_model_downloaded`` higher up
        # doesn't trip the ordering check.
        idx_download = body.find("_ensure_model_downloaded(")
        assert idx_download != -1, (
            "serve_command no longer calls _ensure_model_downloaded() — "
            "the fixture this test pins against has moved; update the "
            "test to match the new boot order."
        )
        assert idx_require < idx_download, (
            "F-H08-INCOMPLETE regression: require_mlx_embeddings_or_exit "
            "fires AFTER _ensure_model_downloaded in serve_command. The "
            "guard must run first so a cold-cache user doesn't pay a "
            "multi-minute download before the install-hint exit."
        )

        # Ordering vs the startup banner.
        idx_banner = body.find("🐆 Rapid-MLX")
        if idx_banner != -1:
            assert idx_require < idx_banner, (
                "F-H08-INCOMPLETE regression: the H-08 guard fires AFTER "
                "the '🐆 Rapid-MLX' startup banner — operators saw the "
                "banner + Features line + Model id before the error, which "
                "looked like a successful boot. Move the guard BEFORE "
                "the banner."
            )

    def test_guard_fires_before_banner_in_server_entrypoint(self):
        """Same invariant for the standalone ``python -m vllm_mlx.server``
        entrypoint. Pre-fix the probe lived after ``configure_logging``
        and the SECURITY CONFIGURATION header; new contract is that
        nothing prints between ``parse_args()`` and the guard."""
        server_file = Path(__file__).resolve().parents[1] / "vllm_mlx" / "server.py"
        source = server_file.read_text()

        # The standalone entrypoint's parse_args sits inside the same
        # function that prints the SECURITY CONFIGURATION banner.
        idx_parse = source.find("args = parser.parse_args()")
        assert idx_parse != -1
        # Confirm the function body actually contains the guard — look
        # for the CALL form so a docstring reference doesn't satisfy
        # the search.
        idx_require = source.find("require_mlx_embeddings_or_exit()", idx_parse)
        assert idx_require != -1, (
            "server.py main entrypoint no longer calls "
            "require_mlx_embeddings_or_exit after parse_args — H-08 "
            "regression."
        )
        idx_security_banner = source.find("SECURITY CONFIGURATION", idx_parse)
        # The first SECURITY CONFIGURATION line after parse_args must
        # come AFTER the guard — i.e. the guard fires before the banner.
        assert idx_security_banner != -1, "SECURITY banner line missing"
        assert idx_require < idx_security_banner, (
            "F-H08-INCOMPLETE regression in server.py: the embedding "
            "extras probe fires AFTER the SECURITY CONFIGURATION banner. "
            "Move it to immediately after parse_args()."
        )

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
                if stripped.startswith("import mlx_embeddings") or stripped.startswith(
                    "from mlx_embeddings"
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
    """H-09: ``POST /v1/embeddings`` must 503 when no embedding model
    is configured. Previously the route silently re-used the chat
    model as if it were an embedding model — vectors shaped right
    (1024-dim for Qwen3-0.6B) but semantically meaningless.

    R11-G: the status code was flipped 400 → 503 because a missing
    ``--embedding-model`` is a configuration / boot gap, not a bad
    client payload. LangChain / LlamaIndex retry policies recognise
    503 as "transient infra issue, back off and retry" — a 400 would
    look like a permanently-bad request and the client would surface
    a misleading user-side error. The envelope also gained the
    machine-readable ``code: "no_embedding_model"`` so downstream
    clients can branch on the code instead of substring-matching
    the message."""

    def test_unconfigured_server_returns_503(self, monkeypatch):
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

        assert r.status_code == 503, r.text
        body = r.json()
        # Canonical OpenAI-shaped envelope, plus the machine-readable
        # code so SDKs can branch without substring-matching the
        # message. Pin both the type AND the code — a future refactor
        # that drops either field would silently break the client
        # branching contract.
        assert "error" in body
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "no_embedding_model"
        msg = body["error"]["message"]
        # The actionable message names the CLI flag and the install
        # hint verbatim so the operator can copy-paste the fix.
        assert "No embedding model loaded" in msg
        assert "--embedding-model" in msg
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
        model — desktop hydrates per-model state from this path."""
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

    def test_retrieve_embedding_model_by_hf_path_id(self, monkeypatch):
        """Codex R1 BLOCKER closure: the configured embedding model is
        almost always a Hugging Face repo id containing ``/`` (e.g.
        ``mlx-community/all-MiniLM-L6-v2-4bit``). The
        ``/v1/models/{model_id:path}`` route must match the raw HF id
        without forcing clients to URL-encode the slash — every other
        rapid-mlx endpoint accepts the bare HF id. Pin the production
        URL shape callers actually use against the production lookup."""
        embed_id = "mlx-community/all-MiniLM-L6-v2-4bit"
        client, restore = self._mount_models_app(
            monkeypatch, embedding_model_locked=embed_id
        )
        try:
            # Raw HF id with slash — no URL-encoding. This is the
            # production wire shape (rapid-desktop and the curl
            # examples in the README both send it this way).
            r = client.get(f"/v1/models/{embed_id}")
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"] == embed_id
        assert "embedding" in body.get("capabilities", [])

    def test_list_models_includes_hf_path_embedding_id(self, monkeypatch):
        """Companion to the per-id test: with a slash-containing
        embedding model configured, the listing also surfaces it with
        the ``"embedding"`` capability tag. Catches the case where
        ``_append`` or the lookup walks a different code path for
        slash-containing ids than alias-shaped ids."""
        embed_id = "mlx-community/all-MiniLM-L6-v2-4bit"
        client, restore = self._mount_models_app(
            monkeypatch, embedding_model_locked=embed_id
        )
        try:
            r = client.get("/v1/models")
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        ids = [e["id"] for e in body["data"]]
        assert embed_id in ids, (
            f"Configured HF-path embedding id {embed_id} missing from /v1/models"
        )
        for entry in body["data"]:
            if entry["id"] == embed_id:
                assert "embedding" in entry.get("capabilities", [])


# ---------------------------------------------------------------------------
# D-EMBED-ALIAS — Sarah F-S2-1
# ---------------------------------------------------------------------------
#
# Sarah F-S2-1 (PyPI 0.8.3): ``rapid-mlx serve <chat-alias>
# --embedding-model embeddinggemma-300m-6bit`` crashed at startup with
# ``mlx_embeddings.utils.ModelNotFoundError: Model not found for path
# or HF repo: embeddinggemma-300m-6bit`` because the positional chat-
# model arg goes through ``resolve_model`` (alias → HF path) but the
# ``--embedding-model`` flag was passed verbatim to
# ``mlx_embeddings.load()``.
#
# Fix: route ``--embedding-model`` through the SAME ``resolve_model``
# call site as the chat positional, AND fail-fast with the standard
# "unknown alias" hint when the value is neither an alias hit nor an
# HF org/name path nor a local directory.


class TestEmbeddingModelAliasResolution:
    """D-EMBED-ALIAS: ``--embedding-model`` must resolve through
    ``resolve_model`` exactly like the positional chat-model arg, and
    surface a clean "unknown alias" error before any
    ``mlx_embeddings.load()`` call happens."""

    def test_resolve_model_round_trips_a_known_alias(self):
        """Codex r5 BLOCKING: this PR ships
        ``embeddinggemma-300m-6bit`` and ``embeddinggemma-300m-8bit``
        in ``aliases.json`` precisely so the CLI's ``resolve_model``
        path round-trips them. Pin the contract unconditionally — a
        future drop of either entry must turn this test red.
        """
        from vllm_mlx.model_aliases import resolve_model

        assert (
            resolve_model("embeddinggemma-300m-6bit")
            == "mlx-community/embeddinggemma-300m-6bit"
        )
        assert (
            resolve_model("embeddinggemma-300m-8bit")
            == "mlx-community/embeddinggemma-300m-8bit"
        )

    def test_resolve_model_passes_through_hf_path(self):
        """A full HF org/name path (``mlx-community/foo``) is already
        the canonical form — ``resolve_model`` must return it
        unchanged so the embedding-model path stays a no-op for
        callers who already pass the full id."""
        from vllm_mlx.model_aliases import resolve_model

        hf = "mlx-community/embeddinggemma-300m-6bit"
        assert resolve_model(hf) == hf

    def test_resolve_model_passes_through_unknown_name(self):
        """An unknown alias / bogus name returns unchanged so the
        downstream "not a known alias and not an HF path" branch can
        emit the actionable error. This is the contract the CLI's
        chat-model path already relies on (cli.py ~5660); the
        embedding-model path inherits the same shape after the fix."""
        from vllm_mlx.model_aliases import resolve_model

        assert resolve_model("bogus-name-no-such-alias") == "bogus-name-no-such-alias"

    def test_load_helper_resolves_alias_before_loader(self, monkeypatch):
        """``_load_embedding_model_or_exit`` MUST route the alias
        through ``resolve_model`` before calling the loader. Sarah
        F-S2-1's reproducer (``--embedding-model
        embeddinggemma-300m-6bit``) crashed because the alias hit
        the loader verbatim; the helper now resolves it to the HF
        path first."""
        from types import SimpleNamespace

        from vllm_mlx.cli import _load_embedding_model_or_exit

        # Pretend the [embeddings] extra is installed so the H-08
        # probe doesn't short-circuit before the alias step.
        monkeypatch.setattr("vllm_mlx.embedding.mlx_embeddings_available", lambda: True)
        captured: dict = {}

        def _fake_loader(name, *, lock):
            captured["name"] = name
            captured["lock"] = lock

        args = SimpleNamespace(embedding_model="embeddinggemma-300m-6bit")
        _load_embedding_model_or_exit(args, _fake_loader)
        # The loader must see the resolved HF path, not the alias.
        assert captured["name"] == "mlx-community/embeddinggemma-300m-6bit", captured
        assert captured["lock"] is True
        # And ``args.embedding_model`` is mutated to the resolved
        # form so downstream banner + config emits the canonical id.
        assert args.embedding_model == "mlx-community/embeddinggemma-300m-6bit"

    def test_load_helper_passes_hf_path_through_unchanged(self, monkeypatch):
        """A caller who already passed ``mlx-community/foo`` (the
        canonical HF id form) must reach the loader unchanged — no
        alias map, no path mutation."""
        from types import SimpleNamespace

        from vllm_mlx.cli import _load_embedding_model_or_exit

        monkeypatch.setattr("vllm_mlx.embedding.mlx_embeddings_available", lambda: True)
        captured: dict = {}

        def _fake_loader(name, *, lock):
            captured["name"] = name

        args = SimpleNamespace(embedding_model="mlx-community/some-embed-7b")
        _load_embedding_model_or_exit(args, _fake_loader)
        assert captured["name"] == "mlx-community/some-embed-7b"
        assert args.embedding_model == "mlx-community/some-embed-7b"

    def test_load_helper_wraps_model_not_found_with_hint(self, monkeypatch, capsys):
        """When the loader raises ``ModelNotFoundError`` (the Sarah
        F-S2-1 surface — mlx_embeddings can't find the repo), the
        helper translates to a clean ``sys.exit(1)`` with the
        actionable hint naming the alias registry + canonical HF id
        format. Codex r0 BLOCKING #3: we let the loader fail rather
        than preflight-rejecting bare names, then translate. Codex r1
        NIT: the wrap path binds to the CONCRETE exception classes
        (no ``"not found"`` substring match), so we exercise the real
        ``ModelNotFoundError`` if installed, else fall back to
        ``FileNotFoundError`` which is in the wrap-tuple too via the
        local-path branch of the loader."""
        from types import SimpleNamespace

        from vllm_mlx.cli import _load_embedding_model_or_exit

        monkeypatch.setattr("vllm_mlx.embedding.mlx_embeddings_available", lambda: True)

        try:
            from mlx_embeddings.utils import (
                ModelNotFoundError as _RealModelNotFoundError,
            )

            exc_cls: type[BaseException] = _RealModelNotFoundError
        except ImportError:
            exc_cls = FileNotFoundError

        def _fake_loader(name, *, lock):
            raise exc_cls(f"Model not found for path or HF repo: {name}.")

        args = SimpleNamespace(embedding_model="definitely-not-an-alias-xyz")
        with pytest.raises(SystemExit) as exc:
            _load_embedding_model_or_exit(args, _fake_loader)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        # The hint must name the alias + the canonical HF id form so
        # the user can copy-paste the fix.
        assert "definitely-not-an-alias-xyz" in out
        assert "embeddinggemma-300m-6bit" in out
        assert "mlx-community/" in out

    def test_load_helper_reraises_unrelated_errors(self, monkeypatch):
        """A loader failure that ISN'T a not-found shape (e.g. a
        corrupt safetensors mid-load, a Metal OOM, an unrelated
        ValueError) must propagate UNCHANGED so the operator sees the
        real trace. Codex r1 NIT: this includes ValueErrors whose
        message HAPPENS to contain ``"not found"`` — the previous
        substring match was too loose and would mis-translate a
        corrupt-model error as the alias/HF-id hint. The wrap now
        binds to concrete classes only."""
        from types import SimpleNamespace

        from vllm_mlx.cli import _load_embedding_model_or_exit

        monkeypatch.setattr("vllm_mlx.embedding.mlx_embeddings_available", lambda: True)

        class CorruptSafetensorsError(RuntimeError):
            pass

        def _fake_loader(name, *, lock):
            raise CorruptSafetensorsError("header mismatch on tensor block 3")

        args = SimpleNamespace(embedding_model="mlx-community/foo")
        with pytest.raises(CorruptSafetensorsError, match="header mismatch"):
            _load_embedding_model_or_exit(args, _fake_loader)

    def test_load_helper_reraises_value_error_with_not_found_substring(
        self, monkeypatch
    ):
        """Codex r1 NIT regression pin: a generic ``ValueError`` whose
        message contains the substring ``"not found"`` MUST propagate
        unchanged. The previous loose substring match would have
        mis-translated this as the alias/HF-id hint. After the codex
        r1 fix the wrap binds to concrete exception CLASSES only."""
        from types import SimpleNamespace

        from vllm_mlx.cli import _load_embedding_model_or_exit

        monkeypatch.setattr("vllm_mlx.embedding.mlx_embeddings_available", lambda: True)

        def _fake_loader(name, *, lock):
            raise ValueError("config field 'rope_theta' not found in tensor map")

        args = SimpleNamespace(embedding_model="mlx-community/foo")
        with pytest.raises(ValueError, match="rope_theta"):
            _load_embedding_model_or_exit(args, _fake_loader)

    def test_load_helper_exits_when_extra_missing(self, monkeypatch, capsys):
        """H-08 regression — when the ``[embeddings]`` extra isn't
        installed, the helper must short-circuit via
        ``require_mlx_embeddings_or_exit`` BEFORE touching the alias
        registry or the loader. ``sys.exit(2)`` with the install hint."""
        from types import SimpleNamespace

        from vllm_mlx.cli import _load_embedding_model_or_exit

        monkeypatch.setattr(
            "vllm_mlx.embedding.mlx_embeddings_available", lambda: False
        )

        def _fake_loader(*a, **kw):
            raise AssertionError("loader must not be reached if extra is missing")

        args = SimpleNamespace(embedding_model="embeddinggemma-300m-6bit")
        with pytest.raises(SystemExit) as exc:
            _load_embedding_model_or_exit(args, _fake_loader)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "[embeddings]" in err

    def test_server_module_routes_through_shared_helper(self):
        """``python -m vllm_mlx.server`` must use the SAME helper as
        the unified CLI so alias-resolution + ModelNotFoundError
        translation behave identically across both entrypoints.

        Source-text verification: ``main_internal`` in ``server.py``
        must reference ``_load_embedding_model_or_exit`` and route the
        ``args.embedding_model`` branch through it. A pure-AST /
        AST-bytecode check (rather than mocking the helper and calling
        the mock — pr_validate codex r1 BLOCKING #1 — which would
        always pass regardless of the server.py wiring). Pinning the
        wire here means a future refactor that re-inlines the
        embedding-load sequence into ``server.py`` would have to
        update this assertion AND prove parity.
        """
        import ast
        import inspect

        from vllm_mlx import server as server_mod

        # Walk every top-level function/method in server.py looking
        # for a call site that names ``_load_embedding_model_or_exit``.
        # If none exists, the standalone entrypoint has DIVERGED from
        # the CLI helper — D-EMBED-ALIAS regression.
        source = inspect.getsource(server_mod)
        tree = ast.parse(source)

        def _names_in_call(node: ast.AST) -> set[str]:
            names: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Name):
                        names.add(func.id)
                    elif isinstance(func, ast.Attribute):
                        names.add(func.attr)
            return names

        all_calls = _names_in_call(tree)
        assert "_load_embedding_model_or_exit" in all_calls, (
            "vllm_mlx/server.py no longer calls _load_embedding_model_or_exit "
            "— the standalone `python -m vllm_mlx.server` entrypoint has "
            "diverged from the CLI's alias-resolution + error-wrapping path. "
            "Either re-route through the shared helper or copy its full "
            "behaviour (alias resolve + concrete-class catch + exit-1 hint) "
            "back in."
        )

        # Belt-and-suspenders: the import edge must exist too, so a
        # local stub of the same name inside server.py can't satisfy
        # the assertion above without ALSO satisfying this.
        assert "from .cli import _load_embedding_model_or_exit" in source, (
            "vllm_mlx/server.py must import _load_embedding_model_or_exit "
            "from vllm_mlx.cli so the alias-resolution + error-wrapping "
            "path stays single-sourced."
        )

    def test_server_main_if_args_embedding_model_block_body(self):
        """Pin the actual ``if args.embedding_model:`` branch body in
        ``server.main()`` (codex r3 BLOCKING #3 — the prior spy-style
        test patched the helper on the cli module and called those
        statements at test scope, which proves the *helper* runs but
        not that ``server.main`` itself routes through it).

        AST walk into ``server.main`` finds the
        ``if args.embedding_model:`` ``If`` node and asserts its body is
        exactly ``ImportFrom(.cli, _load_embedding_model_or_exit)``
        followed by ``Expr(_load_embedding_model_or_exit(args,
        load_embedding_model))`` — byte-for-byte the legacy inline
        block, but now expressed as a single helper call.

        This catches a future regression where the import edge stays
        but the call is silently dropped (or replaced with a stub).
        """
        import ast
        import inspect

        from vllm_mlx import server as server_mod

        source = inspect.getsource(server_mod)
        tree = ast.parse(source)

        # Find the ``main`` function definition (the standalone
        # ``python -m vllm_mlx.server`` entrypoint).
        fn = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "main"
            ),
            None,
        )
        assert fn is not None, "vllm_mlx/server.py must define main"

        # Find ``if args.embedding_model:`` inside main_internal. We
        # match on the test expression so a later cosmetic rename
        # (e.g. ``if getattr(args, ...)``) would fail loudly.
        target_if = None
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (
                isinstance(test, ast.Attribute)
                and test.attr == "embedding_model"
                and isinstance(test.value, ast.Name)
                and test.value.id == "args"
            ):
                target_if = node
                break
        assert target_if is not None, (
            "vllm_mlx/server.py main_internal must contain "
            "`if args.embedding_model:` — the embedding-model gating "
            "branch is gone, which would silently broaden the loader "
            "and re-introduce the D-EMBED-ALIAS regression."
        )

        # Body must be exactly two statements:
        #   from .cli import _load_embedding_model_or_exit
        #   _load_embedding_model_or_exit(args, load_embedding_model)
        body = target_if.body
        assert len(body) == 2, (
            "if args.embedding_model body must be exactly the lazy import "
            "+ helper call; got "
            f"{len(body)} statement(s). Either restore the two-line block "
            "or update this assertion with the new wiring + matching parity "
            "test for the unified CLI."
        )

        stmt_import, stmt_call = body
        assert (
            isinstance(stmt_import, ast.ImportFrom)
            and stmt_import.module == "cli"
            and stmt_import.level == 1
            and any(
                alias.name == "_load_embedding_model_or_exit"
                for alias in stmt_import.names
            )
        ), (
            "First statement of `if args.embedding_model:` must be "
            "`from .cli import _load_embedding_model_or_exit` so the "
            "alias-resolution helper is sourced from the same module the "
            "unified `rapid-mlx serve` CLI uses."
        )

        assert isinstance(stmt_call, ast.Expr) and isinstance(
            stmt_call.value, ast.Call
        ), "Second statement must be the helper CALL, not a re-binding."
        call = stmt_call.value
        assert (
            isinstance(call.func, ast.Name)
            and call.func.id == "_load_embedding_model_or_exit"
        ), (
            "Second statement must invoke `_load_embedding_model_or_exit` "
            "(not a wrapper / not a renamed copy)."
        )
        assert len(call.args) == 2, (
            "Helper invocation must pass exactly (args, load_embedding_model)."
        )
        first_arg, second_arg = call.args
        assert isinstance(first_arg, ast.Name) and first_arg.id == "args"
        assert (
            isinstance(second_arg, ast.Name) and second_arg.id == "load_embedding_model"
        ), (
            "Second arg must be the module-level `load_embedding_model` "
            "symbol from vllm_mlx/server.py — not a renamed import — so "
            "the spy/patch path actually reaches the same loader."
        )
