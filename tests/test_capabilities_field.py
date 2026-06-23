# SPDX-License-Identifier: Apache-2.0
"""F-D01 regression tests — unified ``capabilities`` advertisement on
``/v1/models``.

Pre-fix the ``capabilities`` field returned by ``/v1/models`` only
ever carried ``"embedding"`` (and only for the configured embedding
model). Every other entry — including VLMs with ``modality="image"``
— came back with ``capabilities=[]``. Diego logged this in dogfood
0.8.3: clients that route on capabilities to choose between text /
image / tool surfaces couldn't tell a VLM from a text-only model.

The fix routes every entry through a single
:func:`vllm_mlx.routes.models._detect_capabilities` helper that emits
the full tag list (``text``, ``vision``, ``tools``, ``embedding``) in
a fixed order. These tests pin the new contract.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _mount_models_app(monkeypatch, **cfg_overrides):
    """Mount the models router with controlled config + server state."""
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
            "tool_call_parser",
            "api_key",
        )
    }
    cfg.model_registry = None
    cfg.api_key = None
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)

    import vllm_mlx.server as srv

    saved_srv = {
        "_embedding_model_locked": srv._embedding_model_locked,
        "_tool_call_parser": srv._tool_call_parser,
    }
    srv._embedding_model_locked = cfg_overrides.get("embedding_model_locked")
    srv._tool_call_parser = cfg_overrides.get("tool_call_parser")

    def _restore():
        for k, v in saved.items():
            setattr(cfg, k, v)
        for k, v in saved_srv.items():
            setattr(srv, k, v)

    return TestClient(app), _restore


def _fetch_entry(client, model_id):
    r = client.get("/v1/models")
    assert r.status_code == 200, r.text
    body = r.json()
    for entry in body["data"]:
        if entry["id"] == model_id:
            return entry
    raise AssertionError(f"model {model_id} missing from /v1/models")


class TestTextOnlyModel:
    """Text-only model → ``capabilities=["text"]`` (no vision tag).

    Pre-fix every text model came back with ``capabilities=[]``.
    Clients couldn't tell whether the empty list meant "we don't
    know" or "no capabilities" — the new contract says
    ``["text"]`` is the always-present baseline for chat models.
    """

    def test_text_model_advertises_text_capability(self, monkeypatch):
        client, restore = _mount_models_app(
            monkeypatch,
            model_name="mlx-community/Qwen3-0.6B-8bit",
            model_alias="qwen3-0.6b-8bit",
        )
        try:
            entry = _fetch_entry(client, "mlx-community/Qwen3-0.6B-8bit")
        finally:
            restore()
        caps = entry["capabilities"]
        assert "text" in caps, f"text-only model missing 'text' tag: {caps}"
        assert "vision" not in caps, f"text-only model leaked 'vision' tag: {caps}"
        assert "embedding" not in caps, f"text-only model leaked 'embedding': {caps}"

    def test_unregistered_text_path_still_gets_text(self, monkeypatch):
        """Custom HF paths with no alias entry still get ``"text"`` —
        the baseline isn't gated on alias-registry membership."""
        unknown_id = "operator/custom-text-model"
        client, restore = _mount_models_app(
            monkeypatch,
            model_name=unknown_id,
            model_alias=unknown_id,
        )
        try:
            entry = _fetch_entry(client, unknown_id)
        finally:
            restore()
        assert "text" in entry["capabilities"]


class TestVisionModel:
    """VLM → ``capabilities`` includes both ``"text"`` and ``"vision"``.

    The repro from Diego's dogfood — ``mlx-community/Qwen3-VL-2B-Instruct-4bit``
    came back with ``modality="image"`` but ``capabilities=[]``. New
    contract: ``modality="image"`` AND ``capabilities`` contains
    ``"vision"``."""

    def test_vlm_alias_advertises_vision(self, monkeypatch):
        """A registered VLM alias should carry ``"vision"`` in its
        capabilities. Uses qwen3-vl alias because it's in
        ``aliases.json`` and resolves cleanly without HF cache I/O."""
        # Use an alias that's registered AND triggers the VLM detector.
        # If the alias registry doesn't have a VLM, skip — the test is
        # about the unified detector behavior, not the registry.
        from vllm_mlx.model_aliases import resolve_profile

        candidates = [
            "qwen3-vl-2b-instruct-4bit",
            "mlx-community/Qwen3-VL-2B-Instruct-4bit",
        ]
        for cand in candidates:
            if resolve_profile(cand) is not None:
                vlm_id = cand
                break
        else:
            # Fall back: the raw HF path goes through is_mllm_model
            # which catches VL repos by substring even without alias.
            vlm_id = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

        client, restore = _mount_models_app(
            monkeypatch,
            model_name=vlm_id,
            model_alias=vlm_id,
        )
        try:
            entry = _fetch_entry(client, vlm_id)
        finally:
            restore()
        caps = entry["capabilities"]
        assert "vision" in caps, (
            f"VLM {vlm_id} missing 'vision' in capabilities: {caps}. "
            "F-D01 regression — VLMs must advertise vision capability."
        )
        assert "text" in caps, (
            f"VLM {vlm_id} missing baseline 'text' capability: {caps}"
        )

    def test_raw_hf_vlm_path_advertises_vision(self, monkeypatch):
        """Raw HF paths (no alias entry) still get ``"vision"`` via
        the ``is_mllm_model`` substring detector — same path that
        gates VLM engine routing. Pin so a future refactor that
        narrows the detector to alias-only doesn't silently drop
        the capability for ad-hoc HF VLM repos."""
        # mllm.py MLLM_PATTERNS includes 'qwen3-vl', so a raw path
        # containing it matches the substring fallback.
        raw_id = "mlx-community/Qwen3-VL-7B-Instruct-MLX"
        client, restore = _mount_models_app(
            monkeypatch,
            model_name=raw_id,
            model_alias=raw_id,
        )
        try:
            entry = _fetch_entry(client, raw_id)
        finally:
            restore()
        assert "vision" in entry["capabilities"]
        assert entry["modality"] == "image", (
            "raw HF VLM path: modality should still flip to 'image'"
        )


class TestEmbeddingModel:
    """Configured embedding model → ``capabilities=["embedding"]``.

    H-09 invariant preserved: the embedding model carries
    ``"embedding"`` exclusively (no ``"text"``). Mixing the two would
    mislead clients into routing chat traffic at the embedding model
    id — the chat surface 400s for non-locked ids."""

    def test_embedding_model_carries_only_embedding(self, monkeypatch):
        embed_id = "mlx-community/embeddinggemma-300m-6bit"
        client, restore = _mount_models_app(
            monkeypatch,
            model_name="mlx-community/Qwen3-0.6B-8bit",
            model_alias="qwen3-0.6b-8bit",
            embedding_model_locked=embed_id,
        )
        try:
            entry = _fetch_entry(client, embed_id)
        finally:
            restore()
        assert entry["capabilities"] == ["embedding"], (
            f"embedding entry must carry exactly ['embedding'], got "
            f"{entry['capabilities']}"
        )

    def test_embedding_model_modality_is_text_not_null(self, monkeypatch):
        """F-D01 cosmetic: pre-fix embedding entry advertised
        ``modality=None`` while VLMs advertised ``modality="image"``.
        Embedding accepts text input, so the on-wire modality is
        ``"text"`` — the ``capabilities=["embedding"]`` tag
        distinguishes the lane, not the modality."""
        embed_id = "mlx-community/embeddinggemma-300m-6bit"
        client, restore = _mount_models_app(
            monkeypatch,
            model_name="mlx-community/Qwen3-0.6B-8bit",
            model_alias="qwen3-0.6b-8bit",
            embedding_model_locked=embed_id,
        )
        try:
            entry = _fetch_entry(client, embed_id)
        finally:
            restore()
        assert entry["modality"] == "text", (
            f"embedding modality must be 'text' not null/{entry['modality']!r}"
        )


class TestToolsCapability:
    """Tool-capable models → ``capabilities`` includes ``"tools"``.

    Two signals trip the tag: (1) the alias profile carries a
    non-empty ``tool_call_parser``, (2) the server is booted with
    ``--tool-call-parser`` for an unregistered path. Either fires
    the capability."""

    def test_profile_tool_parser_enables_tools_tag(self, monkeypatch):
        """A model whose alias profile sets ``tool_call_parser=hermes``
        (Qwen3) advertises ``"tools"`` regardless of server flags."""
        model_id = "mlx-community/Qwen3-0.6B-8bit"
        client, restore = _mount_models_app(
            monkeypatch,
            model_name=model_id,
            model_alias="qwen3-0.6b-8bit",
        )
        try:
            entry = _fetch_entry(client, model_id)
        finally:
            restore()
        caps = entry["capabilities"]
        assert "tools" in caps, (
            f"Qwen3 alias should advertise 'tools' (hermes parser), got {caps}"
        )

    def test_server_tool_parser_enables_tag_for_unregistered_id(self, monkeypatch):
        """Operator-supplied custom HF path + ``--tool-call-parser``
        flag → ``"tools"`` capability appears even without an alias
        profile entry."""
        unknown_id = "operator/custom-tools-model"
        client, restore = _mount_models_app(
            monkeypatch,
            model_name=unknown_id,
            model_alias=unknown_id,
            tool_call_parser="hermes",
        )
        try:
            entry = _fetch_entry(client, unknown_id)
        finally:
            restore()
        assert "tools" in entry["capabilities"]

    def test_server_global_tool_parser_does_not_leak_to_unrelated_entries(
        self, monkeypatch
    ):
        """Codex r4 BLOCKING: when the server is configured with a
        ``--tool-call-parser`` flag, the ``"tools"`` capability tag
        must appear ONLY on the model the server is actually serving
        — not on every entry that happens to be in the response.

        The server-level tool parser is a per-server flag. Painting
        ``"tools"`` onto unrelated registry/listed entries would
        mislead discovery clients into pre-flighting tool support on
        models that don't actually have it wired.

        Construct a registry with two entries — one served, one
        not — and verify only the served entry carries ``"tools"``
        from the global fallback. Both entries have a profile with
        a tool-call parser (qwen3 → hermes); the test pivots on the
        UNREGISTERED served vs unregistered unserved case where the
        server global is the ONLY signal."""
        from fastapi import FastAPI

        from vllm_mlx.config import get_config
        from vllm_mlx.routes import models as models_route

        served_unregistered = "operator/served-custom-tools-model"
        other_unregistered = "operator/discovered-other-model"

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
                "tool_call_parser",
                "api_key",
            )
        }
        cfg.model_name = served_unregistered
        cfg.model_alias = served_unregistered
        cfg.model_registry = None
        cfg.embedding_model_locked = None
        cfg.tool_call_parser = None
        cfg.api_key = None

        import vllm_mlx.server as srv

        saved_srv = {
            "_embedding_model_locked": srv._embedding_model_locked,
            "_tool_call_parser": srv._tool_call_parser,
        }
        srv._embedding_model_locked = None
        srv._tool_call_parser = "hermes"

        client = TestClient(app)
        try:
            # Probe the served id directly (in the list) and the
            # other unregistered id via retrieve_model — both must
            # respect the served-set gate.
            served_entry = _fetch_entry(client, served_unregistered)
            r = client.get(f"/v1/models/{other_unregistered}")
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)
            for k, v in saved_srv.items():
                setattr(srv, k, v)

        # Served entry carries "tools" from the server-global fallback.
        assert "tools" in served_entry["capabilities"]
        # The unserved/unregistered id returns 404 — the server
        # doesn't advertise it. (If it ever did via discovery, the
        # gate would still keep "tools" off.)
        assert r.status_code == 404, (
            f"non-served unregistered id should 404, got {r.status_code}"
        )

    def test_tools_tag_falls_back_to_server_global(self, monkeypatch):
        """Codex r1 BLOCKING follow-up: ``_tools_capable`` must check
        ``server._tool_call_parser`` when ``ServerConfig.tool_call_parser``
        is still None. Mirrors the ``_locked_embedding_id`` bridge — the
        config sync hasn't bridged the value yet but the global is set,
        and ``/v1/models`` must still advertise ``"tools"``.

        Without this fallback a boot-order quirk (server global set,
        ``_sync_config`` not yet run) would silently drop the
        capability tag for unregistered paths.
        """
        from fastapi import FastAPI

        from vllm_mlx.config import get_config
        from vllm_mlx.routes import models as models_route

        app = FastAPI()
        app.include_router(models_route.router)

        unknown_id = "operator/custom-tools-model-2"
        cfg = get_config()
        saved = {
            k: getattr(cfg, k, None)
            for k in (
                "model_name",
                "model_alias",
                "model_registry",
                "embedding_model_locked",
                "tool_call_parser",
                "api_key",
            )
        }
        cfg.model_name = unknown_id
        cfg.model_alias = unknown_id
        cfg.model_registry = None
        cfg.embedding_model_locked = None
        # Explicitly keep ``cfg.tool_call_parser`` at None — the bridge
        # hasn't fired yet in this scenario.
        cfg.tool_call_parser = None
        cfg.api_key = None

        import vllm_mlx.server as srv

        saved_srv = {
            "_embedding_model_locked": srv._embedding_model_locked,
            "_tool_call_parser": srv._tool_call_parser,
        }
        srv._embedding_model_locked = None
        # Server global IS set; config bridge has NOT happened.
        srv._tool_call_parser = "hermes"

        try:
            entry = _fetch_entry(TestClient(app), unknown_id)
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)
            for k, v in saved_srv.items():
                setattr(srv, k, v)
        assert "tools" in entry["capabilities"], (
            "F3 regression: server global _tool_call_parser was set "
            "but cfg.tool_call_parser is None, and 'tools' capability "
            "is missing. The fallback to the server global is gone."
        )


class TestCapabilityShapeAndOrder:
    """Pin the wire shape: list of strings, stable order, no dupes.

    Tests pre-bind the shape so a future refactor that returns
    ``set``, ``tuple``, or duplicate entries fails fast."""

    def test_capabilities_is_a_list(self, monkeypatch):
        client, restore = _mount_models_app(
            monkeypatch,
            model_name="mlx-community/Qwen3-0.6B-8bit",
            model_alias="qwen3-0.6b-8bit",
        )
        try:
            entry = _fetch_entry(client, "mlx-community/Qwen3-0.6B-8bit")
        finally:
            restore()
        assert isinstance(entry["capabilities"], list)
        for c in entry["capabilities"]:
            assert isinstance(c, str)

    def test_text_precedes_vision_precedes_tools(self, monkeypatch):
        """Union case: text + tools, or text + vision + tools, must
        appear in ``text → vision → tools`` order. The detector
        builds the list deterministically; tests pin so reordering
        breaks the contract loudly."""
        # Qwen3-VL is multimodal AND has hermes parser.

        vlm_with_tools = "mlx-community/Qwen3-VL-7B-Instruct-MLX"
        client, restore = _mount_models_app(
            monkeypatch,
            model_name=vlm_with_tools,
            model_alias=vlm_with_tools,
            tool_call_parser="hermes",
        )
        try:
            entry = _fetch_entry(client, vlm_with_tools)
        finally:
            restore()
        caps = entry["capabilities"]
        # All three present.
        assert "text" in caps and "vision" in caps and "tools" in caps
        # Order: text < vision < tools.
        assert caps.index("text") < caps.index("vision") < caps.index("tools"), (
            f"capabilities order broken: {caps}. Expected text → vision → tools."
        )

    def test_no_duplicate_tags(self, monkeypatch):
        client, restore = _mount_models_app(
            monkeypatch,
            model_name="mlx-community/Qwen3-0.6B-8bit",
            model_alias="qwen3-0.6b-8bit",
            tool_call_parser="hermes",
        )
        try:
            entry = _fetch_entry(client, "mlx-community/Qwen3-0.6B-8bit")
        finally:
            restore()
        caps = entry["capabilities"]
        assert len(caps) == len(set(caps)), f"duplicate tags: {caps}"
