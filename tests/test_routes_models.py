# SPDX-License-Identifier: Apache-2.0
"""R11-G / H-13 — ``/v1/models`` visibility of the configured embedding model.

H-13 (Bo r11 carry from R8-H3): boot
``rapid-mlx serve --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit``
and the response from ``GET /v1/models`` only listed the *chat* model.
The embedding model id was missing, which broke
``client.models.list()`` auto-discovery for LangChain, LlamaIndex, and
openai-python — those clients enumerate ``/v1/models`` to find
``capabilities`` containing ``"embedding"`` before they will route a
``client.embeddings.create()`` call.

This file pins the discovery contract directly against
``vllm_mlx.routes.models``. The broader H-08+H-09+H-13 regression net
lives in :mod:`tests.test_embeddings_extra_guard` (the same
``ModelsListEmbeddingCapability`` class); this dedicated file exists
because the task spec named ``tests/test_routes_models.py::
test_embedding_model_in_models_list`` explicitly and discovery clients
typically grep for the route file name when wiring up a new transport.

A regression here is a wire-shape break, not a unit-level bug, so we
mount a real :class:`FastAPI` app with the ``vllm_mlx.routes.models``
router and inspect the JSON response — same shape rapid-desktop and
the openai client see in production.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _mount_models_app(*, embedding_model_locked: str | None):
    """Mount a TestClient on the models router with a stubbed config.

    Saves + restores both :class:`ServerConfig` fields AND the
    ``vllm_mlx.server._embedding_model_locked`` global so a test
    interleave can't bleed state across cases.
    """
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
    cfg.model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    cfg.model_alias = "llama-3.2-1b-4bit"
    cfg.model_registry = None
    cfg.embedding_model_locked = embedding_model_locked
    cfg.api_key = None

    import vllm_mlx.server as srv

    saved_srv = {"_embedding_model_locked": srv._embedding_model_locked}
    srv._embedding_model_locked = embedding_model_locked

    def _restore() -> None:
        for k, v in saved.items():
            setattr(cfg, k, v)
        for k, v in saved_srv.items():
            setattr(srv, k, v)

    return TestClient(app), _restore


def test_embedding_model_in_models_list():
    """H-13: when an embedding model is locked at boot, it MUST appear
    in ``/v1/models`` alongside the chat model with
    ``capabilities=["embedding"]``.

    Pre-fix the listing only enumerated ``cfg.model_name`` /
    ``cfg.model_alias`` — the dedicated embedding model id was
    invisible. Discovery clients (``client.models.list()`` in
    langchain / llamaindex / openai-python) iterate this listing and
    pick the embedding id by ``"embedding" in capabilities``; without
    the entry the clients fell back to substring-matching the model
    name, which fails on every aliased id.
    """
    embed_id = "mlx-community/Qwen3-Embedding-0.6B-4bit"
    client, restore = _mount_models_app(embedding_model_locked=embed_id)
    try:
        r = client.get("/v1/models")
    finally:
        restore()

    assert r.status_code == 200, r.text
    body = r.json()

    ids = [entry["id"] for entry in body["data"]]
    # Both cards must be present — the chat model AND the embedding model.
    assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in ids, (
        "Chat model went missing from /v1/models after adding the embedding "
        "entry. Listing regression — H-13 fix must not drop the chat card."
    )
    assert embed_id in ids, (
        f"Configured embedding model {embed_id!r} missing from /v1/models. "
        "client.models.list() auto-discovery is broken for "
        "langchain / llamaindex / openai-python."
    )

    # Exactly one entry carries the embedding capability — chat models
    # MUST NOT silently advertise it. The H-09 route guard would 503
    # ``/v1/embeddings`` for non-locked ids, so claiming a chat card is
    # embedding-capable on the listing would mislead the desktop client.
    embedding_entries = [
        entry for entry in body["data"] if "embedding" in entry.get("capabilities", [])
    ]
    assert len(embedding_entries) == 1, embedding_entries
    embed_card = embedding_entries[0]
    assert embed_card["id"] == embed_id
    assert embed_card["capabilities"] == ["embedding"]
    # The modality on the wire is "text" — embedding models accept
    # text input; the ``capabilities`` tag is what distinguishes the
    # lane (F-D01 cosmetic).
    assert embed_card["modality"] == "text"
    # ``object`` field is OpenAI-canonical "model" — clients that
    # validate the response shape against the OpenAI spec rely on it.
    assert embed_card["object"] == "model"


def test_no_embedding_model_no_embedding_card():
    """Sanity: without ``--embedding-model``, no card carries the
    embedding capability tag. The H-09 route guard already 503s
    ``/v1/embeddings``, so claiming the chat card is embedding-capable
    here would mislead discovery clients into routing real traffic at
    a path that will only error."""
    client, restore = _mount_models_app(embedding_model_locked=None)
    try:
        r = client.get("/v1/models")
    finally:
        restore()
    assert r.status_code == 200, r.text
    body = r.json()
    for entry in body["data"]:
        caps = entry.get("capabilities", [])
        assert "embedding" not in caps, (
            f"Model {entry['id']} advertises embedding capability but no "
            "embedding model is configured — /v1/embeddings would 503."
        )


def test_retrieve_embedding_model_by_path_id():
    """``GET /v1/models/{embed_id}`` must resolve a slash-containing
    HF id directly — every other rapid-mlx endpoint accepts the bare
    HF id, this one should too.

    desktop / rapid-desktop hydrates per-model state from this path
    (R10-D contract); a future refactor that breaks slash handling
    would silently kneecap the per-model UI without touching
    ``/v1/models``.
    """
    embed_id = "mlx-community/Qwen3-Embedding-0.6B-4bit"
    client, restore = _mount_models_app(embedding_model_locked=embed_id)
    try:
        r = client.get(f"/v1/models/{embed_id}")
    finally:
        restore()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == embed_id
    assert "embedding" in body["capabilities"]


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Served-engine modality authority (#1187 / #393).
#
# After the engine auto-degrades a vision-config checkpoint with no usable
# vision tower to the text lane (or the operator passes --no-mllm), the live
# engine's ``is_mllm`` is the truth. A fresh ``is_mllm_model`` re-detect still
# reads config/index — which keep declaring vision — so /v1/models must prefer
# the engine for the served model, or it advertises a vision capability the
# server will reject.
# ---------------------------------------------------------------------------


class _StubEngine:
    def __init__(self, is_mllm: bool):
        self.is_mllm = is_mllm


def _stub_single_serve(monkeypatch, *, model_id: str, engine_is_mllm: bool):
    """Point get_config() at a single-model serve whose live engine reports
    ``engine_is_mllm``. Returns a restore callable."""
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved = {
        k: getattr(cfg, k, None)
        for k in ("model_name", "model_alias", "model_registry", "engine")
    }
    cfg.model_registry = None
    cfg.model_name = model_id
    cfg.model_alias = None
    cfg.engine = _StubEngine(engine_is_mllm)

    def _restore() -> None:
        for k, v in saved.items():
            setattr(cfg, k, v)

    return _restore


def test_reported_modality_prefers_degraded_engine(monkeypatch):
    """Engine degraded to text ⇒ wire says ``text`` even though the static
    detector (reading the still-vision config/index) says otherwise."""
    from vllm_mlx.routes import models as models_route

    restore = _stub_single_serve(
        monkeypatch, model_id="fake/gemma4-optiq-4bit", engine_is_mllm=False
    )
    # Static detector claims vision — the lying config/index. Engine must win.
    monkeypatch.setattr(models_route, "is_mllm_model", lambda _m: True)
    try:
        assert models_route._served_engine_is_mllm("fake/gemma4-optiq-4bit") is False
        assert (
            models_route._reported_modality("fake/gemma4-optiq-4bit", "text", False)
            == "text"
        )
        assert models_route._is_vlm("fake/gemma4-optiq-4bit", "text", False) is False
    finally:
        restore()


def test_reported_modality_engine_authority_is_symmetric(monkeypatch):
    """SYMMETRIC authority: the live engine for the served model wins over the
    static detector BOTH ways. is_mllm=False → text (degrade / --no-mllm), and
    is_mllm=True → image even when the static detector misses it (an explicit
    --mllm that loaded a real vision tower the config/index re-detect can't
    see). Scoping (`_served_engine_is_mllm`) — not asymmetry — is what keeps a
    different alias's leaked engine from contaminating an unrelated model."""
    from vllm_mlx.routes import models as models_route

    # Engine loaded a vision tower; static detector disagrees (says text).
    # The live engine must win and report image/vision.
    restore = _stub_single_serve(
        monkeypatch, model_id="fake/qwen3-vl-4bit", engine_is_mllm=True
    )
    monkeypatch.setattr(models_route, "is_mllm_model", lambda _m: False)
    try:
        assert (
            models_route._reported_modality("fake/qwen3-vl-4bit", "text", False)
            == "image"
        ), "live engine is_mllm=True must win over a text detector verdict"
        assert models_route._is_vlm("fake/qwen3-vl-4bit", "text", False) is True
    finally:
        restore()
    # And when the detector agrees the model IS a VLM, image is preserved.
    restore = _stub_single_serve(
        monkeypatch, model_id="fake/qwen3-vl-4bit", engine_is_mllm=True
    )
    monkeypatch.setattr(models_route, "is_mllm_model", lambda _m: True)
    try:
        assert (
            models_route._reported_modality("fake/qwen3-vl-4bit", "text", False)
            == "image"
        )
        assert models_route._is_vlm("fake/qwen3-vl-4bit", "text", False) is True
    finally:
        restore()


def test_served_engine_authority_only_for_served_model(monkeypatch):
    """A registry-only / unserved id has no live engine ⇒ the helper returns
    None and the static detector remains the decider (no over-reach)."""
    from vllm_mlx.routes import models as models_route

    restore = _stub_single_serve(
        monkeypatch, model_id="fake/served-model", engine_is_mllm=False
    )
    monkeypatch.setattr(models_route, "is_mllm_model", lambda _m: True)
    try:
        assert models_route._served_engine_is_mllm("other/registry-only") is None
        assert (
            models_route._reported_modality("other/registry-only", "text", False)
            == "image"
        )
        assert models_route._is_vlm("other/registry-only", "text", False) is True
    finally:
        restore()


def test_engine_is_mllm_or_none_is_defensive():
    """Only a real bool is authoritative; anything else (None engine, a
    partially-built entry, a test double without ``is_mllm``) yields None so
    /v1/models falls through to the static detector rather than raising."""
    from vllm_mlx.routes import models as models_route

    assert models_route._engine_is_mllm_or_none(None) is None
    assert models_route._engine_is_mllm_or_none(object()) is None  # no is_mllm attr
    assert models_route._engine_is_mllm_or_none(_StubEngine(True)) is True
    assert models_route._engine_is_mllm_or_none(_StubEngine(False)) is False


def test_build_model_info_raw_hf_path_honors_degraded_engine(monkeypatch):
    """The raw-HF-path branch of ``_build_model_info`` (no alias profile — the
    gemma-4 OptiQ case) must, once the served engine has degraded to text,
    report the SAME wire shape as any other served raw-HF text model: the
    baseline ``modality`` (``None`` — raw-HF text ids carry no positive
    modality; that's the established behavior for every non-VLM raw path) and
    NO vision capability. Pinning the exact modality (not just "!= image")
    proves ``modality`` and ``capabilities`` are consistent and that the
    degraded VLM is indistinguishable from a plain text model (#1187)."""
    from vllm_mlx.routes import models as models_route

    model_id = "mlx-community/gemma-4-26B-A4B-it-qat-OptiQ-4bit"

    # Reference: a genuinely-text raw-HF model served by the same stub. Its
    # modality is the raw-HF text baseline the degraded VLM must match.
    ref_id = "mlx-community/some-plain-text-model-4bit"
    restore = _stub_single_serve(monkeypatch, model_id=ref_id, engine_is_mllm=False)
    monkeypatch.setattr(models_route, "is_mllm_model", lambda _m: False)
    try:
        baseline_modality = models_route._build_model_info(ref_id).modality
    finally:
        restore()

    # Pin the baseline EXPLICITLY to None (not just "whatever the function
    # returns") so a shared None -> other-non-image regression can't slip past
    # both the reference and the degraded check (codex NIT, PR #1189).
    assert baseline_modality is None, (
        f"raw-HF text baseline modality must be None; got {baseline_modality!r} "
        "— the reference itself regressed, so the equality check below would be "
        "meaningless"
    )

    restore = _stub_single_serve(monkeypatch, model_id=model_id, engine_is_mllm=False)
    # Static detector still sees the declared vision modality (lying index).
    monkeypatch.setattr(models_route, "is_mllm_model", lambda _m: True)
    try:
        info = models_route._build_model_info(model_id)
        assert info.modality is None, (
            f"degraded raw-HF VLM must report modality None (the raw-HF text "
            f"baseline), not {info.modality!r}"
        )
        assert info.modality == baseline_modality, (
            f"degraded raw-HF VLM must report the same modality as a plain "
            f"raw-HF text model ({baseline_modality!r}); got {info.modality!r}"
        )
        assert info.modality != "image", (
            f"degraded raw-HF VLM must not advertise image modality; "
            f"got {info.modality!r}"
        )
        assert "vision" not in (info.capabilities or []), (
            f"degraded model must not advertise vision capability; "
            f"got {info.capabilities!r}"
        )
    finally:
        restore()
