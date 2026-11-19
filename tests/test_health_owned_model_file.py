# SPDX-License-Identifier: Apache-2.0
"""Readiness must inspect the model code selectors the loader actually uses."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.routes.health import _clear_consent_cache, probe_router


@pytest.mark.parametrize("consent", [None, "0", "1"])
@pytest.mark.parametrize("runtime", ["owned", "unregistered", "custom", "vlm"])
def test_readiness_uses_effective_model_file_config(
    tmp_path, monkeypatch, consent, runtime
):
    from rapid_mlx.utils import tokenizer

    model_type = "custom" if runtime == "custom" else "raon1_nano"
    config = {"model_type": model_type, "model_file": "raon_mlx_model.py"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    # Model the registration completed by the text loader without requiring Metal.
    monkeypatch.setattr(
        tokenizer,
        "_VENDORED_MODEL_TYPES",
        set() if runtime == "unregistered" else {"raon1_nano"},
    )
    monkeypatch.setitem(
        sys.modules, "mlx_lm.models.raon1_nano", ModuleType("raon1_nano")
    )
    if consent is None:
        monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", consent)

    cfg = reset_config()
    cfg.model_path = str(tmp_path)
    cfg.ready = True
    cfg.engine = SimpleNamespace(is_mllm=runtime == "vlm", get_stats=lambda: {})
    app = FastAPI()
    app.include_router(probe_router)
    _clear_consent_cache()
    try:
        client = TestClient(app)
        ready = runtime == "owned" or consent == "1"
        health = client.get("/health").json()
        assert health["ready"] is ready
        assert health["state"] == ("ready" if ready else "blocked")
        for path in ("/health/ready", "/readyz"):
            response = client.get(path)
            assert response.status_code == (200 if ready else 503)
            assert response.json()["ready"] is ready
            if not ready:
                assert health["reason"] == "model_code_consent_required"
                assert response.json()["reason"] == health["reason"]
        assert json.loads((tmp_path / "config.json").read_text()) == config
    finally:
        reset_config()
        _clear_consent_cache()


def test_pinned_readiness_rechecks_after_owned_runtime_registers(tmp_path, monkeypatch):
    from rapid_mlx import model_aliases
    from rapid_mlx.routes.health import _model_consent_blocker
    from rapid_mlx.utils import tokenizer

    revision = "a1" * 20
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        '{"model_type":"raon1_nano","model_file":"raon_mlx_model.py"}'
    )
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda _: revision)
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.resolve_offline_cached_snapshot", lambda _: snapshot
    )
    monkeypatch.setattr(tokenizer, "_VENDORED_MODEL_TYPES", set())
    monkeypatch.setitem(
        sys.modules, "mlx_lm.models.raon1_nano", ModuleType("raon1_nano")
    )
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    cfg = SimpleNamespace(model_path="test/raon")
    _clear_consent_cache()
    try:
        assert _model_consent_blocker(cfg)["reason"] == "model_code_consent_required"
        monkeypatch.setattr(tokenizer, "_VENDORED_MODEL_TYPES", {"raon1_nano"})
        assert _model_consent_blocker(cfg) == {}
    finally:
        _clear_consent_cache()
