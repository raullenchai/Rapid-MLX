# SPDX-License-Identifier: Apache-2.0
"""F-210: ``/v1/audio/transcriptions`` rejects path-shaped model ids
with the canonical 404 ``model_not_found_error`` instead of crashing
inside the codec layer as a generic 500 ``transcription_failed``.

PR #735 (F-165 / F-167) canonicalized bogus *alias* model strings to
the 404 ``model_not_found_error``. The follow-up gap reported by
test-loop-ad3b522d (TODO.md F-210): model strings shaped like a path —
multi-slash (``foo/bar/baz``) or all-slash (``////``) — still slipped
past the simple ``"/" in model`` heuristic, were forwarded to
``STTEngine.load``, and surfaced downstream as 500
``transcription_failed`` once the codec couldn't open them. This fix
restricts the HuggingFace pass-through to the canonical ``<org>/<repo>``
regex shape and rejects everything else with the same 404 the
``does-not-exist`` alias case already returned.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Unit-level — exercise the resolver directly
# ---------------------------------------------------------------------------


class TestPathShapedRejection:
    """Path-shaped / malformed model ids must 404 BEFORE codec open."""

    @pytest.mark.parametrize(
        "model_string",
        [
            "foo/bar/baz",
            "a/b/c/d/e",
            "////",
            "//",
            "/leading-slash",
            "trailing-slash/",
            "double//slash",
            "spaces in/name",
            "name\nwith\nnewline",
            "back\\slash",
            "name?with?qmark",
            "name#with#hash",
        ],
    )
    def test_rejects_path_shaped_model_strings(self, model_string: str):
        from vllm_mlx.routes.audio import _resolve_stt_model

        with pytest.raises(HTTPException) as exc_info:
            _resolve_stt_model(model_string)
        assert exc_info.value.status_code == 404, (
            f"expected 404 for {model_string!r}, got {exc_info.value.status_code}"
        )
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["error"]["type"] == "model_not_found_error"
        assert detail["error"]["code"] == "model_not_found"

    @pytest.mark.parametrize(
        "model_string",
        [
            "org/private-stt",
            "mlx-community/whisper-large-v3-mlx",
            "user-name/repo.name",
            "org/repo-with-hyphens",
            "org/repo_with_underscore",
        ],
    )
    def test_accepts_canonical_hf_repo_ids(self, model_string: str):
        """Single-slash ``<org>/<repo>`` is the canonical HuggingFace
        shape; must continue to pass through unchanged. Allowed char
        class mirrors HF's ``[A-Za-z0-9._-]`` repo-id rule (no ``+``)."""
        from vllm_mlx.routes.audio import _resolve_stt_model

        assert _resolve_stt_model(model_string) == model_string

    def test_rejects_repo_id_with_plus(self):
        """codex-r1 BLOCKING: ``+`` is not a valid HF repo-id character.
        Must be rejected as ``model_not_found_error`` rather than
        passed through to ``STTEngine.load`` (which would 500)."""
        from vllm_mlx.routes.audio import _resolve_stt_model

        with pytest.raises(HTTPException) as exc_info:
            _resolve_stt_model("org/repo+with+plus")
        assert exc_info.value.status_code == 404
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["error"]["type"] == "model_not_found_error"

    @pytest.mark.parametrize(
        "model_string",
        [
            "org/.hidden",
            ".hidden/repo",
            "org/repo.",
            "org/.",
            "org/repo..name",
            "org/repo--name",
            "org/-leading-dash",
            "org/trailing-dash-",
            "org/repo.git",
        ],
    )
    def test_rejects_hf_structurally_invalid_repo_ids(self, model_string: str):
        """codex r2 BLOCKING: HF rejects ``.hidden``, ``..``, ``--``,
        leading/trailing ``.``/``-``, and ``.git`` suffix as repo-id
        components. The bare-regex check accepted these and let them
        through to ``STTEngine.load`` where the HF resolver also fails
        — surfacing as a 500 instead of the intended 404. Per-component
        structural validation matching ``huggingface_hub.utils.
        validate_repo_id``."""
        from vllm_mlx.routes.audio import _resolve_stt_model

        with pytest.raises(HTTPException) as exc_info:
            _resolve_stt_model(model_string)
        assert exc_info.value.status_code == 404, (
            f"expected 404 for {model_string!r}, got {exc_info.value.status_code}"
        )
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["error"]["type"] == "model_not_found_error"

    def test_accepts_notebook_ipynb_repo_id(self):
        """codex r3 BLOCKING: ``.ipynb`` is NOT a HF-reserved suffix
        (only ``.git`` is). Must continue to pass through."""
        from vllm_mlx.routes.audio import _resolve_stt_model

        assert _resolve_stt_model("org/notebook.ipynb") == "org/notebook.ipynb"

    def test_rejects_over_length_repo_id(self):
        """codex r3 BLOCKING: HF's overall repo_id length cap is 96.
        A 193-char ``namespace/repo`` previously slipped past the
        per-component bound and crashed in ``STTEngine.load``."""
        from vllm_mlx.routes.audio import _resolve_stt_model

        # 90 + 1 ('/') + 90 = 181 chars — over the 96-char total cap.
        over_long = "a" * 90 + "/" + "b" * 90
        with pytest.raises(HTTPException) as exc_info:
            _resolve_stt_model(over_long)
        assert exc_info.value.status_code == 404
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["error"]["type"] == "model_not_found_error"

    def test_alias_still_resolves(self):
        """F-165 contract: known aliases continue to map to repos."""
        from vllm_mlx.routes.audio import _resolve_stt_model

        assert _resolve_stt_model("whisper-small") == "mlx-community/whisper-small-mlx"

    def test_empty_string_still_400(self):
        """Empty string must remain a 400 ``invalid_request_error``,
        not get re-classified as a 404."""
        from vllm_mlx.routes.audio import _resolve_stt_model

        with pytest.raises(HTTPException) as exc_info:
            _resolve_stt_model("")
        assert exc_info.value.status_code == 400
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Route-level — exercise the actual /v1/audio/transcriptions endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def _audio_client(monkeypatch):
    """Mount the audio router on a clean FastAPI app for route-level checks."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr("vllm_mlx.middleware.auth.verify_api_key", lambda: None)
    from vllm_mlx.routes.audio import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.parametrize("model_string", ["foo/bar/baz", "////"])
def test_route_path_shaped_model_returns_404_not_500(_audio_client, model_string: str):
    """Live route surface mirrors the BEFORE/AFTER repro in TODO.md F-210.

    Pre-fix: HTTP 500 ``transcription_failed``.
    Post-fix: HTTP 404 ``model_not_found_error`` (matches the
    ``does-not-exist`` control case from PR #735).
    """
    r = _audio_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("silence.wav", b"\x00" * 32, "audio/wav")},
        data={"model": model_string},
    )

    assert r.status_code == 404, r.text
    body = r.json()
    # In the live server, ``middleware/exception_handlers.py`` unwraps
    # ``detail`` so the on-wire envelope is ``{"error": {...}}``. In
    # this unit-level FastAPI app no global handlers are mounted, so
    # the default Starlette renderer ships ``{"detail": {"error":
    # {...}}}``. Accept either shape — the contract under test is
    # "404 + ``model_not_found_error`` code".
    err = body.get("error") or body.get("detail", {}).get("error", {})
    assert err.get("type") == "model_not_found_error", body
    assert err.get("code") == "model_not_found", body
    assert model_string in err.get("message", ""), body
