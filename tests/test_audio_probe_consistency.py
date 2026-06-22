# SPDX-License-Identifier: Apache-2.0
"""F-D05 regression tests — unified ``mlx_audio`` probe.

Pre-fix the three audio endpoints used DIFFERENT probes:

* ``/v1/audio/voices`` never touched ``mlx_audio`` at all and always
  returned a static voice list (200).
* ``/v1/audio/speech`` did a lazy ``from mlx_audio.tts.generate
  import load_model`` inside the request handler — if the runtime
  import broke, it caught ``ImportError`` and 503'd.
* ``/v1/audio/transcriptions`` did its own lazy import.

When the runtime import actually broke (e.g. a fresh ``[vision]``
install where ``mlx-audio`` was pulled in as a transitive but
something downstream was wonky), the endpoints disagreed: voices
said yes, speech said no. Diego logged this in dogfood 0.8.3.

Fix: every audio route consults the SAME
:func:`vllm_mlx.audio.probe.require_mlx_audio` helper. The
``find_spec`` presence check + cached late-import covers both
"extra not installed" and "installed-but-runtime-broken" failure
modes with the same 503 envelope.
"""

from __future__ import annotations

import builtins

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def _reset_audio_probe():
    """Clear the cached probe verdict around each test.

    The probe caches at module level so the import check runs at
    most once per process. Tests that monkeypatch the import path
    need to drop the cache so the next call re-probes.
    """
    from vllm_mlx.audio import probe

    probe._reset_probe_cache()
    yield
    probe._reset_probe_cache()


def _mount_audio_app():
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import audio as audio_route

    app = FastAPI()
    app.include_router(audio_route.router)
    cfg = get_config()
    saved_api = cfg.api_key
    cfg.api_key = None

    def _restore():
        cfg.api_key = saved_api

    return TestClient(app), _restore


# ---------------------------------------------------------------------------
# Both routes succeed when mlx_audio is available
# ---------------------------------------------------------------------------


class TestProbeAgreesWhenAvailable:
    """When ``mlx_audio`` is installed and importable cleanly, every
    audio route reachable without a model load returns 200."""

    def test_voices_returns_200_when_probe_ok(self, _reset_audio_probe):
        pytest.importorskip("mlx_audio")
        client, restore = _mount_audio_app()
        try:
            r = client.get("/v1/audio/voices")
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        assert "voices" in body
        assert isinstance(body["voices"], list)
        assert len(body["voices"]) > 0


# ---------------------------------------------------------------------------
# Both routes 503 with the SAME envelope when mlx_audio is broken
# ---------------------------------------------------------------------------


def _install_broken_mlx_audio(monkeypatch, *, reason="simulated breakage"):
    """Force ``import mlx_audio`` (and sub-modules) to fail."""
    _orig_import = builtins.__import__

    def _broken(name, *args, **kwargs):
        if name == "mlx_audio" or name.startswith("mlx_audio."):
            raise ImportError(reason)
        return _orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken)


def _install_missing_mlx_audio(monkeypatch):
    """Force ``find_spec("mlx_audio")`` to return None — simulates the
    extra not being installed."""
    import importlib.util as _ilu

    real_find_spec = _ilu.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "mlx_audio":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr("importlib.util.find_spec", _fake_find_spec)


class TestProbeAgreesWhenBroken:
    """When ``mlx_audio`` fails at runtime, both ``/v1/audio/speech``
    and ``/v1/audio/voices`` return 503 with the same envelope shape
    — the cross-endpoint inconsistency Diego logged is gone."""

    def test_both_routes_503_when_runtime_import_fails(
        self, monkeypatch, _reset_audio_probe
    ):
        _install_broken_mlx_audio(monkeypatch, reason="torn install simulated")
        client, restore = _mount_audio_app()
        try:
            r_voices = client.get("/v1/audio/voices")
            r_speech = client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "input": "hi", "voice": "af_heart"},
            )
        finally:
            restore()
        assert r_voices.status_code == 503, r_voices.text
        assert r_speech.status_code == 503, r_speech.text
        # Same status, same envelope shape.
        body_v = r_voices.json()
        body_s = r_speech.json()
        assert "detail" in body_v
        assert "detail" in body_s
        # Both detail strings mention the runtime failure reason and
        # the actionable install hint.
        for body in (body_v, body_s):
            assert "torn install simulated" in body["detail"] or (
                "import failed" in body["detail"] or "not installed" in body["detail"]
            )
            assert "rapid-mlx[audio]" in body["detail"]

    def test_both_routes_503_when_extra_not_installed(
        self, monkeypatch, _reset_audio_probe
    ):
        _install_missing_mlx_audio(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r_voices = client.get("/v1/audio/voices")
            r_speech = client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "input": "hi", "voice": "af_heart"},
            )
        finally:
            restore()
        assert r_voices.status_code == 503
        assert r_speech.status_code == 503
        assert "not installed" in r_voices.json()["detail"]
        assert "not installed" in r_speech.json()["detail"]
        # Install hint is uniform.
        assert "rapid-mlx[audio]" in r_voices.json()["detail"]
        assert "rapid-mlx[audio]" in r_speech.json()["detail"]


# ---------------------------------------------------------------------------
# Both routes use the SAME helper (via static analysis)
# ---------------------------------------------------------------------------


class TestProbeWiredFromOneSource:
    """Mechanical guard against re-introducing route-local probes.

    A future refactor that hand-rolls a ``try: import mlx_audio`` in
    a new audio route — or removes the call to ``require_mlx_audio``
    from an existing route — would recreate the exact cross-endpoint
    inconsistency F-D05 fixed. Source-grep the audio routes module:
    every ``POST``/``GET`` audio route must import/call
    ``require_mlx_audio``.
    """

    def _route_body(self, path_marker: str) -> str:
        """Slice the source between a ``@router.<verb>("<path>"...)``
        decorator and the next ``@router.`` decorator (or EOF). Keeps
        the assertion off the module-level docstrings that mention
        the same path strings."""
        from pathlib import Path

        route_file = (
            Path(__file__).resolve().parents[1] / "vllm_mlx" / "routes" / "audio.py"
        )
        source = route_file.read_text()
        decorator = "@router."
        # Find the decorator line whose immediate next argument is the
        # route path we care about. Scan @router. occurrences.
        idx = 0
        while True:
            idx = source.find(decorator, idx)
            if idx == -1:
                raise AssertionError(f"no route decorator for {path_marker}")
            # Look ahead a few lines for the path string.
            line_end = source.find("\n", idx)
            decorator_chunk = source[idx : line_end + 1]
            if path_marker in decorator_chunk:
                break
            idx = line_end + 1
        next_decorator = source.find(decorator, idx + 1)
        end = next_decorator if next_decorator != -1 else len(source)
        return source[idx:end]

    def test_speech_route_calls_require_mlx_audio(self):
        body = self._route_body("/v1/audio/speech")
        assert "require_mlx_audio" in body, (
            "F-D05 regression: /v1/audio/speech no longer calls "
            "require_mlx_audio(). Every audio route must consult the "
            "shared probe — see vllm_mlx/audio/probe.py."
        )

    def test_voices_route_calls_require_mlx_audio(self):
        body = self._route_body("/v1/audio/voices")
        assert "require_mlx_audio" in body, (
            "F-D05 regression: /v1/audio/voices no longer calls "
            "require_mlx_audio(). The voices and speech routes diverged."
        )

    def test_transcriptions_route_calls_require_mlx_audio(self):
        body = self._route_body("/v1/audio/transcriptions")
        assert "require_mlx_audio" in body, (
            "F-D05 regression: /v1/audio/transcriptions no longer "
            "consults the shared probe. Add ``require_mlx_audio()`` "
            "to the handler."
        )

    def test_probe_module_no_top_level_mlx_audio_import(self):
        """The probe itself must not import ``mlx_audio`` at module
        top level — otherwise the base install (no ``[audio]`` extra)
        crashes on ``from vllm_mlx.audio.probe import require_mlx_audio``,
        defeating the whole point of the lazy probe. Mirror the
        invariant H-08's ``test_mlx_embeddings_not_imported_at_module_top_level``
        already pins for embeddings."""
        from pathlib import Path

        probe_file = (
            Path(__file__).resolve().parents[1] / "vllm_mlx" / "audio" / "probe.py"
        )
        for lineno, line in enumerate(probe_file.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if stripped != line:
                # Indented — inside a function, that's the lazy form.
                continue
            assert not stripped.startswith("import mlx_audio"), (
                f"probe.py:{lineno}: top-level ``import mlx_audio`` — "
                "the lazy probe must keep mlx_audio out of the base "
                "install's import surface."
            )
            assert not stripped.startswith("from mlx_audio"), (
                f"probe.py:{lineno}: top-level ``from mlx_audio`` — "
                "same lazy-import constraint as above."
            )


# ---------------------------------------------------------------------------
# Probe covers BOTH TTS and STT sub-modules
# ---------------------------------------------------------------------------


class TestProbeCoversBothLanes:
    """Codex r1 BLOCKING follow-up: the probe must import BOTH the
    TTS sub-module (``mlx_audio.tts.generate``) AND the STT
    sub-module (``mlx_audio.stt.utils``) so a transcription-only
    breakage doesn't pass the probe then 500 inside the STT route
    with a different envelope."""

    def test_stt_submodule_failure_trips_probe(self, monkeypatch, _reset_audio_probe):
        """Simulate an install where ``mlx_audio.tts.generate``
        imports cleanly but ``mlx_audio.stt.utils`` is broken. The
        probe must return ok=False so the transcriptions route
        returns the SAME 503 envelope the speech/voices routes use,
        rather than letting the STT route reach
        ``STTEngine.load()`` and 500 with a different shape.

        Codex r2 BLOCKING: clearing the probe's cached verdict isn't
        enough — if an earlier test imported ``mlx_audio.stt.*``,
        ``__import__("mlx_audio.stt.utils")`` returns the cached
        module without re-running the import machinery and the
        simulated breakage is never exercised. Drop the cached STT
        modules from ``sys.modules`` first so the next ``__import__``
        call has to go through the import system (and therefore hit
        our monkeypatched ``__import__``).
        """
        import sys

        # Evict any cached STT modules so the broken-import monkeypatch
        # actually fires when the probe re-imports.
        for name in list(sys.modules):
            if name == "mlx_audio.stt" or name.startswith("mlx_audio.stt."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        _orig_import = builtins.__import__

        def _broken_stt(name, *args, **kwargs):
            if name == "mlx_audio.stt.utils" or name.startswith("mlx_audio.stt"):
                raise ImportError("simulated stt breakage")
            return _orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _broken_stt)

        from vllm_mlx.audio import probe

        v = probe.mlx_audio_available("stt")
        assert v.ok is False, "STT-only breakage must trip the STT probe — F2 BLOCKING."
        assert "stt" in v.reason.lower(), (
            f"verdict reason should name the failing submodule, got {v.reason!r}"
        )

    def test_stt_breakage_does_not_trip_tts_lane(self, monkeypatch, _reset_audio_probe):
        """Codex r3 BLOCKING: an STT-only breakage must NOT 503 the
        TTS routes. Lane separation closes the regression where a
        torn STT install masked TTS-usable installs as fully broken.
        """
        import sys

        for name in list(sys.modules):
            if name == "mlx_audio.stt" or name.startswith("mlx_audio.stt."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        _orig_import = builtins.__import__

        def _broken_stt_only(name, *args, **kwargs):
            if name == "mlx_audio.stt.utils" or name.startswith("mlx_audio.stt"):
                raise ImportError("simulated stt breakage")
            return _orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _broken_stt_only)

        from vllm_mlx.audio import probe

        # TTS lane must still work — only STT is broken.
        v_tts = probe.mlx_audio_available("tts")
        assert v_tts.ok is True, (
            "Codex r3 regression: STT-only breakage tripped the TTS "
            f"probe (verdict={v_tts}). Lane separation broken."
        )
        # STT lane reports the breakage.
        v_stt = probe.mlx_audio_available("stt")
        assert v_stt.ok is False
        assert "stt" in v_stt.reason.lower()

    def test_tts_breakage_does_not_trip_stt_lane(self, monkeypatch, _reset_audio_probe):
        """Mirror of the previous test: TTS-only breakage must NOT
        503 transcriptions. Lane separation works both directions."""
        import sys

        for name in list(sys.modules):
            if name == "mlx_audio.tts" or name.startswith("mlx_audio.tts."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        _orig_import = builtins.__import__

        def _broken_tts_only(name, *args, **kwargs):
            if name == "mlx_audio.tts.generate" or name.startswith("mlx_audio.tts"):
                raise ImportError("simulated tts breakage")
            return _orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _broken_tts_only)

        from vllm_mlx.audio import probe

        v_stt = probe.mlx_audio_available("stt")
        assert v_stt.ok is True, (
            "Codex r3 regression: TTS-only breakage tripped the STT "
            f"probe (verdict={v_stt}). Lane separation broken."
        )
        v_tts = probe.mlx_audio_available("tts")
        assert v_tts.ok is False
        assert "tts" in v_tts.reason.lower()

    def test_probe_source_lists_both_submodules(self):
        """Source-pin so a future refactor that removes the STT
        sub-module from the probe is caught immediately."""
        from pathlib import Path

        probe_file = (
            Path(__file__).resolve().parents[1] / "vllm_mlx" / "audio" / "probe.py"
        )
        source = probe_file.read_text()
        assert "mlx_audio.tts" in source, "TTS submodule probe missing"
        assert "mlx_audio.stt" in source, "STT submodule probe missing — F2 regression"


# ---------------------------------------------------------------------------
# Probe verdict caching
# ---------------------------------------------------------------------------


class TestProbeCaching:
    """The probe runs the runtime import at most once per process,
    then re-uses the verdict. Tests pin the cache behavior so a
    future refactor that re-imports on every request doesn't quietly
    add per-request import latency."""

    def test_verdict_is_cached_after_first_call(self, _reset_audio_probe):
        pytest.importorskip("mlx_audio")
        from vllm_mlx.audio import probe

        # First call populates the cache for the TTS lane.
        v1 = probe.mlx_audio_available("tts")
        assert v1.ok is True
        # Cache is populated (per-lane dict).
        assert probe._cached_verdict, "lane-keyed cache should be non-empty"
        assert "tts" in probe._cached_verdict
        v2 = probe.mlx_audio_available("tts")
        # Same object (cached, not re-computed).
        assert v1 is v2

    def test_reset_cache_forces_reprobe(self, monkeypatch, _reset_audio_probe):
        from vllm_mlx.audio import probe

        # Force a False verdict to be cached.
        _install_missing_mlx_audio(monkeypatch)
        v1 = probe.mlx_audio_available()
        assert v1.ok is False
        # Reset and let the real find_spec answer this time.
        probe._reset_probe_cache()
        monkeypatch.undo()
        # If mlx_audio isn't installed in this venv, skip — the test
        # is about cache-reset behavior, not the True path.
        pytest.importorskip("mlx_audio")
        v2 = probe.mlx_audio_available()
        assert v2.ok is True
        assert v1 is not v2
