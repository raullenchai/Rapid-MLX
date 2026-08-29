# SPDX-License-Identifier: Apache-2.0
"""Kokoro G2P preparation and inference-boundary contracts."""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import types
from typing import cast

import pytest

from vllm_mlx.audio import probe, runtime_requirements
from vllm_mlx.audio.probe import (
    _KOKORO_G2P_SPACY_MODEL,
    _ensure_kokoro_g2p_model_ready,
    _kokoro_voice_needs_en_g2p,
    _probe_kokoro_g2p_model,
    _reset_g2p_model_state,
    require_kokoro_runtime,
)
from vllm_mlx.audio.registry import (
    AudioRuntimeRequirement,
    AudioRuntimeRequirementKind,
)
from vllm_mlx.audio.runtime_requirements import (
    AudioRuntimePreparationError,
    _installer_env,
    prepare_runtime_requirement,
)


@pytest.fixture(autouse=True)
def _fresh_verdict():
    _reset_g2p_model_state()
    yield
    _reset_g2p_model_state()


@pytest.mark.parametrize(
    "source,prefix,is_venv,expected",
    [
        (
            {"VIRTUAL_ENV": "/other", "PATH": "/bin"},
            "/opt/venv",
            True,
            {"VIRTUAL_ENV": "/opt/venv", "PATH": "/bin"},
        ),
        (
            {"PATH": "/bin"},
            "/opt/venv",
            True,
            {"PATH": "/bin", "VIRTUAL_ENV": "/opt/venv"},
        ),
        ({}, "/usr", False, {}),
        (
            {"VIRTUAL_ENV": "/outer", "PATH": "/bin"},
            "/usr",
            False,
            {"PATH": "/bin"},
        ),
    ],
)
def test_installer_env_targets_running_interpreter(source, prefix, is_venv, expected):
    original = dict(source)
    assert _installer_env(source, prefix, is_venv) == expected
    assert source == original


def _fake_spacy(monkeypatch, state):
    util = types.ModuleType("spacy.util")
    util.is_package = lambda name: bool(state.get("installed"))
    spacy = types.ModuleType("spacy")
    spacy.util = util
    monkeypatch.setitem(sys.modules, "spacy", spacy)
    monkeypatch.setitem(sys.modules, "spacy.util", util)


def _requirement() -> AudioRuntimeRequirement:
    return AudioRuntimeRequirement(kind="spacy_pipeline", name=_KOKORO_G2P_SPACY_MODEL)


def test_pull_preparer_noops_when_pipeline_present(monkeypatch):
    _fake_spacy(monkeypatch, {"installed": True})
    calls = []
    monkeypatch.setattr(
        runtime_requirements, "_run_installer", lambda *args: calls.append(args)
    )

    prepare_runtime_requirement(_requirement())

    assert calls == []


def test_pull_preparer_installs_absent_pipeline_then_verifies(monkeypatch):
    state = {"installed": False}
    _fake_spacy(monkeypatch, state)
    seen = {}

    def fake_install(cmd, env, timeout):
        seen.update(cmd=cmd, env=env, timeout=timeout)
        state["installed"] = True

    monkeypatch.setattr(runtime_requirements, "_run_installer", fake_install)

    prepare_runtime_requirement(_requirement())

    assert seen["cmd"] == [
        sys.executable,
        "-m",
        "spacy",
        "download",
        _KOKORO_G2P_SPACY_MODEL,
    ]
    assert seen["timeout"] == 300
    assert seen["env"] is not None


def test_pull_preparer_sanitizes_installer_failure(monkeypatch, caplog):
    _fake_spacy(monkeypatch, {"installed": False})
    secret = "https://user:token@internal.pkgs.example/simple"

    def fail(cmd, env, timeout):
        raise subprocess.CalledProcessError(7, cmd, stderr=secret)

    monkeypatch.setattr(runtime_requirements, "_run_installer", fail)
    with (
        caplog.at_level("ERROR"),
        pytest.raises(
            AudioRuntimePreparationError, match=_KOKORO_G2P_SPACY_MODEL
        ) as excinfo,
    ):
        prepare_runtime_requirement(_requirement())

    assert secret not in str(excinfo.value)
    assert secret not in caplog.text
    assert "CalledProcessError (exit 7)" in caplog.text


def test_pull_preparer_fails_when_pipeline_remains_invisible(monkeypatch):
    _fake_spacy(monkeypatch, {"installed": False})
    monkeypatch.setattr(
        runtime_requirements, "_run_installer", lambda *args, **kwargs: None
    )

    with pytest.raises(AudioRuntimePreparationError, match="reported success"):
        prepare_runtime_requirement(_requirement())


def test_pull_preparer_sanitizes_post_install_import_failure(monkeypatch):
    calls = 0

    def availability(_package):
        nonlocal calls
        calls += 1
        if calls == 1:
            return False
        raise ImportError("/private/secret/path/broken.dylib")

    monkeypatch.setattr(runtime_requirements, "spacy_pipeline_available", availability)
    monkeypatch.setattr(
        runtime_requirements, "_run_installer", lambda *args, **kwargs: None
    )

    with pytest.raises(AudioRuntimePreparationError) as excinfo:
        prepare_runtime_requirement(_requirement())

    assert "/private/secret/path" not in str(excinfo.value)


def test_pull_preparer_fails_closed_when_spacy_runtime_is_broken(monkeypatch):
    def broken(_package):
        raise ImportError("/private/secret/path/broken.dylib")

    monkeypatch.setattr(runtime_requirements, "spacy_pipeline_available", broken)
    monkeypatch.setattr(
        runtime_requirements,
        "_run_installer",
        lambda *args, **kwargs: pytest.fail("broken spaCy must not run installer"),
    )

    with pytest.raises(AudioRuntimePreparationError) as excinfo:
        prepare_runtime_requirement(_requirement())

    assert "spaCy runtime is not importable" in str(excinfo.value)
    assert "/private/secret/path" not in str(excinfo.value)


def test_preparer_rejects_unvalidated_requirement_kind():
    requirement = AudioRuntimeRequirement(
        kind=cast(AudioRuntimeRequirementKind, "shell"), name="payload"
    )

    with pytest.raises(AudioRuntimePreparationError, match="Unsupported"):
        prepare_runtime_requirement(requirement)


def test_inference_probe_is_check_only_when_pipeline_missing(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime_requirements,
        "spacy_pipeline_available",
        lambda name: calls.append(name) or False,
    )
    monkeypatch.setattr(
        runtime_requirements,
        "_run_installer",
        lambda *args: pytest.fail("inference must not install packages"),
    )

    ready, reason = _probe_kokoro_g2p_model()

    assert ready is False
    assert calls == [_KOKORO_G2P_SPACY_MODEL]
    assert "rapid-mlx pull kokoro" in reason


def test_inference_probe_accepts_prepared_pipeline(monkeypatch):
    monkeypatch.setattr(
        runtime_requirements, "spacy_pipeline_available", lambda name: True
    )
    assert _probe_kokoro_g2p_model() == (True, None)


def test_inference_probe_fails_closed_on_broken_spacy(monkeypatch):
    def broken(_name):
        raise ImportError("private dylib path")

    monkeypatch.setattr(runtime_requirements, "spacy_pipeline_available", broken)
    ready, reason = _probe_kokoro_g2p_model()
    assert ready is False
    assert _KOKORO_G2P_SPACY_MODEL in reason


def test_ensure_missing_pipeline_is_503_and_failure_is_not_cached(monkeypatch):
    from fastapi import HTTPException

    calls = []

    def probe_fn():
        calls.append(1)
        return (False, "run pull") if len(calls) == 1 else (True, None)

    monkeypatch.setattr(probe, "_probe_kokoro_g2p_model", probe_fn)
    with pytest.raises(HTTPException) as excinfo:
        _ensure_kokoro_g2p_model_ready()
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "run pull"

    _ensure_kokoro_g2p_model_ready()
    _ensure_kokoro_g2p_model_ready()
    assert len(calls) == 2


def test_require_kokoro_runtime_propagates_model_503(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(probe, "_ensure_kokoro_g2p_ready", lambda: None)
    monkeypatch.setattr(
        probe,
        "_ensure_kokoro_g2p_model_ready",
        lambda: (_ for _ in ()).throw(
            HTTPException(status_code=503, detail="model missing")
        ),
    )
    with pytest.raises(HTTPException) as excinfo:
        require_kokoro_runtime()
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "model missing"


def test_is_kokoro_family_covers_explicit_and_default():
    from vllm_mlx.audio.tts import is_kokoro_family_model

    assert is_kokoro_family_model("mlx-community/Kokoro-82M-bf16") is True
    assert is_kokoro_family_model("acme/MysteryTTS-v2") is True


def test_gate_matches_engine_family_for_every_registered_tts_model():
    from vllm_mlx.audio.registry import tts_aliases
    from vllm_mlx.audio.tts import TTSEngine, is_kokoro_family_model

    for alias, hf_id in tts_aliases().items():
        expect_gated = TTSEngine(hf_id)._detect_family(hf_id) == "kokoro"
        assert is_kokoro_family_model(hf_id) is expect_gated
        assert is_kokoro_family_model(alias) is expect_gated


def test_detect_family_method_matches_module_ssot():
    from vllm_mlx.audio.tts import TTSEngine, detect_tts_family

    for name in (
        "mlx-community/Kokoro-82M-bf16",
        "acme/MysteryTTS-v2",
        "mlx-community/chatterbox-turbo-fp16",
    ):
        assert TTSEngine(name)._detect_family(name) == detect_tts_family(name)


def test_dry_run_tts_contains_systemexit(monkeypatch):
    from vllm_mlx.audio import tts as tts_mod

    class _FakeEngine:
        def __init__(self, name):
            pass

        def load(self):
            pass

        def generate(self, *args, **kwargs):
            raise SystemExit("No virtual environment found")

    monkeypatch.setattr(tts_mod, "TTSEngine", _FakeEngine)
    ok, reason = probe._dry_run_tts("mlx-community/Kokoro-82M-bf16")
    assert ok is False
    assert "SystemExit" in reason


def test_voice_language_gate_classifies():
    assert _kokoro_voice_needs_en_g2p("af_heart") is True
    assert _kokoro_voice_needs_en_g2p("bm_george") is True
    for voice in (
        "jf_alpha",
        "zf_xiaobei",
        "ef_dora",
        "ff_siwis",
        "if_sara",
        "pf_dora",
    ):
        assert _kokoro_voice_needs_en_g2p(voice) is False
    assert _kokoro_voice_needs_en_g2p(None) is True
    assert _kokoro_voice_needs_en_g2p("") is True


def test_require_kokoro_runtime_skips_english_pipeline_for_other_voices(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(probe, "_ensure_kokoro_g2p_ready", lambda: None)
    calls = []
    monkeypatch.setattr(
        probe, "_ensure_kokoro_g2p_model_ready", lambda: calls.append(1)
    )

    require_kokoro_runtime("jf_alpha")
    assert calls == []
    require_kokoro_runtime("af_heart")
    assert calls == [1]


def test_installer_kills_process_group_on_timeout(monkeypatch):
    killed = {}

    class _FakeProc:
        pid = 4321
        returncode = None
        stdout = stderr = stdin = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="spacy", timeout=timeout)

        def kill(self):
            killed["direct"] = True

    popen_kwargs = {}

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: killed.__setitem__("pgid", (pgid, sig))
    )
    with pytest.raises(subprocess.TimeoutExpired):
        runtime_requirements._run_installer(["x"], {}, timeout=1)
    assert popen_kwargs["start_new_session"] is True
    assert killed["pgid"] == (4321, signal.SIGKILL)
    assert "direct" not in killed


def test_installer_timeout_falls_back_and_closes_every_pipe(monkeypatch):
    killed = {}

    class _Pipe:
        def __init__(self, *, fail=False):
            self.fail = fail
            self.closed = 0

        def close(self):
            self.closed += 1
            if self.fail:
                raise OSError("already closed")

    class _FakeProc:
        pid = 4321
        returncode = None
        stdout = _Pipe()
        stderr = _Pipe(fail=True)
        stdin = _Pipe()

        def __init__(self):
            self.communications = 0

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise subprocess.TimeoutExpired(cmd="spacy", timeout=timeout)
            return "", ""

        def kill(self):
            killed["direct"] = True

    proc = _FakeProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(PermissionError())
    )

    with pytest.raises(subprocess.TimeoutExpired):
        runtime_requirements._run_installer(["x"], {}, timeout=1)

    assert killed == {"direct": True}
    assert proc.communications == 2
    assert proc.stdout.closed == 1
    assert proc.stderr.closed == 1
    assert proc.stdin.closed == 1


def test_installer_raises_on_nonzero_exit(monkeypatch):
    class _FakeProc:
        pid = 1
        returncode = 7

        def communicate(self, timeout=None):
            return "", "boom stderr"

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _FakeProc())
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        runtime_requirements._run_installer(["x"], {}, timeout=1)
    assert excinfo.value.returncode == 7
