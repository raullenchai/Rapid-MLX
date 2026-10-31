# SPDX-License-Identifier: Apache-2.0
"""Audio ready-banner examples produce usable requests."""

import base64
import json
import os
import re
import subprocess

import pytest

import rapid_mlx.server as server
from rapid_mlx.api.models import AudioSpeechRequest
from rapid_mlx.config import get_config


def banner(monkeypatch, capsys, model, served_alias=None):
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", "127.0.0.1")
    monkeypatch.setattr(cfg, "bind_port", 18732)
    monkeypatch.setattr(cfg, "model_alias", served_alias or model)
    monkeypatch.setattr(cfg, "model_name", model)
    monkeypatch.setattr(server, "_served_model_name_set", False)
    server.print_ready_banner()
    return capsys.readouterr().out


def command(output, label):
    return re.search(rf"^\s*{label}\s{{2,}}(.+)$", output, re.MULTILINE).group(1)


@pytest.mark.parametrize(
    ("model", "served_alias"),
    [
        (model, alias)
        for model in ("indextts", "qwen3-tts-clone")
        for alias in (None, 'indextts "alias" $HOME `false` \\')
    ]
    + [("mlx-community/IndexTTS-1.5", "gateway-tts")],
)
@pytest.mark.parametrize("large_audio", [False, True], ids=["small", "large"])
def test_reference_speech_commands(
    monkeypatch, capsys, tmp_path, model, served_alias, large_audio
):
    output = banner(monkeypatch, capsys, model, served_alias)
    audio = os.urandom(3 * 1024 * 1024) if large_audio else b"reference audio\x00\xff"
    (tmp_path / "reference.wav").write_bytes(audio)
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        'exec "' + os.sys.executable + '" -c '
        '\'import json,sys; open("curl-args.json", "w").write(json.dumps(sys.argv[1:])); '
        'open("curl-stdin.json", "w").write(sys.stdin.read() if "@-" in sys.argv else "")\' "$@"\n'
    )
    curl.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    subprocess.run(
        ["bash", "-c", command(output, "Reference") + "\n" + command(output, "Speak")],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    argv = json.loads((tmp_path / "curl-args.json").read_text())
    assert argv[argv.index("--data-binary") + 1] == "@-"
    payload = json.loads((tmp_path / "curl-stdin.json").read_text())
    assert payload["model"] == (served_alias or model)
    assert base64.b64decode(payload["ref_audio"], validate=True) == audio
    if model == "qwen3-tts-clone" or served_alias == "gateway-tts":
        assert payload["ref_text"].strip()
    else:
        assert "ref_text" not in payload
    AudioSpeechRequest(**payload)
    assert "--fail" in argv


def test_kokoro_speech_is_bare(monkeypatch, capsys):
    speak = command(banner(monkeypatch, capsys, "kokoro"), "Speak")
    assert "--fail" in speak
    assert "ref_audio" not in speak


@pytest.mark.parametrize("model", ["qwen3-aligner", "qwen3-forced-aligner"])
def test_aligner_command(monkeypatch, capsys, model):
    output = banner(monkeypatch, capsys, model)
    align = command(output, "Align")
    assert f"model={model}" in align
    assert "text=Exact transcript of audio.wav" in align
    assert "Transcribe" not in output


def test_whisper_transcription_command(monkeypatch, capsys):
    output = banner(monkeypatch, capsys, "whisper-base")
    assert "text=" not in command(output, "Transcribe")
    assert "Align" not in output
