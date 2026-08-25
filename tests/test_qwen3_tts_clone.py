# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS **Base** zero-shot voice cloning (ref_audio + ref_text).

Hermetic: a fake mlx_audio model is injected at ``TTSEngine.model``. We assert
the engine forwards ``ref_audio``/``ref_text`` to the ``qwen3_tts`` generate
call ONLY when supplied — so one family serves both CustomVoice (named speaker)
and Base (cloning) without cross-contaminating the call surface.
"""

import types

import numpy as np

from vllm_mlx.audio.registry import resolve_audio_alias
from vllm_mlx.audio.tts import TTSEngine

BASE_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
_UNSET = object()


class _CloneCapturingModel:
    """Fake Qwen3-TTS Base whose ``generate`` accepts the cloning surface.
    Explicit signature (no ``**kwargs``) so an unexpected keyword raises here;
    sentinel defaults record ref_audio/ref_text ONLY when actually passed.

    ``max_tokens`` mirrors the real backend (``max_tokens: int = 4096``). Qwen3
    decodes its whole waveform before yielding, so #2305 bounds it with a token
    budget; a double that rejected the kwarg would make the request fail
    closed. Recorded apart from ``rec`` so the cloning assertions stay about
    the cloning kwargs."""

    def __init__(self):
        self.calls: list[dict] = []
        self.token_budgets: list[int] = []
        # See _CapturingModel in test_qwen3_tts_audio: the engine measures the
        # stride off the loaded model before it will generate.
        self.sample_rate = 24000
        self.speech_tokenizer = types.SimpleNamespace(decode_upsample_rate=1920)

    def generate(
        self,
        *,
        text,
        voice=None,
        speed=1.0,
        lang_code=None,
        instruct=_UNSET,
        ref_audio=_UNSET,
        ref_text=_UNSET,
        max_tokens=4096,
    ):
        self.token_budgets.append(max_tokens)
        rec = {"text": text, "voice": voice, "lang_code": lang_code}
        if instruct is not _UNSET:
            rec["instruct"] = instruct
        if ref_audio is not _UNSET:
            rec["ref_audio"] = ref_audio
        if ref_text is not _UNSET:
            rec["ref_text"] = ref_text
        self.calls.append(rec)
        return iter(
            [
                types.SimpleNamespace(
                    audio=np.zeros(240, dtype=np.float32), sample_rate=24000
                )
            ]
        )


def _clone_engine():
    engine = TTSEngine(BASE_BF16)
    engine.model = _CloneCapturingModel()
    engine._loaded = True
    return engine


def test_clone_alias_points_at_base_variant():
    entry = resolve_audio_alias("qwen3-tts-clone")
    assert entry is not None
    assert entry.type == "tts"
    assert entry.family == "qwen3_tts"
    assert entry.hf_id == BASE_BF16


def test_base_repo_detected_as_qwen3_family():
    assert _clone_engine()._model_family == "qwen3_tts"


def test_ref_audio_and_text_forwarded_when_given():
    eng = _clone_engine()
    eng.generate("你好世界", ref_audio="narrator_ref.wav", ref_text="这是参考音频")
    call = eng.model.calls[0]
    assert call["ref_audio"] == "narrator_ref.wav"
    assert call["ref_text"] == "这是参考音频"
    assert call["lang_code"] == "auto"  # qwen3 auto-detects language


def test_no_cloning_kwargs_without_ref():
    eng = _clone_engine()
    eng.generate("你好世界", voice="Serena")
    call = eng.model.calls[0]
    assert "ref_audio" not in call
    assert "ref_text" not in call


def test_ref_text_omitted_when_only_ref_audio():
    eng = _clone_engine()
    eng.generate("你好世界", ref_audio="narrator_ref.wav")
    call = eng.model.calls[0]
    assert call["ref_audio"] == "narrator_ref.wav"
    assert "ref_text" not in call
