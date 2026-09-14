from pathlib import Path

ROOT = Path(__file__).parents[1]
SIDECAR_BUILD = ROOT / "apps" / "rapid-mac" / "scripts" / "build-sidecar.sh"


def test_audio_pruning_keeps_qwen3_transitive_codec_closure() -> None:
    script = SIDECAR_BUILD.read_text()

    assert "-not -name qwen3_tts -not -name chatterbox -not -name __pycache__" in script
    assert "from mlx_audio.tts.models.qwen3_tts import Model" in script
    assert "mlx-audio>=0.5.3,<0.6" in (ROOT / "pyproject.toml").read_text()
    assert (
        "mlx-audio==0.5.3"
        in (
            ROOT / "apps" / "rapid-mac" / "scripts" / "sidecar-constraints.txt"
        ).read_text()
    )
