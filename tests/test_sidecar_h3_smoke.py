"""Keep the post-prune sidecar smoke aligned with the H3 adapters' imports."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
BUILD = ROOT / "apps/rapid-mac/scripts/build-sidecar.sh"


def video_smoke_tree():
    script = BUILD.read_text()
    video_block = script.split('    VIDEO_OUT="$(')[1].split("' 2>&1)")[0]
    source = video_block.split("'import importlib.metadata", 1)[1]
    return ast.parse("import importlib.metadata" + source)


def imports(tree):
    return {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def test_video_smoke_covers_h3_runtime_imports_after_pruning():
    required = set()
    for name in ("minimax_h3", "pika_h3_flex"):
        tree = ast.parse((ROOT / f"rapid_mlx/video/{name}.py").read_text())
        required.update(
            (module, symbol)
            for module, symbol in imports(tree)
            if module and module.startswith("mlx_vlm")
        )
    required.update(
        {
            ("safetensors", "safe_open"),
            ("huggingface_hub", "try_to_load_from_cache"),
            ("rapid_mlx.video.minimax_h3", "MiniMaxH3VideoEngine"),
            ("rapid_mlx.video.pika_h3_flex", "PikaH3FlexVideoEngine"),
        }
    )
    assert required <= imports(video_smoke_tree())


def test_video_smoke_exercises_synchronized_audio_encoding():
    calls = [
        node
        for node in ast.walk(video_smoke_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "encode_rgb_video"
    ]
    assert any(
        {keyword.arg for keyword in call.keywords} >= {"audio", "sampling_rate"}
        for call in calls
    )


def test_reference_media_bundle_contract():
    script = BUILD.read_text()
    config = script.split("./configure", 1)[1].split("if ! MACOSX", 1)[0]
    assert "--enable-ffprobe" in config
    assert all(
        flag not in config
        for flag in (
            "--enable-gpl",
            "--enable-nonfree",
            "--enable-version3",
            "--disable-swresample",
        )
    )
    decoders = config.split("--enable-decoder=", 1)[1].split()[0].split(",")
    assert "aac_at" in decoders
    assert "aac" not in decoders
    assert "ffmpeg ffprobe" in script
    assert "s/^configuration: //p" in script
    assert 'ln -s ../bin/ffmpeg "$STAGE/media-bin/ffmpeg"' in script
    assert 'ln -s ../bin/ffprobe "$STAGE/media-bin/ffprobe"' in script
    block = script.split('    VIDEO_OUT="$(')[1]
    assert 'PATH="$STAGE/media-bin:/usr/bin:/bin"' in block
    assert "decode_video_with_ffmpeg(reference_video)" in block
    assert "h3_processing.decode_audio(reference)" in block
    assert "h3_processing.decode_video_soundtrack(audio_output)" in block
    shim = (BUILD.parent / "sidecar-shim.sh").read_text()
    assert 'export PATH="$ROOT/media-bin:$PATH"' in shim
    assert 'export FFPROBE_BINARY="$ROOT/bin/ffprobe"' in shim
