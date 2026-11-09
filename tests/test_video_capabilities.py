from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.routes import video


@pytest.mark.parametrize(
    "alias,engine",
    [
        (
            "cogvideox-fun-5b-q4",
            SimpleNamespace(video_family="cogvideox-fun"),
        ),
        (
            "dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q8",
            SimpleNamespace(video_family="cogvideox-fun"),
        ),
        (
            "dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx",
            SimpleNamespace(video_family="cogvideox-fun"),
        ),
        (
            "wan2.2-ti2v-5b-q8",
            SimpleNamespace(
                video_family="wan",
                native_fps=24,
                _wan_engine=SimpleNamespace(model_type="ti2v", max_area=None),
            ),
        ),
        (
            "rickylin20260522/Wan2.2-TI2V-5B-mlx",
            SimpleNamespace(
                video_family="wan",
                native_fps=24,
                _wan_engine=SimpleNamespace(model_type="ti2v", max_area=None),
            ),
        ),
        (
            "wan2.2-i2v-a14b-q8",
            SimpleNamespace(
                video_family="wan",
                native_fps=24,
                _wan_engine=SimpleNamespace(model_type="i2v", max_area=None),
            ),
        ),
        (
            "wan2.2-t2v-a14b-bf16",
            SimpleNamespace(
                video_family="wan",
                native_fps=24,
                _wan_engine=SimpleNamespace(model_type="t2v", max_area=None),
            ),
        ),
        ("ltx-2.3-mlx-q4", SimpleNamespace(video_family="ltx-2.3")),
        ("ltx-2.5-mlx-q8", SimpleNamespace(video_family="ltx-2.5")),
    ],
)
def test_builtin_pre_serve_modes_match_live_capabilities(alias, engine) -> None:
    profile = resolve_profile(alias)
    assert profile is not None
    engine.model_name = profile.hf_path

    assert list(profile.video_modes or ()) == video._video_capabilities(engine)["modes"]


@pytest.mark.asyncio
async def test_ltx_capabilities_expose_combined_workload_budget(monkeypatch) -> None:
    engine = SimpleNamespace(
        model_name="notapalindrome/ltx23-mlx-av-q4",
        video_family="ltx-2.3",
        native_fps=24,
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert body["cancellation"] == {"queued": True, "in_progress": True}
    assert body["object"] == "video.capabilities"
    assert body["model"] == engine.model_name
    assert body["modality"] == "video-gen"
    assert body["modes"] == ["text-to-video", "image-to-video"]
    assert body["limits"]["seconds"]["maximum"] == 20
    assert body["limits"]["fps"] == {
        "minimum": 1,
        "maximum": 60,
        "default": 24,
        "fixed": False,
    }
    assert body["limits"]["frames"]["step"] == 8
    assert body["limits"]["workload"]["maximum"] == 768 * 512 * 97
    assert body["controls"]["conditioning_strength"]["maximum"] == 1.0


@pytest.mark.asyncio
async def test_cogvideox_capabilities_report_fixed_mvp_shape(monkeypatch) -> None:
    engine = SimpleNamespace(
        model_name="dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4",
        video_family="cogvideox-fun",
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert body["modes"] == ["text-to-video"]
    assert body["limits"]["size"] == {"type": "fixed", "values": ["672x384"]}
    assert body["limits"]["seconds"]["maximum"] == 1
    assert body["limits"]["fps"]["default"] == 5
    assert body["limits"]["workload"]["dimension_rounding"] == "none"
    assert body["controls"]["conditioning_strength"] is None


@pytest.mark.asyncio
async def test_wan_capabilities_include_registered_community_benchmark_size(
    monkeypatch,
) -> None:
    engine = SimpleNamespace(
        model_name="Anes1032/Wan2.2-TI2V-5B-mlx-q8",
        video_family="wan",
        native_fps=24,
        _wan_engine=SimpleNamespace(model_type="ti2v", max_area=832 * 512),
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert "832x480" in body["limits"]["size"]["also_supported"]


@pytest.mark.asyncio
async def test_wan_capabilities_require_aligned_canvas_area_for_benchmark_size(
    monkeypatch,
) -> None:
    engine = SimpleNamespace(
        model_name="Anes1032/Wan2.2-TI2V-5B-mlx-q8",
        video_family="wan",
        native_fps=24,
        _wan_engine=SimpleNamespace(model_type="ti2v", max_area=832 * 512 - 1),
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert "832x480" not in body["limits"]["size"]["also_supported"]


@pytest.mark.asyncio
async def test_wan_capabilities_use_checkpoint_limits(monkeypatch) -> None:
    engine = SimpleNamespace(
        model_name="Anes1032/Wan2.2-TI2V-5B-mlx-q8",
        video_family="wan",
        native_fps=24,
        _wan_engine=SimpleNamespace(model_type="i2v", max_area=901_120),
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert body["modes"] == ["image-to-video"]
    assert body["limits"]["size"]["maximum_area"] == 901_120
    assert body["limits"]["size"]["also_supported"] == ["832x480"]
    assert body["limits"]["fps"] == {
        "minimum": 24,
        "maximum": 24,
        "default": 24,
        "fixed": True,
    }
    assert body["limits"]["frames"]["step"] == 4


@pytest.mark.asyncio
async def test_wan_only_lists_openai_sizes_accepted_after_alignment(
    monkeypatch,
) -> None:
    engine = SimpleNamespace(
        model_name="custom/wan",
        video_family="wan",
        native_fps=16,
        _wan_engine=SimpleNamespace(model_type="t2v", max_area=1280 * 768),
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert body["limits"]["size"]["also_supported"] == [
        "1280x720",
        "720x1280",
        "832x480",
    ]


def test_capabilities_route_precedes_dynamic_video_id_route() -> None:
    routes = [route.path for route in video.router.routes]
    assert routes.index("/v1/videos/capabilities") < routes.index(
        "/v1/videos/{video_id}"
    )


@pytest.mark.parametrize(
    "available",
    [
        frozenset(),
        frozenset({"video"}),
        frozenset({"audio"}),
        frozenset({"video", "audio"}),
    ],
)
def test_ref2va_only_advertises_available_decoders(monkeypatch, available):
    monkeypatch.setattr(video, "_reference_decoder_kinds", lambda: available)
    engine = SimpleNamespace(
        model_name="h3",
        video_family="minimax-h3",
        native_fps=24,
        _minimax_engine=SimpleNamespace(supports_references=True),
    )
    limits = video._video_capabilities(engine)["limits"]["references"]
    assert "image" in limits
    for kind in ("video", "audio"):
        assert (kind in limits) == (kind in available)


@pytest.mark.parametrize(
    "missing", ["ffprobe", "ffmpeg", "cv2", "miniaudio", "mlx_audio.audio_io", None]
)
def test_reference_decoder_dependencies(monkeypatch, missing):
    from types import SimpleNamespace

    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=" D.V.L. h264 H.264\n DEA.L. aac AAC (decoders: aac_at)\n"
        ),
    )
    monkeypatch.setenv("FFMPEG_BINARY", "/output-only/ffmpeg")
    monkeypatch.setattr(
        video.shutil, "which", lambda name: None if name == missing else "/bin/" + name
    )

    def import_module(name):
        if name == missing:
            raise ImportError(name)

    monkeypatch.setattr(video.importlib, "import_module", import_module)
    video._ffmpeg_decodable_codecs.cache_clear()
    video._reference_decoder_kinds.cache_clear()
    try:
        actual = video._reference_decoder_kinds()
        expected = {"video", "audio"}
        if missing in {"ffprobe", "ffmpeg"}:
            expected.clear()
        elif missing in {"miniaudio", "mlx_audio.audio_io"}:
            expected.remove("audio")
        assert actual == expected
    finally:
        video._ffmpeg_decodable_codecs.cache_clear()
        video._reference_decoder_kinds.cache_clear()


@pytest.mark.parametrize(
    "codecs,expected",
    [
        (" D.V.L. h264 H264\n", {"video"}),
        (" DEAIL. aac AAC\n", {"audio"}),
        (
            " DEA.L. aac  AAC (Advanced Audio Coding) (decoders: aac_at)\n",
            {"audio"},
        ),
        (" .EV.L. h264 H264\n .EA.L. aac AAC\n", set()),
        (" D.V.L. vp9 VP9\n D.A.L. mp3 MP3 (decoders: mp3float)\n", set()),
        (" D..... = Decoding supported\n", set()),
        (None, set()),
    ],
)
def test_reference_decoder_codec_inventory(monkeypatch, codecs, expected):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(video.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(video.importlib, "import_module", lambda name: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command == ["/bin/ffmpeg", "-hide_banner", "-codecs"]
        assert kwargs == dict(capture_output=True, text=True, timeout=5, check=True)
        if codecs is None:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=codecs)

    monkeypatch.setattr(video.subprocess, "run", run)
    video._ffmpeg_decodable_codecs.cache_clear()
    video._reference_decoder_kinds.cache_clear()
    try:
        assert video._reference_decoder_kinds() == expected
        assert video._reference_decoder_kinds() == expected
        assert "=" not in video._ffmpeg_decodable_codecs()
        assert len(calls) == 1
    finally:
        video._ffmpeg_decodable_codecs.cache_clear()
        video._reference_decoder_kinds.cache_clear()


@pytest.mark.parametrize(
    "failure", [OSError("missing"), subprocess.TimeoutExpired("ffmpeg", 5)]
)
def test_codec_inventory_fails_closed(monkeypatch, failure):
    monkeypatch.setattr(video.shutil, "which", lambda name: "/bin/" + name)

    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(video.subprocess, "run", run)
    video._ffmpeg_decodable_codecs.cache_clear()
    try:
        assert video._ffmpeg_decodable_codecs() == frozenset()
    finally:
        video._ffmpeg_decodable_codecs.cache_clear()
