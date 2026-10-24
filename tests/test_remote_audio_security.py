"""Mocked transport regressions for audio and vendored decoder boundaries."""

import base64
import socket
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from rapid_mlx.models import mllm


@pytest.fixture
def inputs():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from rapid_mlx.models.mlx_vlm_vendored import inputs

    return inputs


class Response:
    is_redirect = False
    is_permanent_redirect = False

    def __init__(self, body=b"audio", headers=None):
        self.body = body
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        self.closed = True


@pytest.fixture
def transport(monkeypatch, tmp_path):
    state = {
        "requests": [],
        "dns": [],
        "addresses": ["93.184.216.34"],
        "response": Response(),
    }

    def resolve(host, port, **kwargs):
        state["dns"].append(host)
        addresses = [host] if host == "169.254.169.254" else state["addresses"]
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in addresses
        ]

    class Session:
        def mount(self, *args):
            pass

        def request(self, method, url, **kwargs):
            state["requests"].append((method, url, kwargs, self.trust_env))
            return state["response"]

        def close(self):
            state["session_closed"] = True

    monkeypatch.setattr(mllm.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(mllm.requests, "Session", Session)
    original_temp = mllm.tempfile.NamedTemporaryFile
    monkeypatch.setattr(
        mllm.tempfile,
        "NamedTemporaryFile",
        lambda **kw: original_temp(dir=tmp_path, **kw),
    )
    manager = mllm.TempFileManager()
    monkeypatch.setattr(mllm, "_temp_manager", manager)
    state["manager"] = manager
    yield state
    manager.cleanup_all()


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["::1"],
        ["93.184.216.34", "192.168.1.2"],
    ],
)
def test_audio_rejects_blocked_and_mixed_dns_before_transport(transport, addresses):
    transport["addresses"] = addresses
    with pytest.raises(mllm.RemoteMediaFetchError):
        mllm.process_audio_input("https://audio.example/clip.wav")
    assert transport["requests"] == []
    assert transport["session_closed"]


def test_audio_redirect_to_metadata_is_not_requested(transport):
    response = transport["response"]
    response.is_redirect = True
    response.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
    with pytest.raises(mllm.RemoteMediaFetchError):
        mllm.download_audio("https://audio.example/clip.wav")
    assert len(transport["requests"]) == 1
    assert response.closed
    assert transport["session_closed"]


def test_audio_uses_validated_address_and_closes_transport(transport):
    path = Path(mllm.download_audio("https://audio.example/clip.wav"))
    assert path.read_bytes() == b"audio"
    assert transport["dns"] == ["audio.example"]
    method, url, options, trust_env = transport["requests"][0]
    assert url == "https://93.184.216.34/clip.wav"
    assert options["headers"]["Host"] == "audio.example"
    assert options["allow_redirects"] is False
    assert options["verify"] is True
    assert trust_env is False
    assert transport["response"].closed and transport["session_closed"]


@pytest.mark.parametrize("headers", [{}, {"content-length": "100"}])
def test_audio_size_limit_cleans_up(transport, tmp_path, headers):
    transport["response"] = Response(body=b"12345", headers=headers)
    with pytest.raises(mllm.FileSizeExceededError):
        mllm.download_audio("https://audio.example/clip.wav", max_size=4)
    assert not list(tmp_path.iterdir())
    assert transport["response"].closed and transport["session_closed"]


def test_audio_decoder_receives_data_bytes_and_local_path(inputs, monkeypatch):
    audio_io = pytest.importorskip("mlx_audio.audio_io")
    seen = []

    def read(source, dtype):
        seen.append(source.getvalue() if isinstance(source, BytesIO) else source)
        return np.array([0.25], dtype=np.float32), 16000

    monkeypatch.setattr(audio_io, "read", read)
    inputs.load_audio(
        "data:audio/wav;base64," + base64.b64encode(b"wave").decode(), 16000
    )
    inputs.load_audio(Path("/local/audio.wav"), 16000)
    assert seen == [b"wave", "/local/audio.wav"]


def test_vendored_audio_rejects_private_url_before_decoder(
    inputs, transport, monkeypatch
):
    audio_io = pytest.importorskip("mlx_audio.audio_io")
    monkeypatch.setattr(
        audio_io, "read", lambda *a, **k: pytest.fail("decoder reached")
    )
    transport["addresses"] = ["127.0.0.1"]
    with pytest.raises(ValueError, match="Failed to load audio from URL") as exc:
        inputs.load_audio("http://localhost/audio.wav", 16000)
    assert isinstance(exc.value.__cause__, mllm.RemoteMediaFetchError)
    assert transport["requests"] == []


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("scheme", ["http", "HTTP", "https", "HTTPS"])
def test_vendored_other_decoders_reject_private_url(inputs, transport, kind, scheme):
    transport["addresses"] = ["127.0.0.1"]
    with pytest.raises(mllm.RemoteMediaFetchError):
        getattr(inputs, f"load_{kind}")(f"{scheme}://localhost/media.{kind}")
    assert transport["requests"] == []


def test_decoder_protocols_do_not_bypass_http_guard(inputs):
    with pytest.raises(mllm.RemoteMediaFetchError):
        mllm.process_audio_input("ftp://internal/audio.wav")
    with pytest.raises(ValueError, match="http"):
        inputs.load_video("rtsp://internal/video")


def test_prepare_inputs_blocks_audio_before_processor(inputs, transport):
    transport["addresses"] = ["169.254.169.254"]
    with pytest.raises(ValueError, match="Failed to load audio from URL"):
        inputs.prepare_inputs(
            object(), audio=["http://metadata/audio.wav"], prompts="transcribe"
        )
    assert transport["requests"] == []


def test_inline_audio_bound_and_array_are_preserved(inputs, monkeypatch):
    monkeypatch.setattr(mllm, "MAX_AUDIO_SIZE", 3)
    with pytest.raises(mllm.FileSizeExceededError):
        mllm.process_audio_input(
            "data:audio/wav;base64," + base64.b64encode(b"four").decode()
        )
    values = np.array([0.25, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(inputs.load_audio(values, 16000), values)


@pytest.mark.parametrize("uri", ["data:audio/wav", "data:audio/wav,AAAA"])
def test_audio_data_uri_requires_base64(uri):
    with pytest.raises(ValueError, match="base64"):
        mllm.process_audio_input(uri)


@pytest.mark.parametrize("scheme", ["data", "DATA"])
def test_inline_audio_size_checks_and_positive(monkeypatch, scheme):
    monkeypatch.setattr(mllm, "MAX_AUDIO_SIZE", 2)
    assert (
        mllm.process_audio_input(scheme + ":audio/wav;base64,YWI=").getvalue() == b"ab"
    )
    for payload in ("YWJj", "YWJjZA=="):
        with pytest.raises(mllm.FileSizeExceededError):
            mllm.process_audio_input(scheme + ":audio/wav;base64," + payload)


def test_audio_unrecognized_protocol_refused():
    with pytest.raises(mllm.RemoteMediaFetchError):
        mllm.process_audio_input("ftp://example/audio")


def test_audio_extension_default_and_local_preserved(transport):
    path = Path(mllm.download_audio("https://audio.example/no-extension"))
    assert path.suffix == ".wav"
    assert path.read_bytes() == b"audio"
    assert mllm.process_audio_input("recording.wav") == "recording.wav"


def test_audio_invalid_length_closes_response(transport):
    transport["response"].headers["content-length"] = "invalid"
    with pytest.raises(mllm.RemoteMediaFetchError, match="Content-Length"):
        mllm.download_audio("https://audio.example/clip.wav")
    assert transport["response"].closed


def test_remote_image_decodes_valid_bytes(inputs, transport, tmp_path):
    from PIL import Image

    payload = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(payload, format="PNG")
    transport["response"] = Response(payload.getvalue(), {"content-type": "image/png"})
    result = inputs.load_image("https://media.example/image.png")
    assert result.size == (2, 2)
    assert result.getpixel((0, 0)) == (255, 0, 0)
    assert not transport["manager"]._files
    assert not list(tmp_path.iterdir())


def test_remote_audio_decodes_valid_wav(inputs, transport):
    import wave

    payload = BytesIO()
    with wave.open(payload, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16)
    transport["response"] = Response(payload.getvalue())
    result = inputs.load_audio("https://media.example/audio.wav", 16000)
    assert result.shape == (16,)
    np.testing.assert_array_equal(result, np.zeros(16))


@pytest.mark.parametrize("source_kind", ["https", "file", "FILE"])
def test_remote_video_decodes_valid_frames(inputs, transport, tmp_path, source_kind):
    cv2 = pytest.importorskip("cv2")
    local = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(local), cv2.VideoWriter_fourcc(*"mp4v"), 2, (16, 16))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.release()
    transport["response"] = Response(local.read_bytes(), {"content-type": "video/mp4"})
    source = (
        "https://media.example/video.mp4"
        if source_kind == "https"
        else f"{source_kind}://{local}"
    )
    frames, metadata = inputs.load_video(
        source,
        inputs.VideoSampling(nframes=1, min_frames=1, max_frames=1, frame_factor=1),
    )
    assert len(frames) == 1
    assert metadata.total_num_frames == 2
    assert not transport["manager"]._files
    assert list(tmp_path.iterdir()) == [local]
    if source_kind == "https":
        assert transport["response"].closed
    else:
        assert transport["requests"] == []


def test_local_colon_audio_filename_is_not_a_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "http:take.wav"
    path.write_bytes(b"local")
    assert mllm.process_audio_input(path.name) == str(path)


@pytest.mark.parametrize("prefix", ["", "file://", "FILE://"])
def test_local_colon_image_filename_decodes(inputs, tmp_path, monkeypatch, prefix):
    from PIL import Image

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "http:frame.png"
    Image.new("RGB", (2, 2)).save(path)
    assert inputs.load_image(prefix + str(path) if prefix else path.name).size == (2, 2)


@pytest.mark.parametrize("prefix", ["", "file://", "FILE://"])
def test_local_colon_video_filename_reaches_local_decoder(
    inputs, tmp_path, monkeypatch, prefix
):
    cv2 = pytest.importorskip("cv2")
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "take:01.mp4"
    path.write_bytes(b"local")
    seen = []

    class Capture:
        def __init__(self, source):
            seen.append(source)

        def isOpened(self):  # noqa: N802 - OpenCV capture API
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    with pytest.raises(ValueError, match="Cannot open video"):
        inputs.load_video(prefix + str(path) if prefix else path.name)
    assert seen == [str(path)]


@pytest.mark.parametrize(
    "name", ["take.wav", "http:take.wav", "http://media.example/take.wav"]
)
@pytest.mark.parametrize("scheme", ["file", "FILE"])
def test_audio_file_uri_resolves_to_local_path(tmp_path, monkeypatch, name, scheme):
    monkeypatch.chdir(tmp_path)
    path = (tmp_path / name).resolve()
    assert mllm.process_audio_input(
        path.as_uri().replace("file:", scheme + ":", 1)
    ) == str(path)


def test_remote_audio_without_extension_preserves_bytes_and_cleans_temp(
    transport, tmp_path
):
    body = b"\x1a\x45\xdf\xa3" + b"container data"
    transport["response"] = Response(body)
    result = mllm.process_audio_input("https://media.example/download")
    assert isinstance(result, BytesIO)
    assert result.getvalue() == body
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["transport", "size", "decoder"])
def test_remote_audio_preserves_value_error_boundary(
    inputs, transport, monkeypatch, failure
):
    import requests

    audio_io = pytest.importorskip("mlx_audio.audio_io")
    error = {
        "transport": requests.ConnectionError("unavailable"),
        "size": mllm.FileSizeExceededError("too large"),
        "decoder": RuntimeError("bad audio"),
    }[failure]

    def fail(*args, **kwargs):
        raise error

    if failure == "decoder":
        monkeypatch.setattr(audio_io, "read", fail)
    else:
        monkeypatch.setattr(mllm, "process_audio_input", fail)
    with pytest.raises(ValueError, match="Failed to load audio from URL") as exc:
        inputs.load_audio("https://media.example/audio.wav", 16000)
    assert exc.value.__cause__ is error


@pytest.mark.parametrize("read_error", [False, True])
def test_remote_audio_repeated_reads_leave_no_files_or_registry_entries(
    transport, tmp_path, monkeypatch, read_error
):
    manager = mllm.TempFileManager()
    monkeypatch.setattr(mllm, "_temp_manager", manager)
    if read_error:

        def fail_read(_path):
            raise OSError("audio read failed")

        monkeypatch.setattr(Path, "read_bytes", fail_read)
    for _ in range(3):
        transport["response"] = Response()
        if read_error:
            with pytest.raises(OSError, match="audio read failed"):
                mllm.process_audio_input("https://media.example/audio.wav")
        else:
            assert (
                mllm.process_audio_input("https://media.example/audio.wav").getvalue()
                == b"audio"
            )
        assert manager._files == set()
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["read", "decode"])
def test_remote_image_failure_releases_download(
    inputs, transport, tmp_path, monkeypatch, failure
):
    if failure == "read":

        def fail_read(path):
            raise OSError("read failed")

        monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(ValueError):
        inputs.load_image("https://media.example/image.png")
    assert not transport["manager"]._files
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["constructor", "open", "release"])
def test_remote_video_failure_releases_download(
    inputs, transport, tmp_path, monkeypatch, failure
):
    cv2 = pytest.importorskip("cv2")

    class Capture:
        def __init__(self, source):
            if failure == "constructor":
                raise RuntimeError("constructor failed")

        def isOpened(self):  # noqa: N802 - OpenCV API
            return False

        def release(self):
            if failure == "release":
                raise RuntimeError("release failed")

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    with pytest.raises((RuntimeError, ValueError)):
        inputs.load_video("https://media.example/video.mp4")
    assert not transport["manager"]._files
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("scheme", ["file", "FILE"])
def test_image_file_uri_decodes_escaped_local_path(inputs, tmp_path, scheme):
    from PIL import Image

    path = tmp_path / "local image.png"
    Image.new("RGB", (2, 2), color="red").save(path)
    uri = path.as_uri().replace("file:", scheme + ":", 1)
    assert "%20" in uri
    result = inputs.load_image(uri)
    assert result.getpixel((0, 0)) == (255, 0, 0)
    assert path.is_file()


@pytest.mark.parametrize("authority", ["", "localhost", "LOCALHOST"])
@pytest.mark.parametrize("kind", ["audio", "video", "image"])
def test_media_file_uri_escaped_paths(inputs, tmp_path, monkeypatch, authority, kind):
    path = tmp_path / "local clip.wav"
    path.write_bytes(b"local")
    uri = (
        "FILE://" + authority + str(path).replace(" ", "%20") + "?ignored=yes#fragment"
    )
    if kind == "audio":
        assert mllm.process_audio_input(uri) == str(path)
    elif kind == "image":
        from PIL import Image

        Image.new("RGB", (2, 2)).save(path, format="PNG")
        assert inputs.load_image(uri).size == (2, 2)
    else:
        cv2 = pytest.importorskip("cv2")
        seen = []

        class Capture:
            def __init__(self, source):
                seen.append(source)

            def isOpened(self):  # noqa: N802 - OpenCV API
                return False

            def release(self):
                pass

        monkeypatch.setattr(cv2, "VideoCapture", Capture)
        with pytest.raises(ValueError, match="Cannot open video"):
            inputs.load_video(uri)
        assert seen == [str(path)]
    assert path.is_file()


@pytest.mark.parametrize("kind", ["audio", "video", "image"])
def test_media_file_uri_rejects_remote_authority(inputs, kind):
    with pytest.raises(ValueError, match="authority"):
        if kind == "audio":
            mllm.process_audio_input("file://remote.example/share/media.wav")
        else:
            getattr(inputs, "load_" + kind)("file://remote.example/share/media")
