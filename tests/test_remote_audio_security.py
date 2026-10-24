"""Mocked transport regressions for audio and vendored decoder boundaries."""

import base64
import socket
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="Vendored audio inputs require MLX")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.models import mllm
from rapid_mlx.models.mlx_vlm_vendored import inputs


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
    return state


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


def test_audio_decoder_receives_data_bytes_and_local_path(monkeypatch):
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


def test_vendored_audio_rejects_private_url_before_decoder(transport, monkeypatch):
    audio_io = pytest.importorskip("mlx_audio.audio_io")
    monkeypatch.setattr(
        audio_io, "read", lambda *a, **k: pytest.fail("decoder reached")
    )
    transport["addresses"] = ["127.0.0.1"]
    with pytest.raises(mllm.RemoteMediaFetchError):
        inputs.load_audio("http://localhost/audio.wav", 16000)
    assert transport["requests"] == []


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("scheme", ["http", "HTTP"])
def test_vendored_other_decoders_reject_private_url(transport, kind, scheme):
    transport["addresses"] = ["127.0.0.1"]
    with pytest.raises(mllm.RemoteMediaFetchError):
        getattr(inputs, f"load_{kind}")(f"{scheme}://localhost/media.{kind}")
    assert transport["requests"] == []


def test_decoder_protocols_do_not_bypass_http_guard():
    with pytest.raises(mllm.RemoteMediaFetchError):
        mllm.process_audio_input("ftp://internal/audio.wav")
    with pytest.raises(ValueError, match="http"):
        inputs.load_video("rtsp://internal/video")


def test_prepare_inputs_blocks_audio_before_processor(transport):
    transport["addresses"] = ["169.254.169.254"]
    with pytest.raises(mllm.RemoteMediaFetchError):
        inputs.prepare_inputs(
            object(), audio=["http://metadata/audio.wav"], prompts="transcribe"
        )
    assert transport["requests"] == []


def test_inline_audio_bound_and_array_are_preserved(monkeypatch):
    monkeypatch.setattr(mllm, "MAX_AUDIO_SIZE", 3)
    with pytest.raises(mllm.FileSizeExceededError):
        mllm.process_audio_input(
            "data:audio/wav;base64," + base64.b64encode(b"four").decode()
        )
    values = np.array([0.25, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(inputs.load_audio(values, 16000), values)
