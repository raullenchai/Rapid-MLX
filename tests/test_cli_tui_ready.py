import json
import urllib.error

from vllm_mlx.cli import _wait_for_server_ready


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_wait_for_server_ready_waits_until_model_loaded(monkeypatch):
    responses = [
        urllib.error.URLError("not listening"),
        {"status": "healthy", "model_loaded": False},
        {"status": "healthy", "model_loaded": True},
    ]
    sleeps = []

    def fake_urlopen(url, timeout):
        next_response = responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return _FakeResponse(next_response)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    _wait_for_server_ready("http://127.0.0.1:8010", timeout_s=5)

    assert sleeps == [0.25, 0.25]
    assert responses == []
