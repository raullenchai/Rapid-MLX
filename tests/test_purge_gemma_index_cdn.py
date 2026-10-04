import hashlib
import io
import json
import urllib.error

import pytest

from scripts import purge_gemma_index_cdn as purge


class Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _index_bytes() -> bytes:
    # Pad a valid JSON string to the exact production length. JSON permits
    # trailing whitespace, and the checksum constant is patched per test.
    raw = json.dumps({"metadata": {"total_size": purge.EXPECTED_TOTAL_SIZE}}).encode()
    return raw + b" " * (purge.EXPECTED_SIZE - len(raw))


def test_scope_is_verified_before_exact_hardcoded_purge(monkeypatch) -> None:
    calls = []
    index = _index_bytes()
    monkeypatch.setattr(purge, "EXPECTED_SHA256", hashlib.sha256(index).hexdigest())

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.get_method(), request.data))
        if request.full_url.endswith("/user/tokens/verify"):
            return Response(b'{"success":true,"result":{"status":"active"}}')
        if request.full_url.endswith("/zones/zone-id"):
            return Response(
                b'{"success":true,"result":{"name":"rapidmlx.com","status":"active"}}'
            )
        if request.full_url.endswith("/zones/zone-id/purge_cache"):
            return Response(b'{"success":true,"result":{"id":"purge-id"}}')
        assert request.full_url in purge.PURGE_URLS
        return Response(index)

    monkeypatch.setattr(purge.urllib.request, "urlopen", fake_urlopen)
    purge.verify_scope("token", "zone-id")
    purge.purge("token", "zone-id")
    purge.verify_public_bytes()

    assert calls[2][1] == "POST"
    assert json.loads(calls[2][2]) == {"files": list(purge.PURGE_URLS)}
    assert [call[0] for call in calls[3:]] == list(purge.PURGE_URLS)


def test_wrong_zone_fails_before_purge(monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/user/tokens/verify"):
            return Response(b'{"success":true,"result":{"status":"active"}}')
        return Response(
            b'{"success":true,"result":{"name":"example.com","status":"active"}}'
        )

    monkeypatch.setattr(purge.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(purge.OperationError, match="not rapidmlx.com"):
        purge.verify_scope("token", "zone-id")
    assert not any(url.endswith("/purge_cache") for url in calls)


def test_purge_rejection_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        purge.urllib.request,
        "urlopen",
        lambda request, timeout: Response(
            b'{"success":false,"errors":[{"message":"forbidden"}]}'
        ),
    )
    with pytest.raises(purge.OperationError):
        purge.purge("token", "zone-id")


def test_auth_http_error_fails_closed_without_followup(monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr(purge.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(purge.OperationError, match="request failed"):
        purge.verify_scope("token", "zone-id")
    assert len(calls) == 1


def test_wrong_public_bytes_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        purge.urllib.request, "urlopen", lambda request, timeout: Response(b"stale")
    )
    with pytest.raises(purge.OperationError, match="is 5 bytes"):
        purge.verify_public_bytes()
