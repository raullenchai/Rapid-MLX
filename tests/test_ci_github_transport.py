"""Fresh evidence reads over a reused, authenticated, repository-bound connection."""

import json
from urllib.parse import parse_qsl, urlsplit

import pytest

from scripts import ci_candidate_admission as admission
from scripts import ci_github_transport as transport
from scripts import queue_tree_evidence as evidence
from tests.test_ci_candidate_consumer import setup
from tests.test_queue_tree_evidence import REPO


class Response:
    def __init__(self, value=None, link="", status=200, raw=None):
        self.status = status
        self.raw = json.dumps(value).encode() if raw is None else raw
        self.link = link

    def read(self, maximum):
        return self.raw[:maximum]

    def getheader(self, key, default):
        return self.link or default


class Connection:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False

    def request(self, method, path, headers):
        self.requests.append((method, path, dict(headers)))

    def getresponse(self):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def client(monkeypatch, responses):
    connection = Connection(responses)
    created = []

    def create(host, timeout):
        created.append((host, timeout))
        return connection

    monkeypatch.setattr(transport.http.client, "HTTPSConnection", create)
    return transport.PersistentGitHubClient(REPO, "fake-token"), connection, created


def test_same_connection_reads_every_response_and_closes(monkeypatch):
    c, conn, created = client(monkeypatch, [Response(1), Response(2)])
    with c:
        assert c.json(f"repos/{REPO}/git/ref/heads/main") == 1
        assert c.json(f"repos/{REPO}/git/ref/heads/main") == 2
    assert created == [("api.github.com", 12)] and conn.closed
    assert len(conn.requests) == 2
    assert all(r[0] == "GET" for r in conn.requests)
    assert conn.requests[0][2]["Authorization"] == "Bearer fake-token"


@pytest.mark.parametrize("alias", [f"repos/{REPO}", "repositories/42"])
def test_all_pages_use_original_repository_and_preserve_filters(monkeypatch, alias):
    path = f"repos/{REPO}/actions/runs/20/jobs"
    link = f'<https://api.github.com/{alias}/actions/runs/20/jobs?page=2>; rel="next"'
    c, conn, _ = client(
        monkeypatch, [Response({"jobs": [1]}, link), Response({"jobs": [2]})]
    )
    assert c.json(path, "filter=all", "per_page=100", paginate=True) == [
        {"jobs": [1]},
        {"jobs": [2]},
    ]
    assert conn.requests[1][1] == f"/{path}?filter=all&per_page=100&page=2"


@pytest.mark.parametrize(
    "next_url",
    [
        "http://api.github.com/repos/owner/repo/actions/runs?page=2",
        "https://evil.example/repos/owner/repo/actions/runs?page=2",
        "https://api.github.com:443/repos/owner/repo/actions/runs?page=2",
        "https://token@api.github.com/repos/owner/repo/actions/runs?page=2",
        "https://api.github.com/repos/other/repo/actions/runs?page=2",
        "https://api.github.com/repos/owner/repo/other?page=2",
        "https://api.github.com/repos/owner/repo/actions/runs?page=1",
        "https://api.github.com/repos/owner/repo/actions/runs?page=101",
        "https://api.github.com/repos/owner/repo/actions/runs?page=2&page=3",
        "https://api.github.com/repos/owner/repo/actions/runs?page=2#secret",
        "https://api.github.com/repos/owner/repo/actions/runs",
        "https://api.github.com/repos/owner/repo/actions/runs?page=two",
        "https://[invalid",
    ],
)
def test_untrusted_next_page_rejects_without_second_authenticated_request(
    monkeypatch, next_url
):
    c, conn, _ = client(monkeypatch, [Response({}, f'<{next_url}>; rel="next"')])
    with pytest.raises(evidence.EvidenceError), c:
        c.json(f"repos/{REPO}/actions/runs", paginate=True)
    assert len(conn.requests) == 1 and conn.closed


@pytest.mark.parametrize(
    "response",
    [
        Response(status=302),
        Response(status=401),
        Response(status=429),
        Response(raw=b"not-json"),
        Response(raw=b"x" * 16_000_001),
        OSError("network failed"),
        transport.http.client.HTTPException("closed"),
    ],
)
def test_transport_errors_fail_closed_no_retry_or_secret_in_error(
    monkeypatch, response
):
    c, conn, _ = client(monkeypatch, [response])
    with pytest.raises(evidence.EvidenceError) as error, c:
        c.json(f"repos/{REPO}/actions/runs")
    assert len(conn.requests) == 1 and conn.closed
    assert "fake-token" not in str(error.value)


@pytest.mark.parametrize(
    "endpoint,fields",
    [
        ("https://evil.example", ()),
        ("repos/other/repo/actions/runs", ()),
        (f"repos/{REPO}/../private", ()),
        (f"repos/{REPO}/actions/runs?token=x", ()),
        (f"repos/{REPO}/actions/runs#x", ()),
        (f"repos/{REPO}/actions\\runs", ()),
        (f"repos/{REPO}/actions/runs", ("page=2",)),
        (f"repos/{REPO}/actions/runs", ("missing",)),
        (f"repos/{REPO}/actions/runs", ("=value",)),
    ],
)
def test_invalid_paths_queries_make_no_request(monkeypatch, endpoint, fields):
    c, conn, _ = client(monkeypatch, [])
    with pytest.raises(evidence.EvidenceError), c:
        c.json(endpoint, *fields)
    assert not conn.requests


@pytest.mark.parametrize("token", ["", "token\nheader", "token space", "秘密"])
def test_invalid_token_never_opens_connection(monkeypatch, token):
    monkeypatch.setattr(
        transport.http.client, "HTTPSConnection", lambda *a, **k: pytest.fail("created")
    )
    with pytest.raises(evidence.EvidenceError):
        transport.PersistentGitHubClient(REPO, token)


@pytest.mark.parametrize("change", [None, "main", "index", "attempt", "closed"])
def test_real_zip_and_full_contract_remain_live_on_persistent_transport(
    monkeypatch, change
):
    raw, _, status, producer, _ = setup(monkeypatch)
    reads = []
    response = None

    class FixtureConnection(Connection):
        def request(self, method, path, headers):
            nonlocal response
            parsed = urlsplit(path)
            endpoint = parsed.path.lstrip("/")
            fields = tuple(f"{k}={v}" for k, v in parse_qsl(parsed.query))
            reads.append(endpoint)
            if endpoint.endswith("/jobs"):
                value = {"jobs": raw.job_records[int(endpoint.split("/")[-2])]}
            elif "/git/commits/" in endpoint:
                value = raw.commit(endpoint.split("/")[-1])
            else:
                value = raw.json(endpoint, *fields)
            response = Response(value)

        def getresponse(self):
            return response

    connection = FixtureConnection([])
    monkeypatch.setattr(
        transport.http.client, "HTTPSConnection", lambda *a, **k: connection
    )
    c = transport.PersistentGitHubClient(REPO, "fake-token")
    first = admission.verify_admission(c, 100)
    assert first["verified"]
    if change == "main":
        raw.responses[f"repos/{REPO}/git/ref/heads/main"]["object"]["sha"] = "e" * 40
    elif change == "index":
        status["state"] = "failure"
    elif change == "attempt":
        producer["run_attempt"] = 2
    elif change == "closed":
        raw.responses[f"repos/{REPO}/pulls"] = []
    assert admission.verify_admission(c, 100)["verified"] is (change is None)
    assert reads.count(f"repos/{REPO}/git/ref/heads/main") >= 4
    c.__exit__()
    assert connection.closed


def test_ambiguous_pages_and_malformed_job_records_reject(monkeypatch):
    url = f"https://api.github.com/repos/{REPO}/actions/runs/20/jobs?page=2"
    c, _, _ = client(
        monkeypatch, [Response({}, f'<{url}>; rel="next", <{url}>; rel="next"')]
    )
    with pytest.raises(evidence.EvidenceError, match="ambiguous"), c:
        c.jobs(20)
    c, _, _ = client(monkeypatch, [Response({"jobs": [1]})])
    with pytest.raises(evidence.EvidenceError, match="malformed job"), c:
        c.jobs(20)


def test_completed_bad_proof_is_terminal_not_a_readiness_retry(monkeypatch):
    raw, _, _, _, _ = setup(monkeypatch)
    raw.responses[f"repos/{REPO}/git/ref/heads/main"]["object"]["sha"] = "e" * 40
    monkeypatch.setattr(
        admission.time, "sleep", lambda *a: pytest.fail("stale proof retried")
    )
    result = admission.verify_source_admission(raw, 20, 1)
    assert not result["verified"] and "base is stale" in result["reason"]
