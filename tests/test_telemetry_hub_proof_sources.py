# SPDX-License-Identifier: Apache-2.0
"""What counts as a Hub round trip — and what only looks like one.

``telemetry_model_id`` may report a non-catalog ``org/name`` ONLY on proof
that the repo is public, and the only proof it accepts is a Hub call that
succeeded with no token. PR #3606's adversarial round found that one of the
three sites recording that proof could be satisfied *from the local cache*,
which makes the "success" worthless:

``huggingface_hub.hf_hub_download`` catches a failed HEAD — including the
401/403 a gated repo answers an anonymous client — into ``head_call_error``
and then returns the cached pointer file if one exists ("Couldn't make a
HEAD call => let's try to find a local file", ``file_download.py``). So a
GATED repo whose ``config.json`` was cached by an earlier token-authenticated
pull "succeeded" with no token in sight, and the persisted marker made every
later process report ``acme/gated-model``.

These tests run against a local HTTP server that answers 401 GatedRepo to
everything — no real network, no real token — with a hand-built HF cache, so
the cached-fallback path is exercised for real rather than simulated.
"""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rapid_mlx.telemetry import model_id as mid

_GATED_REPO = "acme/gated-model"
_COMMIT = "a" * 40


class _GatedHandler(BaseHTTPRequestHandler):
    """Everything is 401 GatedRepo, exactly as the Hub answers an
    anonymous client for a gated repository."""

    def _deny(self):
        self.send_response(401)
        self.send_header("content-length", "0")
        self.send_header("x-error-code", "GatedRepo")
        self.end_headers()

    # ``BaseHTTPRequestHandler`` dispatches on these exact names.
    def do_GET(self):  # noqa: N802 — stdlib-mandated spelling
        self._deny()

    def do_HEAD(self):  # noqa: N802 — stdlib-mandated spelling
        self._deny()

    def log_message(self, *_args):
        pass


@pytest.fixture
def gated_hub(tmp_path, monkeypatch):
    """A 401-only Hub endpoint plus a cache already holding the repo's
    ``config.json`` — the state left behind by an earlier authenticated
    pull of a repo that is (or has become) gated."""
    server = HTTPServer(("127.0.0.1", 0), _GatedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"

    cache = tmp_path / "hub"
    repo_dir = cache / "models--acme--gated-model"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "snapshots" / _COMMIT).mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(_COMMIT, encoding="utf-8")
    (repo_dir / "snapshots" / _COMMIT / "config.json").write_text(
        '{"model_type": "qwen3"}', encoding="utf-8"
    )

    import huggingface_hub.constants as hf_constants

    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.setattr(hf_constants, "ENDPOINT", endpoint, raising=False)
    # ``HfApi`` reads ``constants.ENDPOINT`` at construction, but
    # ``hf_hub_url`` formats a TEMPLATE that was built from the endpoint at
    # import time — patch both or the download still goes to huggingface.co.
    monkeypatch.setattr(
        hf_constants,
        "HUGGINGFACE_CO_URL_TEMPLATE",
        endpoint + "/{repo_id}/resolve/{revision}/{filename}",
        raising=False,
    )
    monkeypatch.setattr(
        hf_constants, "HUGGINGFACE_CO_URL_HOME", endpoint + "/", raising=False
    )
    # ``from huggingface_hub import model_info`` is a bound method of a
    # module-level ``HfApi`` built at import time — the very object
    # ``server._prefetch_routing_metadata`` calls. Repoint it too.
    import huggingface_hub.hf_api as hf_api

    monkeypatch.setattr(hf_api.api, "endpoint", endpoint, raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    # The suite runs hermetic (HF_HUB_OFFLINE), which would short-circuit
    # every request before it reached the fake endpoint — and offline mode
    # takes the SAME cached-fallback branch, so the 401 path would never be
    # exercised. Point huggingface_hub at 127.0.0.1 and let it talk.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False, raising=False)
    for var in mid._HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Definitively anonymous: this is the dangerous state, because a
    # "success" here would be read as proof.
    monkeypatch.setattr(mid, "hf_auth_state", lambda: False)
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.hub_offline_mode_active", lambda: False
    )
    mid._reset_for_tests()
    # Self-check: if the stub endpoint is not answering, every test below
    # would "pass" for the wrong reason (no proof because the call blew up
    # on connect). Fail here instead, loudly.
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{endpoint}/api/models/{_GATED_REPO}", timeout=5)
        raise AssertionError("stub endpoint must answer 401, not 200")
    except urllib.error.HTTPError as exc:
        assert exc.code == 401, exc.code

    try:
        yield endpoint, cache
    finally:
        mid._reset_for_tests()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_the_cached_fallback_really_happens(gated_hub):
    """Pin the upstream behaviour the rest of this file depends on.

    Without this, a later test could pass merely because the download
    raised — proving nothing about our rule.
    """
    from huggingface_hub import hf_hub_download

    _endpoint, cache = gated_hub
    path = hf_hub_download(_GATED_REPO, "config.json", cache_dir=str(cache))
    assert str(cache) in path, path
    assert os.path.exists(path)


def test_a_gated_repo_served_from_cache_is_never_proof(gated_hub, monkeypatch):
    """THE P0: gated repo, anonymous probe, cached ``config.json``.

    Before the fix this recorded a marker and ``telemetry_model_id``
    returned ``acme/gated-model`` — including after a fresh process
    re-read the marker from disk.
    """
    from rapid_mlx import server

    # Present the CANONICAL endpoint to the proof guard. The stub Hub this
    # fixture runs is necessarily a non-canonical endpoint, and the round-2
    # guard would refuse to record proof for that reason alone — which would
    # make this test green even if the call site came back. Isolating the
    # guard keeps the mutation "restore note_hub_fetch here" red.
    monkeypatch.setattr(mid, "hub_endpoint_is_canonical", lambda: True)

    server._prefetch_config_for_text_lane_guard(_GATED_REPO)

    assert mid.telemetry_model_id(_GATED_REPO) == "<custom>"
    marker = mid._marker_path(_GATED_REPO)
    assert marker is not None and not os.path.exists(marker)
    mid._reset_for_tests()  # a fresh process re-reading the cache
    assert mid.telemetry_model_id(_GATED_REPO) == "<custom>"


def test_model_info_has_no_cache_fallback_so_it_stays_a_proof_source(gated_hub):
    """The audit of the other two sites: both go through ``model_info``,
    a plain API call that raises on the gated 401 instead of quietly
    answering from the cache."""
    from huggingface_hub.utils import GatedRepoError

    from rapid_mlx import _download_gate

    with pytest.raises(GatedRepoError):
        _download_gate._model_info_with_timeout(_GATED_REPO, 10.0)
    assert mid.telemetry_model_id(_GATED_REPO) == "<custom>"


def test_bare_model_info_also_raises_for_the_routing_prefetch(gated_hub):
    """``server._prefetch_routing_metadata`` depends on the same primitive."""
    from huggingface_hub import model_info
    from huggingface_hub.utils import GatedRepoError

    with pytest.raises(GatedRepoError):
        model_info(_GATED_REPO)
    assert mid.telemetry_model_id(_GATED_REPO) == "<custom>"


def test_a_foreign_endpoint_is_not_a_proof_source(gated_hub):
    """Round-2 P1: this stub IS a foreign endpoint, and an anonymous 200
    from one proves nothing about huggingface.co. Belt to the braces
    above: even if a call site recorded here, the endpoint guard refuses."""
    assert mid.hub_endpoint_is_canonical() is False
    mid.note_hub_fetch("acme/secret-internal-finetune")
    assert mid.telemetry_model_id("acme/secret-internal-finetune") == "<custom>"
