# SPDX-License-Identifier: Apache-2.0
"""``telemetry_model_id`` — the model identity rules, and the leak they close.

Before this, telemetry reported ``request.model``: a caller-controlled
string that on a ``--served-model-name acme-internal-bot`` server put a
private product name straight on the wire. Each test below pins ONE rule
and fails if that rule is removed.
"""

from __future__ import annotations

import pytest

from rapid_mlx.telemetry import model_id as mid

#: The real token probe, captured before the autouse fixture stubs it out.
_REAL_HF_AUTH_IN_USE = mid.hf_auth_in_use


@pytest.fixture(autouse=True)
def _isolated_hub_cache(tmp_path, monkeypatch):
    """Point the marker store at a temp dir and drop the in-process cache.

    Without this, a proof marker written by a real download on the dev
    machine could make a "no proof" test pass for the wrong reason.
    """
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    for var in mid._HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mid, "hf_auth_in_use", lambda: False)
    mid._reset_for_tests()
    yield
    mid._reset_for_tests()


# ------------------------------------------------------------------ catalog


def test_catalog_alias_by_alias_name():
    assert mid.telemetry_model_id("qwen3.5-9b-4bit") == "qwen3.5-9b-4bit"


def test_catalog_alias_matched_by_hf_path():
    """A user who typed the full catalog repo id still reports as the alias."""
    assert mid.telemetry_model_id("mlx-community/Qwen3.5-9B-4bit") == "qwen3.5-9b-4bit"


def test_catalog_alias_hf_path_match_is_case_insensitive():
    assert mid.telemetry_model_id("MLX-Community/qwen3.5-9b-4BIT") == "qwen3.5-9b-4bit"


def test_user_alias_is_not_catalog_identity(monkeypatch):
    """A name the USER invented is user data, not catalog identity."""
    import rapid_mlx.user_aliases as user_aliases

    monkeypatch.setattr(
        user_aliases,
        "validated_user_aliases",
        lambda *a, **k: {"my-secret-tune": "qwen3.5-9b-4bit"},
    )
    # ``resolve_profile`` WOULD resolve this; ``catalog_alias_for`` must not.
    assert mid.telemetry_model_id("my-secret-tune") == "<custom>"


# -------------------------------------------------------------- proof rules


def test_private_looking_repo_without_proof_is_custom():
    """The core leak: a non-catalog ``org/name`` we cannot prove public."""
    assert mid.telemetry_model_id("acme-corp/internal-support-model") == "<custom>"


def test_repo_reported_only_after_an_anonymous_hub_fetch():
    repo = "someone/public-community-mlx"
    assert mid.telemetry_model_id(repo) == "<custom>"
    mid.note_hub_fetch(repo)
    assert mid.telemetry_model_id(repo) == repo


def test_token_authenticated_fetch_is_no_proof(monkeypatch):
    """A fetch that used a token proves nothing — the token may be what
    opened a private/gated repo."""
    monkeypatch.setattr(mid, "hf_auth_in_use", lambda: True)
    repo = "acme-corp/gated-weights"
    mid.note_hub_fetch(repo)
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_proof_survives_a_new_process(monkeypatch):
    """Warm starts keep the proof: it lives in the repo's cache dir, not
    only in this process."""
    repo = "someone/public-community-mlx"
    mid.note_hub_fetch(repo)
    mid._reset_for_tests()  # simulate a fresh process
    assert mid.telemetry_model_id(repo) == repo


def test_hf_auth_detected_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert _REAL_HF_AUTH_IN_USE() is True


# ------------------------------------------------------------- local / caps


def test_local_path_is_local(tmp_path):
    assert mid.telemetry_model_id("/Users/alice/private-checkout") == "<local>"
    assert mid.telemetry_model_id("./my-model") == "<local>"
    real = tmp_path / "org-shaped" / "dir"
    real.mkdir(parents=True)
    assert mid.telemetry_model_id(str(real)) == "<local>"


def test_empty_and_non_string_are_custom():
    assert mid.telemetry_model_id("") == "<custom>"
    assert mid.telemetry_model_id(None) == "<custom>"
    assert mid.telemetry_model_id(object()) == "<custom>"


def test_pattern_and_length_cap(monkeypatch):
    """Defence in depth: even a "proven" id must look like an id and be
    short, or it collapses to ``<custom>``."""
    monkeypatch.setattr(mid, "is_proven_public", lambda _r: True)
    assert mid.telemetry_model_id("org/" + "n" * 200) == "<custom>"
    # Off-pattern values never survive as themselves; which sentinel they
    # collapse to (``<custom>`` via the cap, ``<local>`` via the path
    # heuristic) is not the contract — "not the raw string" is.
    for hostile in ("org/name?query=secret", "org/name with spaces", "a/b/c"):
        got = mid.telemetry_model_id(hostile)
        assert got in ("<custom>", "<local>"), got


def test_cap_applies_even_to_a_catalog_alias(monkeypatch):
    """The cap is the LAST gate, after every other rule said yes.

    Exercised through the catalog branch because the repo-id branch is
    additionally guarded by the repo-id pattern; a hostile ``aliases.json``
    entry (or a future catalog source) is the path that reaches the cap on
    its own.
    """
    import rapid_mlx.model_aliases as aliases

    monkeypatch.setattr(aliases, "catalog_alias_for", lambda _n: "a" * 200)
    assert mid.telemetry_model_id("qwen3.5-9b-4bit") == "<custom>"
    monkeypatch.setattr(aliases, "catalog_alias_for", lambda _n: "alias with spaces")
    assert mid.telemetry_model_id("qwen3.5-9b-4bit") == "<custom>"


def test_sentinels_pass_through():
    assert mid.telemetry_model_id("<local>") == "<local>"
    assert mid.telemetry_model_id("<custom>") == "<custom>"


# ------------------------------------------------------- the served-name leak


class _Entry:
    def __init__(self, telemetry_model_id, model_path):
        self.telemetry_model_id = telemetry_model_id
        self.model_path = model_path


class _Registry:
    def __init__(self, entry):
        self._entry = entry

    def __bool__(self):
        return True

    def get_entry(self, _name=None):
        return self._entry


@pytest.fixture
def _config(monkeypatch):
    from rapid_mlx.config.server_config import get_config, reset_config

    reset_config()
    yield get_config()
    reset_config()


def test_served_model_id_ignores_the_caller_string(_config):
    """``--served-model-name`` and ``request.model`` never reach the id."""
    _config.model_registry = _Registry(
        _Entry("qwen3.5-9b-4bit", "mlx-community/Qwen3.5-9B-4bit")
    )
    _config.model_name = "acme-internal-support-bot"  # served name
    got = mid.served_model_id("acme-internal-support-bot")
    assert got == "qwen3.5-9b-4bit"
    assert "acme" not in got


def test_served_model_id_falls_back_to_resolved_path_not_served_name(_config):
    _config.model_registry = None
    _config.model_name = "acme-internal-support-bot"
    _config.model_path = "mlx-community/Qwen3.5-9B-4bit"
    assert mid.served_model_id("whatever-the-client-typed") == "qwen3.5-9b-4bit"


def test_served_model_id_is_custom_with_nothing_loaded(_config):
    _config.model_registry = None
    _config.model_path = None
    assert mid.served_model_id("acme-internal-support-bot") == "<custom>"


def test_registry_entry_default_is_custom():
    """An entry built by a path that never stamped an id reports nothing."""
    from rapid_mlx.runtime.model_registry import ModelEntry

    entry = ModelEntry(engine=object(), model_name="x", model_path="acme/private")
    assert entry.telemetry_model_id == "<custom>"


# ------------------------------------------------------- the emit boundary


def test_emit_request_reapplies_the_rule(opted_in_queue):
    """Even a call site that regressed to ``request.model`` cannot leak."""
    from rapid_mlx.telemetry import emit

    emit.request(
        endpoint="/v1/chat/completions",
        model_alias="acme-corp/internal-support-model",
        stream=False,
        tool_call_used=False,
        prompt_tokens=1,
        completion_tokens=1,
        ttft_ms=1.0,
        tps=1.0,
        status=200,
    )
    payload = opted_in_queue[0]
    assert payload["request"]["model_alias"] == "<custom>"
    assert "acme" not in repr(payload)


@pytest.fixture
def opted_in_queue(tmp_path, monkeypatch):
    """Telemetry on, sampling forced, queue captured in memory."""
    import importlib

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    monkeypatch.setenv("RAPID_MLX_TELEMETRY_REQUEST_SAMPLE", "1")

    import rapid_mlx.telemetry.emit as emit
    import rapid_mlx.telemetry.state as state

    importlib.reload(state)
    importlib.reload(emit)
    emit._reset_for_tests()
    state.record_consent(True, rapid_mlx_version="0.0.0+test")

    captured: list = []

    class _Q:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _Q())
    yield captured
    emit._reset_for_tests()
