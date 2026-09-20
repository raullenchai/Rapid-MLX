# SPDX-License-Identifier: Apache-2.0
"""``telemetry_model_id`` — the model identity rules, and the leak they close.

Before this, telemetry reported ``request.model``: a caller-controlled
string that on a ``--served-model-name acme-internal-bot`` server put a
private product name straight on the wire. Each test below pins ONE rule
and fails if that rule is removed.
"""

from __future__ import annotations

from types import SimpleNamespace

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


# ----------------------------------------------- fail-soft / edge coverage
#
# ``telemetry_model_id`` sits on the request and model-load paths, so every
# branch below is a "telemetry must never break the product" guarantee: each
# one has to be exercised, or we are shipping an error path nobody ran.


def test_hf_auth_assumes_authenticated_when_hub_is_unavailable(monkeypatch):
    """No huggingface_hub → we cannot prove anonymity, so we assume a token."""
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert _REAL_HF_AUTH_IN_USE() is True


def test_hf_auth_assumes_authenticated_when_get_token_raises(monkeypatch):
    import huggingface_hub

    def _boom():
        raise RuntimeError("unreadable token file")

    monkeypatch.setattr(huggingface_hub, "get_token", _boom)
    assert _REAL_HF_AUTH_IN_USE() is True


def test_hf_auth_false_when_no_token_anywhere(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    assert _REAL_HF_AUTH_IN_USE() is False


def test_marker_path_is_none_when_the_cache_layout_is_unavailable(monkeypatch):
    import rapid_mlx._download_gate as gate

    def _boom(*_a, **_k):
        raise RuntimeError("no cache")

    monkeypatch.setattr(gate, "rapid_cache_marker_path", _boom)
    assert mid._marker_path("org/name") is None
    # Both callers degrade to "no proof" rather than raising.
    mid.note_hub_fetch("org/name")
    mid._reset_for_tests()
    assert mid.is_proven_public("org/name") is False


def test_note_hub_fetch_ignores_non_repo_references():
    mid.note_hub_fetch("qwen3.5-9b-4bit")  # bare alias, not a repo id
    mid.note_hub_fetch("/Users/alice/model")
    mid.note_hub_fetch(None)  # type: ignore[arg-type]
    assert mid.is_proven_public("qwen3.5-9b-4bit") is False


def test_note_hub_fetch_is_idempotent_and_marker_aware():
    repo = "someone/public-community-mlx"
    mid.note_hub_fetch(repo)  # writes the marker
    mid.note_hub_fetch(repo)  # in-process short circuit
    mid._reset_for_tests()
    mid.note_hub_fetch(repo)  # marker already on disk
    assert mid.is_proven_public(repo) is True


def test_note_hub_fetch_survives_an_unwritable_cache(monkeypatch):
    monkeypatch.setattr(
        mid.os,
        "makedirs",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only")),
    )
    mid.note_hub_fetch("someone/public-community-mlx")  # must not raise


def test_is_proven_public_swallows_a_broken_filesystem(monkeypatch):
    monkeypatch.setattr(
        mid.os.path, "exists", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom"))
    )
    assert mid.is_proven_public("someone/public-community-mlx") is False


def test_unreadable_reference_is_treated_as_local(monkeypatch):
    """An ``os.path.exists`` that refuses the string (NUL byte, bad encoding)
    resolves to ``<local>``, never to the string itself."""
    assert mid.telemetry_model_id("org/na\x00me") == "<local>"


def test_telemetry_model_id_swallows_an_internal_bug(monkeypatch):
    import rapid_mlx.model_aliases as aliases

    def _boom(_name):
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(aliases, "catalog_alias_for", _boom)
    assert mid.telemetry_model_id("someone/public-community-mlx") == "<custom>"


def test_served_model_id_when_the_config_cannot_be_imported(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "rapid_mlx.config.server_config", None)
    assert mid.served_model_id("anything") == "<custom>"


def test_served_model_id_when_the_registry_lookup_raises(_config):
    class _Raising:
        def __bool__(self):
            return True

        def get_entry(self, _name=None):
            raise KeyError("no models loaded")

    _config.model_registry = _Raising()
    _config.model_path = "mlx-community/Qwen3.5-9B-4bit"
    # The registry could not answer, so the resolved checkpoint does.
    assert mid.served_model_id("qwen3.5-9b-4bit") == "qwen3.5-9b-4bit"


def test_served_model_id_recomputes_when_the_entry_has_no_stamp(_config):
    _config.model_registry = _Registry(_Entry("", "mlx-community/Qwen3.5-9B-4bit"))
    assert mid.served_model_id(None) == "qwen3.5-9b-4bit"


def test_served_model_id_caps_a_stamped_value(_config):
    _config.model_registry = _Registry(_Entry("a" * 200, "acme/private"))
    assert mid.served_model_id("x") == "<custom>"


def test_served_model_id_passes_sentinels_through(_config):
    _config.model_registry = _Registry(_Entry("<local>", "/Users/alice/model"))
    assert mid.served_model_id("x") == "<local>"


def test_served_model_id_swallows_a_broken_config(monkeypatch, _config):
    class _Exploding:
        def __bool__(self):
            raise RuntimeError("config is broken")

    _config.model_registry = _Exploding()
    assert mid.served_model_id("x") == "<custom>"


# --------------------------------------------------- catalog lookup itself


def test_catalog_alias_for_rejects_non_names():
    from rapid_mlx.model_aliases import catalog_alias_for

    assert catalog_alias_for("") is None
    assert catalog_alias_for(None) is None  # type: ignore[arg-type]
    assert catalog_alias_for(123) is None  # type: ignore[arg-type]


def test_catalog_alias_for_is_fail_soft(monkeypatch):
    import rapid_mlx.model_aliases as aliases

    def _boom():
        raise RuntimeError("aliases.json is corrupt")

    monkeypatch.setattr(aliases, "_load", _boom)
    assert aliases.catalog_alias_for("qwen3.5-9b-4bit") is None


def test_catalog_alias_for_reads_the_live_registry(monkeypatch):
    """A test elsewhere that swaps in a temporary registry must not be able to
    leave a stale reverse index answering for the real catalog."""
    import rapid_mlx.model_aliases as aliases
    from rapid_mlx.model_profile import ModelProfile

    monkeypatch.setattr(
        aliases, "_aliases", {"temp-alias": ModelProfile(hf_path="org/Temp-Model")}
    )
    assert aliases.catalog_alias_for("org/temp-model") == "temp-alias"
    assert aliases.catalog_alias_for("mlx-community/Qwen3.5-9B-4bit") is None


# ------------------------------------------------ where the proof is taken
#
# The proof is only worth anything if the download path actually records it.
# These drive the two Hub touch points that call ``note_hub_fetch``.


def test_model_info_probe_records_the_proof(monkeypatch):
    """``_model_info_with_timeout`` is our first anonymous Hub touch on a cold
    pull; a success there is what licenses reporting the repo id."""
    from rapid_mlx import _download_gate

    class _FakeApi:
        def model_info(self, repo_id, files_metadata=False):
            return SimpleNamespace(sha="deadbeef", siblings=[])

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    repo = "someone/public-community-mlx"
    assert mid.telemetry_model_id(repo) == "<custom>"
    _download_gate._model_info_with_timeout(repo, 5.0)
    assert mid.telemetry_model_id(repo) == repo


def test_text_lane_config_prefetch_records_the_proof(monkeypatch):
    import huggingface_hub

    from rapid_mlx import server

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: "/tmp/config.json"
    )
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.hub_offline_mode_active", lambda: False
    )
    repo = "someone/other-public-mlx"
    server._prefetch_config_for_text_lane_guard(repo)
    assert mid.telemetry_model_id(repo) == repo


# ---------------------------------------------- user intent always wins
#
# The fail-soft wrappers swallow bugs, never the user's Ctrl-C. Each of the
# three entry points is checked separately because each owns its own
# ``except (KeyboardInterrupt, SystemExit): raise`` clause.


def _raise_keyboard_interrupt(*_a, **_k):
    raise KeyboardInterrupt


def test_note_hub_fetch_does_not_swallow_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(mid, "hf_auth_in_use", _raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        mid.note_hub_fetch("someone/public-community-mlx")


def test_telemetry_model_id_does_not_swallow_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(mid, "_looks_local", _raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        mid.telemetry_model_id("someone/public-community-mlx")


def test_served_model_id_does_not_swallow_keyboard_interrupt(monkeypatch, _config):
    class _Interrupting:
        def __bool__(self):
            raise KeyboardInterrupt

    _config.model_registry = _Interrupting()
    with pytest.raises(KeyboardInterrupt):
        mid.served_model_id("x")


def test_a_reference_the_filesystem_refuses_is_local(monkeypatch):
    """``os.path.exists`` can raise on an undecodable name; that is not a
    reason to report the name."""

    def _boom(_path):
        raise OSError("name too long")

    monkeypatch.setattr(mid.os.path, "exists", _boom)
    assert mid._looks_local("some-unreadable-name") is True
