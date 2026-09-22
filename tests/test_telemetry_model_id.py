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

#: The real probes, captured before the autouse fixture stubs them out.
_REAL_HF_AUTH_IN_USE = mid.hf_auth_in_use
_REAL_HF_AUTH_STATE = mid.hf_auth_state


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
    # Patch the tri-state PROBE, not the boolean read-side: that keeps the
    # latch and the fail-closed mapping under test instead of stubbed out.
    monkeypatch.setattr(mid, "hf_auth_state", lambda: False)
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
    monkeypatch.setattr(mid, "hf_auth_state", lambda: True)
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


# ------------------------------------------ proof is a lease (codex P0 #3600)
#
# A repo that was public when we pulled it can be made private or gated
# afterwards. Each test below pins one of the three rules that stop a stale
# marker from naming such a repo.


def test_auth_appearing_later_revokes_a_stored_proof(monkeypatch):
    """THE P0: anonymous proof, then a token appears — the id must go dark.

    Not "we stop adding proof": the id already stored must stop being
    reported, and must stay dark across a fresh process.
    """
    repo = "someone/public-community-mlx"
    mid.note_hub_fetch(repo)
    assert mid.telemetry_model_id(repo) == repo

    monkeypatch.setattr(mid, "hf_auth_state", lambda: True)
    assert mid.telemetry_model_id(repo) == "<custom>"

    # And it must stay dark for a fresh process that re-reads the marker
    # with NO token in sight — i.e. the marker itself must be gone, not
    # merely out-voted by the flag that is still patched on.
    mid.note_hub_fetch(repo)  # the authenticated re-open revokes
    monkeypatch.setattr(mid, "hf_auth_state", lambda: False)
    mid._reset_for_tests()
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_an_authenticated_fetch_deletes_the_marker(monkeypatch):
    """The authenticated load actively revokes; the proof does not survive it."""
    repo = "someone/public-community-mlx"
    mid.note_hub_fetch(repo)
    marker = mid._marker_path(repo)
    assert marker is not None and mid.os.path.exists(marker)

    monkeypatch.setattr(mid, "hf_auth_state", lambda: True)
    mid.note_hub_fetch(repo)  # the gated re-open
    assert not mid.os.path.exists(marker)

    # Even with the token gone again, there is nothing left to report.
    monkeypatch.setattr(mid, "hf_auth_state", lambda: False)
    mid._reset_for_tests()
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_revocation_survives_a_cache_without_a_marker_path(monkeypatch):
    import rapid_mlx._download_gate as gate

    monkeypatch.setattr(
        gate,
        "rapid_cache_marker_path",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no cache")),
    )
    monkeypatch.setattr(mid, "hf_auth_state", lambda: True)
    mid.note_hub_fetch("someone/public-community-mlx")  # must not raise


def test_proof_expires(monkeypatch):
    """Stored proof is a lease, not a deed: past the TTL it is not proof."""
    repo = "someone/public-community-mlx"
    mid.note_hub_fetch(repo)
    assert mid.telemetry_model_id(repo) == repo

    later = mid.time.time() + mid.PUBLIC_PROOF_TTL_SECONDS + 1
    monkeypatch.setattr(mid, "time", SimpleNamespace(time=lambda: later))
    # Stale in memory AND stale on disk — neither may answer "public".
    assert mid.telemetry_model_id(repo) == "<custom>"
    mid._reset_for_tests()
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_a_fresh_anonymous_fetch_renews_an_expired_proof(monkeypatch):
    repo = "someone/public-community-mlx"
    mid.note_hub_fetch(repo)
    later = mid.time.time() + mid.PUBLIC_PROOF_TTL_SECONDS + 1
    monkeypatch.setattr(mid, "time", SimpleNamespace(time=lambda: later))
    assert mid.telemetry_model_id(repo) == "<custom>"
    mid.note_hub_fetch(repo)  # re-proved, at the new "now"
    assert mid.telemetry_model_id(repo) == repo


def _write_raw_marker(repo: str, contents: str) -> None:
    path = mid._marker_path(repo)
    assert path is not None
    mid.os.makedirs(mid.os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(contents)


@pytest.mark.parametrize("contents", ["nan", "NaN", "inf", "Infinity", "1e999"])
def test_a_non_finite_marker_is_not_proof(contents):
    """``float()`` parses all of these, and a NaN makes every comparison
    False — so without an explicit finiteness check a garbage marker is
    proof that NEVER expires, which is the exact lease the TTL removes."""
    repo = "someone/public-community-mlx"
    _write_raw_marker(repo, contents)
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_a_future_dated_marker_is_not_proof():
    """The window is closed at BOTH ends. ``now - stamp > TTL`` is False for
    a future stamp, so a backwards clock step would otherwise restore an
    indefinite lease."""
    repo = "someone/public-community-mlx"
    future = mid.time.time() + mid.PUBLIC_PROOF_TTL_SECONDS
    _write_raw_marker(repo, str(int(future)))
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_a_marker_inside_the_clock_skew_grace_is_still_proof():
    """An ordinary few-seconds clock adjustment must not throw the proof
    away — the grace window exists so the fix is not a regression."""
    repo = "someone/public-community-mlx"
    _write_raw_marker(repo, str(int(mid.time.time() + 30)))
    assert mid.telemetry_model_id(repo) == repo


def test_a_content_free_marker_is_not_proof():
    """The pre-TTL marker format carried no timestamp; it cannot be trusted."""
    repo = "someone/public-community-mlx"
    path = mid._marker_path(repo)
    assert path is not None
    mid.os.makedirs(mid.os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("")
    assert mid.telemetry_model_id(repo) == "<custom>"


# ------------------------------- proof only from the canonical Hub (r2 P1)


def test_proof_needs_the_canonical_endpoint(monkeypatch):
    """An anonymous 200 from an internal registry or a LAN mirror says
    nothing about public readability on huggingface.co — and such a host
    answers 200 for a repo that is PRIVATE on the real Hub."""
    import huggingface_hub.constants as hf_constants

    repo = "acme-corp/secret-internal-finetune"
    monkeypatch.setattr(hf_constants, "ENDPOINT", "https://hf.acme-corp.internal")
    mid.note_hub_fetch(repo)

    marker = mid._marker_path(repo)
    assert marker is not None and not mid.os.path.exists(marker)
    assert mid.telemetry_model_id(repo) == "<custom>"
    # And nothing is left for a later process on the canonical endpoint.
    monkeypatch.setattr(hf_constants, "ENDPOINT", "https://huggingface.co")
    mid._reset_for_tests()
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_proof_needs_the_canonical_endpoint_via_env(monkeypatch):
    """Same guard when the library exposes no constant to read."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "ENDPOINT", "")
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.acme-corp.internal")
    assert mid.hub_endpoint_is_canonical() is False
    repo = "acme-corp/secret-internal-finetune"
    mid.note_hub_fetch(repo)
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_a_redirecting_endpoint_cannot_mint_proof_for_another_repo(monkeypatch):
    """The second manifestation: the endpoint 307s ``acme/private-one`` to a
    genuinely public repo and answers 200. The proof would be recorded
    under the REQUESTED id, naming the private repo."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "ENDPOINT", "https://redirector.acme.internal")
    requested = "acme-corp/private-one"
    mid.note_hub_fetch(requested)  # the 200 was really for someone else
    assert mid.telemetry_model_id(requested) == "<custom>"


def test_the_canonical_endpoint_still_records_proof(monkeypatch):
    """The guard must not silently switch proof off everywhere."""
    import huggingface_hub.constants as hf_constants

    repo = "someone/public-community-mlx"
    monkeypatch.setattr(hf_constants, "ENDPOINT", "https://huggingface.co")
    assert mid.hub_endpoint_is_canonical() is True
    mid.note_hub_fetch(repo)
    assert mid.telemetry_model_id(repo) == repo


@pytest.mark.parametrize(
    "spelling", ["https://huggingface.co/", "HTTPS://HuggingFace.co", "", None]
)
def test_canonical_endpoint_spellings(monkeypatch, spelling):
    """Trailing slash, case, and "unset" all mean the real Hub."""
    import huggingface_hub.constants as hf_constants

    if spelling is None:
        monkeypatch.delattr(hf_constants, "ENDPOINT", raising=False)
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
    else:
        monkeypatch.setattr(hf_constants, "ENDPOINT", spelling)
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert mid.hub_endpoint_is_canonical() is True


@pytest.mark.parametrize("bogus", [123, b"https://huggingface.co", object()])
def test_a_non_string_endpoint_is_not_canonical(monkeypatch, bogus):
    """The docstring promises "anything we cannot read answers False" —
    make that literally true rather than leaving an ``AttributeError`` for
    a caller's blanket handler to convert."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "ENDPOINT", bogus)
    assert mid.hub_endpoint_is_canonical() is False


def test_unreadable_hub_constants_fall_back_to_the_env(monkeypatch):
    """No importable ``huggingface_hub`` — the env var still decides, and a
    foreign endpoint still blocks proof."""
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setenv("HF_ENDPOINT", "https://elsewhere.invalid")
    assert mid.hub_endpoint_is_canonical() is False


# ------------------------------------------------- latch / never-raise (r2)


def test_note_hub_fetch_latches_the_token_it_observed(monkeypatch):
    """Record the observation, do not re-probe for it.

    Re-probing lost the latch when the token was cleared between the two
    probes — the exact laundering the latch exists to prevent.
    """
    states = iter([True, False, False, False])
    monkeypatch.setattr(mid, "hf_auth_state", lambda: next(states))
    mid.note_hub_fetch("acme-corp/gated-weights")  # observes True
    assert mid.hf_auth_in_use() is True  # latched, without re-probing


def test_hf_auth_in_use_swallows_a_probe_explosion(monkeypatch):
    """Never-raise reaches this entry point too, and fails closed."""

    def _boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(mid, "hf_auth_state", _boom)
    assert mid.hf_auth_in_use() is True


def test_hf_auth_in_use_does_not_swallow_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(mid, "hf_auth_state", _raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        mid.hf_auth_in_use()


def test_auth_detection_latches_for_the_process(monkeypatch):
    """A token seen once stays seen: clearing ``HF_TOKEN`` mid-run must not
    launder a gated repo into a reportable public name."""
    monkeypatch.setattr(mid, "hf_auth_state", _REAL_HF_AUTH_STATE)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert _REAL_HF_AUTH_IN_USE() is True
    monkeypatch.delenv("HF_TOKEN")
    assert _REAL_HF_AUTH_IN_USE() is True


def test_cannot_tell_is_fail_closed_but_never_latches(monkeypatch):
    """ "Cannot tell" blocks reporting while it lasts, and only while it
    lasts — it is not an observation, so it must not stick."""
    monkeypatch.setattr(mid, "hf_auth_state", lambda: None)
    assert mid.hf_auth_in_use() is True
    monkeypatch.setattr(mid, "hf_auth_state", lambda: False)
    assert mid.hf_auth_in_use() is False


def test_cannot_tell_never_records_proof(monkeypatch):
    """The other half of the tri-state. "Cannot tell" most often means a
    token IS configured and ``get_token()`` merely blew up reading it — so
    it can never be read as "definitely anonymous"."""
    repo = "acme-corp/gated-finetune"
    monkeypatch.setattr(mid, "hf_auth_state", lambda: None)
    mid.note_hub_fetch(repo)
    marker = mid._marker_path(repo)
    assert marker is not None and not mid.os.path.exists(marker)

    monkeypatch.setattr(mid, "hf_auth_state", lambda: False)
    mid._reset_for_tests()
    assert mid.telemetry_model_id(repo) == "<custom>"


def test_cannot_tell_never_destroys_proof(monkeypatch):
    """THE regression this tri-state exists for.

    ``hf_auth_state`` is fail-closed: an unreadable token file answers
    "cannot tell". Combined with a latch and with destructive revocation,
    one transient probe failure would delete — for every future process,
    on shared disk — a marker that a genuinely anonymous pull earned.
    """
    repo = "someone/public-community-mlx"
    mid.note_hub_fetch(repo)
    marker = mid._marker_path(repo)
    assert marker is not None and mid.os.path.exists(marker)

    monkeypatch.setattr(mid, "hf_auth_state", lambda: None)  # NFS hiccup
    mid.note_hub_fetch(repo)
    assert mid.os.path.exists(marker), "a transient probe failure ate the proof"

    # The hiccup passes; the proof is still there for the next process.
    monkeypatch.setattr(mid, "hf_auth_state", lambda: False)
    mid._reset_for_tests()
    assert mid.telemetry_model_id(repo) == repo


def test_hf_auth_detected_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert _REAL_HF_AUTH_STATE() is True


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


# ----------------------------------------------- fail-soft / edge coverage
#
# ``telemetry_model_id`` sits on the request and model-load paths, so every
# branch below is a "telemetry must never break the product" guarantee: each
# one has to be exercised, or we are shipping an error path nobody ran.


def test_hf_auth_assumes_authenticated_when_hub_is_unavailable(monkeypatch):
    """No huggingface_hub → we cannot TELL, and the read side fails closed."""
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert _REAL_HF_AUTH_STATE() is None
    monkeypatch.setattr(mid, "hf_auth_state", _REAL_HF_AUTH_STATE)
    assert _REAL_HF_AUTH_IN_USE() is True


def test_hf_auth_assumes_authenticated_when_get_token_raises(monkeypatch):
    import huggingface_hub

    def _boom():
        raise RuntimeError("unreadable token file")

    monkeypatch.setattr(huggingface_hub, "get_token", _boom)
    assert _REAL_HF_AUTH_STATE() is None
    monkeypatch.setattr(mid, "hf_auth_state", _REAL_HF_AUTH_STATE)
    assert _REAL_HF_AUTH_IN_USE() is True


def test_hf_auth_false_when_no_token_anywhere(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    assert _REAL_HF_AUTH_STATE() is False


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


def test_marker_write_survives_a_hostile_cleanup(monkeypatch):
    """A cache that refuses the temp-file cleanup must not lose the marker
    that ``os.replace`` already put in place."""
    repo = "someone/public-community-mlx"
    real_remove = mid.os.remove

    def _refuse(path):
        if ".rapidmlx-public." in str(path):
            raise PermissionError("sticky bit")
        return real_remove(path)

    monkeypatch.setattr(mid.os, "remove", _refuse)
    mid.note_hub_fetch(repo)
    mid._reset_for_tests()
    assert mid.telemetry_model_id(repo) == repo


def test_note_hub_fetch_survives_an_unwritable_cache(monkeypatch):
    monkeypatch.setattr(
        mid.os,
        "makedirs",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only")),
    )
    mid.note_hub_fetch("someone/public-community-mlx")  # must not raise


def test_is_proven_public_swallows_an_unreadable_marker(monkeypatch):
    """The marker is there, the filesystem refuses it: no proof, no raise."""
    repo = "someone/public-community-mlx"
    mid.note_hub_fetch(repo)
    mid._reset_for_tests()

    def _boom(*_a, **_k):
        raise OSError("boom")

    monkeypatch.setattr("builtins.open", _boom)
    assert mid.is_proven_public(repo) is False


def test_is_proven_public_swallows_an_internal_bug(monkeypatch):
    def _boom(_repo):
        raise RuntimeError("marker reader exploded")

    monkeypatch.setattr(mid, "_read_marker", _boom)
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


def test_served_model_id_rechecks_a_stamped_repo_id(monkeypatch, _config):
    """A stamp taken at load time is not a permanent licence."""
    repo = "someone/public-community-mlx"
    _config.model_registry = _Registry(_Entry(repo, repo))
    mid.note_hub_fetch(repo)
    assert mid.served_model_id("x") == repo
    monkeypatch.setattr(mid, "hf_auth_state", lambda: True)
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


def test_text_lane_config_prefetch_records_no_proof(monkeypatch):
    """A ``hf_hub_download`` that returns is NOT evidence of a public repo.

    ``huggingface_hub`` swallows a failed HEAD — the 401/403 a gated repo
    answers an anonymous client included — and returns the CACHED file if
    one is there. A gated repo whose ``config.json`` was cached by an
    earlier token-authenticated pull therefore "succeeds" here with no
    token in sight. This site must record nothing.
    """
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
    assert mid.telemetry_model_id(repo) == "<custom>"
    mid._reset_for_tests()
    assert mid.telemetry_model_id(repo) == "<custom>"


# ---------------------------------------------- user intent always wins
#
# The fail-soft wrappers swallow bugs, never the user's Ctrl-C. Each of the
# three entry points is checked separately because each owns its own
# ``except (KeyboardInterrupt, SystemExit): raise`` clause.


def _raise_keyboard_interrupt(*_a, **_k):
    raise KeyboardInterrupt


def test_note_hub_fetch_does_not_swallow_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(mid, "hf_auth_state", _raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        mid.note_hub_fetch("someone/public-community-mlx")


def test_is_proven_public_does_not_swallow_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(mid, "_read_marker", _raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        mid.is_proven_public("someone/public-community-mlx")


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
