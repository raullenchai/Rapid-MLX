# SPDX-License-Identifier: Apache-2.0
"""``telemetry_model_id`` — the ONLY model identity telemetry may report.

The field telemetry used to send was ``request.model``: whatever string the
HTTP client typed. On a server started with ``--served-model-name
acme-internal-support-bot`` — or one serving a private fine-tune by its
``org/name`` — that string went on the wire verbatim. This module replaces
it with an identity that is safe by construction:

1. **Catalog alias** — the model is in the shipped text/image or audio
   ``aliases.json`` (matched by alias spelling, or by its resolved Hub path
   case-insensitively). Public by definition: we publish the catalogs.
2. **``org/name``** — only when we have *fresh* proof the repo is public,
   i.e. a Hugging Face Hub call for it succeeded while **no token was in
   use** (see :func:`note_hub_fetch`), within the last
   :data:`PUBLIC_PROOF_TTL_SECONDS`, **and** no Hub token is in use in this
   process right now. Non-catalog public models people actually run are the
   demand signal telemetry exists to find.
3. **``"<local>"``** — a local checkout / path. Never the path itself.
4. **``"<custom>"``** — everything else: gated repos, token-authenticated
   fetches, not-found repos, user aliases, anything we cannot prove public,
   and anything that fails the pattern/length cap.

``--served-model-name`` can never influence the result because the id is
computed from the *resolved checkpoint* at load time (``ModelEntry.model_path``),
not from the name the server advertises or the client requests.

**Proof is a lease, not a deed** (codex P0 on #3600). A repo that was public
when we pulled it can be made private or gated later, and the next load will
open it with a token. Three rules keep a stale "public" marker from naming
such a repo:

* **Auth wins, always.** While any HF token is visible to this process — or
  while we cannot tell — a non-catalog repo id is ``<custom>`` no matter
  what proof is stored. A *positively observed* token latches: once seen,
  this process never reports a non-catalog repo id again, so clearing
  ``HF_TOKEN`` mid-run cannot launder a gated repo into a reportable name.
  "Cannot tell" is fail-closed for reading but never latches and never
  writes (:func:`hf_auth_state`).
* **An authenticated fetch revokes.** ``note_hub_fetch`` on an authenticated
  round trip deletes the proof — in memory and on disk — instead of merely
  declining to add one.
* **Proof expires.** The marker stores the timestamp of the anonymous
  success, and it is believed only inside a window closed at BOTH ends —
  no older than :data:`PUBLIC_PROOF_TTL_SECONDS`, no further ahead than
  :data:`_CLOCK_SKEW_GRACE_SECONDS`. A content-free marker (the pre-TTL
  format), a non-finite one (``nan`` / ``inf`` / ``1e999`` all parse as
  floats) and a future-dated one are none of them proof.

**What counts as a Hub round trip.** A call to the CANONICAL Hub that
CANNOT be satisfied from the local cache. Both halves are load-bearing.

*Canonical.* ``huggingface_hub`` routes everything through
``constants.ENDPOINT`` / ``HF_ENDPOINT``, so an operator can point the
library at an internal HF-compatible registry or a LAN mirror. Such a host
answers an anonymous 200 for a repo that is *private on the real Hub* — and
can even redirect one repo id to another, so the 200 may not even describe
the repo we asked for. Proof is therefore recorded only when the effective
endpoint is ``https://huggingface.co``; on any other endpoint we record
nothing and the model reports ``<custom>``. Revocation is deliberately not
gated this way: dropping proof is always safe.

*Not from cache.* The two remaining ``note_hub_fetch`` call sites —
``_download_gate._model_info_with_timeout`` and
``server._prefetch_routing_metadata`` — both go through
``huggingface_hub``'s ``model_info``, a plain API call with no cache
fallback that raises ``GatedRepoError`` on the 401 a gated repo answers to
an anonymous client.

``server._prefetch_config_for_text_lane_guard`` used to be a third site and
is NOT one any more (PR #3606 adversarial round). It calls
``hf_hub_download``, which swallows a failed HEAD — 401/403 included — into
``head_call_error`` and then returns the cached pointer file if one exists.
A gated repo whose ``config.json`` had been cached by an earlier
token-authenticated pull therefore "succeeded" there with no token in
sight, and the recorded marker made this module report ``org/gated-name``
indefinitely, across processes. A success that the cache can fabricate is
not evidence.

**Why not ``token=False``?** The obvious hardening — take the proof from a
*forced*-anonymous request rather than inferring anonymity afterwards — is
not available without changing what the product downloads. Both remaining
call sites are on the download path; forcing ``token=False`` there would
make an authenticated user's private/gated pull fail 401 where it succeeds
today. Passing it only when we already believe there is no token would
produce a byte-identical request (``huggingface_hub`` sends no
``Authorization`` header when it cannot resolve a token), so it would buy no
safety. The ambient check stays, fail-closed, backed by the rules above.

**Limit, stated honestly.** An operator who sets ``HF_ENDPOINT`` gets no
non-catalog model names in telemetry at all — the demand signal is traded
away for the guarantee, in the same conservative direction as everything
else here. Proof of anonymity is recorded when we touch the
Hub, so a model that was already in the HF cache before this shipped (or was
copied in out of band) has no proof and reports ``"<custom>"`` forever unless
some later Hub call for it succeeds anonymously. That is the conservative
direction: we under-report demand rather than over-report identity. We do
NOT infer "public" from "no token is set right now" — the weights on disk may
have been pulled months ago by an authenticated ``huggingface-cli download``.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time

#: Model is local, or we cannot prove anything about it.
LOCAL = "<local>"
CUSTOM = "<custom>"

_SENTINELS: frozenset[str] = frozenset({LOCAL, CUSTOM})

# Defence in depth on top of the rules above: even a value that passed them
# must look like a catalog alias or a repo id, and be short. A catalog alias
# that somehow grew a query string, or a 4 KB repo name, collapses to
# ``<custom>`` rather than riding out on a payload.
_MAX_ID_LEN = 96
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}(?:/[A-Za-z0-9._-]{1,64})?$")

# Repo-id shape, i.e. the ``org/name`` branch of the pattern above.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}/[A-Za-z0-9._-]{1,64}$")

# Env vars huggingface_hub reads for an API token. Checked directly (rather
# than only through ``get_token()``) so a missing/newer huggingface_hub does
# not silently turn an authenticated environment into a "proven public" one.
_HF_TOKEN_ENV_VARS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_API_TOKEN",
)

_PUBLIC_MARKER_NAME = "public-anon-fetch"

#: The only Hub whose "this repo is readable anonymously" we believe.
#:
#: Proof means "the public Hugging Face Hub served this repo to a client
#: with no credentials". An anonymous 200 from somewhere else says nothing
#: about that: an internal HF-compatible registry, or a LAN mirror that is
#: simply unauthenticated, answers 200 for a PRIVATE fine-tune — and the
#: recorded marker would then put ``acme/secret-internal-finetune`` on the
#: wire and keep it there for the whole TTL.
_CANONICAL_HF_ENDPOINT = "https://huggingface.co"

#: How long one anonymous Hub success licenses reporting a repo id.
#:
#: The exposure this bounds is narrow — one ``org/name`` string for a repo
#: that was public when we pulled it and was made private or gated
#: afterwards — but an unbounded marker is an indefinite authorisation, and
#: that is what the finding objects to. Thirty days is the shortest window
#: that still answers the question the marker exists for: the proof is taken
#: on the *cold pull*, and the telemetry that wants to name the model comes
#: from every warm start after it. A server pulled once and left running (or
#: restarted daily) for a few weeks is the normal case, so a one-week lease
#: would silently drop most of the demand signal; a one-year lease would be
#: the indefinite marker with extra steps. Thirty days also bounds the stale
#: window to roughly one release cycle, so a visibility flip is re-tested by
#: the next cold pull of that repo rather than never.
PUBLIC_PROOF_TTL_SECONDS = 30 * 24 * 60 * 60

#: How far into the future a stamp may sit and still be believed.
#:
#: The window has to be closed at BOTH ends. ``now - stamp > TTL`` is False
#: for a future-dated stamp, so with only a lower bound a backwards clock
#: step — an NTP correction after the marker was written, a VM restored
#: from a snapshot, a dual-boot machine with a local-time RTC — turns every
#: marker written "in the future" back into the indefinite authorisation the
#: TTL exists to remove. Five minutes absorbs ordinary clock adjustment
#: without leaving that hole open.
_CLOCK_SKEW_GRACE_SECONDS = 300

_proven_lock = threading.Lock()
#: ``repo_id`` -> unix timestamp of the anonymous success that proved it.
_proven_public: dict[str, float] = {}
#: Latched once any Hub token is observed; cleared only by tests.
_auth_seen = False


# ------------------------------------------------------------------- tokens


def hub_endpoint_is_canonical() -> bool:
    """Whether the Hub calls that carry proof go to the real Hub.

    ``huggingface_hub`` routes every request through ``constants.ENDPOINT``
    (seeded from ``HF_ENDPOINT``), so an operator can point the whole
    library at an internal registry or a mirror. Both remaining
    ``note_hub_fetch`` call sites inherit that — which makes their
    "anonymous success" evidence about *that* host, not about public
    readability on huggingface.co.

    Fail-closed: anything we cannot read, or cannot recognise as the
    canonical endpoint, answers ``False`` and no proof is recorded.
    """
    raw = ""
    try:
        from huggingface_hub import constants as hf_constants

        raw = getattr(hf_constants, "ENDPOINT", "") or ""
    except Exception:
        raw = ""
    if not raw:
        raw = os.environ.get("HF_ENDPOINT") or ""
    if not isinstance(raw, str):
        # A non-str ``ENDPOINT`` is a library state we cannot read. The
        # docstring above promises ``False`` for that, so return it here
        # instead of raising ``AttributeError`` into a caller's blanket
        # handler and relying on that handler to keep us fail-closed.
        return False
    normalised = raw.strip().rstrip("/").lower()
    if not normalised:
        # Neither set: huggingface_hub's own default is the canonical Hub.
        return True
    return normalised == _CANONICAL_HF_ENDPOINT


def hf_auth_state() -> bool | None:
    """Tri-state look at the ambient Hub credentials.

    * ``True``  — a token is visible to this process.
    * ``False`` — definitively no token.
    * ``None``  — **cannot tell**: no ``huggingface_hub`` to ask, or the
      token file would not read.

    The three answers are deliberately distinct. "Cannot tell" must be
    fail-closed on the *read* side (:func:`hf_auth_in_use`) but must NOT
    be treated as evidence on the *write* side: a momentary probe failure
    is not a reason to latch the process or to delete proof that another
    process earned (see :func:`note_hub_fetch`).
    """
    for var in _HF_TOKEN_ENV_VARS:
        if (os.environ.get(var) or "").strip():
            return True
    try:
        from huggingface_hub import get_token
    except Exception:
        return None
    try:
        return bool(get_token())
    except Exception:
        return None


def hf_auth_in_use() -> bool:
    """Whether a Hugging Face token is or has been available to this process.

    Fail-closed: "cannot tell" answers ``True`` — "assume authenticated" —
    because the only thing this answer gates is permission to report a
    repo id.

    A *positively observed* token **latches** for the life of the process.
    A token that was visible at load time may have opened a gated repo
    whose weights are now resident; dropping ``HF_TOKEN`` from the
    environment afterwards must not turn that repo into a reportable
    public name. "Cannot tell" does not latch — it is not an observation.
    """
    global _auth_seen
    if _auth_seen:
        return True
    try:
        state = hf_auth_state()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        # The module's never-raise contract reaches here too: a probe that
        # explodes is the definition of "cannot tell", so fail closed.
        return True
    if state is True:
        _auth_seen = True
        return True
    return state is None


# -------------------------------------------------------------------- proof


def _marker_path(repo_id: str) -> str | None:
    try:
        from rapid_mlx._download_gate import rapid_cache_marker_path

        return rapid_cache_marker_path(repo_id, _PUBLIC_MARKER_NAME)
    except Exception:
        return None


def _write_marker(repo_id: str, stamp: float) -> None:
    """Persist ``stamp`` as the proof marker for ``repo_id``.

    Atomic now that the content is load-bearing: a truncated marker would
    read back as "no proof", which is safe, but a torn *timestamp* could
    read back as a future date and never expire.
    """
    path = _marker_path(repo_id)
    if path is None:
        return
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=directory, prefix=".rapidmlx-public.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(f"{int(stamp)}\n")
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            # The ordinary case is FileNotFoundError: ``os.replace``
            # already consumed the temp file. Any other OSError here means
            # the cache directory turned hostile between write and
            # cleanup; leaking one stray ``.rapidmlx-public.*.tmp`` beats
            # unwinding a marker we may have successfully replaced.
            pass


def _read_marker(repo_id: str) -> float | None:
    """The timestamp stored in ``repo_id``'s marker, or ``None``.

    ``None`` covers every "this is not proof" case identically: no marker,
    an unreadable cache, and the pre-TTL content-free marker written by the
    first version of this module.
    """
    path = _marker_path(repo_id)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read(64).strip()
    except OSError:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _is_fresh(stamp: float, now: float) -> bool:
    """Whether ``stamp`` is inside the window that licenses reporting.

    The window is closed at BOTH ends, and that is load-bearing twice
    over. A future-dated stamp passes ``age <= TTL`` trivially, so without
    the lower bound a backwards clock step restores an indefinite lease.
    And the two-sided form is also what rejects the garbage ``float()``
    accepts — ``"nan"``, ``"inf"`` and ``"1e999"`` all parse, but every
    comparison against NaN is False and ±inf lands outside the window, so
    neither can become proof that never expires. No explicit ``isfinite``
    check: it would be a line no test could kill.
    """
    age = now - stamp
    return -_CLOCK_SKEW_GRACE_SECONDS <= age <= PUBLIC_PROOF_TTL_SECONDS


def _revoke_proof(repo_id: str) -> None:
    """Drop any stored proof for ``repo_id``, in memory and on disk.

    On-disk revocation is **best effort**: an unwritable or read-only cache
    swallows the ``OSError`` and the marker survives. In-process that is
    masked — the auth latch outranks any stored proof for the rest of this
    process — but a later *anonymous* process sharing that cache will still
    believe the marker until the TTL retires it.
    """
    with _proven_lock:
        _proven_public.pop(repo_id, None)
    path = _marker_path(repo_id)
    if path is None:
        return
    try:
        os.remove(path)
    except OSError:
        return


def note_hub_fetch(repo_id: str) -> None:
    """Record — or revoke — proof that ``repo_id`` is public.

    Call this ONLY after a real Hub round trip returned successfully. It
    decides for itself whether that round trip was anonymous:

    * **Anonymous success** (a token was definitively absent) — proof,
      stamped with the current time.
    * **Authenticated success** (a token was positively observed) — the
      opposite of proof. The token may be exactly what opened the repo, so
      any earlier proof is *revoked* (memory + marker), not merely left
      alone. This is the codex P0 case: a repo pulled anonymously while
      public, made private afterwards, and then re-opened with a token
      must stop naming itself.
    * **Cannot tell** — nothing happens. Neither recorded nor revoked.

    The proof is persisted as a marker file inside the repo's own HF cache
    directory, because the load that proves a model public is usually the
    cold pull, while the telemetry that wants to name it comes from every
    later warm start. The marker dies with the cached repo, and expires on
    its own after :data:`PUBLIC_PROOF_TTL_SECONDS`.

    Never raises: this sits on the download path.
    """
    global _auth_seen
    try:
        if not isinstance(repo_id, str) or not _HF_REPO_RE.match(repo_id):
            return
        state = True if _auth_seen else hf_auth_state()
        if state is True:
            # Record the observation we just made. Re-probing here (the
            # old ``hf_auth_in_use()`` call) could miss it: a token
            # cleared between the two probes left the latch unset.
            _auth_seen = True
            _revoke_proof(repo_id)
            return
        if state is None:
            # We could not tell whether this round trip carried a token.
            # That is not proof — and it is not grounds to destroy proof
            # either. Revoking on "cannot tell" would let one transient
            # unreadable-token-file error delete, for every future
            # process, a marker that a genuinely anonymous pull earned.
            return
        if not hub_endpoint_is_canonical():
            # Anonymous 200 from an endpoint we do not control is not
            # evidence of public readability on huggingface.co. Revocation
            # above is deliberately NOT gated on this — that direction is
            # always safe.
            return
        now = time.time()
        with _proven_lock:
            _proven_public[repo_id] = now
        _write_marker(repo_id, now)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return


def is_proven_public(repo_id: str) -> bool:
    """Whether an anonymous Hub fetch of ``repo_id`` succeeded *recently*.

    "Recently" is :data:`PUBLIC_PROOF_TTL_SECONDS` and no further into the
    future than :data:`_CLOCK_SKEW_GRACE_SECONDS`; anything outside that
    window — including a non-finite stamp — is not proof until a fresh
    anonymous fetch re-proves it.

    This answers "does fresh proof exist", NOT "may we report this id" —
    the token check lives in :func:`telemetry_model_id`, which is the only
    place that decides what goes on the wire.
    """
    try:
        now = time.time()
        with _proven_lock:
            remembered = _proven_public.get(repo_id)
        if remembered is not None and _is_fresh(remembered, now):
            return True
        stamp = _read_marker(repo_id)
        if stamp is None or not _is_fresh(stamp, now):
            return False
        with _proven_lock:
            _proven_public[repo_id] = stamp
        return True
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return False


def _reset_for_tests() -> None:
    """Drop the in-process proof cache and the auth latch. Tests-only seam."""
    global _auth_seen
    _auth_seen = False
    with _proven_lock:
        _proven_public.clear()


# ----------------------------------------------------------------- the rule


def _looks_local(ref: str) -> bool:
    from rapid_mlx.telemetry.redact import normalize_model_path

    if normalize_model_path(ref) == LOCAL:
        return True
    try:
        return os.path.exists(ref)
    except (OSError, ValueError):
        return True


def _capped(value: str) -> str:
    """Pattern + length cap — the last gate before a value can be reported."""
    if len(value) > _MAX_ID_LEN or not _MODEL_ID_RE.match(value):
        return CUSTOM
    return value


def telemetry_model_id(model_ref: object) -> str:
    """Map a model reference to its privacy-safe telemetry identity.

    ``model_ref`` should be the RESOLVED checkpoint (``ModelEntry.model_path``
    / the output of ``resolve_model``), never a served name and never a
    client-supplied ``request.model``. Passing one of those is not unsafe —
    the rules below reject anything unproven — it just yields ``<custom>``.

    Total function: never raises, always returns one of a catalog alias, a
    proven-public ``org/name``, ``"<local>"`` or ``"<custom>"``.
    """
    try:
        if not isinstance(model_ref, str):
            return CUSTOM
        ref = model_ref.strip()
        if not ref:
            return CUSTOM
        if ref in _SENTINELS:
            return ref
        if _looks_local(ref):
            return LOCAL
        from rapid_mlx.model_aliases import catalog_alias_for

        alias = catalog_alias_for(ref)
        if alias is not None:
            return _capped(alias)
        from rapid_mlx.audio.registry import resolve_audio_alias

        audio_entry = resolve_audio_alias(ref)
        if audio_entry is not None:
            return _capped(audio_entry.alias)
        if _HF_REPO_RE.match(ref):
            # Auth outranks stored proof: while a token is in use we cannot
            # tell a public repo from one the token opened, and the repo may
            # have become gated since the proof was taken.
            if hf_auth_in_use():
                return CUSTOM
            if is_proven_public(ref):
                return _capped(ref)
        return CUSTOM
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return CUSTOM


def _entry_telemetry_id(entry: object) -> str:
    """The reportable id of a registry ``entry``.

    Prefers the stamp computed once at load time; re-derives it from the
    resolved checkpoint when an entry predates the stamp.
    """
    stored = getattr(entry, "telemetry_model_id", None)
    if isinstance(stored, str) and stored:
        # Already the product of the full rule, applied at load time.
        # Re-running it here would only re-stat the disk on every request;
        # the cap is kept as the cheap part.
        if stored in _SENTINELS:
            return stored
        if _HF_REPO_RE.match(stored):
            # Except for a stamped repo id, whose licence can have lapsed
            # since load: a token may have appeared, or the proof expired.
            # Re-run the full rule.
            return telemetry_model_id(stored)
        return _capped(stored)
    return telemetry_model_id(getattr(entry, "model_path", None))


def engine_telemetry_id(engine: object) -> str:
    """The telemetry id of the model ``engine`` is, captured at selection.

    Codex P1 on #3600: :func:`served_model_id` re-resolves the registry *by
    name* at request completion, so a resident-model swap or a
    ``set_default`` between "start the request" and "emit the event" made
    the event name the wrong model — most concretely for the Anthropic
    surface, where ``claude-*`` names deliberately fall through to the
    default engine.

    This looks the entry up by **engine identity** instead, at request
    start, and the routes carry the returned string to the terminal emit.
    The engine object is the thing that actually ran the tokens, so no
    later registry mutation can repoint it; an entry that has been unloaded
    by the time we look is reported as ``<custom>`` rather than guessed at.

    Never raises: a telemetry lookup may not break a request.
    """
    try:
        from rapid_mlx.config.server_config import get_config

        cfg = get_config()
    except Exception:
        return CUSTOM
    try:
        registry = getattr(cfg, "model_registry", None)
        if registry:
            for entry in registry.list_entries():
                if getattr(entry, "engine", None) is engine:
                    return _entry_telemetry_id(entry)
            return CUSTOM
        # Single-model server with no registry: ``model_path`` is the
        # resolved checkpoint, ``model_name`` may be a served name.
        return telemetry_model_id(getattr(cfg, "model_path", None))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return CUSTOM


def served_model_id(requested: object = None) -> str:
    """The telemetry id of the model a request for ``requested`` routes to.

    **Prefer :func:`engine_telemetry_id` on the request path.** This
    function resolves the registry live, so between request start and
    request completion its answer can change; it remains the fallback for
    call sites that have no engine in hand.

    Route call sites pass the client's ``request.model`` purely as a routing
    key: it selects which loaded model answered, exactly as
    ``ModelRegistry.get_engine`` does, and the id returned is the one
    computed from that entry's resolved checkpoint at LOAD time. The
    caller's string never reaches a payload.
    """
    try:
        from rapid_mlx.config.server_config import get_config

        cfg = get_config()
    except Exception:
        return CUSTOM
    try:
        registry = getattr(cfg, "model_registry", None)
        if registry:
            key = requested if isinstance(requested, str) and requested else None
            try:
                entry = registry.get_entry(key)
            except Exception:
                entry = None
            if entry is not None:
                return _entry_telemetry_id(entry)
        # Single-model server with no registry: ``model_path`` is the
        # resolved checkpoint, ``model_name`` may be a served name — only
        # the former may be consulted.
        return telemetry_model_id(getattr(cfg, "model_path", None))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return CUSTOM
