# SPDX-License-Identifier: Apache-2.0
"""Telemetry v2 event registry — loader + strict validator.

``events.json`` next to this module is the single source of truth (see
telemetry-design.md sec 1.3). Python reads it here, Swift reads a
byte-identical copy bundled into the desktop app, and a pytest drift
check keeps the two from diverging.

**Nothing here transmits.** This module only answers "is this event
shape allowed on the wire?". Block 4 wires it to a transport.

Strict semantics, copied from Orca's ``src/main/telemetry/validator.ts``
(``.strict()`` on every per-event schema + ``safeParse``). Every failure
drops the WHOLE event; we never strip the offending key and send the
rest, because a caller that got one key wrong has told us nothing about
whether the rest is trustworthy:

- unknown event name                        -> ``None``
- unknown property key                      -> ``None`` (event dropped)
- missing required property                 -> ``None``
- wrong type / out-of-enum / out-of-range   -> ``None``
- string past its length cap or pattern     -> ``None``

:func:`validate` never raises. A malformed event must not be able to
take ``serve`` down (design sec 1.1, "failure suppression unchanged"),
and a misbehaving caller must not be able to DoS stderr either, so the
debug log fires at most once per event name per process.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import Any

# The two property kinds that may appear ONLY in ``common_props``. Event
# properties are restricted to the closed set below them; the drift test
# (tests/test_telemetry_registry_drift.py) pins both lists.
_COMMON_ONLY_KINDS: frozenset[str] = frozenset({"version", "uuid"})
_EVENT_KINDS: frozenset[str] = frozenset({"enum", "bool", "int", "model_id"})

# RFC 4122 shape, any version. ``install_id`` / ``session_id`` are the
# only fields carrying one and both are generated locally.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Rate-limit ledger for the debug log: one line per event name per
# process, as Orca's validator does per 60 s window. Bounded by the
# registry's event count plus one slot for unknown names.
_LOGGED: set[str] = set()
_UNKNOWN_LOG_KEY = "<unknown-event>"


def registry_path() -> Any:
    """The ``importlib.resources`` traversable for ``events.json``.

    Kept public so the drift check can hash the exact bytes the wheel
    ships rather than a path it reconstructed itself.
    """

    return files("rapid_mlx.telemetry").joinpath("events.json")


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    """Parse ``events.json`` once per process.

    Raises only if the shipped file is missing or unparseable — that is a
    packaging bug, not a runtime condition, and it must be loud.
    """

    parsed: dict[str, Any] = json.loads(registry_path().read_text(encoding="utf-8"))
    return parsed


def registry_version() -> int:
    return int(load_registry()["registry_version"])


def event_names() -> frozenset[str]:
    return frozenset(load_registry()["events"])


def _log_once(key: str, message: str) -> None:
    """Debug-only, once per ``key`` per process. Never raises."""

    if key in _LOGGED:
        return
    _LOGGED.add(key)
    try:
        from rapid_mlx.telemetry.transport import _log

        _log(f"registry: {message}")
    except Exception:  # pragma: no cover - logging must never propagate
        pass


def _reset_log_state_for_tests() -> None:
    """Test-only: clear the once-per-process log ledger."""

    _LOGGED.clear()


def _check_value(spec: dict[str, Any], value: Any, reg: dict[str, Any]) -> bool:
    """One property against one spec. Returns True when it may ship."""

    kind = spec.get("kind")

    if kind == "bool":
        # ``isinstance(True, int)`` is True in Python, so bool must be
        # checked before int and int must exclude bool (below).
        return isinstance(value, bool)

    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        low: int = spec["min"]
        high: int = spec["max"]
        return low <= value <= high

    if kind == "enum":
        if not isinstance(value, str):
            return False
        return value in reg["enums"][spec["enum"]]["values"]

    if kind == "model_id":
        model_id = reg["model_id"]
        if not isinstance(value, str) or len(value) > model_id["max_length"]:
            return False
        return re.fullmatch(model_id["pattern"], value) is not None

    if kind in ("version", "uuid"):
        if not isinstance(value, str) or len(value) > spec.get("max_length", 64):
            return False
        if kind == "uuid":
            return _UUID_RE.fullmatch(value) is not None
        return re.fullmatch(spec["pattern"], value) is not None

    # An unrecognised kind means the registry itself is malformed. Fail
    # closed rather than letting an unchecked value through — pinned by
    # tests/test_telemetry_registry.py::test_unknown_kind_fails_closed.
    return False


def _validate_props(
    props: Any,
    specs: dict[str, Any],
    reg: dict[str, Any],
    log_key: str,
    label: str,
) -> dict[str, Any] | None:
    if not isinstance(props, dict):
        _log_once(log_key, f"{label}: props is {type(props).__name__}, not a dict")
        return None

    for key in props:
        if key not in specs:
            # Whole event dropped, NOT the key stripped: see module docstring.
            _log_once(log_key, f"{label}: unknown property {key!r} — event dropped")
            return None

    # ``specs`` arrives already stripped of the ``_``-prefixed documentation
    # keys — both callers filter them, so there is nothing to skip here.
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        if name not in props:
            if spec.get("required", False):
                _log_once(log_key, f"{label}: missing required property {name!r}")
                return None
            continue
        value = props[name]
        if not _check_value(spec, value, reg):
            _log_once(log_key, f"{label}: property {name!r} rejected")
            return None
        out[name] = value

    return out


def validate(event_name: str, props: dict[str, Any]) -> dict[str, Any] | None:
    """Return the accepted property dict, or ``None`` to drop the event.

    Never raises: an unexpected registry or input shape drops the event
    like any other rejection.
    """

    try:
        reg = load_registry()
        event = reg["events"].get(event_name)
        if event is None:
            _log_once(_UNKNOWN_LOG_KEY, f"unknown event {event_name!r}")
            return None
        specs = {
            name: spec
            for name, spec in event["props"].items()
            if not name.startswith("_")
        }
        return _validate_props(props, specs, reg, event_name, event_name)
    except Exception as exc:  # pragma: no cover - defence in depth
        _log_once(f"error:{event_name}", f"{event_name}: {type(exc).__name__}: {exc}")
        return None


def validate_common(props: dict[str, Any]) -> dict[str, Any] | None:
    """Same contract as :func:`validate`, for the sec 1.2 common props.

    Orca validates these before the transport initialises so a bad
    envelope fails closed instead of stamping every event with junk.
    """

    try:
        reg = load_registry()
        specs = {
            name: spec
            for name, spec in reg["common_props"].items()
            if not name.startswith("_")
        }
        return _validate_props(props, specs, reg, "<common>", "common_props")
    except Exception as exc:  # pragma: no cover - defence in depth
        _log_once("error:<common>", f"common_props: {type(exc).__name__}: {exc}")
        return None
