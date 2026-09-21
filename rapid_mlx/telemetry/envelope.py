# SPDX-License-Identifier: Apache-2.0
"""PostHog Cloud batch envelope builder — pure dict shaping, no network.

Telemetry v2 ships Orca-style product events to PostHog Cloud's
``/batch/`` endpoint, whose wire contract is (all JSON)::

    {"api_key": <project api key>, "batch": [<item>, ...]}

where every item is exactly::

    {"event": <name>, "distinct_id": <install_id>,
     "timestamp": <ISO-8601 UTC with "Z", second precision>,
     "properties": {...}}

This module owns that SHAPE and nothing else. Inputs are validated here
by :mod:`rapid_mlx.telemetry.registry`; like the registry, everything is
strict and fail-closed — a rejected input yields ``None`` for the WHOLE
item or batch, never a partially valid payload, and no function here
ever raises, opens a socket, or touches the filesystem. The sender that
POSTs the envelope is a later block.

Two ``$`` properties are stamped onto EVERY item, and both are privacy
load-bearing:

- ``$geoip_disable`` — PostHog discards the ``$ip`` property server-side
  by default, but that discard does not stop GeoIP enrichment from
  reading the CONNECTING IP. Without this flag an install's coarse
  location would silently attach to every event, and location is not in
  the schema (design sec 1.5).
- ``$process_person_profile`` — PostHog creates a person profile per
  ``distinct_id`` by default. Installs are anonymous counters, not
  profiles; one profile per install would both add persistence the
  design forbids and burn PostHog's per-project profile quota.

Both are stamped AFTER the common/event merge, so no validated property
can displace them; a collision between event and common props drops the
whole item instead, so an event prop can never shadow a common one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from rapid_mlx.telemetry import registry

# PostHog accepts at most 100 items per /batch/ request; callers with
# more events chunk themselves (the sender owns that policy).
MAX_BATCH_ITEMS = 100

_GEOIP_DISABLE_KEY = "$geoip_disable"
_PROCESS_PERSON_PROFILE_KEY = "$process_person_profile"


def _utc_timestamp(moment: datetime) -> str:
    """ISO-8601 UTC, ``Z`` suffix, second precision — PostHog's shape."""

    if moment.tzinfo is None:
        # A naive datetime carries no zone: read it as UTC. Handing it
        # to astimezone() would silently read it as LOCAL time and move
        # the stamp for everyone not running on UTC.
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    # isoformat rather than strftime: the year is zero-padded on every
    # platform, unlike strftime for years < 1000.
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def build_batch_item(
    event_name: str,
    props: Mapping[str, object],
    common: Mapping[str, object],
    *,
    occurred_at: datetime | None = None,
) -> dict[str, object] | None:
    """Shape one registry-validated event into one PostHog batch item.

    ``props`` and ``common`` are validated by the registry first; EITHER
    failing drops the whole item. ``occurred_at`` defaults to now (UTC);
    a naive value is read as UTC and any non-datetime drops the item.
    Never raises: hostile mappings (methods that explode) drop the item
    like any other rejection.
    """

    try:
        event_props: dict[str, object] | None = registry.validate(
            event_name, dict(props)
        )
        common_props: dict[str, object] | None = registry.validate_common(dict(common))
        if event_props is None or common_props is None:
            return None

        # Registry kinds keep the two vocabularies disjoint today, but a
        # future registry edit could overlap them; a shared key would let
        # one side shadow the other, so fail closed instead of merging.
        if event_props.keys() & common_props.keys():
            return None

        if occurred_at is None:
            moment = datetime.now(timezone.utc)
        elif isinstance(occurred_at, datetime):
            moment = occurred_at
        else:
            return None

        properties: dict[str, object] = dict(common_props)
        properties.update(event_props)
        # Stamped last: nothing validated above can displace these.
        properties[_GEOIP_DISABLE_KEY] = True
        properties[_PROCESS_PERSON_PROFILE_KEY] = False
        return {
            "event": event_name,
            "distinct_id": common_props["install_id"],
            "timestamp": _utc_timestamp(moment),
            "properties": properties,
        }
    except Exception:
        # Garbage in, None out — telemetry must never take serve down.
        return None


def _snapshot_item(item: Mapping[str, object]) -> dict[str, object]:
    """One-level copy of an item: the item itself plus its ``properties``.

    ``build_batch_item`` nests exactly one level, so copying those two
    layers is what makes the envelope immune to later caller edits: the
    sender queues items between build and POST, and a write into
    ``item["properties"]`` in that window would otherwise reach the wire
    without ever passing the registry — the one non-fail-closed seam in
    this module.
    """

    snapshot = dict(item)
    properties = snapshot.get("properties")
    if isinstance(properties, Mapping):
        snapshot["properties"] = dict(properties)
    return snapshot


def build_batch(
    items: Sequence[Mapping[str, object]],
    api_key: str,
) -> dict[str, object] | None:
    """Frame already-built items into the ``/batch/`` envelope.

    ``None`` for an empty sequence, an empty or non-str ``api_key``, or
    more than :data:`MAX_BATCH_ITEMS` items (the caller chunks). Inputs
    are never mutated, and the envelope is isolated from later edits:
    each item is copied one level deep — the item itself plus its
    nested ``properties`` mapping when present (see
    :func:`_snapshot_item`) — so writes into ``item["properties"]``
    after the call cannot reach the wire. Never raises.
    """

    try:
        if not isinstance(api_key, str) or not api_key:
            return None
        if len(items) == 0 or len(items) > MAX_BATCH_ITEMS:
            return None
        batch: list[dict[str, object]] = [_snapshot_item(item) for item in items]
        return {"api_key": api_key, "batch": batch}
    except Exception:
        # Garbage in, None out — see build_batch_item.
        return None
