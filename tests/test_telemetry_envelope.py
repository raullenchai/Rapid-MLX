# SPDX-License-Identifier: Apache-2.0
"""Contract pins for ``rapid_mlx.telemetry.envelope``.

The envelope builder is the last pure gate before the wire: it must
shape registry-validated events into exactly PostHog's ``/batch/`` item
shape, stamp the two mandatory ``$`` privacy keys after the merge, fail
closed on any invalid input, and never raise. Each test here guards ONE
of those rules and is written so that removing the rule turns it red —
that was verified by fault injection before merge, not assumed.

Valid ``common`` props are CONSTRUCTED from ``events.json`` (enums from
the registry's own value lists, ints from the declared bounds) rather
than copied by hand, so a registry edit that re-meaning a vocabulary
breaks these tests loudly instead of letting a stale fixture pass.
"""

from __future__ import annotations

import copy
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import NoReturn

import pytest

from rapid_mlx.telemetry import envelope as env
from rapid_mlx.telemetry import registry as reg

_UUID = "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
_SESSION = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# The registry's version kind has a per-key pattern, so each version
# field carries a concrete string matching ITS pattern; the two pin
# tests below run the samples back through the registry itself, so a
# registry edit that invalidates a sample fails HERE, loudly.
_VERSION_SAMPLES = {
    "app_version": "0.15.0",
    "os_version": "25.3",
    "python_version": "3.12",
}


def _common_sample() -> dict[str, object]:
    """A valid common-props dict derived from events.json, not guessed."""

    loaded = reg.load_registry()
    sample: dict[str, object] = {}
    for name, spec in loaded["common_props"].items():
        if name.startswith("_"):
            continue
        kind = spec["kind"]
        if kind == "enum":
            sample[name] = loaded["enums"][spec["enum"]]["values"][0]
        elif kind == "int":
            sample[name] = spec["min"]
        elif kind == "uuid":
            sample[name] = _UUID if name == "install_id" else _SESSION
        else:  # kind == "version", the only remaining shipped kind
            sample[name] = _VERSION_SAMPLES[name]
    return sample


def _served_sample() -> dict[str, object]:
    """Valid ``model_served`` props, likewise derived from the registry."""

    loaded = reg.load_registry()
    specs = loaded["events"]["model_served"]["props"]
    return {
        "model": loaded["model_id"]["reserved"][0],
        "model_type": loaded["enums"][specs["model_type"]["enum"]]["values"][0],
        "auto_selected": True,
        "quant": loaded["enums"][specs["quant"]["enum"]]["values"][0],
    }


def _props(item: dict[str, object]) -> dict[str, object]:
    """The item's ``properties`` dict, with a shape guard on the way in."""

    props = item["properties"]
    assert isinstance(props, dict)
    return props


@pytest.fixture
def _tokyo_local_time(monkeypatch):
    """Pin local time to UTC+9 so 'local vs UTC' is observable.

    ``time.tzset()`` is POSIX; the repo's supported hosts (macOS, Linux
    CI) all have it. The fixture un-pins on the way out so later tests
    see the host's real zone again.
    """

    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is required to pin a non-UTC local zone")
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


class _Exploding:
    """Every mapping/sequence protocol method raises."""

    def __iter__(self) -> NoReturn:
        raise RuntimeError("exploded __iter__")

    def keys(self) -> NoReturn:
        raise RuntimeError("exploded keys")

    def items(self) -> NoReturn:
        raise RuntimeError("exploded items")

    def __getitem__(self, key: object) -> NoReturn:
        raise RuntimeError("exploded __getitem__")

    def __len__(self) -> NoReturn:
        raise RuntimeError("exploded __len__")


# ------------------------------------------------------------- sample pin


def test_common_sample_round_trips_through_the_registry():
    """The constructed common sample must be valid by the book."""

    sample = _common_sample()
    assert reg.validate_common(sample) == sample


def test_served_sample_round_trips_through_the_registry():
    sample = _served_sample()
    assert reg.validate("model_served", sample) == sample


# ------------------------------------------------------------- happy path


def test_item_shape_for_an_event_with_props():
    item = env.build_batch_item("model_served", _served_sample(), _common_sample())
    assert item is not None
    # The privacy red line: exactly these five top-level fields.
    assert set(item) == {"uuid", "event", "distinct_id", "timestamp", "properties"}
    item_uuid = item["uuid"]
    assert isinstance(item_uuid, str)
    assert str(uuid.UUID(item_uuid)) == item_uuid
    assert item["event"] == "model_served"
    assert item["distinct_id"] == _UUID
    timestamp = item["timestamp"]
    assert isinstance(timestamp, str)
    assert _TIMESTAMP_RE.fullmatch(timestamp)
    properties = _props(item)
    assert set(properties) == (
        set(_common_sample())
        | set(_served_sample())
        | {"$geoip_disable", "$process_person_profile"}
    )
    assert properties["model"] == _served_sample()["model"]
    assert properties["install_id"] == _UUID
    assert properties["$geoip_disable"] is True
    assert properties["$process_person_profile"] is False


def test_item_shape_for_an_event_without_props():
    item = env.build_batch_item("app_opened", {}, _common_sample())
    assert item is not None
    assert item["event"] == "app_opened"
    properties = _props(item)
    assert set(properties) == set(_common_sample()) | {
        "$geoip_disable",
        "$process_person_profile",
    }
    # install_id is named exactly once inside properties and REFERENCED
    # (never copied under a second name) as distinct_id.
    assert item["distinct_id"] == properties["install_id"]
    assert "distinct_id" not in properties


def test_each_item_gets_a_fresh_uuid():
    first = env.build_batch_item("app_opened", {}, _common_sample())
    second = env.build_batch_item("app_opened", {}, _common_sample())
    assert first is not None
    assert second is not None
    assert first["uuid"] != second["uuid"]


def test_item_snapshots_the_validated_props_not_the_caller_dicts():
    props = _served_sample()
    common = _common_sample()
    item = env.build_batch_item("model_served", props, common)
    assert item is not None
    props["model"] = "tampered"
    common["app_version"] = "9.9.9"
    properties = _props(item)
    assert properties["model"] == _served_sample()["model"]
    assert properties["app_version"] == "0.15.0"


# -------------------------------------------------------------- timestamp


def test_timestamp_defaults_to_utc_now():
    item = env.build_batch_item("app_opened", {}, _common_sample())
    assert item is not None
    timestamp = item["timestamp"]
    assert isinstance(timestamp, str)
    assert _TIMESTAMP_RE.fullmatch(timestamp)
    parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    assert abs(parsed - datetime.now(timezone.utc)) < timedelta(seconds=30)


def test_aware_non_utc_timestamp_is_converted_to_utc():
    moment = datetime(2024, 6, 1, 12, 30, 45, tzinfo=timezone(timedelta(hours=2)))
    item = env.build_batch_item("app_opened", {}, _common_sample(), occurred_at=moment)
    assert item is not None
    assert item["timestamp"] == "2024-06-01T10:30:45Z"


def test_timestamp_is_truncated_to_second_precision():
    moment = datetime(2024, 6, 1, 12, 30, 45, 999999, tzinfo=timezone.utc)
    item = env.build_batch_item("app_opened", {}, _common_sample(), occurred_at=moment)
    assert item is not None
    assert item["timestamp"] == "2024-06-01T12:30:45Z"


def test_naive_timestamp_is_read_as_utc_never_local(_tokyo_local_time):
    """12:30:45 naive must ship as 12:30:45Z even with local at +09:00.

    If the builder ever ran a naive datetime through ``astimezone()``,
    Python would read it as LOCAL time and this assertion would flip to
    03:30:45Z — that injected fault is what this test pins against.
    """

    item = env.build_batch_item(
        "app_opened",
        {},
        _common_sample(),
        occurred_at=datetime(2024, 6, 1, 12, 30, 45),
    )
    assert item is not None
    assert item["timestamp"] == "2024-06-01T12:30:45Z"


# -------------------------------------------------------------- fail closed


def test_unknown_event_name_drops_the_item():
    assert env.build_batch_item("model_teleported", {}, _common_sample()) is None


def test_unknown_event_prop_key_drops_the_item():
    props = {**_served_sample(), "prompt": "free-form must not sneak in"}
    assert env.build_batch_item("model_served", props, _common_sample()) is None


def test_invalid_common_props_drop_the_item():
    bad = _common_sample()
    bad["install_id"] = "not-a-uuid"
    assert env.build_batch_item("app_opened", {}, bad) is None


@pytest.mark.parametrize("bad", ["now", 20240601, 20240601.5, object()])
def test_non_datetime_occurred_at_drops_the_item(bad):
    assert (
        env.build_batch_item(  # type: ignore[arg-type]
            "app_opened", {}, _common_sample(), occurred_at=bad
        )
        is None
    )


def test_event_prop_cannot_shadow_a_common_prop(monkeypatch):
    """A key shared by event and common props drops the WHOLE item.

    The shipped registry keeps the two vocabularies disjoint by kind, so
    the validator seams are patched here to produce the overlap the
    envelope must defend against.
    """

    monkeypatch.setattr(
        env.registry, "validate", lambda name, props: {"install_id": "event-wins"}
    )
    monkeypatch.setattr(
        env.registry, "validate_common", lambda props: {"install_id": _UUID}
    )
    assert env.build_batch_item("model_served", {}, {}) is None


def test_disjoint_props_merge_under_the_same_patch(monkeypatch):
    """Companion to the collision test: no overlap -> the item ships."""

    monkeypatch.setattr(env.registry, "validate", lambda name, props: {"model": "q"})
    monkeypatch.setattr(
        env.registry, "validate_common", lambda props: {"install_id": _UUID}
    )
    item = env.build_batch_item("model_served", {}, {})
    assert item is not None
    assert _props(item) == {
        "model": "q",
        "install_id": _UUID,
        "$geoip_disable": True,
        "$process_person_profile": False,
    }


def test_dollar_keys_cannot_be_shadowed_by_validated_props(monkeypatch):
    """The $ keys are stamped after the merge, so they always win."""

    monkeypatch.setattr(
        env.registry,
        "validate",
        lambda name, props: {
            "$geoip_disable": False,
            "$process_person_profile": True,
        },
    )
    item = env.build_batch_item("model_served", {}, _common_sample())
    assert item is not None
    properties = _props(item)
    assert properties["$geoip_disable"] is True
    assert properties["$process_person_profile"] is False


# -------------------------------------------------------------- build_batch


def test_build_batch_frames_items_for_the_wire():
    item = env.build_batch_item("app_opened", {}, _common_sample())
    assert item is not None
    assert env.build_batch([item], "phc_test_key") == {
        "api_key": "phc_test_key",
        "batch": [item],
    }


@pytest.mark.parametrize(
    "bad_uuid",
    [None, 123, "", "not-a-uuid", object()],
)
def test_build_batch_rejects_missing_or_malformed_item_uuid(bad_uuid):
    item = env.build_batch_item("app_opened", {}, _common_sample())
    assert item is not None
    if bad_uuid is None:
        del item["uuid"]
    else:
        item["uuid"] = bad_uuid
    assert env.build_batch([item], "k") is None


def test_snapshot_item_rejects_uuid_object_instead_of_coercing_it():
    item = {"uuid": uuid.UUID(_UUID), "event": "app_opened", "properties": {}}
    assert env._snapshot_item(item) is None


@pytest.mark.parametrize(
    "spelling",
    [
        "6F1B1D3E-4A2B-4C9D-8E7F-0A1B2C3D4E5F",
        "{6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f}",
        "urn:uuid:6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f",
        "6f1b1d3e4a2b4c9d8e7f0a1b2c3d4e5f",
    ],
)
def test_build_batch_normalizes_uuid_spellings(spelling):
    item = env.build_batch_item("app_opened", {}, _common_sample())
    assert item is not None
    item["uuid"] = spelling
    batch = env.build_batch([item], "k")
    assert batch is not None
    assert batch["batch"][0]["uuid"] == "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"


def test_build_batch_rejects_an_empty_sequence():
    assert env.build_batch([], "phc_test_key") is None


@pytest.mark.parametrize("key", [None, 123, b"phc", ""])
def test_build_batch_rejects_a_bad_api_key(key):
    assert env.build_batch([{"event": "app_opened"}], key) is None  # type: ignore[arg-type]


def test_build_batch_enforces_the_100_item_cap():
    # 100 is PostHog's external /batch/ contract, not an internal knob:
    # pin the constant itself, otherwise raising it leaves this suite
    # green (both sides would read the same constant).
    assert env.MAX_BATCH_ITEMS == 100
    item = env.build_batch_item("app_opened", {}, _common_sample())
    assert item is not None
    assert env.build_batch([item] * env.MAX_BATCH_ITEMS, "k") is not None
    assert env.build_batch([item] * (env.MAX_BATCH_ITEMS + 1), "k") is None


def test_build_batch_does_not_mutate_its_inputs():
    item = env.build_batch_item("app_opened", {}, _common_sample())
    assert item is not None
    items: list[dict[str, object]] = [dict(item)]
    snapshot = copy.deepcopy(items)
    batch = env.build_batch(items, "k")
    assert batch is not None
    # Edits after the call cannot reach the input, in either direction:
    # the envelope is a copy and the input list is untouched.
    batch["batch"][0]["event"] = "edited-after-the-fact"
    assert items == snapshot


def test_build_batch_snapshots_nested_properties_against_later_writes():
    """Writes into ``item["properties"]`` after build_batch must not leak.

    The sender queues items between build and POST; with a shallow copy
    the nested ``properties`` dict stays aliased to the caller, so a
    write in that window reaches the wire without ever passing the
    registry — the one non-fail-closed seam in the module. Red before
    the one-level snapshot was added: both writes below showed up
    inside ``batch["batch"][0]["properties"]`` (same object).
    """

    item = env.build_batch_item("app_opened", {}, _common_sample())
    assert item is not None
    items: list[dict[str, object]] = [dict(item)]
    batch = env.build_batch(items, "k")
    assert batch is not None
    envelope_snapshot = copy.deepcopy(batch)

    queued_properties = items[0]["properties"]
    assert isinstance(queued_properties, dict)
    queued_properties["chip"] = "LEAKED /Users/alice/secret"  # overwrite
    queued_properties["prompt"] = "user typed this"  # inject new key

    assert batch == envelope_snapshot


def test_build_batch_copies_items_without_a_properties_mapping():
    """An item with no ``properties`` key still gets its one-level copy.

    build_batch is structural: it does not require build_batch_item's
    exact shape, so an item without a nested mapping simply has nothing
    extra to snapshot.
    """

    items: list[dict[str, object]] = [{"uuid": _UUID, "event": "app_opened"}]
    assert env.build_batch(items, "k") == {
        "api_key": "k",
        "batch": [{"uuid": _UUID, "event": "app_opened"}],
    }


# --------------------------------------------------------------- never raise


@pytest.mark.parametrize("garbage", [None, 42, "app_opened", _Exploding()])
def test_build_batch_item_never_raises_on_garbage(garbage):
    """Garbage in ANY position comes out as None, never an exception."""

    assert env.build_batch_item(garbage, garbage, garbage, occurred_at=garbage) is None
    assert env.build_batch_item("app_opened", garbage, _common_sample()) is None
    assert env.build_batch_item("app_opened", _common_sample(), garbage) is None


@pytest.mark.parametrize("garbage", [None, 42, _Exploding()])
def test_build_batch_never_raises_on_garbage(garbage):
    assert env.build_batch(garbage, "k") is None
    assert env.build_batch([{"event": "app_opened"}], garbage) is None  # type: ignore[arg-type]


def test_build_batch_never_raises_on_a_hostile_item():
    assert env.build_batch([_Exploding()], "k") is None


def test_build_batch_never_raises_on_a_hostile_api_key():
    class _ExplodingKey(str):
        def __len__(self) -> int:
            raise RuntimeError("exploded __len__")

    assert env.build_batch([{"event": "app_opened"}], _ExplodingKey("k")) is None
