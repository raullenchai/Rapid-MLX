#!/usr/bin/env python3
"""Assert menu-card AX evidence, scoped to each control and its descendants."""

import json
import math
import sys
from pathlib import Path

PREFIX = "MenuBar.Resources."


def validate(payload, state, physical_bytes):
    assert payload.get("success") is True, "AX dump failed"
    data = payload["data"]
    assert data["walk"]["complete"] is True, "incomplete AX walk"
    elements = data["ui_elements"]

    def scope(identifier):
        indexes = [
            i for i, e in enumerate(elements) if e.get("identifier") == identifier
        ]
        assert len(indexes) == 1, f"expected one {identifier}"
        start = indexes[0]
        end = start + 1
        while end < len(elements) and elements[end]["depth"] > elements[start]["depth"]:
            end += 1
        return elements[start:end]

    def text(identifier):
        return " ".join(
            str(e.get(key, ""))
            for e in scope(identifier)
            for key in ("title", "description", "value")
        )

    def enabled(identifier, expected):
        assert scope(identifier)[0].get("enabled") is expected, identifier

    rows = {
        e.get("identifier")
        for e in elements
        if str(e.get("identifier", "")).startswith(PREFIX + "Model.")
    }
    expected = set() if state == "idle" else {PREFIX + "Model.fake-alias"}
    if state == "two":
        expected.add(PREFIX + "Model.fake-evictable-alias")
    assert rows == expected, f"wrong model rows: {rows}"
    used = {"idle": "0.00", "one": "2.50", "two": "3.50"}[state]
    memory = f"{used} GB / {physical_bytes / 1e9:.2f} GB"
    assert memory in text(PREFIX + "Memory"), "incorrect used/physical memory"
    # Native progress indicators expose a numeric AXValue, even when SwiftUI
    # supplies a spoken memory label. Check the fill against its native range.
    meter = scope(PREFIX + "Meter")[0]
    value, minimum, maximum = (
        meter.get(key) for key in ("value", "min_value", "max_value")
    )
    assert all(
        type(v) in (int, float) and math.isfinite(v) for v in (value, minimum, maximum)
    ), "missing numeric meter range"
    assert minimum <= value <= maximum and maximum > minimum, "invalid meter range"
    expected_fraction = min(1, float(used) * 1e9 / physical_bytes)
    assert math.isclose(
        (value - minimum) / (maximum - minimum), expected_fraction, abs_tol=1e-6
    ), "incorrect meter memory"
    if state == "idle":
        assert "Idle" in text(PREFIX + "Activity"), "idle copy missing"
        # The empty message has no identifier; scope it between the meter and
        # whole-pool action, excluding unrelated status/sidebar text.
        meter = elements.index(scope(PREFIX + "Meter")[0])
        unload = elements.index(scope(PREFIX + "UnloadAll")[0])
        assert any(
            "Unloaded" in str(e.get(k, ""))
            for e in elements[meter:unload]
            for k in ("title", "description", "value")
        )
    else:
        assert "Generating" in text(PREFIX + "Activity")
        assert "42.5 tok/s" in text(PREFIX + "Activity")
        assert "fake-alias" in text(PREFIX + "Model.fake-alias")
        assert "2.50 GB" in text(PREFIX + "Model.fake-alias")
        enabled(PREFIX + "Unload.fake-alias", False)
        if state == "two":
            assert "fake-evictable-alias" in text(PREFIX + "Model.fake-evictable-alias")
            assert "1.00 GB" in text(PREFIX + "Model.fake-evictable-alias")
            enabled(PREFIX + "Unload.fake-evictable-alias", True)
    enabled(PREFIX + "UnloadAll", False)


if __name__ == "__main__":
    try:
        validate(
            json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], int(sys.argv[3])
        )
    except (AssertionError, KeyError, ValueError) as error:
        sys.exit(str(error) or "resource card assertion failed")
