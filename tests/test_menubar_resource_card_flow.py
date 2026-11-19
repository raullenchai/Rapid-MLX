# SPDX-License-Identifier: Apache-2.0
"""Wire and negative-observation contracts for the menu-card golden journey."""

import importlib.util
import json
import threading
from copy import deepcopy
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "apps/rapid-mac/scripts"
SPEC = importlib.util.spec_from_file_location(
    "resource_card", SCRIPTS / "gui-resource-card.py"
)
CARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CARD)


def evidence(state, *, layout="inherited"):
    elements = []

    def add(identifier, text="", enabled=None):
        node = {"depth": 2, "identifier": CARD.PREFIX + identifier, "value": text}
        if enabled is not None:
            node["enabled"] = enabled
        elements.append(node)

    used = {"idle": "0.00", "two": "3.50", "one": "2.50"}[state]
    add("Activity", "Idle" if state == "idle" else "Generating · 42.5 tok/s")
    add("Memory", f"{used} GB / 8.00 GB · Total RAM")
    add("Meter")
    elements[-1].update(
        role="AXProgressIndicator",
        value=float(used) / 8,
        min_value=0,
        max_value=1,
    )
    if state == "idle":
        elements.append({"depth": 2, "value": "Unloaded"})
    else:

        def row(alias, amount, enabled):
            # macOS can flatten the HStack and propagate its identifier to
            # both text nodes and the button, overriding the button's ID.
            if layout == "container":
                add("Model." + alias)
                elements[-1]["role"] = "AXGroup"
            for role, value in [
                ("AXStaticText", alias),
                ("AXStaticText", amount),
                ("AXButton", ""),
            ]:
                identifier = (
                    "Unload." + alias
                    if layout == "container" and role == "AXButton"
                    else "Model." + alias
                )
                add(identifier, value, enabled if role == "AXButton" else True)
                elements[-1].update(role=role, depth=3 if layout == "container" else 2)
                if layout == "container" and role == "AXStaticText":
                    del elements[-1]["identifier"]
                if role == "AXButton":
                    elements[-1]["description"] = "Unload " + alias

        row("fake-alias", "2.50 GB", False)
        if state == "two":
            row("fake-evictable-alias", "1.00 GB", True)
    action = "Unload model" if state == "one" else "Unload all models"
    add("UnloadAll", f"{action} and free {used} GB", enabled=False)
    elements[-1]["role"] = "AXButton"
    return {
        "success": True,
        "data": {"walk": {"complete": True}, "ui_elements": elements},
    }


@pytest.mark.parametrize("state", ["idle", "two", "one"])
@pytest.mark.parametrize("layout", ["inherited", "container"])
def test_card_accepts_each_observed_state(state, layout):
    CARD.validate(evidence(state, layout=layout), state, 8_000_000_000)


@pytest.mark.parametrize("layout", ["inherited", "container"])
def test_card_two_matches_ci_memory_and_row_identity(layout):
    tree = evidence("two", layout=layout)
    nodes = tree["data"]["ui_elements"]
    nodes[1]["value"] = "3.50 GB / 17.18 GB, ·, Total RAM"
    nodes[2]["value"] = 3_500_000_000 / 17_179_869_184
    CARD.validate(tree, "two", 17_179_869_184)
    nodes[-1]["value"] = "Unload all models and free 3.26 GB"
    with pytest.raises(AssertionError, match="incorrect unload-all memory"):
        CARD.validate(tree, "two", 17_179_869_184)


def test_card_accepts_container_with_inherited_descendant_identifiers():
    tree = evidence("two", layout="container")
    for node in tree["data"]["ui_elements"]:
        if node.get("value") in {"fake-alias", "2.50 GB"}:
            node["identifier"] = CARD.PREFIX + "Model.fake-alias"
    CARD.validate(tree, "two", 8_000_000_000)


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "missing-model",
        "wrong-memory",
        "wrong-meter",
        "missing-meter-range",
        "wrong-rate",
        "pinned-enabled",
        "evictable-disabled",
        "unrelated-copy",
        "stale-row",
        "binary-unload-amount",
        "missing-unload-amount",
        "missing-row-button",
        "duplicate-row-button",
        "discontiguous-row",
        "unrelated-row-size",
    ],
)
def test_card_rejects_false_positive_evidence(mutation):
    tree = evidence("two")
    nodes = tree["data"]["ui_elements"]
    if mutation == "incomplete":
        tree["data"]["walk"]["complete"] = False
    elif mutation == "missing-model":
        nodes[:] = [
            n for n in nodes if n.get("identifier") != CARD.PREFIX + "Model.fake-alias"
        ]
    elif mutation == "wrong-memory":
        nodes[1]["value"] = "3.50 GB / 32.00 GB"
    elif mutation == "wrong-meter":
        nodes[2]["value"] = 0
    elif mutation == "missing-meter-range":
        del nodes[2]["max_value"]
    elif mutation == "wrong-rate":
        nodes[0]["value"] = "Generating · 0.0 tok/s"
    elif mutation in {"pinned-enabled", "evictable-disabled"}:
        target = (
            "fake-alias" if mutation == "pinned-enabled" else "fake-evictable-alias"
        )
        next(
            n
            for n in nodes
            if n.get("identifier") == CARD.PREFIX + "Model." + target
            and n.get("role") == "AXButton"
        )["enabled"] = mutation == "pinned-enabled"
    elif mutation == "unrelated-copy":
        nodes[0]["value"] = "Idle"
        nodes.append({"depth": 2, "value": "Generating · 42.5 tok/s"})
    elif mutation == "stale-row":
        nodes.append({"depth": 2, "identifier": CARD.PREFIX + "Model.stale"})
    elif mutation in {"binary-unload-amount", "missing-unload-amount"}:
        nodes[-1]["value"] = (
            "Unload all models and free 3.26 GB"
            if mutation == "binary-unload-amount"
            else "Unload all models"
        )
    elif mutation == "missing-row-button":
        nodes[:] = [n for n in nodes if n.get("description") != "Unload fake-alias"]
    elif mutation == "duplicate-row-button":
        index = next(
            i
            for i, n in enumerate(nodes)
            if n.get("description") == "Unload fake-alias"
        )
        nodes.insert(index, deepcopy(nodes[index]))
    elif mutation == "discontiguous-row":
        nodes.append(deepcopy(nodes[3]))
    elif mutation == "unrelated-row-size":
        next(n for n in nodes if n.get("value") == "2.50 GB")["value"] = "wrong size"
        nodes.append({"depth": 2, "value": "2.50 GB"})
    with pytest.raises(AssertionError):
        CARD.validate(tree, "two", 8_000_000_000)


@pytest.mark.parametrize("state", ["idle", "two", "one"])
def test_card_accepts_native_meter_with_percentage_range(state):
    tree = evidence(state)
    meter = tree["data"]["ui_elements"][2]
    meter["value"] *= 100
    meter["max_value"] = 100
    CARD.validate(tree, state, 8_000_000_000)


def test_one_model_state_rejects_an_evicted_row_still_visible():
    with pytest.raises(AssertionError):
        CARD.validate(evidence("two"), "one", 8_000_000_000)


def test_opt_in_fixture_serves_status_and_deletes_only_evictable_once(
    monkeypatch, tmp_path
):
    source = (
        (SCRIPTS / "fake-rapid-mlx.sh")
        .read_text()
        .split("<<'PYEOF'\n", 1)[1]
        .rsplit("\nPYEOF", 1)[0]
    )
    namespace = {"__name__": "fixture_contract"}
    exec(compile(source, str(SCRIPTS / "fake-rapid-mlx.sh"), "exec"), namespace)
    monkeypatch.setenv("FAKE_RESOURCE_CARD", "1")
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("FAKE_EVENT_LOG", str(events))
    namespace["SERVED_ALIAS"] = "fake-alias"
    server = ThreadingHTTPServer(("127.0.0.1", 0), namespace["Handler"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with urlopen(base + path, timeout=3) as response:
            return json.load(response)

    def delete(model):
        with urlopen(
            Request(base + "/v1/models/" + model, method="DELETE"), timeout=3
        ) as response:
            return response.status

    try:
        original = deepcopy(get("/v1/models/residency"))
        assert original["memory_used_bytes"] == 3_500_000_000
        assert [
            (m["primary"], m["pinned"], m["active_requests"])
            for m in original["models"]
        ] == [(True, True, 0), (False, False, 0)]
        assert get("/v1/status")["num_running"] == 0
        namespace["ACTIVE_CHAT_REQUESTS"].begin()
        assert get("/v1/status") == {
            "num_running": 1,
            "num_waiting": 0,
            "generation_tps": 42.5,
        }
        assert get("/v1/models/residency")["models"][0]["active_requests"] == 1
        with pytest.raises(HTTPError) as error:
            delete("fake-alias")
        assert error.value.code == 409
        assert delete("fake%2Devictable%2Dalias") == 204
        remaining = get("/v1/models/residency")
        assert [m["id"] for m in remaining["models"]] == ["fake-alias"]
        assert remaining["memory_used_bytes"] == 2_500_000_000
        with pytest.raises(HTTPError) as error:
            delete("fake-evictable-alias")
        assert error.value.code == 409
        namespace["ACTIVE_CHAT_REQUESTS"].end()
        assert get("/v1/status")["generation_tps"] == 0
        deletes = [json.loads(line) for line in events.read_text().splitlines()]
        assert [e["model"] for e in deletes if e["event"] == "model_delete"] == [
            "fake-alias",
            "fake-evictable-alias",
            "fake-evictable-alias",
        ]
        monkeypatch.delenv("FAKE_RESOURCE_CARD")
        assert len(get("/v1/models/residency")["models"]) == 1
        with pytest.raises(HTTPError) as error:
            get("/v1/status")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_journey_is_routed_to_models_and_collects_evidence():
    from scripts.select_gui_flows import select

    selected = select(["apps/rapid-mac/Sources/Rapid/UI/MenuBarResourceCard.swift"])
    assert "menubar-resource-card" in selected
    source = (SCRIPTS / "gui-golden-flows.sh").read_text()
    flow = source.split("flow_menubar_resource_card() {", 1)[1].split("\n}", 1)[0]
    assert "wait_resource_card idle " in flow
    assert "wait_resource_card two " in flow
    assert flow.count("wait_resource_card one ") == 2
    assert flow.count('"$AX_DRIVER" press-status-menu') == 1
    assert "($deletes | length) == 1" in flow
    assert "menubar-resource-card.$state.observed.txt" in source


@pytest.mark.parametrize("valid", [True, False])
def test_card_command_reports_observation_result(tmp_path, valid, monkeypatch):
    import runpy
    import sys

    payload = evidence("idle")
    payload["success"] = valid
    tree = tmp_path / "card.json"
    tree.write_text(json.dumps(payload))
    monkeypatch.setattr(
        sys, "argv", ["gui-resource-card.py", str(tree), "idle", "8000000000"]
    )
    if valid:
        runpy.run_path(str(SCRIPTS / "gui-resource-card.py"), run_name="__main__")
    else:
        with pytest.raises(SystemExit, match="AX dump failed"):
            runpy.run_path(str(SCRIPTS / "gui-resource-card.py"), run_name="__main__")


def test_resource_artifacts_upload_on_success_and_standard_failure_path_remains():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/rapid-mac-ci.yml").read_text())
    steps = workflow["jobs"]["gui-golden-flows"]["steps"]
    upload = next(
        s for s in steps if s.get("name") == "Upload menu bar resource evidence"
    )
    assert (
        upload["if"]
        == "success() && contains(fromJSON(matrix.gui_flows), 'menubar-resource-card')"
    )
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["path"].endswith("/golden/menubar-resource-card")
    assert (
        next(s for s in steps if s.get("name") == "Upload AX evidence")["if"]
        == "failure()"
    )
