"""The bounded, batched accessibility walk and its plumbing (no screen)."""

from __future__ import annotations

import sys
import time
import types

import pytest

from rapid_mlx.computer_use import ax_driver, backend, guards


class FakeAS:
    """ApplicationServices stand-in: batched reads over a dict tree."""

    kAXValueCGPointType = 1  # noqa: N815
    kAXValueCGSizeType = 2  # noqa: N815

    def __init__(self, tree, *, batch=True):
        self.tree = tree
        self.batch = batch
        self.requests: list[tuple[object, tuple[str, ...]]] = []
        self.timeouts: list[tuple[object, float]] = []

    def AXUIElementSetMessagingTimeout(self, element, seconds):  # noqa: N802
        self.timeouts.append((element, seconds))

    def AXUIElementCopyMultipleAttributeValues(self, element, attributes, options, _):  # noqa: N802
        self.requests.append((element, tuple(attributes)))
        if not self.batch:
            return -25200, None
        node = self.tree.get(element, {})
        return 0, [node.get(attribute) for attribute in attributes]


def _geometry(monkeypatch):
    def get_value(value, kind, _):
        x, y, w, h = value
        return 0, (
            types.SimpleNamespace(x=x, y=y)
            if kind == 1
            else types.SimpleNamespace(width=w, height=h)
        )

    monkeypatch.setattr(ax_driver, "AXValueGetValue", get_value)


@pytest.fixture
def tree(monkeypatch):
    nodes: dict = {}
    fake = FakeAS(nodes)
    monkeypatch.setattr(ax_driver, "AS", fake)
    monkeypatch.setattr(
        ax_driver, "_action_names", lambda e: list(nodes.get(e, {}).get("actions", []))
    )
    monkeypatch.setattr(ax_driver, "_get", lambda e, a: nodes.get(e, {}).get(a))
    _geometry(monkeypatch)
    return nodes, fake


def _walk(root, **kw):
    out: list[dict] = []
    ax_driver._walk(root, 0, out, [0], **kw)
    return out


def test_secure_field_text_is_never_requested(tree):
    nodes, fake = tree
    nodes.update(
        {
            "win": {"AXRole": "AXWindow", "AXChildren": ["pw", "name"]},
            "pw": {
                "AXRole": "AXTextField",
                "AXSubrole": "AXSecureTextField",
                "AXValue": "hunter2",
            },
            "name": {
                "AXRole": "AXTextField",
                "AXValue": "Ada",
                "AXPlaceholderValue": "Full name",
            },
        }
    )
    out = _walk("win")
    assert [t["text"] for t in out] == ["[secure text redacted]", "Ada"]
    assert "hunter2" not in repr(out)
    pw_requests = [attrs for element, attrs in fake.requests if element == "pw"]
    assert pw_requests and all(
        "AXValue" not in attrs and "AXTitle" not in attrs for attrs in pw_requests
    )
    assert any(
        "AXValue" in attrs for element, attrs in fake.requests if element == "name"
    )
    assert out[1]["placeholder"] == "Full name"
    assert ("pw", ax_driver.NODE_MESSAGING_TIMEOUT_S) in fake.timeouts


def test_fields_only_the_user_fills_are_named_and_never_show_their_value(tree):
    nodes, _ = tree
    nodes.update(
        {
            "win": {"AXRole": "AXWindow", "AXChildren": ["otp", "pin", "q"]},
            # No name of its own: without the placeholder, the label would be
            # the code the user typed.
            "otp": {
                "AXRole": "AXTextField",
                "AXValue": "482913",
                "AXPlaceholderValue": "Verification code",
            },
            "pin": {"AXRole": "AXTextField", "AXTitle": "PIN", "AXValue": ""},
            "q": {"AXRole": "AXSearchField", "AXValue": "milk"},
        }
    )
    out = _walk("win")
    assert "482913" not in repr(out)
    assert ax_driver.USER_VALUE == guards.USER_VALUE
    assert [(t["text"], t["value"]) for t in out] == [
        ("Verification code", "[entered by the user]"),
        ("PIN", ""),
        ("milk", None),
    ]


def test_unbatched_reads_fall_back_per_attribute_and_still_skip_secrets(
    tree, monkeypatch
):
    nodes, fake = tree
    fake.batch = False
    nodes["pw"] = {
        "AXRole": "AXTextField",
        "AXSubrole": "AXSecureTextField",
        "AXValue": "hunter2",
    }
    read = []
    monkeypatch.setattr(
        ax_driver, "_get", lambda e, a: read.append(a) or nodes.get(e, {}).get(a)
    )
    assert _walk("pw")[0]["text"] == "[secure text redacted]"
    assert "AXValue" not in read and "AXRole" in read


def test_read_node_survives_a_messaging_timeout_failure(tree, monkeypatch):
    nodes, fake = tree
    nodes["b"] = {"AXRole": "AXButton", "AXTitle": "Go"}

    def boom(element, seconds):
        raise TypeError("pyobjc variant")

    monkeypatch.setattr(fake, "AXUIElementSetMessagingTimeout", boom, raising=False)
    assert ax_driver._read_node("b")["AXTitle"] == "Go"


def test_states_values_and_labels(tree):
    nodes, _ = tree
    nodes.update(
        {
            "win": {"AXRole": "AXWindow", "AXChildren": ["web"]},
            "web": {
                "AXRole": "AXWebArea",
                "AXChildren": ["c1", "c2", "c3", "r", "step", "pop", "text", "f"],
            },
            "c1": {
                "AXRole": "AXCheckBox",
                "AXTitle": "Gift",
                "AXValue": 1,
                "AXEnabled": False,
            },
            "c2": {
                "AXRole": "AXCheckBox",
                "AXTitle": "Mixed",
                "AXValue": 2,
                "AXExpanded": True,
            },
            "c3": {
                "AXRole": "AXCheckBox",
                "AXTitle": "Off",
                "AXValue": 0,
                "AXSelected": True,
            },
            "r": {"AXRole": "AXRadioButton", "AXTitle": "Bool", "AXValue": True},
            "step": {
                "AXRole": "AXIncrementor",
                "AXDescription": "Qty",
                "AXValue": 2.0,
                "actions": ["AXIncrement"],
            },
            "pop": {
                "AXRole": "AXPopUpButton",
                "AXTitle": "Size",
                "AXValue": "",
                "actions": ["AXPress"],
            },
            "text": {"AXRole": "AXStaticText", "AXValue": "x" * 400},
            "f": {
                "AXRole": "AXTextField",
                "AXDescription": "Search",
                "AXFocused": True,
                "AXPosition": (1, 2, 0, 0),
                "AXSize": (0, 0, 30, 40),
            },
        }
    )
    out = {t["text"][:5]: t for t in _walk("win")}
    assert out["Gift"]["states"] == ["disabled", "checked"]
    assert out["Mixed"]["states"] == ["expanded", "mixed"]
    assert out["Off"]["states"] == ["selected", "unchecked"]
    assert out["Bool"]["states"] == []
    assert out["Qty"]["value"] == "2"
    assert out["Size"]["value"] is None  # an empty choice is not a value
    assert (
        len(out["xxxxx"]["text"]) == 300
    )  # static text keeps more than a control name
    assert out["Searc"]["states"] == ["focused"]
    assert out["Searc"]["rect"] == (1.0, 2.0, 30.0, 40.0)
    assert out["Gift"]["rect"] is None


def test_web_content_keeps_document_order_and_native_groups_put_controls_first(tree):
    nodes, _ = tree
    nodes.update(
        {
            "win": {"AXRole": "AXWindow", "AXChildren": ["table", "toolbar", "label"]},
            "table": {
                "AXRole": "AXTable",
                "AXVisibleRows": ["row1"],
                "AXChildren": ["row1", "row2"],
            },
            "row1": {
                "AXRole": "AXRow",
                "AXTitle": "Visible row",
                "actions": ["AXPress"],
            },
            "row2": {
                "AXRole": "AXRow",
                "AXTitle": "Scrolled row",
                "actions": ["AXPress"],
            },
            "toolbar": {"AXRole": "AXToolbar", "AXChildren": ["btn"]},
            "btn": {"AXRole": "AXButton", "AXTitle": "Back"},
            "label": {"AXRole": "AXStaticText", "AXValue": "Status"},
            "page": {"AXRole": "AXWindow", "AXChildren": ["web"]},
            "web": {"AXRole": "AXWebArea", "AXChildren": ["p", "go"]},
            "p": {"AXRole": "AXStaticText", "AXValue": "Total $5"},
            "go": {"AXRole": "AXButton", "AXTitle": "Pay now"},
        }
    )
    assert [t["text"] for t in _walk("win")] == ["Back", "Status", "Visible row"]
    assert [t["text"] for t in _walk("page")] == ["Total $5", "Pay now"]
    assert _walk("page")[1]["parent_role"] == "AXWebArea"


def test_walk_records_tree_paths_page_membership_and_dialogs(tree):
    nodes, _ = tree
    nodes.update(
        {
            # Few siblings outside the page: read up front and sorted (the
            # toolbar first), but each keeps its own child position.
            "win": {"AXRole": "AXWindow", "AXChildren": ["web", "toolbar"]},
            "toolbar": {"AXRole": "AXToolbar", "AXChildren": ["back"]},
            "back": {"AXRole": "AXButton", "AXTitle": "Back"},
            "web": {"AXRole": "AXWebArea", "AXChildren": ["dlg", "buy"]},
            "dlg": {
                "AXRole": "AXGroup",
                "AXSubrole": "AXApplicationDialog",
                "AXChildren": ["x"],
            },
            "x": {"AXRole": "AXButton", "AXTitle": "Close"},
            "buy": {"AXRole": "AXButton", "AXTitle": "Buy"},
        }
    )
    out = {t["text"] or t["subrole"]: t for t in _walk("win")}
    assert out["Back"]["path"] == [1, 0] and out["Back"]["web"] is False
    # An unnamed web dialog is still a row, so an observation can say so.
    assert out["AXApplicationDialog"]["path"] == [0, 0]
    assert out["Close"]["path"] == [0, 0, 0] and out["Close"]["web"] is True
    assert out["Buy"]["path"] == [0, 1]


def test_secure_field_is_named_and_says_whether_it_is_filled(tree):
    nodes, fake = tree
    nodes.update(
        {
            "win": {"AXRole": "AXWindow", "AXChildren": ["pw", "pin"]},
            "pw": {
                "AXRole": "AXTextField",
                "AXSubrole": "AXSecureTextField",
                "AXDescription": "Password",
                "AXTitle": "hunter2",  # some apps mirror the contents here
                "AXValue": "hunter2",
                "AXNumberOfCharacters": 7,
            },
            "pin": {
                "AXRole": "AXTextField",
                "AXSubrole": "AXSecureTextField",
                "AXPlaceholderValue": "PIN",
                "AXNumberOfCharacters": 0,
            },
        }
    )
    pw, pin = _walk("win")
    assert "hunter2" not in repr([pw, pin])
    assert (pw["text"], pw["field_name"], pw["filled"]) == (
        "[secure text redacted]",
        "Password",
        True,
    )
    assert (pin["field_name"], pin["filled"]) == ("PIN", False)
    for element, attrs in fake.requests:
        if element in {"pw", "pin"}:
            assert "AXValue" not in attrs and "AXTitle" not in attrs
    nodes["pw"]["AXNumberOfCharacters"] = None
    nodes["pw"]["AXDescription"] = None
    pw = _walk("pw")[0]
    assert (pw["field_name"], pw["filled"]) == ("", None)


def test_walk_marks_truncation_at_every_cap(tree, monkeypatch):
    nodes, _ = tree
    nodes.update(
        {
            "win": {"AXRole": "AXWindow", "AXChildren": ["a", "deep"]},
            "a": {"AXRole": "AXButton", "AXTitle": "A"},
            "deep": {"AXRole": "AXGroup", "AXChildren": ["b"]},
            "b": {"AXRole": "AXButton", "AXTitle": "B"},
        }
    )
    budget = {"deadline": time.monotonic() + 60, "truncated": False}
    assert len(_walk("win", budget=budget)) == 2 and budget["truncated"] is False

    monkeypatch.setattr(ax_driver, "MAX_DEPTH", 1)
    budget = {"deadline": time.monotonic() + 60, "truncated": False}
    assert [t["text"] for t in _walk("win", budget=budget)] == ["A"]
    assert budget["truncated"] is True and budget["depth_cap"] is True
    assert "deadline_hit" not in budget

    monkeypatch.setattr(ax_driver, "MAX_DEPTH", 60)
    monkeypatch.setattr(ax_driver, "MAX_NODES", 1)
    # The node cap stops the walk; collect reports it as node_cap.
    assert (
        len(
            _walk("win", budget={"deadline": time.monotonic() + 60, "truncated": False})
        )
        == 1
    )

    monkeypatch.setattr(ax_driver, "MAX_NODES", 4000)
    budget = {"deadline": time.monotonic() - 1, "truncated": False}
    assert _walk("win", budget=budget) == [] and budget["truncated"] is True
    assert budget["deadline_hit"] is True and "depth_cap" not in budget


def test_wedged_siblings_cannot_outlast_the_walk_budget(tree, monkeypatch):
    nodes, _ = tree
    nodes["win"] = {"AXRole": "AXWindow", "AXChildren": ["a", "b", "c", "d"]}
    for key in "abcd":
        nodes[key] = {"AXRole": "AXButton", "AXTitle": key.upper()}
    clock = [0.0]
    reads = []

    def slow_read(element):
        reads.append(element)
        clock[0] += 1.0  # each wedged node costs its messaging timeout
        return dict(nodes[element])

    monkeypatch.setattr(ax_driver, "_read_node", slow_read)
    monkeypatch.setattr(
        ax_driver, "time", types.SimpleNamespace(monotonic=lambda: clock[0])
    )
    budget = {"deadline": 2.5, "truncated": False}
    out = _walk("win", budget=budget)
    assert out == [] and reads == ["win", "a", "b"]
    assert budget["truncated"] is True and budget["deadline_hit"] is True


def test_the_caps_fit_a_whole_web_page():
    assert ax_driver.MAX_NODES == 4000
    assert ax_driver.MAX_DEPTH == 60
    assert ax_driver.WALK_BUDGET_S == 3.0
    assert backend.ACTION_WALK_BUDGET_S > ax_driver.WALK_BUDGET_S


def test_collect_reports_an_exhausted_budget(monkeypatch):
    monkeypatch.setattr(ax_driver, "_app_element", lambda name, **kw: "app")
    monkeypatch.setattr(ax_driver, "_app_windows", lambda app: ["w"])
    seen = {}

    def walk(window, depth, out, counter, roles_seen=None, *, budget):
        seen["remaining"] = budget["deadline"] - time.monotonic()
        out.append(
            {
                "target_id": "t000",
                "role": "AXButton",
                "text": "Go",
                "rect": (0, 0, 10, 10),
                "element": "e",
            }
        )
        counter[0] += 1
        budget["truncated"] = True
        budget["deadline_hit"] = True

    monkeypatch.setattr(ax_driver, "_walk", walk)
    status: dict = {}
    collected = ax_driver.collect(
        "A", keep_elements=True, budget_s=0.5, walk_status=status
    )
    assert collected and status == {
        "budget_exhausted": True,
        "depth_cap": False,
        "node_cap": False,
    }
    assert 0 < seen["remaining"] <= 0.5
    monkeypatch.setattr(ax_driver, "MAX_NODES", 1)
    ax_driver.collect("A", keep_elements=True, walk_status=status)
    assert status["node_cap"] is True


def test_frame_of_handles_missing_and_unreadable_geometry(monkeypatch):
    assert ax_driver._frame_of({}) is None
    assert ax_driver._frame_of({"AXPosition": "p"}) is None
    monkeypatch.setattr(ax_driver, "AS", FakeAS({}))
    monkeypatch.setattr(
        ax_driver, "AXValueGetValue", lambda *a: (_ for _ in ()).throw(ValueError("x"))
    )
    assert ax_driver._frame_of({"AXPosition": "p", "AXSize": "s"}) is None


def test_is_ax_error_is_false_for_plain_values():
    assert ax_driver._is_ax_error("text") is False


def test_is_ax_error_recognizes_an_ax_error_value(monkeypatch):
    # A batched read reports a missing attribute as an AXValue of error type.
    cf = types.ModuleType("CoreFoundation")
    cf.CFGetTypeID = lambda value: 5 if value in ("err", "point") else 1
    monkeypatch.setitem(sys.modules, "CoreFoundation", cf)
    monkeypatch.setattr(
        ax_driver,
        "AS",
        types.SimpleNamespace(
            AXValueGetTypeID=lambda: 5,
            AXValueGetType=lambda value: 9 if value == "err" else 1,
            kAXValueAXErrorType=9,
        ),
    )
    assert ax_driver._is_ax_error("err") is True
    assert ax_driver._is_ax_error("point") is False  # an AXValue, not an error
    assert ax_driver._is_ax_error("text") is False


# -- backend plumbing -----------------------------------------------------------


def test_live_elements_are_kept_for_recent_snapshots_only(monkeypatch):
    monkeypatch.setattr(backend, "_LIVE_ELEMENTS", {})
    for i in range(backend._LIVE_ELEMENTS_KEEP + 2):
        backend._remember_live_elements(f"s{i}", [f"live{i}"])
    assert backend.live_elements({"snapshot_id": "s0"}) is None
    assert backend.live_elements({"snapshot_id": "s1"}) is None
    last = backend._LIVE_ELEMENTS_KEEP + 1
    assert backend.live_elements({"snapshot_id": f"s{last}"}) == [f"live{last}"]
    assert backend.live_elements({}) is None


def test_get_app_state_exposes_states_placeholders_and_live_refs(monkeypatch):
    app_info = {"name": "Chrome", "bundleId": "com.google.Chrome", "pid": 42}
    window = {
        "window_id": "cg:5",
        "index": 0,
        "title": "Shop",
        "x": 0,
        "y": 0,
        "width": 800,
        "height": 600,
    }
    target = {
        "target_id": "t000",
        "role": "AXTextField",
        "subrole": "",
        "text": "Search",
        "actions": [],
        "rect": (10, 10, 100, 20),
        "states": ["focused"],
        "placeholder": "Search Mart",
        "source_window_id": "cg:5",
        "element": "live-field",
        "path": [0, 2],
        "web": True,
    }
    secure = {
        "target_id": "t001",
        "role": "AXTextField",
        "subrole": "AXSecureTextField",
        "text": "[secure text redacted]",
        "actions": [],
        "rect": (10, 40, 100, 20),
        "field_name": "Password",
        "filled": True,
    }

    def collect(*a, collection_status=None, **k):
        collection_status["partial"] = True
        collection_status["budget_exhausted"] = True
        return [target, secure]

    monkeypatch.setattr(backend, "_resolve_app", lambda *a, **k: (object(), app_info))
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: window)
    monkeypatch.setattr(backend, "_window_records", lambda app: [window])
    monkeypatch.setattr(backend, "_collect_with_timeout", collect)
    snapshot = backend.get_app_state("Chrome", screenshot=False, use_cache=False)
    element = snapshot["elements"][0]
    assert element["states"] == ["focused"] and element["placeholder"] == "Search Mart"
    assert snapshot["budget_exhausted"] is True and snapshot["truncated"] is True
    assert backend.live_elements(snapshot) == ["live-field", None]
    assert (element["path"], element["web"]) == ([0, 2], True)
    assert "field_name" not in element
    hidden = snapshot["elements"][1]
    assert (hidden["label"], hidden["field_name"], hidden["filled"]) == (
        "[secure text redacted]",
        "Password",
        True,
    )
    assert "path" not in hidden


def test_collect_watchdog_marks_a_budget_cut_walk_partial(monkeypatch):
    def collect(*args, walk_status, budget_s, **kwargs):
        assert budget_s == 1.5
        walk_status["budget_exhausted"] = True
        return [{"target_id": "t000"}]

    monkeypatch.setattr(backend.ax_driver, "collect", collect)
    status: dict = {}
    assert backend._collect_with_timeout(
        "A", timeout_s=2, collection_status=status, budget_s=1.5
    )
    assert status == {"partial": True, "budget_exhausted": True}

    def deep(*args, walk_status, budget_s, **kwargs):
        walk_status["depth_cap"] = True
        return [{"target_id": "t000"}]

    monkeypatch.setattr(backend.ax_driver, "collect", deep)
    status = {}
    backend._collect_with_timeout("A", timeout_s=2, collection_status=status)
    assert status == {"partial": True}


def test_numeric_controls_are_written_as_numbers():
    assert backend._numeric_request(3, "4") == 4.0
    assert backend._numeric_request(2.5, "x") is None
    assert backend._numeric_request(True, "1") is None
    assert backend._numeric_request("3", "4") is None


def test_readback_waits_for_an_asynchronous_write(monkeypatch):
    values = iter(["", "", "2"])
    monkeypatch.setattr(backend, "_read_value", lambda live: next(values))
    monkeypatch.setattr(backend.time, "sleep", lambda s: None)
    assert backend._await_readback("live", "2", None) == "2"

    numbers = iter([1, 1.0, 2])
    monkeypatch.setattr(backend.ax_driver, "_get", lambda live, a: next(numbers))
    assert backend._await_readback("live", "2", 2.0) == "2"

    monkeypatch.setattr(backend, "AX_WRITE_READBACK_S", 0)
    monkeypatch.setattr(backend, "_read_value", lambda live: "22")
    assert backend._await_readback("live", "2", None) == "22"
    monkeypatch.setattr(backend.ax_driver, "_get", lambda live, a: True)
    assert backend._await_readback("live", "1", 1.0) is None
    monkeypatch.setattr(backend.ax_driver, "_get", lambda live, a: "3")
    assert backend._await_readback("live", "2", 2.0) == "3"


def _set_value_snapshot(role, subrole=""):
    window = {
        "index": 0,
        "window_id": "cg:101",
        "title": "Main",
        "x": 0,
        "y": 0,
        "width": 100,
        "height": 100,
    }
    return {
        "snapshot_id": "planned",
        "observed_at": 100.0,
        "app": {"name": "A", "bundleId": "b", "pid": 4},
        "window_index": 0,
        "window_id": "cg:101",
        "window": window,
        "elements": [{"index": 0, "role": role, "subrole": subrole, "center": [1, 2]}],
    }


@pytest.mark.parametrize(
    ("role", "subrole", "current", "written", "reads_current"),
    [
        ("AXIncrementor", "", 1, 2.0, True),
        ("AXSlider", "", "loud", "2", True),
        ("AXTextField", "", None, "2", False),
        ("AXTextField", "AXSecureTextField", None, "2", False),
    ],
)
def test_set_value_writes_numbers_to_numeric_controls_only(
    monkeypatch, role, subrole, current, written, reads_current
):
    snapshot = _set_value_snapshot(role, subrole)
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    writes, reads = [], []
    module = types.ModuleType("ApplicationServices")
    module.kAXValueAttribute = "AXValue"
    module.AXUIElementSetAttributeValue = lambda live, attr, value: (
        writes.append(value) or 0
    )
    monkeypatch.setitem(sys.modules, "ApplicationServices", module)

    def get(live, attribute):
        reads.append(attribute)
        return writes[-1] if writes else current

    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(backend, "_read_value", lambda live: "2")
    assert backend.set_value("A", 0, "2")["verified"] is True
    assert writes == [written]
    assert bool(reads) is reads_current
