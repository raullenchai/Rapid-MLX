"""Menus, modifier clicks, the focus guard, Electron, hidden renderers, AX scroll.

Everything here runs without a screen: the AX server, SkyLight and Quartz are
replaced by fakes, the same way test_computer_use_offspace.py does it.
"""

# ruff: noqa: N802 - PyObjC test doubles intentionally mirror Objective-C names.

import sys
import time
import types

import pytest

from rapid_mlx.computer_use import ax_driver, backend, background_input, errors


def _snapshot(*, elements=None, bundle="com.google.chrome"):
    window = {
        "index": 0,
        "window_id": "cg:101",
        "title": "Main",
        "x": 0,
        "y": 0,
        "width": 400,
        "height": 300,
    }
    return {
        "snapshot_id": "s1",
        "observed_at": time.time(),
        "app": {"name": "App", "bundleId": bundle, "pid": 4},
        "window_index": 0,
        "window_id": "cg:101",
        "window": window,
        "elements": elements or [],
    }


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.time = time.time
        self.time_ns = time.time_ns

    def sleep(self, seconds):
        self.now += max(float(seconds), 0.01)

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(backend, "time", fake)
    monkeypatch.setattr(background_input, "time", fake)
    monkeypatch.setattr(ax_driver, "time", fake)
    return fake


@pytest.fixture
def attrs(monkeypatch):
    """Dict-backed AX attributes: ``attrs[element][attribute]``."""
    table: dict = {}

    def checked(element, attribute):
        value = table.get(element, {}).get(attribute)
        return (False, None) if value is _UNREADABLE else (True, value)

    monkeypatch.setattr(ax_driver, "_get", lambda e, a: table.get(e, {}).get(a))
    monkeypatch.setattr(ax_driver, "_get_checked", checked)
    return table


_UNREADABLE = object()  # an attribute whose read fails (not merely absent)


@pytest.fixture
def calls(monkeypatch, clock):
    """Background delivery with recorded focus moves, validations and AX actions."""
    log: list[tuple] = []
    monkeypatch.setenv(background_input.DELIVERY_ENV, "background")
    monkeypatch.setattr(background_input, "skylight_available", lambda: True)
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (999, 555))
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 101)
    monkeypatch.setattr(backend, "_sheet_owner_id", lambda pid: None)
    monkeypatch.setattr(
        background_input,
        "activate_without_raise",
        lambda *a, **k: log.append(("activate", a)) or True,
    )
    monkeypatch.setattr(
        backend, "_restore_user_focus", lambda *a: log.append(("restore", a)) or True
    )
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda snap, *a, **k: log.append(("validate", a, k)),
    )
    monkeypatch.setattr(
        background_input,
        "press_key",
        lambda *a, **k: log.append(("press_key", a)) or True,
    )
    monkeypatch.setattr(backend, "_finish_action", lambda app, snap, d, **k: d)
    monkeypatch.setattr(backend, "_clipboard_change_count", lambda: None)
    return log


def _ax_actions(monkeypatch, log, results=None):
    """Install ApplicationServices with a recording AXUIElementPerformAction."""
    results = {} if results is None else results

    def perform(element, action):
        log.append(("ax", element, action))
        return results.get((element, action), 0)

    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=perform,
    )
    monkeypatch.setattr(ax_driver, "AXUIElementPerformAction", perform)
    return perform


EXACT = {"require_active_app": False, "require_exact_window_id": True}


# --- menu titles and item discovery ------------------------------------------------


def test_menu_title_key_ignores_case_and_trailing_ellipsis():
    assert backend._menu_title_key(" Save As… ") == "save as"
    assert backend._menu_title_key("Find...") == "find"
    assert backend._menu_title_key(None) == ""


def test_menu_item_title_falls_back_to_value_then_description(attrs):
    attrs["a"] = {"AXTitle": "Copy"}
    attrs["b"] = {"AXTitle": "", "AXValue": "Option 2"}
    attrs["c"] = {"AXDescription": "Zoom"}
    assert [backend._menu_item_title(x) for x in "abcd"] == [
        "Copy",
        "Option 2",
        "Zoom",
        "",
    ]


def test_menu_items_under_searches_a_few_levels(attrs):
    attrs["popup"] = {"AXChildren": ["menu", "label"]}
    attrs["menu"] = {"AXRole": "AXMenu", "AXChildren": ["one", "group"]}
    attrs["label"] = {"AXRole": "AXStaticText"}
    attrs["one"] = {"AXRole": "AXMenuItem"}
    attrs["group"] = {"AXRole": "AXGroup", "AXChildren": ["two"]}
    attrs["two"] = {"AXRole": "AXMenuItem"}
    assert sorted(backend._menu_items_under("popup")) == ["one", "two"]
    assert backend._menu_items_under(None) == []


# --- open-menu counting -----------------------------------------------------------


def _quartz_windows(monkeypatch, windows):
    return _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: windows(),
        kCGNullWindowID=0,
        kCGWindowListOptionAll=0,
    )


def test_open_menu_count_counts_menu_layer_windows_of_the_pid(monkeypatch):
    _quartz_windows(
        monkeypatch,
        lambda: [
            {"kCGWindowOwnerPID": 4, "kCGWindowLayer": 101},
            {"kCGWindowOwnerPID": 4, "kCGWindowLayer": 0},
            {"kCGWindowOwnerPID": 9, "kCGWindowLayer": 101},
        ],
    )
    assert backend._open_menu_count(4) == 1


def test_open_menu_count_is_none_when_unreadable(monkeypatch):
    _quartz_windows(monkeypatch, lambda: 1 / 0)
    assert backend._open_menu_count(4) is None


def test_menus_before_skips_foreground_and_fails_closed_for_a_choice(monkeypatch):
    monkeypatch.setattr(backend, "_background_delivery", lambda s: False)
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: None)
    assert backend._menus_before(_snapshot(), None) is None
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._menus_before(_snapshot(), "Copy")
    assert exc.value.code == "action_failed"
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 2)
    assert backend._menus_before(_snapshot(), "Copy") == 2


def test_menus_before_fails_closed_for_a_background_menu_gesture(monkeypatch):
    monkeypatch.setattr(backend, "_background_delivery", lambda s: True)
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: None)
    assert backend._menus_before(_snapshot(), None) is None
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._menus_before(_snapshot(), None, expect_menu=True)
    assert exc.value.code == "action_failed"


# --- closing and settling menus ---------------------------------------------------


def test_close_menus_cancels_then_escapes_until_gone(monkeypatch, calls, attrs):
    _ax_actions(monkeypatch, calls)
    attrs["popup"] = {"AXChildren": ["menu"]}
    attrs["menu"] = {"AXRole": "AXMenu"}
    counts = iter([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: next(counts, 0))
    assert backend._close_menus(4, 0, "popup") is True
    assert calls[0] == ("ax", "menu", "AXCancel")
    assert ("press_key", (4, backend.KEY_ALIASES["escape"])) in calls


def test_close_menus_reports_a_menu_that_stays_open(monkeypatch, calls):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 1)
    assert backend._close_menus(4, 0, None) is False
    escapes = [c for c in calls if c[0] == "press_key"]
    assert len(escapes) == 2  # bounded
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: None)
    assert backend._close_menus(4, 0, None) is False


def test_close_menus_escape_spi_error_is_action_failed(monkeypatch, calls):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 1)
    monkeypatch.setattr(
        background_input, "press_key", lambda *a: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._close_menus(4, 0, None)
    assert exc.value.code == "action_failed"


def test_settle_menus_without_a_menu(monkeypatch, calls):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
    snap = _snapshot()
    assert backend._settle_menus(snap, 0, None, None, expect_menu=True) is None
    assert backend._settle_menus(snap, 0, 0, None, expect_menu=False) is None
    report = backend._settle_menus(snap, 0, 0, None, expect_menu=True)
    assert report["opened"] is False
    with pytest.raises(backend._NoMenuOpenedError) as exc:
        backend._settle_menus(snap, 3, 0, "Copy", expect_menu=True)
    assert exc.value.code == "element_not_found"
    assert exc.value.snapshot is snap and exc.value.element_index == 3


def test_settle_menus_never_reads_an_uncountable_menu_as_none(monkeypatch, calls):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: None)
    snap = _snapshot()
    report = backend._settle_menus(snap, 0, 0, None, expect_menu=False)
    assert report["opened"] is None and report["closed"] is False
    assert "may still be open" in report["warning"]
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._settle_menus(snap, 3, 0, "Copy", expect_menu=True)
    assert exc.value.code == "action_failed"
    assert not isinstance(exc.value, backend._NoMenuOpenedError)
    assert "may still be open" in exc.value.message
    assert not [c for c in calls if c[0] == "press_key"]  # no blind Escape


def test_menu_opened_despite_error(monkeypatch, clock):
    snap = _snapshot()
    assert backend._menu_opened_despite_error(snap, None) is False
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 1)
    assert backend._menu_opened_despite_error(snap, 0) is True
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
    assert backend._menu_opened_despite_error(snap, 0) is False
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: None)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._menu_opened_despite_error(snap, 0)
    assert exc.value.code == "action_failed"
    # A menu that opens late, within the regular open window, still counts.
    start = clock.now
    monkeypatch.setattr(
        backend, "_open_menu_count", lambda pid: 1 if clock.now - start > 0.45 else 0
    )
    assert backend._menu_opened_despite_error(snap, 0) is True


def test_choose_from_ax_menu_closes_a_late_menu_after_a_failed_press(
    monkeypatch, native_popup, calls
):
    real = sys.modules["ApplicationServices"].AXUIElementPerformAction

    def perform(element, action):
        calls.append(("ax", element, action))
        return -25204 if (element, action) == ("popup", "AXPress") else 0

    sys.modules["ApplicationServices"].AXUIElementPerformAction = perform
    # teardown wait, lingering check, open wait, then the late look.
    menus = iter([None, None, None, "menu"])
    monkeypatch.setattr(backend, "_open_menu_of", lambda live, timeout: next(menus))
    closed = []
    monkeypatch.setattr(
        backend, "_close_menu", lambda live, menu, pid: closed.append(menu) or True
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._choose_from_ax_menu("popup", "B", 4)
    assert exc.value.code == "accessibility_error"
    assert closed == ["menu"]
    sys.modules["ApplicationServices"].AXUIElementPerformAction = real


@pytest.fixture
def open_menu(monkeypatch, calls, attrs):
    """A popup whose menu (Copy, Paste[disabled]) is open until cancelled."""
    state = {"open": 1}
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: state["open"])
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "popup")
    attrs["popup"] = {"AXChildren": ["menu"]}
    attrs["menu"] = {"AXRole": "AXMenu", "AXChildren": ["copy", "paste"]}
    attrs["copy"] = {"AXRole": "AXMenuItem", "AXTitle": "Copy"}
    attrs["paste"] = {"AXRole": "AXMenuItem", "AXTitle": "Paste", "AXEnabled": False}
    results: dict = {}

    def perform(element, action):
        calls.append(("ax", element, action))
        if action == "AXCancel":
            state["open"] = 0
        return results.get((element, action), 0)

    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=perform,
    )
    return types.SimpleNamespace(results=results, state=state)


def test_settle_menus_reads_and_closes_an_opened_menu(open_menu, calls):
    report = backend._settle_menus(_snapshot(), 0, 0, None, expect_menu=True)
    assert report["closed"] is True
    assert report["items"] == ["Copy", "Paste"]
    assert ("ax", "menu", "AXCancel") in calls
    assert not any(c[2] == "AXPress" for c in calls if c[0] == "ax")


def test_settle_menus_chooses_an_item_and_closes(open_menu, calls):
    report = backend._settle_menus(_snapshot(), 0, 0, "copy", expect_menu=True)
    assert report == {"closed": True, "chosen": "Copy"}
    assert ("ax", "copy", "AXPress") in calls


def test_settle_menus_refuses_a_missing_or_disabled_item(open_menu, calls):
    for wanted in ("Paste", "Delete"):
        open_menu.state["open"] = 1
        with pytest.raises(errors.ComputerUseError) as exc:
            backend._settle_menus(_snapshot(), 0, 0, wanted, expect_menu=True)
        assert exc.value.code == "element_not_found"
        assert "Copy" in exc.value.message
    assert not any(c[2] == "AXPress" for c in calls if c[0] == "ax")


def test_settle_menus_matches_a_unique_start_or_part_and_refuses_ambiguity(
    open_menu, calls, attrs
):
    attrs["menu"]["AXChildren"] = ["visa", "checking", "savings", "add", "settings"]
    attrs["settings"] = {"AXRole": "AXMenuItem", "AXTitle": "Payment settings"}
    attrs["visa"] = {"AXRole": "AXMenuItem", "AXTitle": "Visa ending 4242 (2.95% fee)"}
    attrs["checking"] = {
        "AXRole": "AXMenuItem",
        "AXTitle": "Checking ending 6789 (no fee)",
    }
    attrs["savings"] = {"AXRole": "AXMenuItem", "AXTitle": "Savings ending 1111"}
    attrs["add"] = {"AXRole": "AXMenuItem", "AXTitle": "Add a new payment method…"}
    report = backend._settle_menus(_snapshot(), 0, 0, "checking", expect_menu=True)
    assert report["chosen"] == "Checking ending 6789 (no fee)"
    open_menu.state["open"] = 1
    report = backend._settle_menus(_snapshot(), 0, 0, "6789", expect_menu=True)
    assert report["chosen"] == "Checking ending 6789 (no fee)"
    open_menu.state["open"] = 1
    report = backend._settle_menus(_snapshot(), 0, 0, "Sa", expect_menu=True)
    assert report["chosen"] == "Savings ending 1111"
    for wanted, why in (
        ("ending", "has 3 items matching"),  # in three titles
        # One title starts with it, another holds it: still ambiguous.
        ("pay", "has 2 items matching"),
        ("ng", "has no enabled item"),  # two letters choose only as a start
        ("c", "has no enabled item"),  # too short to choose by
        ("Mastercard", "has no enabled item"),
    ):
        open_menu.state["open"] = 1
        presses = len([c for c in calls if c[2:] == ("AXPress",)])
        with pytest.raises(errors.ComputerUseError) as exc:
            backend._settle_menus(_snapshot(), 0, 0, wanted, expect_menu=True)
        assert exc.value.code == "element_not_found"
        assert why in exc.value.message and "Savings ending 1111" in exc.value.message
        assert len([c for c in calls if c[2:] == ("AXPress",)]) == presses


def test_settle_menus_reports_a_failed_press(open_menu, calls):
    open_menu.results[("copy", "AXPress")] = -25200
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._settle_menus(_snapshot(), 0, 0, "Copy", expect_menu=True)
    assert exc.value.code == "accessibility_error"
    assert ("ax", "menu", "AXCancel") in calls


def test_settle_menus_errors_say_when_the_menu_stayed_open(
    monkeypatch, open_menu, calls
):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 1)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._settle_menus(_snapshot(), 0, 0, "Delete", expect_menu=True)
    assert "may still be open" in exc.value.message
    open_menu.results[("copy", "AXPress")] = -25200
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._settle_menus(_snapshot(), 0, 0, "Copy", expect_menu=True)
    assert "may still be open" in exc.value.message


def test_settle_menus_closes_the_menu_when_reading_it_fails(
    monkeypatch, open_menu, calls
):
    def drifted(*a, **k):
        raise errors.ComputerUseError("stale_snapshot", "element moved")

    monkeypatch.setattr(backend, "_live_element", drifted)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._settle_menus(_snapshot(), 0, 0, None, expect_menu=True)
    assert exc.value.code == "stale_snapshot"
    # No element to cancel through: closed by Escape to the owning pid.
    assert ("press_key", (4, backend.KEY_ALIASES["escape"])) in calls
    # The menu ignored the Escapes, and the error says so (once).
    assert exc.value.message.count("may still be open") == 1
    assert str(exc.value) == exc.value.message


def test_settle_menus_wraps_an_unexpected_failure_when_the_menu_stays_open(
    monkeypatch, open_menu, calls
):
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: 1 / 0)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._settle_menus(_snapshot(), 0, 0, None, expect_menu=True)
    assert exc.value.code == "action_failed"
    assert "may still be open" in exc.value.message
    assert isinstance(exc.value.__cause__, ZeroDivisionError)
    # A failure whose menu did close propagates unchanged.
    monkeypatch.setattr(backend, "_close_menus", lambda *a: True)
    with pytest.raises(ZeroDivisionError):
        backend._settle_menus(_snapshot(), 0, 0, None, expect_menu=True)


def test_settle_menus_warns_when_the_menu_stays_open(monkeypatch, open_menu, calls):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 1)
    report = backend._settle_menus(_snapshot(), None, 0, None, expect_menu=False)
    assert report["closed"] is False and "warning" in report


# --- click: menu_item, modifiers ------------------------------------------------


def _popup_snapshot(role="AXPopUpButton"):
    return _snapshot(
        elements=[
            {
                "index": 0,
                "role": role,
                "center": [10, 10],
                "actions": ["AXPress"],
            }
        ]
    )


def test_click_with_menu_item_presses_then_chooses(monkeypatch, open_menu, calls):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
    real_perform = sys.modules["ApplicationServices"].AXUIElementPerformAction

    def perform(element, action):
        result = real_perform(element, action)
        if (element, action) == ("popup", "AXPress"):
            monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 1)
        if action == "AXCancel":
            monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
        return result

    sys.modules["ApplicationServices"].AXUIElementPerformAction = perform
    result = backend.click(
        "App", element_index=0, expected_snapshot=_popup_snapshot(), menu_item="Copy"
    )
    assert result["mode"] == "AXPress"
    assert result["menu"] == {"closed": True, "chosen": "Copy"}
    pressed = [c[1] for c in calls if c[0] == "ax" and c[2] == "AXPress"]
    assert pressed == ["popup", "copy"]


def test_click_menu_item_on_a_popup_without_menu_sets_the_value(
    monkeypatch, open_menu, calls
):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
    seen = {}

    def set_value(app, index, value, **kwargs):
        seen.update(index=index, value=value, snapshot=kwargs["expected_snapshot"])
        return {"mode": "AXValue", "warning": None}

    monkeypatch.setattr(backend, "set_value", set_value)
    snap = _popup_snapshot()
    result = backend.click(
        "App", element_index=0, expected_snapshot=snap, menu_item="B"
    )
    assert seen == {"index": 0, "value": "B", "snapshot": snap}
    assert result["warning"] == "the popup opened no menu; its value was set instead"


def test_click_menu_item_on_a_non_popup_without_menu_is_refused(
    monkeypatch, open_menu, calls
):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
    monkeypatch.setattr(
        backend, "set_value", lambda *a, **k: pytest.fail("set_value on a button")
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click(
            "App",
            element_index=0,
            expected_snapshot=_popup_snapshot("AXMenuButton"),
            menu_item="B",
        )
    assert exc.value.code == "element_not_found"


def test_click_argument_errors(calls):
    snap = _popup_snapshot()
    for kwargs in (
        {"focus_only": True, "modifiers": "shift"},
        {"focus_only": True, "menu_item": "A"},
    ):
        with pytest.raises(errors.ComputerUseError) as exc:
            backend.click("App", element_index=0, expected_snapshot=snap, **kwargs)
        assert exc.value.code == "invalid_argument"
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", element_index=0, expected_snapshot=snap, modifiers="hyper")
    assert exc.value.code == "unsupported_key"
    # A bare point has no element to read a menu under: refused before acting.
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", x=10, y=10, expected_snapshot=snap, menu_item="A")
    assert exc.value.code == "invalid_argument"
    assert calls == []


def test_modifier_flags_parse_lists_and_chords():
    flags = backend.MODIFIER_FLAGS
    assert backend._modifier_flags(None) == 0
    assert backend._modifier_flags("Shift+cmd") == flags["shift"] | flags["cmd"]
    assert backend._modifier_flags(["option", " ", "ctrl"]) == (
        flags["option"] | flags["ctrl"]
    )


def test_modifier_click_takes_the_pixel_route_with_flags(monkeypatch, calls):
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
    _ax_actions(monkeypatch, calls)
    seen = {}

    def pixel_click(snapshot, x, y, **kwargs):
        seen.update(kwargs)
        return {"mode": "SkyLight-click"}

    monkeypatch.setattr(backend, "_pixel_click", pixel_click)
    backend.click(
        "App",
        element_index=0,
        expected_snapshot=_popup_snapshot("AXButton"),
        modifiers=["cmd"],
    )
    assert seen["flags"] == backend.MODIFIER_FLAGS["cmd"]
    assert not [c for c in calls if c[0] == "ax"]  # no AXPress with modifiers


def test_foreground_modifier_click_is_refused(monkeypatch):
    monkeypatch.setattr(backend, "_background_delivery", lambda s: False)
    monkeypatch.setattr(
        ax_driver, "_cg_click", lambda *a, **k: pytest.fail("held user modifiers")
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._pixel_click(_snapshot(), 1.0, 1.0, flags=1 << 17)
    assert exc.value.code == "synthetic_input_blocked"


# --- menu-bar commands ------------------------------------------------------------


@pytest.fixture
def menubar(monkeypatch, attrs):
    monkeypatch.setattr(backend, "_pid_app_element", lambda app: "app")
    attrs["app"] = {"AXMenuBar": "bar"}
    attrs["bar"] = {"AXChildren": ["edit-title"]}
    attrs["edit-title"] = {"AXTitle": "Edit", "AXChildren": ["edit-menu"]}
    attrs["edit-menu"] = {"AXRole": "AXMenu", "AXChildren": ["copy", "find"]}
    attrs["copy"] = {
        "AXTitle": "Copy",
        "AXMenuItemCmdChar": "C",
        "AXMenuItemCmdModifiers": 0,
    }
    attrs["find"] = {"AXTitle": "Find", "AXChildren": ["find-menu"]}
    attrs["find-menu"] = {"AXRole": "AXMenu", "AXChildren": ["find-item"]}
    attrs["find-item"] = {"AXTitle": "Find…"}
    return attrs


def test_menu_item_by_path_resolves_and_reports_the_failing_step(menubar):
    app = {"pid": 4}
    assert backend._menu_item_by_path(app, ["edit", "Find", "find..."]) == "find-item"
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._menu_item_by_path(app, ["Edit", "Paste"])
    assert exc.value.code == "element_not_found"
    assert "under Edit" in exc.value.message and "Copy" in exc.value.message
    menubar["app"] = {}
    with pytest.raises(errors.ComputerUseError):
        backend._menu_item_by_path(app, ["Edit", "Copy"])


def test_menu_rejects_short_paths_and_submenus(monkeypatch, menubar, calls):
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: _snapshot())
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.menu("App", "Edit")
    assert exc.value.code == "invalid_argument"
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.menu("App", "Edit > Find")
    assert exc.value.code == "invalid_argument"
    assert calls == []


def test_menu_presses_through_ax_with_the_target_forced_key(
    monkeypatch, menubar, calls
):
    _ax_actions(monkeypatch, calls)
    monkeypatch.setattr(backend, "_process_is_active", lambda snap: False)
    result = backend.menu("App", ["Edit", "Copy"], expected_snapshot=_snapshot())
    assert result["mode"] == "AXMenuPress" and result["menu_item"] == "Edit > Copy"
    assert result["focus_restored"] is True
    # Forced even though AX already names 101 as the key window.
    assert [c[0] for c in calls] == ["activate", "validate", "ax", "restore"]
    assert calls[1] == ("validate", (), EXACT)


def test_menu_uses_the_key_equivalent_while_the_app_is_active(
    monkeypatch, menubar, calls
):
    monkeypatch.setattr(backend, "_process_is_active", lambda snap: True)
    clip = iter([7, 8])
    monkeypatch.setattr(backend, "_clipboard_change_count", lambda: next(clip))
    finished = {}
    monkeypatch.setattr(
        backend,
        "_finish_action",
        lambda app, snap, d, **k: finished.update(k) or d,
    )
    result = backend.menu("App", "Edit > Copy", expected_snapshot=_snapshot())
    assert result["mode"] == "SkyLight-menu-chord"
    assert result["clipboard_changed"] is True
    # The chord route is not claimed as an Accessibility press, nor verified.
    assert finished["verified"] is None
    assert "key equivalent" in finished["verification"]
    keycode = ax_driver._keycode_for("c")
    assert ("press_key", (4, keycode, backend.MODIFIER_FLAGS["cmd"])) in calls


def test_menu_in_the_foreground_borrows_then_validates_exactly(
    monkeypatch, menubar, calls
):
    _ax_actions(monkeypatch, calls)
    monkeypatch.setattr(backend, "_background_delivery", lambda s: False)
    monkeypatch.setattr(
        backend, "_borrow_foreground", lambda snap: calls.append(("borrow",))
    )
    backend.menu("App", "Edit > Copy", expected_snapshot=_snapshot())
    assert [c[0] for c in calls] == ["borrow", "validate", "ax"]
    assert calls[1] == ("validate", (), {"require_exact_window_id": True})


def test_press_menu_item_refuses_disabled_and_reports_errors(
    monkeypatch, menubar, calls
):
    results: dict = {}
    _ax_actions(monkeypatch, calls, results)
    monkeypatch.setattr(backend, "_process_is_active", lambda snap: False)
    snap = _snapshot()
    menubar["copy"]["AXEnabled"] = False
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._press_menu_item("App", snap, "copy", "Copy", False)
    assert exc.value.code == "synthetic_input_blocked"
    assert calls[-1][0] == "restore"  # focus handed back on refusal
    menubar["copy"]["AXEnabled"] = True
    results[("copy", "AXPress")] = -25204  # ran a modal loop
    result = backend._press_menu_item("App", snap, "copy", "Copy", False)
    assert "still running" in result["warning"]
    results[("copy", "AXPress")] = -25200
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._press_menu_item("App", snap, "copy", "Copy", False)
    assert exc.value.code == "accessibility_error"


def test_menu_item_chord_maps_modifiers_and_skips_glyph_keys(attrs):
    flags = backend.MODIFIER_FLAGS
    attrs["a"] = {"AXMenuItemCmdChar": "S", "AXMenuItemCmdModifiers": 0x1 | 0x2 | 0x4}
    assert backend._menu_item_chord("a") == (
        ax_driver._keycode_for("s"),
        flags["cmd"] | flags["shift"] | flags["option"] | flags["ctrl"],
    )
    attrs["b"] = {"AXMenuItemCmdChar": "K", "AXMenuItemCmdModifiers": 0x8}
    attrs["c"] = {"AXMenuItemCmdChar": "", "AXMenuItemCmdModifiers": 0}
    attrs["d"] = {"AXMenuItemCmdChar": "S"}
    assert [backend._menu_item_chord(x) for x in "bcd"] == [None, None, None]


def test_clipboard_change_count_is_optional(monkeypatch):
    _install_module(
        monkeypatch,
        "AppKit",
        NSPasteboard=types.SimpleNamespace(
            generalPasteboard=lambda: types.SimpleNamespace(changeCount=lambda: 3)
        ),
    )
    assert backend._clipboard_change_count() == 3
    monkeypatch.setitem(sys.modules, "AppKit", None)
    assert backend._clipboard_change_count() is None


def test_hotkey_presses_the_resolved_menu_item_of_an_inactive_app(
    monkeypatch, menubar, calls
):
    _ax_actions(monkeypatch, calls)
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: _snapshot())
    monkeypatch.setattr(backend, "_process_is_active", lambda snap: False)
    result = backend.hotkey("App", "cmd+c", window_id="cg:101")
    assert result["mode"] == "AXMenuPress" and result["menu_item"] == "cmd+c (Copy)"
    assert [c[0] for c in calls] == ["activate", "validate", "ax", "restore"]
    assert not [c for c in calls if c[0] == "press_key"]


def test_hotkey_sends_an_unresolvable_menu_chord_to_the_keyed_target_of_an_active_app(
    monkeypatch, menubar, calls
):
    _ax_actions(monkeypatch, calls)
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: _snapshot())
    monkeypatch.setattr(backend, "_process_is_active", lambda snap: True)
    # An unreadable menu may hold Cmd+Z: not ruled out, and not resolvable.
    menubar["bar"]["AXChildren"] = ["edit-title", "view-title"]
    menubar["view-title"] = {"AXRole": "AXMenuBarItem", "AXChildren": _UNREADABLE}
    backend.hotkey("App", "cmd+z", window_id="cg:101")
    keycode = ax_driver._keycode_for("z")
    assert ("press_key", (4, keycode, backend.MODIFIER_FLAGS["cmd"])) in calls
    assert not [c for c in calls if c[0] == "ax"]
    # Delivered only after the exact target window was validated.
    names = [c[0] for c in calls]
    assert names.index("validate") < names.index("press_key")
    assert calls[names.index("validate")][2] == EXACT


def test_menu_equivalent_lookup_keeps_scanning_past_unreadable_modifiers(
    menubar,
):
    cmd = backend.MODIFIER_FLAGS["cmd"]
    menubar["edit-menu"]["AXChildren"] = ["odd", "copy"]
    menubar["odd"] = {"AXMenuItemCmdChar": "C"}  # unreadable modifiers
    assert backend._menu_equivalent_lookup({"pid": 4}, "c", cmd) == (True, "copy")
    menubar["edit-menu"]["AXChildren"] = ["odd"]
    assert backend._menu_equivalent_lookup({"pid": 4}, "c", cmd) == (True, None)
    assert backend._menu_equivalent_lookup({"pid": 4}, "z", cmd) == (False, None)


def test_menu_equivalent_lookup_fails_closed_on_unenumerable_menus(menubar):
    cmd = backend.MODIFIER_FLAGS["cmd"]
    # A menu whose children fail to read may hide the chord: fail closed,
    # but still resolve an item that was found elsewhere.
    menubar["bar"]["AXChildren"] = ["edit-title", "view-title"]
    menubar["view-title"] = {"AXRole": "AXMenuBarItem", "AXChildren": _UNREADABLE}
    assert backend._menu_equivalent_lookup({"pid": 4}, "z", cmd) == (True, None)
    assert backend._menu_equivalent_lookup({"pid": 4}, "c", cmd) == (True, "copy")
    # A menu bar with no readable items cannot rule anything out.
    menubar["bar"]["AXChildren"] = []
    assert backend._menu_equivalent_lookup({"pid": 4}, "z", cmd) == (True, None)
    menubar["app"]["AXMenuBar"] = None
    assert backend._menu_equivalent_lookup({"pid": 4}, "z", cmd) == (True, None)


# --- focus guard and keyed target ------------------------------------------------


def test_guard_user_focus_restores_only_a_move_into_the_target(monkeypatch, calls):
    fronts = iter([(999, 555), (4, 101)])
    monkeypatch.setattr(backend, "_frontmost_window", lambda: next(fronts))
    with backend._guard_user_focus(_snapshot()) as state:
        pass
    assert state == {"focus_restored": True}
    # The user switched to another app, or to another window of the target's
    # app, meanwhile: theirs to keep.
    for moved_to in ((77, 1), (4, 202)):
        fronts = iter([(999, 555), moved_to])
        with backend._guard_user_focus(_snapshot()) as state:
            pass
        assert state == {}
    # Unchanged focus: nothing to do.
    fronts = iter([(999, 555), (999, 555)])
    with backend._guard_user_focus(_snapshot()) as state:
        pass
    assert state == {}
    assert [c[0] for c in calls] == ["restore"]


def test_guard_user_focus_reports_a_failed_restore(monkeypatch, calls):
    fronts = iter([(999, 555), (4, 101)])
    monkeypatch.setattr(backend, "_frontmost_window", lambda: next(fronts))
    monkeypatch.setattr(
        backend, "_restore_user_focus", lambda *a: (_ for _ in ()).throw(OSError())
    )
    with backend._guard_user_focus(_snapshot()) as state:
        pass
    assert state == {"focus_restored": False}
    assert backend._focus_fields(state)["warning"]


def test_guard_user_focus_is_inert_in_the_foreground(monkeypatch):
    monkeypatch.setattr(backend, "_background_delivery", lambda s: False)
    monkeypatch.setattr(
        backend, "_frontmost_window", lambda: pytest.fail("probed focus")
    )
    with backend._guard_user_focus(_snapshot()) as state:
        pass
    assert state == {}


def test_keyed_target_reposts_after_an_appkit_bounce_back(monkeypatch, calls):
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 555))
    # Key moves to 101 only after a second switch.
    keys = iter([555] + [555] * 20 + [101] * 5)
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: next(keys, 101))
    with backend._keyed_target(_snapshot()):
        calls.append(("body",))
    names = [c[0] for c in calls]
    assert names.count("activate") >= 2
    assert names[-3:] == ["validate", "body", "restore"]
    assert names.count("validate") == 1


def test_keyed_target_falls_back_to_a_final_exact_validation(monkeypatch, calls):
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 555)
    with backend._keyed_target(_snapshot()):
        calls.append(("body",))
    names = [c[0] for c in calls]
    assert names.count("activate") == 4  # first switch + 3 bounded reposts
    assert names[-3:] == ["validate", "body", "restore"]


# --- focus without commit ---------------------------------------------------------


def test_focus_without_commit_polls_and_accepts_electron_element_focus(
    monkeypatch, attrs, clock
):
    monkeypatch.setattr(ax_driver, "kAXErrorSuccess", 0, raising=False)
    monkeypatch.setattr(ax_driver, "AXUIElementSetAttributeValue", lambda *a: 0)
    focused = iter([None, None, "live"])
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: next(focused))
    assert backend._focus_without_commit(_snapshot(), "live") == "AXFocused"
    # No app-level focused element: the element's own AXFocused counts.
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: None)
    attrs["live"] = {"AXFocused": True}
    assert backend._focus_without_commit(_snapshot(), "live") == "AXFocused"
    attrs["live"] = {}
    assert backend._focus_without_commit(_snapshot(), "live") is None


# --- Electron and hidden renderers ------------------------------------------------


def test_is_electron_detects_the_framework_once_per_process(monkeypatch, tmp_path):
    bundle = tmp_path / "Slack.app"
    (bundle / "Contents/Frameworks/Electron Framework.framework").mkdir(parents=True)
    lookups = []

    def running(pid):
        lookups.append(pid)
        path = bundle if pid == 4 else tmp_path / "Native.app"
        return types.SimpleNamespace(
            bundleURL=lambda: types.SimpleNamespace(path=lambda: str(path))
        )

    monkeypatch.setattr(backend, "_ELECTRON_BY_PID", {})
    monkeypatch.setattr(ax_driver, "_application_for_pid", running)
    slack = {"pid": 4, "bundleId": "com.tinyspeck.slackmacgap"}
    assert backend._is_electron(slack) and backend._is_electron(slack)
    assert lookups == [4]
    assert backend._needs_web_content_retry(slack)
    assert not backend._is_electron({"pid": 5, "bundleId": "com.apple.TextEdit"})
    assert not backend._is_electron({"pid": "x"})
    monkeypatch.setattr(ax_driver, "_application_for_pid", lambda pid: 1 / 0)
    assert not backend._is_electron({"pid": 6})


def test_await_ax_actions_ready_waits_out_the_chromium_warmup(monkeypatch, clock):
    ages = {4: 0.5}
    monkeypatch.setattr(ax_driver, "exposure_age", lambda pid: ages.get(pid))
    backend._await_ax_actions_ready({"pid": 4, "bundleId": "com.google.chrome"})
    assert clock.now == pytest.approx(1.5)
    backend._await_ax_actions_ready({"pid": 9, "bundleId": "com.google.chrome"})
    monkeypatch.setattr(backend, "_ELECTRON_BY_PID", {(4, None): False})
    backend._await_ax_actions_ready({"pid": 4, "bundleId": "com.apple.TextEdit"})
    assert clock.now == pytest.approx(1.5)


def test_rouse_woken_renderer_scrolls_the_target_once(monkeypatch, clock):
    performed = []
    monkeypatch.setattr(
        ax_driver, "AXUIElementPerformAction", lambda e, a: performed.append((e, a))
    )
    monkeypatch.setattr(ax_driver, "_WOKEN_PIDS", {4})
    backend._rouse_woken_renderer(_snapshot(), "live")
    backend._rouse_woken_renderer(_snapshot(), "live")
    assert performed == [("live", "AXScrollToVisible")]
    assert not ax_driver.renderer_was_woken(4)


def test_rouse_woken_renderer_keeps_the_mark_when_the_scroll_fails(monkeypatch, clock):
    monkeypatch.setattr(ax_driver, "_WOKEN_PIDS", {4})
    monkeypatch.setattr(ax_driver, "AXUIElementPerformAction", lambda e, a: -25200)
    backend._rouse_woken_renderer(_snapshot(), "live")
    assert ax_driver.renderer_was_woken(4)
    monkeypatch.setattr(ax_driver, "AXUIElementPerformAction", lambda e, a: 1 / 0)
    backend._rouse_woken_renderer(_snapshot(), "live")
    assert ax_driver.renderer_was_woken(4)


def test_exposure_clock_and_woken_marks(monkeypatch, clock):
    monkeypatch.setattr(ax_driver, "_EXPOSED", {})
    monkeypatch.setattr(ax_driver, "_WOKEN_PIDS", set())

    class App:
        def processIdentifier(self):
            return 4

        def launchDate(self):
            return types.SimpleNamespace(timeIntervalSince1970=lambda: 100.0)

    assert ax_driver.exposure_age(4) is None
    assert ax_driver.first_exposure(App())
    clock.now += 3.0
    assert ax_driver.exposure_age(4) == pytest.approx(3.0)
    monkeypatch.setattr(ax_driver, "_pid_of", lambda e: 4)
    ax_driver._restart_exposure("app")
    assert ax_driver.exposure_age(4) == pytest.approx(0.0)
    ax_driver._mark_woken("app")
    assert ax_driver.renderer_was_woken(4)
    ax_driver.clear_woken(4)
    assert not ax_driver.renderer_was_woken(4)
    # An unreadable pid changes nothing.
    monkeypatch.setattr(ax_driver, "_pid_of", lambda e: None)
    clock.now += 1.0
    ax_driver._restart_exposure("app")
    ax_driver._mark_woken("app")
    assert ax_driver.exposure_age(4) == pytest.approx(1.0)
    assert not ax_driver.renderer_was_woken(4)


def _quartz_window_info(monkeypatch, info):
    return _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda option, wid: info(wid),
        kCGWindowListOptionIncludingWindow=8,
    )


def test_window_is_onscreen(monkeypatch):
    _quartz_window_info(
        monkeypatch, lambda wid: [{"kCGWindowIsOnscreen": wid == 1}] if wid else []
    )
    assert ax_driver.window_is_onscreen(1) is True
    assert ax_driver.window_is_onscreen(2) is False
    assert ax_driver.window_is_onscreen(0) is None


def test_wake_hidden_renderer_grows_and_restores_an_offscreen_window(monkeypatch):
    sizes = []
    monkeypatch.setattr(
        ax_driver,
        "AS",
        types.SimpleNamespace(
            kAXValueCGSizeType=2,
            AXValueCreate=lambda kind, size: size,
            AXUIElementSetAttributeValue=lambda w, a, v: sizes.append((a, v)) or 0,
        ),
    )
    monkeypatch.setattr(background_input, "ax_window_id", lambda w: 7)
    monkeypatch.setattr(ax_driver, "_point_size", lambda w: (0, 0, 300.0, 200.0))
    monkeypatch.setattr(ax_driver, "window_is_onscreen", lambda wid: False)
    assert ax_driver._wake_hidden_renderer("win")
    assert sizes == [("AXSize", (301.0, 200.0)), ("AXSize", (300.0, 200.0))]
    # Visible (or unknown) windows are never touched.
    sizes.clear()
    for onscreen in (True, None):
        monkeypatch.setattr(ax_driver, "window_is_onscreen", lambda wid, o=onscreen: o)
        assert not ax_driver._wake_hidden_renderer("win")
    assert sizes == []
    monkeypatch.setattr(ax_driver, "window_is_onscreen", lambda wid: False)
    monkeypatch.setattr(
        ax_driver.AS, "AXUIElementSetAttributeValue", lambda w, a, v: -25200
    )
    assert not ax_driver._wake_hidden_renderer("win")


def test_wake_hidden_renderer_retries_restoring_the_size(monkeypatch, clock):
    results = iter([0, -25200, 0])
    sizes = []

    def set_size(window, attribute, value):
        sizes.append(value)
        return next(results, -1)

    monkeypatch.setattr(
        ax_driver,
        "AS",
        types.SimpleNamespace(
            kAXValueCGSizeType=2,
            AXValueCreate=lambda kind, size: size,
            AXUIElementSetAttributeValue=set_size,
        ),
    )
    monkeypatch.setattr(background_input, "ax_window_id", lambda w: 7)
    monkeypatch.setattr(ax_driver, "_point_size", lambda w: (0, 0, 300.0, 200.0))
    monkeypatch.setattr(ax_driver, "window_is_onscreen", lambda wid: False)
    assert ax_driver._wake_hidden_renderer("win")
    assert sizes == [(301.0, 200.0), (300.0, 200.0), (300.0, 200.0)]
    # A restore that never lands is retried a bounded number of times, then reported.
    results = iter([0])
    sizes.clear()
    assert not ax_driver._wake_hidden_renderer("win")
    assert sizes == [(301.0, 200.0)] + [(300.0, 200.0)] * 3


# --- native popups ------------------------------------------------------------------


@pytest.fixture
def native_popup(monkeypatch, calls, attrs):
    state = {"open": False, "value": "A"}
    attrs["popup"] = {}

    def refresh():
        attrs["popup"] = {
            "AXChildren": ["menu"] if state["open"] else [],
            "AXValue": state["value"],
        }

    attrs["menu"] = {"AXRole": "AXMenu", "AXChildren": ["a", "b", "c"]}
    attrs["a"] = {"AXRole": "AXMenuItem", "AXTitle": "A"}
    attrs["b"] = {"AXRole": "AXMenuItem", "AXTitle": "B"}
    attrs["c"] = {"AXRole": "AXMenuItem", "AXTitle": "C", "AXEnabled": False}
    refresh()

    def perform(element, action):
        calls.append(("ax", element, action))
        if element == "popup" and action == "AXPress":
            state["open"] = True
        elif action == "AXCancel":
            state["open"] = False
        elif action == "AXPress" and element in ("a", "b"):
            state["open"] = False
            state["value"] = attrs[element]["AXTitle"]
        refresh()
        return 0

    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=perform,
    )
    return state


def test_choose_from_ax_menu_presses_the_item_and_reads_back(native_popup, calls):
    # Titles match case-insensitively; the readback is the item's own title.
    assert backend._choose_from_ax_menu("popup", "b", 4) == "B"
    assert [c[1:] for c in calls] == [("popup", "AXPress"), ("b", "AXPress")]


def test_choose_from_ax_menu_cancels_an_unknown_or_disabled_option(native_popup, calls):
    for value in ("C", "Z"):
        with pytest.raises(errors.ComputerUseError) as exc:
            backend._choose_from_ax_menu("popup", value, 4)
        assert exc.value.code == "value_not_settable"
        assert "A, B" in exc.value.message
        assert calls[-1][1:] == ("menu", "AXCancel")
    assert native_popup["value"] == "A"


def test_close_menu_escapes_a_menu_that_ignores_cancel(monkeypatch, calls, attrs):
    attrs["popup"] = {"AXChildren": ["menu"]}
    attrs["menu"] = {"AXRole": "AXMenu"}
    _ax_actions(monkeypatch, calls)
    # The menu ignores everything: two bounded Escapes, then report it open.
    assert backend._close_menu("popup", "menu", 4) is False
    escape = ("press_key", (4, backend.KEY_ALIASES["escape"], 0))
    assert calls.count(escape) == 2
    # An Escape that closes it is confirmed.
    calls.clear()
    monkeypatch.setattr(
        background_input,
        "press_key",
        lambda *a, **k: calls.append(("press_key", a)) or attrs["popup"].clear(),
    )
    attrs["popup"] = {"AXChildren": ["menu"]}
    assert backend._close_menu("popup", "menu", 4) is True
    assert calls.count(escape) == 1


def test_choose_from_ax_menu_never_leaves_its_menu_open(
    monkeypatch, native_popup, calls, attrs
):
    real = sys.modules["ApplicationServices"].AXUIElementPerformAction
    stuck = {"on": False}

    def perform(element, action):
        result = real(element, action)
        if element == "b" and action == "AXPress":
            stuck["on"] = True
        if stuck["on"]:
            attrs["popup"]["AXChildren"] = ["menu"]
        return result

    monkeypatch.setattr(
        sys.modules["ApplicationServices"], "AXUIElementPerformAction", perform
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._choose_from_ax_menu("popup", "B", 4)
    assert exc.value.code == "action_failed"
    assert "stayed open" in exc.value.message
    assert ("ax", "menu", "AXCancel") in calls
    # Unknown option with a menu that will not close: the error says so.
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._choose_from_ax_menu("popup", "Z", 4)
    assert "may still be open" in exc.value.message


def test_choose_from_ax_menu_closes_a_menu_that_opens_late(
    monkeypatch, native_popup, calls
):
    # teardown wait, lingering check, open timeout, then the late look.
    menus = iter([None, None, None, "menu"])
    monkeypatch.setattr(backend, "_open_menu_of", lambda live, timeout: next(menus))
    closed = []
    monkeypatch.setattr(
        backend, "_close_menu", lambda live, menu, pid: closed.append(menu) or True
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._choose_from_ax_menu("popup", "B", 4)
    assert exc.value.code == "action_failed"
    assert "may still be open" not in exc.value.message
    assert closed == ["menu"]
    # A late menu that cannot be closed is called out.
    # teardown wait, lingering check, open timeout, then the late look.
    menus = iter([None, None, None, "menu"])
    monkeypatch.setattr(backend, "_close_menu", lambda live, menu, pid: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._choose_from_ax_menu("popup", "B", 4)
    assert "may still be open" in exc.value.message


def test_choose_from_ax_menu_closes_a_previous_menu_before_pressing(
    monkeypatch, native_popup, calls
):
    # The previous menu outlives the teardown wait.
    menus = iter(["old"] * 40)
    monkeypatch.setattr(backend, "_open_menu_of", lambda live, timeout: next(menus))
    monkeypatch.setattr(backend, "_close_menu", lambda live, menu, pid: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._choose_from_ax_menu("popup", "B", 4)
    assert exc.value.code == "action_failed"
    assert "previous menu did not close" in exc.value.message
    assert not [c for c in calls if c[0] == "ax"]


def test_open_menu_of_times_out(attrs, clock):
    attrs["popup"] = {"AXChildren": ["label"]}
    assert backend._open_menu_of("popup", timeout=0.2) is None


def test_set_value_chooses_a_native_popup_through_its_menu(
    monkeypatch, native_popup, calls
):
    snap = _snapshot(
        bundle="com.apple.TextEdit",
        elements=[{"index": 0, "role": "AXPopUpButton", "center": [1, 1]}],
    )
    monkeypatch.setattr(backend, "_ELECTRON_BY_PID", {(4, None): False})
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "popup")
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        kAXValueAttribute="AXValue",
        AXUIElementSetAttributeValue=lambda *a: pytest.fail("set a native popup"),
        AXUIElementPerformAction=sys.modules[
            "ApplicationServices"
        ].AXUIElementPerformAction,
    )
    monkeypatch.setattr(
        backend,
        "_finish_action",
        lambda app, snap, d, **k: {**d, "verified": k["verified"]},
    )
    result = backend.set_value("App", 0, "b", expected_snapshot=snap)
    assert result["mode"] == "AXMenuChoose" and result["actual"] == "B"
    assert "focus_restored" in result  # always reported, None when focus never moved
    # A case-insensitive choice that landed is verified, not reported as a miss.
    assert result["verified"] is True


# --- perform_secondary_action -----------------------------------------------------


def test_show_menu_is_refused_when_menus_cannot_be_counted(monkeypatch, calls):
    snap = _snapshot(
        elements=[{"index": 0, "role": "AXButton", "actions": ["AXShowMenu"]}]
    )
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snap)
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: None)
    _ax_actions(monkeypatch, calls)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.perform_secondary_action("App", 0, "AXShowMenu")
    assert exc.value.code == "action_failed"
    assert not [c for c in calls if c[0] == "ax"]


def test_secondary_action_resolves_the_index_in_the_expected_snapshot(
    monkeypatch, calls
):
    snap = _snapshot(
        elements=[{"index": 0, "role": "AXButton", "actions": ["AXShowMenu"]}]
    )

    def fresh(*a, **k):
        raise AssertionError("an expected snapshot is never re-observed")

    resolved = []
    _ax_actions(monkeypatch, calls)
    monkeypatch.setattr(backend, "get_app_state", fresh)
    monkeypatch.setattr(
        backend, "_live_element", lambda s, i, **k: resolved.append(s) or None
    )
    with pytest.raises(errors.ComputerUseError):
        backend.perform_secondary_action("App", 0, "AXShowMenu", expected_snapshot=snap)
    assert resolved == [snap]


def test_show_menu_action_never_leaves_the_menu_open(monkeypatch, open_menu, calls):
    snap = _snapshot(
        elements=[{"index": 0, "role": "AXButton", "actions": ["AXShowMenu"]}]
    )
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snap)
    monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
    real = sys.modules["ApplicationServices"].AXUIElementPerformAction

    def perform(element, action):
        result = real(element, action)
        if action == "AXShowMenu":
            monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 1)
        if action == "AXCancel":
            monkeypatch.setattr(backend, "_open_menu_count", lambda pid: 0)
        return result

    sys.modules["ApplicationServices"].AXUIElementPerformAction = perform
    result = backend.perform_secondary_action("App", 0, "AXShowMenu")
    assert result["menu"]["closed"] is True
    assert result["menu"]["items"] == ["Copy", "Paste"]


def test_show_menu_error_after_opening_still_closes_the_menu(
    monkeypatch, open_menu, calls
):
    snap = _snapshot(
        elements=[{"index": 0, "role": "AXButton", "actions": ["AXShowMenu"]}]
    )
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snap)
    open_menu.state["open"] = 0
    real = sys.modules["ApplicationServices"].AXUIElementPerformAction

    def perform(element, action):
        result = real(element, action)
        if action == "AXShowMenu":
            open_menu.state["open"] = 1
            return -25204  # kAXErrorCannotComplete after the menu opened
        return result

    sys.modules["ApplicationServices"].AXUIElementPerformAction = perform
    result = backend.perform_secondary_action("App", 0, "AXShowMenu")
    assert result["ax_error"] == -25204
    assert result["menu"]["closed"] is True and open_menu.state["open"] == 0
    assert "focus_restored" in result

    def failing(element, action):
        calls.append(("ax", element, action))
        return -25200

    sys.modules["ApplicationServices"].AXUIElementPerformAction = failing
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.perform_secondary_action("App", 0, "AXShowMenu")
    assert exc.value.code == "accessibility_error"


def test_semantic_click_error_after_opening_a_menu_settles_it(
    monkeypatch, open_menu, calls
):
    snap = _popup_snapshot()
    monkeypatch.setattr(backend, "_finish_action", lambda app, s, d, **k: d)
    open_menu.state["open"] = 0
    real = sys.modules["ApplicationServices"].AXUIElementPerformAction

    def perform(element, action):
        result = real(element, action)
        if action == "AXPress" and element == "popup":
            open_menu.state["open"] = 1
            return -25204
        return result

    sys.modules["ApplicationServices"].AXUIElementPerformAction = perform
    result = backend.click("App", element_index=0, expected_snapshot=snap)
    assert result["mode"] == "AXPress" and result["ax_error"] == -25204
    assert result["menu"]["closed"] is True and open_menu.state["open"] == 0
    assert not [c for c in calls if c[0] == "click"]  # not clicked again


def test_cancel_menu_swallows_an_axcancel_exception():
    seen = []

    def perform(element, action):
        seen.append((element, action))
        raise RuntimeError("pyobjc bridge error")

    backend._cancel_menu(perform, "menu")  # Escape fallback still runs after it
    assert seen == [("menu", "AXCancel")]


def test_choose_from_ax_menu_closes_a_menu_its_failed_press_opened(
    monkeypatch, native_popup, calls
):
    real = sys.modules["ApplicationServices"].AXUIElementPerformAction

    def perform(element, action):
        result = real(element, action)
        return -25204 if (element, action) == ("popup", "AXPress") else result

    sys.modules["ApplicationServices"].AXUIElementPerformAction = perform
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._choose_from_ax_menu("popup", "B", 4)
    assert exc.value.code == "accessibility_error"
    assert "may still be open" not in exc.value.message
    assert native_popup["open"] is False
    assert ("ax", "menu", "AXCancel") in calls


# --- window titles without Screen Recording ---------------------------------------


def test_ax_window_titles_map_cg_ids(monkeypatch, attrs):
    monkeypatch.setattr(ax_driver, "AS", object())
    monkeypatch.setattr(ax_driver, "AXUIElementCreateApplication", lambda pid: "app")
    attrs["app"] = {"AXWindows": ["w1", "w2", "w3"]}
    attrs["w1"] = {"AXTitle": "Inbox"}
    attrs["w2"] = {"AXTitle": None}
    ids = {"w1": 11, "w2": 12}
    monkeypatch.setattr(background_input, "ax_window_id", lambda w: ids.get(w))
    assert backend._ax_window_titles(4) == {11: "Inbox"}
    monkeypatch.setattr(
        background_input, "ax_window_id", lambda w: (_ for _ in ()).throw(OSError())
    )
    assert backend._ax_window_titles(4) == {}
    monkeypatch.setattr(ax_driver, "AS", None)
    assert backend._ax_window_titles(4) == {}


def test_cg_window_names_visible_follows_screen_capture_access(monkeypatch):
    _install_module(monkeypatch, "Quartz", CGPreflightScreenCaptureAccess=lambda: False)
    assert backend._cg_window_names_visible() is False
    _install_module(monkeypatch, "Quartz")
    assert backend._cg_window_names_visible() is True


def test_window_records_fill_titles_from_ax(monkeypatch):
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: [
            {
                "kCGWindowNumber": 101,
                "kCGWindowOwnerPID": 4,
                "kCGWindowLayer": 0,
                "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 500, "Height": 400},
            }
        ],
        kCGNullWindowID=0,
        kCGWindowListExcludeDesktopElements=1,
        kCGWindowListOptionOnScreenOnly=2,
    )
    monkeypatch.setattr(backend, "_ax_window_titles", lambda pid: {101: "Inbox"})
    monkeypatch.setattr(backend, "_offscreen_ax_windows", lambda app, seen: [])
    records = backend._window_records({"pid": 4, "name": "App"})
    assert [r["title"] for r in records] == ["Inbox"]


def test_offscreen_candidates_use_size_when_titles_are_hidden(monkeypatch, attrs):
    cg = [
        {
            "kCGWindowNumber": number,
            "kCGWindowOwnerPID": 4,
            "kCGWindowLayer": 0,
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": w, "Height": h},
        }
        for number, w, h in ((303, 800, 600), (404, 1, 1), (505, 900, 20))
    ]
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: cg,
        kCGNullWindowID=0,
        kCGWindowListOptionAll=0,
    )
    monkeypatch.setattr(backend, "_cg_window_names_visible", lambda: False)
    monkeypatch.setattr(ax_driver, "AXUIElementCreateApplication", lambda pid: "app")
    monkeypatch.setattr(ax_driver, "_app_windows", lambda app: [])
    scans = []
    monkeypatch.setattr(
        ax_driver,
        "discover_remote_windows",
        lambda pid, wanted: scans.append(set(wanted)) or set(),
    )
    monkeypatch.setattr(backend, "_REMOTE_SCANNED", {})
    backend._offscreen_ax_windows_unchecked({"pid": 4}, set())
    assert scans == [{303}]


# --- AX scroll of a window the window server is not compositing --------------------


class _Scroller:
    """A scroll view whose rows move by ``offset`` when scrolled into view."""

    def __init__(self, attrs, rows=10, row_height=100.0, viewport=300.0):
        self.attrs = attrs
        self.offset = 0.0
        self.rows = [f"row{i}" for i in range(rows)]
        self.row_height = row_height
        self.viewport = viewport
        attrs["scroller"] = {"AXChildren": list(self.rows)}

    def frame(self, element):
        if element == "scroller":
            return (0.0, 0.0, 400.0, self.viewport)
        if element in self.rows:
            i = self.rows.index(element)
            return (0.0, i * self.row_height - self.offset, 400.0, self.row_height)
        return None

    def scroll_to(self, element):
        _, y, _, h = self.frame(element)
        if y + h > self.viewport:
            self.offset += y + h - self.viewport
        elif y < 0:
            self.offset += y


@pytest.fixture
def scroller(monkeypatch, attrs, clock):
    view = _Scroller(attrs)
    monkeypatch.setattr(ax_driver, "_point_size", view.frame)
    monkeypatch.setattr(
        ax_driver,
        "AXUIElementPerformAction",
        lambda e, a: view.scroll_to(e) if a == "AXScrollToVisible" else None,
    )
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "scroller")
    return view


def _scroll_snapshot():
    return _snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXGroup",
                "x": 0,
                "y": 0,
                "width": 800,
                "height": 900,
            },
            {
                "index": 1,
                "role": "AXScrollArea",
                "x": 0,
                "y": 0,
                "width": 400,
                "height": 300,
            },
            {
                "index": 2,
                "role": "AXButton",
                "x": 500,
                "y": 0,
                "width": 10,
                "height": 10,
            },
        ]
    )


def test_ax_scroll_target_prefers_the_furthest_row_within_reach(scroller):
    lo, hi, node = backend._ax_scroll_target("scroller", True, True, 300.0)
    assert (lo, hi, node) == (0.0, 300.0, "row5")
    assert backend._ax_scroll_target("scroller", True, True, 50.0)[2] == "row3"
    assert backend._ax_scroll_target("scroller", True, False, 300.0) is None
    assert backend._ax_scroll_target("missing", True, True, 300.0) is None


def test_ax_scroll_target_takes_clipped_rows_in_document_order(monkeypatch, attrs):
    frames = {"s": (0.0, 0.0, 100.0, 300.0), "a": (0.0, 300.0, 100.0, 0.0)}
    frames["b"] = (0.0, 300.0, 100.0, 0.0)
    attrs["s"] = {"AXChildren": ["a", "b"]}
    monkeypatch.setattr(ax_driver, "_point_size", frames.get)
    assert backend._ax_scroll_target("s", True, True, 300.0)[2] == "a"


def test_ax_scroll_moves_about_a_page_on_the_smallest_scroller(scroller):
    moved = backend._ax_scroll(_scroll_snapshot(), (10.0, 10.0), "down", 1.0)
    assert moved["scroller"] == 1
    assert 250 <= moved["points"] <= 350
    assert scroller.offset > 0
    back = backend._ax_scroll(_scroll_snapshot(), (10.0, 10.0), "up", 5.0)
    assert back is not None and scroller.offset == 0


def test_ax_scroll_returns_none_when_nothing_moves(monkeypatch, scroller):
    monkeypatch.setattr(ax_driver, "AXUIElementPerformAction", lambda e, a: None)
    assert backend._ax_scroll(_scroll_snapshot(), (10.0, 10.0), "down", 1.0) is None
    # An element that drifted is skipped, not acted on.
    monkeypatch.setattr(
        backend,
        "_live_element",
        lambda *a, **k: (_ for _ in ()).throw(
            errors.ComputerUseError("target_drift", "moved")
        ),
    )
    assert backend._ax_scroll(_scroll_snapshot(), (10.0, 10.0), "down", 1.0) is None


def test_scroll_uses_ax_for_a_window_that_is_not_composited(monkeypatch, scroller):
    monkeypatch.setenv(background_input.DELIVERY_ENV, "background")
    monkeypatch.setattr(background_input, "skylight_available", lambda: True)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(backend, "_finish_action", lambda app, snap, d, **k: d)
    monkeypatch.setattr(
        background_input, "scroll", lambda *a, **k: pytest.fail("wheel posted")
    )
    monkeypatch.setattr(ax_driver, "window_is_onscreen", lambda wid: False)
    result = backend.scroll(
        "App", "down", x=10, y=10, expected_snapshot=_scroll_snapshot()
    )
    assert result["mode"] == "AX-scroll" and result["scroller"] == 1
    # A composited window keeps the wheel route.
    wheels = []
    monkeypatch.setattr(ax_driver, "window_is_onscreen", lambda wid: True)
    monkeypatch.setattr(
        background_input, "scroll", lambda *a, **k: wheels.append(a) or True
    )
    result = backend.scroll(
        "App", "down", x=10, y=10, expected_snapshot=_scroll_snapshot()
    )
    assert result["mode"] == "SkyLight-scroll" and len(wheels) == 1


# --- remaining branches ------------------------------------------------------------


def test_keyed_target_waits_while_ax_focus_trails_key_status(monkeypatch, calls):
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 555))
    trailing = {"left": 1}

    def validate(snap, *a, **k):
        calls.append(("validate", a, k))
        if trailing["left"]:
            trailing["left"] -= 1
            raise errors.ComputerUseError("stale_snapshot", "focused window trails")

    monkeypatch.setattr(backend, "_validate_focused_window", validate)
    with backend._keyed_target(_snapshot()):
        calls.append(("body",))
    names = [c[0] for c in calls]
    assert names.count("validate") == 2
    assert names[-2:] == ["body", "restore"]


def test_settle_menus_reports_a_close_that_raises(monkeypatch, open_menu, calls):
    def drifted(*a, **k):
        raise errors.ComputerUseError("stale_snapshot", "element moved")

    def broken_close(*a):
        raise RuntimeError("window list unavailable")

    monkeypatch.setattr(backend, "_live_element", drifted)
    monkeypatch.setattr(backend, "_close_menus", broken_close)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._settle_menus(_snapshot(), 0, 0, None, expect_menu=True)
    assert exc.value.code == "stale_snapshot"
    assert "may still be open" in exc.value.message


def test_right_click_at_a_point_settles_its_context_menu(monkeypatch, open_menu, calls):
    open_menu.state["open"] = 0

    def pixel_click(snapshot, x, y, **kwargs):
        open_menu.state["open"] = 1
        return {"mode": "SkyLight-click"}

    monkeypatch.setattr(backend, "_pixel_click", pixel_click)
    result = backend.click(
        "App", x=10, y=10, mouse_button="right", expected_snapshot=_snapshot()
    )
    # No element to read items under: closed by Escape (which this fake
    # menu ignores, so the report says it stayed open).
    assert result["menu"]["closed"] is False and "warning" in result["menu"]
    assert ("press_key", (4, backend.KEY_ALIASES["escape"])) in calls


def test_choose_from_ax_menu_closes_the_menu_when_the_option_press_fails(
    monkeypatch, native_popup, calls
):
    real = sys.modules["ApplicationServices"].AXUIElementPerformAction

    def perform(element, action):
        if (element, action) == ("b", "AXPress"):
            calls.append(("ax", element, action))
            return -25200
        return real(element, action)

    sys.modules["ApplicationServices"].AXUIElementPerformAction = perform
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._choose_from_ax_menu("popup", "B", 4)
    assert exc.value.code == "accessibility_error"
    assert "may still be open" not in exc.value.message
    assert native_popup["open"] is False and native_popup["value"] == "A"


def test_choose_from_ax_menu_waits_for_the_value_to_land(
    monkeypatch, native_popup, calls
):
    values = iter(["A", "A", "B"])
    monkeypatch.setattr(backend, "_read_value", lambda live: next(values))
    assert backend._choose_from_ax_menu("popup", "B", 4) == "B"


def test_press_menu_item_reports_an_unsynthesized_chord(monkeypatch, menubar, calls):
    monkeypatch.setattr(backend, "_process_is_active", lambda snap: True)
    monkeypatch.setattr(background_input, "press_key", lambda *a, **k: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._press_menu_item("App", _snapshot(), "copy", "Copy", False)
    assert exc.value.code == "action_failed"
    assert calls[-1][0] == "restore"


def test_menu_item_chord_without_a_keycode(monkeypatch, attrs):
    attrs["a"] = {"AXMenuItemCmdChar": "S", "AXMenuItemCmdModifiers": 0}
    monkeypatch.setattr(ax_driver, "_keycode_for", lambda char: None)
    assert backend._menu_item_chord("a") is None


def test_ax_scroll_target_skips_nodes_without_a_frame(monkeypatch, attrs):
    frames = {"s": (0.0, 0.0, 100.0, 300.0), "b": (0.0, 200.0, 100.0, 200.0)}
    attrs["s"] = {"AXChildren": ["a", "b"]}
    monkeypatch.setattr(ax_driver, "_point_size", frames.get)
    assert backend._ax_scroll_target("s", True, True, 300.0)[2] == "b"


def test_ax_scroll_skips_unusable_scrollers(monkeypatch, attrs, clock):
    snap = _scroll_snapshot()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: None)
    assert backend._ax_scroll(snap, (10.0, 10.0), "down", 1.0) is None
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "s")
    monkeypatch.setattr(ax_driver, "_point_size", lambda e: None)
    assert backend._ax_scroll(snap, (10.0, 10.0), "down", 1.0) is None
    # A scroller with nothing past either edge moves in neither direction.
    frames = {"s": (0.0, 0.0, 100.0, 300.0)}
    monkeypatch.setattr(ax_driver, "_point_size", frames.get)
    monkeypatch.setattr(
        ax_driver, "AXUIElementPerformAction", lambda *a: pytest.fail("scrolled")
    )
    assert backend._ax_scroll(snap, (10.0, 10.0), "down", 1.0) is None
    assert backend._ax_scroll(snap, (10.0, 10.0), "up", 1.0) is None


def test_collect_wakes_a_hidden_renderer_once(monkeypatch, clock):
    monkeypatch.setattr(ax_driver, "_app_element", lambda *a, **k: "app")
    monkeypatch.setattr(ax_driver, "_app_windows", lambda app: ["win"])
    walks = {"n": 0}

    def walk(element, depth, out, counter, seen=None, **kwargs):
        walks["n"] += 1
        role = "AXWebArea" if walks["n"] > 1 else "AXGroup"
        if seen is not None:
            seen.add(role)
        out.append({"role": role, "element": element})
        counter[0] += 1

    events = []
    monkeypatch.setattr(ax_driver, "_walk", walk)
    monkeypatch.setattr(ax_driver, "_wake_hidden_renderer", lambda w: True)
    monkeypatch.setattr(ax_driver, "_restart_exposure", lambda a: events.append("x"))
    monkeypatch.setattr(ax_driver, "_mark_woken", lambda a: events.append("w"))
    collected = ax_driver.collect("A", retry_web_content=True)
    assert [e["role"] for e in collected] == ["AXWebArea"]
    assert events == ["x", "w"]
    # Nothing to wake: no marks, the plain retry wait.
    walks["n"] = 0
    events.clear()
    monkeypatch.setattr(ax_driver, "_wake_hidden_renderer", lambda w: False)
    ax_driver.collect("A", retry_web_content=True)
    assert events == []
