"""Perception session behaviour that needs no screen (backend faked)."""

from __future__ import annotations

import itertools
import sys
import time
import types

import pytest

from rapid_mlx.computer_use import perception
from rapid_mlx.computer_use.errors import ComputerUseError


def _element(index, role, label, value=None, width=100, height=20):
    return {
        "index": index,
        "role": role,
        "label": label,
        "value": value,
        "width": width,
        "height": height,
        "center": [10, 10 + index],
    }


def _snapshot(elements):
    return {
        "app": {"name": "Chrome"},
        "window_id": "cg:1",
        "window": {"title": "t"},
        "elements": elements,
    }


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(perception.backend, "live_elements", lambda snapshot: None)
    s = perception.PerceptionSession()
    monkeypatch.setattr(s, "keep_awake", lambda: None)
    monkeypatch.setattr(perception, "_require_display_awake", lambda: None)
    return s


def _observe_from(session, monkeypatch, snapshots):
    """Make each observe return the next snapshot (the last one repeats)."""
    queue = list(snapshots)

    def get_app_state(app, **kw):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(perception.backend, "get_app_state", get_app_state)


def test_visually_hidden_sliver_is_folded_off_screen(session, monkeypatch):
    _observe_from(
        session,
        monkeypatch,
        [
            _snapshot(
                [
                    _element(0, "AXHeading", "Skip to", width=1, height=24),
                    _element(1, "AXLink", "main content", width=1, height=20),
                    _element(2, "AXLink", "Your Orders"),
                ]
            )
        ],
    )
    obs = session.observe("Chrome", "cg:1")
    text = obs.render()
    assert '"Your Orders"' in text
    assert '"main content"' not in text
    assert "2 more elements off screen" in text
    assert '"Skip to"' not in text  # a hidden menu is not a section to scroll to
    assert "(off screen)" in obs.find("main content")


def test_wait_ignores_typed_text_and_waits_for_the_reply(session, monkeypatch):
    monkeypatch.setattr(perception, "WAIT_POLL_S", 0.01)
    monkeypatch.setattr(perception, "WAIT_QUIET_S", 0.05)
    before = [
        _element(0, "AXStaticText", "Hi, how can I help?"),
        _element(1, "AXTextField", "Message", ""),
    ]
    cleared = [
        _element(0, "AXStaticText", "Hi, how can I help?"),
        _element(1, "AXTextField", "Message", "x"),
    ]
    reply = [*cleared, _element(2, "AXStaticText", "It arrives Oct 11.")]
    snaps = [_snapshot(before)] + [_snapshot(cleared)] * 20 + [_snapshot(reply)]
    _observe_from(session, monkeypatch, snaps)
    session.observe("Chrome", "cg:1")
    out = session.wait("cg:1", timeout=5)
    assert out["met"]
    assert any(r.label == "It arrives Oct 11." for r in out["observation"].rows)


def test_window_not_yet_observed_is_resolved_by_owner(session, monkeypatch):
    _observe_from(session, monkeypatch, [_snapshot([_element(0, "AXLink", "Home")])])
    monkeypatch.setattr(perception, "_window_owner", lambda wid: "pid:42")
    assert session._window_obs("cg:1").window_id == "cg:1"


def test_return_in_unfocused_web_field_clicks_it_first(session, monkeypatch):
    calls = []

    def press_key(app, key, window_id=None, expected_snapshot=None, element_index=None):
        calls.append(("key", element_index))
        if element_index is not None:
            raise ComputerUseError("target_drift", "focus is elsewhere")
        return {"effect": "unverifiable"}

    monkeypatch.setattr(perception.backend, "press_key", press_key)
    monkeypatch.setattr(
        perception.backend,
        "click",
        lambda app, element_index=None, **kw: (
            calls.append(("click", element_index)) or {}
        ),
    )
    row = perception.Row("e1", "AXTextField", "Search", "milk", (), 3, (0, 0))
    session._dispatch("key", "Chrome", {}, "cg:1", row, {"key": "Return"})
    assert calls == [("key", 3), ("click", 3), ("key", None)]

    button = perception.Row("e2", "AXButton", "Go", None, (), 4, (0, 0))
    with pytest.raises(ComputerUseError):
        session._dispatch("key", "Chrome", {}, "cg:1", button, {"key": "Return"})


# ---------------------------------------------------------------------------
# A scripted screen: windows hold elements keyed by a stable live identity,
# so the session's ref tracking sees the same "AX element" across walks.
# ---------------------------------------------------------------------------


def E(  # noqa: N802
    key,
    role,
    label="",
    value=None,
    states=(),
    width=100,
    height=20,
    subrole="",
    **extra,  # path, web, field_name, filled: as the walk reports them
):
    return {
        "key": key,
        "role": role,
        "label": label,
        "value": value,
        "states": list(states),
        "width": width,
        "height": height,
        "subrole": subrole,
        **extra,
    }


class Screen:
    """Stands in for backend: observations and the input routes."""

    def __init__(self, monkeypatch):
        self.windows: dict[str, dict] = {}
        self.calls: list[tuple[str, dict]] = []
        self.handlers: dict = {}
        self.front = ["com.google.Chrome"]
        self.serial = 0
        backend = perception.backend
        monkeypatch.setattr(backend, "get_app_state", self.get_app_state)
        monkeypatch.setattr(backend, "live_elements", lambda snap: snap.get("_lives"))
        for op in (
            "click",
            "set_value",
            "type_text",
            "press_key",
            "hotkey",
            "scroll",
            "perform_secondary_action",
        ):
            monkeypatch.setattr(backend, op, self._route(op))
        monkeypatch.setattr(perception, "_frontmost_bundle", lambda: self.front[0])

    def show(self, elements, wid="cg:1", title="Shop", truncated=False):
        self.windows[wid] = {
            "elements": list(elements),
            "title": title,
            "truncated": truncated,
        }

    def get_app_state(self, app, *, window_id=None, **kw):
        if window_id is None:
            wid = next(iter(self.windows))
        else:
            wid = f"cg:{perception.backend._cg_window_id(window_id)}"
        window = self.windows[wid]
        self.serial += 1
        elements = []
        for i, spec in enumerate(window["elements"]):
            element = {k: v for k, v in spec.items() if k != "key"}
            element.update(index=i, center=[10, 10 + 30 * i])
            elements.append(element)
        return {
            "snapshot_id": f"s{self.serial}",
            "app": {"name": "Chrome", "pid": 7},
            "window_id": wid,
            "window": {"title": window["title"]},
            "elements": elements,
            "truncated": window["truncated"],
            "visible_window_ids": list(self.windows),
            "_lives": [("live", wid, spec["key"]) for spec in window["elements"]],
        }

    def _route(self, op):
        def call(*args, **kwargs):
            self.calls.append((op, kwargs | {"args": args}))
            handler = self.handlers.get(op)
            return handler(*args, **kwargs) if handler else {"effect": "unverifiable"}

        return call


@pytest.fixture
def screen(monkeypatch, session):
    monkeypatch.setattr(perception, "SETTLE_POLL_S", 0)
    monkeypatch.setattr(perception, "SETTLE_MIN_S", 0)
    monkeypatch.setattr(perception, "SETTLE_CAP_S", 0.05)
    monkeypatch.setattr(perception, "SETTLE_CAP_SLOW_S", 0.05)
    monkeypatch.setattr(perception, "WAIT_POLL_S", 0)
    # The app's live focus is a harmless field unless a test says otherwise.
    monkeypatch.setattr(perception, "_focused_secret", lambda app_info: None)
    return Screen(monkeypatch)


def _ref(obs, label):
    return next(row.ref for row in obs.rows if row.label == label)


# -- rendering, folding, find --------------------------------------------------


def test_render_marks_new_rows_values_states_and_hides_extension_buttons(
    session, screen
):
    screen.show(
        [
            E("a", "AXButton", "Add to cart"),
            E("x", "AXPopUpButton", "Grammarly has access to this site"),
        ]
    )
    first = session.observe("Chrome", "cg:1")
    assert first.previous is None and first.changes == []
    screen.show(
        [
            E("a", "AXButton", "Add to cart", states=("disabled",)),
            E("q", "AXTextField", "Qty", value="2"),
            E("x", "AXPopUpButton", "Grammarly has access to this site"),
        ],
        truncated=True,
    )
    obs = session.observe("Chrome", "cg:1")
    text = obs.render()
    assert "TRUNCATED" in text
    assert f"changes since {first.obs_id}: +1 -0 ~1" in text
    assert f' {_ref(obs, "Add to cart")} button "Add to cart" [disabled]' in text
    assert f"*{_ref(obs, 'Qty')} textfield \"Qty\" = '2'" in text
    assert "Grammarly" not in text.split("elements:")[1].split("(")[0]
    assert "(1 browser-extension buttons hidden)" in text
    assert "elements:" not in obs.render(full=False)
    assert all("Grammarly" not in t for t in obs.texts())
    assert "find 'grammarly'" in obs.find("grammarly") and ": 0 matches" in obs.find(
        "grammarly"
    )


def test_off_screen_rows_fold_into_named_sections_and_stay_findable(
    session, screen, monkeypatch
):
    monkeypatch.setattr(perception, "MAX_FOLD_SECTIONS", 2)
    monkeypatch.setattr(perception, "MAX_FIND_HITS", 2)
    screen.show(
        [
            E("top", "AXLink", "Deals"),
            E("h1", "AXHeading", "Reviews", width=0, height=0),
            E("h1b", "AXHeading", "Reviews", width=0, height=0),
            E("h2", "AXHeading", "Questions", width=0, height=0),
            E("h3", "AXHeading", "Related items", width=0, height=0),
            E("r1", "AXStaticText", "Great milk", width=0, height=0),
            E("r2", "AXStaticText", "Milk was warm", width=0, height=0),
            E("r3", "AXStaticText", "milk again", width=0, height=0),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    text = obs.render()
    assert '"Deals"' in text and '"Great milk"' not in text
    assert (
        '7 more elements off screen; sections: "Reviews" · "Questions" +1 more' in text
    )
    everything = obs.render(everything=True)
    assert '"Great milk"' in everything and "off screen" not in everything
    found = obs.find("MILK")
    assert "3 matches" in found and "(off screen)" in found
    assert "… 1 more; narrow the text" in found


def test_choices_name_radio_groups_checked_boxes_and_menu_values(session, screen):
    screen.show(
        [
            E("leg", "AXStaticText", "Sat, Oct 10"),
            E("r1", "AXRadioButton", "10–12 AM", states=("unchecked",)),
            E("r2", "AXRadioButton", "12–2 PM", states=("checked",)),
            E("cb", "AXCheckBox", "Leave at door", states=("checked",)),
            E("cb2", "AXCheckBox", "Gift wrap", states=("unchecked",)),
            E("tip", "AXPopUpButton", "Driver tip", value="$5"),
            E("chrome", "AXPopUpButton", "Chrome", value="menu"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    assert obs.choices() == [
        "Sat, Oct 10 12–2 PM",
        "[x] Leave at door",
        "Driver tip: $5",
    ]
    assert "Driver tip $5" in obs.texts()


# -- refs and diffs ------------------------------------------------------------


def test_refs_follow_the_live_element_and_diffs_name_what_changed(
    session, screen, monkeypatch
):
    screen.show(
        [
            E("a", "AXButton", "Cart (0)"),
            E("b", "AXLink", "Home"),
            E("c", "AXCheckBox", "Box", states=("unchecked",)),
        ]
    )
    first = session.observe("Chrome", "cg:1")
    cart = _ref(first, "Cart (0)")
    # Renamed and reordered: same element, same ref; a role change is a new element.
    screen.show(
        [
            E("n", "AXStaticText", "Added"),
            E("a", "AXButton", "Cart (1)"),
            E("c", "AXCheckBox", "Box", states=("checked",)),
            E("b", "AXButton", "Home"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    assert _ref(obs, "Cart (1)") == cart
    assert _ref(obs, "Home") != _ref(first, "Home")
    assert obs.change_counts == (2, 1, 2)
    joined = "\n".join(obs.changes)
    assert f'~ {cart} button "Cart (1)" label "Cart (0)" -> "Cart (1)"' in joined
    assert "[unchecked] -> [checked]" in joined
    assert "+ *e" in joined and "- e" in joined
    # A live element walked twice in one snapshot gets a second ref.
    screen.show([E("a", "AXButton", "Cart (1)"), E("a", "AXButton", "Cart (1)")])
    dup = session.observe("Chrome", "cg:1")
    assert len({row.ref for row in dup.rows}) == 2
    # Value changes and the change-line cap.
    monkeypatch.setattr(perception, "MAX_CHANGE_LINES", 2)
    screen.show(
        [E("a", "AXButton", "Cart (1)", value="v")]
        + [E(f"k{i}", "AXLink", f"L{i}") for i in range(4)]
    )
    capped = session.observe("Chrome", "cg:1")
    assert "value None -> 'v'" in capped.changes[0]
    assert capped.changes[-1].startswith("... ") and len(capped.changes) == 3


def test_without_live_elements_every_walk_mints_new_refs(session, monkeypatch):
    _observe_from(session, monkeypatch, [_snapshot([_element(0, "AXLink", "Home")])])
    first = session.observe("Chrome", "cg:1")
    second = session.observe("Chrome", "cg:1")
    assert first.rows[0].ref != second.rows[0].ref
    assert second.change_counts == (1, 1, 0)


def test_remembered_elements_are_bounded(session, screen, monkeypatch):
    monkeypatch.setattr(perception, "MAX_REMEMBERED_ELEMENTS", 3)
    screen.show([E(f"k{i}", "AXLink", f"L{i}") for i in range(3)])
    session.observe("Chrome", "cg:1")
    screen.show([E(f"n{i}", "AXLink", f"N{i}") for i in range(2)])
    obs = session.observe("Chrome", "cg:1")
    assert len(session._refs) == 2
    assert {entry[0] for entry in session._refs.values()} == {
        row.ref for row in obs.rows
    }


def test_pruning_keeps_the_refs_other_windows_still_show(session, screen, monkeypatch):
    monkeypatch.setattr(perception, "MAX_REMEMBERED_ELEMENTS", 3)
    screen.show([E("o", "AXLink", "Other")], wid="cg:2")
    other = session.observe("Chrome", "cg:2")
    screen.show([E(f"k{i}", "AXLink", f"L{i}") for i in range(2)])
    session.observe("Chrome", "cg:1")
    screen.show([E(f"n{i}", "AXLink", f"N{i}") for i in range(2)])
    obs = session.observe("Chrome", "cg:1")
    assert {entry[0] for entry in session._refs.values()} == {
        row.ref for row in [*obs.rows, *other.rows]
    }


def test_stale_ref_is_refused_with_the_latest_observations(session, screen):
    with pytest.raises(ComputerUseError) as err:
        session.act("click", "e99")
    assert err.value.code == "stale_ref" and "latest: none" in err.value.message
    screen.show([E("a", "AXLink", "Home")])
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("click", "e99")
    assert f"{obs.obs_id} (window cg:1)" in err.value.message


# -- receipts and the unresolved gate -------------------------------------------


def test_click_receipt_reports_target_change_new_windows_and_front(session, screen):
    screen.show([E("a", "AXButton", "Open"), E("t", "AXStaticText", "x")])
    obs = session.observe("Chrome", "cg:1")

    def click(app, **kw):
        screen.show([E("a", "AXButton", "Opened"), E("t", "AXStaticText", "x")])
        screen.show([E("p", "AXButton", "OK")], wid="cg:2", title="Popup")
        screen.front[0] = "com.apple.Safari"
        return {
            "effect": "dispatched",
            "verification": "v",
            "warnings": ["w"],
            "menu": {"m": 1},
            "mode": "AXPress",
        }

    screen.handlers["click"] = click
    out = session.act("click", _ref(obs, "Open"))
    receipt = out["receipt"]
    assert receipt["effect"] == "target_changed" and receipt["target_changed"] is True
    assert receipt["settled"] is True
    assert receipt["new_windows"] == ["cg:2"]
    assert receipt["frontmost_changed"] == ["com.google.Chrome", "com.apple.Safari"]
    assert receipt["verification"] == "v" and receipt["warnings"] == ["w"]
    assert receipt["menu"] == {"m": 1} and receipt["mode"] == "AXPress"
    assert "unresolved" not in receipt
    assert out["observation"].previous == obs.obs_id
    kwargs = screen.calls[0][1]
    assert kwargs["element_index"] == 0 and kwargs["window_id"] == "cg:1"


def test_silence_blocks_input_until_the_window_is_observed_again(session, screen):
    screen.show([E("a", "AXButton", "Send")])
    obs = session.observe("Chrome", "cg:1")
    out = session.act("click", _ref(obs, "Send"))
    assert out["receipt"]["effect"] == "no_visible_change"
    assert "unresolved" in out["receipt"]
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Send"))
    assert err.value.code == "unresolved_outcome"
    session.observe("Chrome", "cg:1")
    screen.handlers["click"] = lambda app, **kw: {"effect": "confirmed"}
    assert session.act("click", _ref(obs, "Send"))["receipt"]["effect"] == "confirmed"


def test_other_changes_resolve_and_a_suspected_noop_does_not(session, screen):
    screen.show([E("a", "AXButton", "Send"), E("m", "AXStaticText", "hi")])
    obs = session.observe("Chrome", "cg:1")

    def click(app, **kw):
        screen.show(
            [
                E("a", "AXButton", "Send"),
                E("m", "AXStaticText", "hi"),
                E("r", "AXStaticText", "sent"),
            ]
        )
        return {}

    screen.handlers["click"] = click
    out = session.act("click", _ref(obs, "Send"))
    assert (
        out["receipt"]["effect"] == "other_changes_only"
        and "unresolved" not in out["receipt"]
    )
    screen.handlers["click"] = lambda app, **kw: {"effect": "suspected_noop"}
    out = session.act("click", _ref(out["observation"], "Send"))
    assert (
        out["receipt"]["effect"] == "suspected_noop" and "unresolved" in out["receipt"]
    )


def test_unsettled_page_with_unverified_outcome_is_unresolved(
    session, screen, monkeypatch
):
    screen.show([E("a", "AXButton", "Spin")])
    obs = session.observe("Chrome", "cg:1")
    ticks = itertools.count()
    real = screen.get_app_state

    def moving(app, **kw):
        snap = real(app, **kw)
        snap["elements"][0]["value"] = str(next(ticks))
        return snap

    monkeypatch.setattr(perception.backend, "get_app_state", moving)
    screen.handlers["click"] = lambda app, **kw: {"effect": "dispatched"}
    out = session.act("click", _ref(obs, "Spin"))
    assert out["receipt"]["settled"] is False
    assert out["receipt"]["effect"] == "target_changed"
    assert "unresolved" in out["receipt"]


def test_fill_is_confirmed_by_the_observed_value_even_if_the_route_refused(
    session, screen
):
    screen.show([E("q", "AXTextField", "Search", value="")])
    obs = session.observe("Chrome", "cg:1")

    def set_value(app, index, text, **kw):
        screen.show([E("q", "AXTextField", "Search", value=text)])
        raise ComputerUseError("action_failed", "second route refused")

    screen.handlers["set_value"] = set_value
    out = session.act("fill", _ref(obs, "Search"), text="milk")
    assert out["receipt"]["effect"] == "confirmed"
    assert out["receipt"]["error"]["code"] == "action_failed"
    assert out["receipt"]["settled"] is True
    # A value that was already there confirms nothing: the refusal stands.
    screen.handlers["set_value"] = _raise("action_failed", "refused")
    out = session.act("fill", _ref(out["observation"], "Search"), text="milk")
    assert out["receipt"]["effect"] == "refused"


def test_refused_action_is_reported_not_unresolved(session, screen):
    screen.show([E("a", "AXButton", "Go")])
    obs = session.observe("Chrome", "cg:1")
    screen.handlers["click"] = lambda app, **kw: (_ for _ in ()).throw(
        ComputerUseError("target_occluded", "covered")
    )
    out = session.act("click", _ref(obs, "Go"))
    assert out["receipt"]["effect"] == "refused"
    assert out["receipt"]["error"] == {"code": "target_occluded", "message": "covered"}
    assert "unresolved" not in out["receipt"]


def test_menu_item_click_settles_on_the_chosen_value(session, screen):
    screen.show([E("p", "AXPopUpButton", "Size", value="S")])
    obs = session.observe("Chrome", "cg:1")

    def click(app, **kw):
        screen.show([E("p", "AXPopUpButton", "Size", value=kw["menu_item"])])
        return {"effect": "confirmed", "menu": "closed"}

    screen.handlers["click"] = click
    out = session.act("click", _ref(obs, "Size"), menu_item="M")
    assert (
        out["receipt"]["effect"] == "confirmed" and out["receipt"]["menu"] == "closed"
    )


# -- a shifted live page -------------------------------------------------------


def test_shifted_page_retries_once_on_a_fresh_snapshot(session, screen):
    screen.show([E("a", "AXLink", "Deal 1"), E("b", "AXButton", "Buy it")])
    obs = session.observe("Chrome", "cg:1")
    target = _ref(obs, "Buy it")
    screen.show([E("z", "AXLink", "Deal 2"), E("b", "AXButton", "Buy it")])
    attempts = []

    def click(app, **kw):
        attempts.append(kw["expected_snapshot"]["snapshot_id"])
        if len(attempts) == 1:
            raise ComputerUseError("element_not_found", "index moved")
        return {"effect": "confirmed"}

    screen.handlers["click"] = click
    out = session.act("click", target)
    assert out["receipt"]["effect"] == "confirmed"
    assert len(attempts) == 2 and attempts[0] != attempts[1]


def test_shifted_page_without_the_target_is_refused(session, screen):
    screen.show([E("b", "AXButton", "Buy it")])
    obs = session.observe("Chrome", "cg:1")
    screen.show([E("b", "AXButton", "Sold out")])
    screen.handlers["click"] = lambda app, **kw: (_ for _ in ()).throw(
        ComputerUseError("stale_observation", "moved")
    )
    out = session.act("click", _ref(obs, "Buy it"))
    assert out["receipt"]["effect"] == "refused"
    assert len([c for c in screen.calls if c[0] == "click"]) == 1


def test_other_refusals_are_not_retried(session, screen):
    screen.show([E("b", "AXButton", "Go")])
    obs = session.observe("Chrome", "cg:1")
    screen.handlers["click"] = lambda app, **kw: (_ for _ in ()).throw(
        ComputerUseError("target_drift", "x")
    )
    assert session.act("click", _ref(obs, "Go"))["receipt"]["effect"] == "refused"
    assert len(screen.calls) == 1


# -- guards: secrets and card numbers --------------------------------------------


def test_secret_fields_are_handed_to_the_user(session, screen):
    screen.show(
        [
            E(
                "pw",
                "AXTextField",
                "[secure text redacted]",
                subrole="AXSecureTextField",
            ),
            E("otp", "AXTextField", "One-time code"),
            E("name", "AXTextField", "Name"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    for label in ("[secure text redacted]", "One-time code"):
        for op in ("fill", "type"):
            with pytest.raises(ComputerUseError) as err:
                session.act(op, _ref(obs, label), text="123456")
            assert err.value.code == "needs_human" and "handoff" in err.value.message
    assert not screen.calls
    assert session.act("fill", _ref(obs, "Name"), text="Ada")["receipt"][
        "action"
    ].startswith("fill")


def test_typing_without_a_ref_is_guarded_by_the_focused_field(
    session, screen, monkeypatch
):
    monkeypatch.setattr(perception, "_focused_secret", lambda app_info: None)
    screen.show(
        [
            E("pw", "AXTextField", "Password", states=("focused",)),
            E("x", "AXLink", "Help"),
        ]
    )
    session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("type", None, text="hunter2", window_id="cg:1")
    assert err.value.code == "needs_human"
    screen.show(
        [
            E("pw", "AXTextField", "Password"),
            E("n", "AXTextField", "Note", states=("focused",)),
        ]
    )
    session.observe("Chrome", "cg:1")
    session.act("type", None, text="hello", window_id="cg:1")
    assert [c[0] for c in screen.calls] == ["type_text"]
    # Focus that moved after the observation is read live.
    monkeypatch.setattr(
        perception, "_focused_secret", lambda app_info: "a password field"
    )
    session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("type", None, text="hunter2", window_id="cg:1")
    assert err.value.code == "needs_human" and "focused field" in err.value.message
    # Naming a harmless ref does not help: typed text goes to the focus.
    monkeypatch.setattr(perception, "_focused_secret", lambda app_info: None)
    screen.show(
        [
            E("pw", "AXTextField", "Password", states=("focused",)),
            E("n", "AXTextField", "Note"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("type", _ref(obs, "Note"), text="hunter2")
    assert err.value.code == "needs_human"
    monkeypatch.setattr(
        perception, "_focused_secret", lambda app_info: "a password field"
    )
    screen.show([E("pw", "AXTextField", "Password"), E("n", "AXTextField", "Note")])
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("type", _ref(obs, "Note"), text="hunter2")
    assert err.value.code == "needs_human"
    assert [c[0] for c in screen.calls] == ["type_text"]


def test_focused_secret_reads_names_never_values(monkeypatch):
    attrs = {
        "AXRole": "AXTextField",
        "AXSubrole": "AXSecureTextField",
        "AXValue": "hunter2",
    }
    read = []

    def get(element, attribute):
        read.append(attribute)
        return attrs.get(attribute)

    from rapid_mlx.computer_use import ax_driver

    focus = {"readable": True, "element": "focused"}
    failing: set[str] = set()

    def get_checked(element, attribute):
        if element == "app":
            return focus["readable"], focus["element"] if focus["readable"] else None
        read.append(attribute)
        if attribute in failing:
            return False, None
        return True, attrs.get(attribute)

    monkeypatch.setattr(ax_driver, "_get_checked", get_checked)
    monkeypatch.setattr(perception.backend, "_pid_app_element", lambda info: "app")
    unchecked = "a field whose focus could not be checked"
    assert perception._focused_secret({"pid": 7}) == "a password field"
    attrs.update(AXSubrole="", AXDescription="", AXTitle="Verification code")
    assert perception._focused_secret({"pid": 7}).startswith("a secret field")
    attrs.update(AXTitle="Search")
    assert perception._focused_secret({"pid": 7}) is None
    assert "AXValue" not in read
    assert perception._focused_secret({}) is None
    # Nothing has focus: typing goes nowhere.
    focus["element"] = None
    assert perception._focused_secret({"pid": 7}) is None
    # Focus, or a name of the focused element, that cannot be read is not
    # assumed harmless.
    focus.update(readable=False, element="focused")
    assert perception._focused_secret({"pid": 7}) == unchecked
    focus["readable"] = True
    failing.add("AXTitle")
    assert perception._focused_secret({"pid": 7}) == unchecked
    monkeypatch.setattr(
        perception.backend,
        "_pid_app_element",
        lambda info: (_ for _ in ()).throw(RuntimeError("no AX")),
    )
    assert perception._focused_secret({"pid": 7}) == unchecked


def test_secrets_cannot_be_spelled_key_by_key(session, screen, monkeypatch):
    screen.show(
        [
            E("pin", "AXTextField", "PIN", states=("focused",)),
            E("q", "AXTextField", "Search"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    for key in ("4", "shift+a", "A", "option+a", "alt+shift+2"):
        with pytest.raises(ComputerUseError) as err:
            session.act("key", None, key=key, window_id="cg:1")
        assert err.value.code == "needs_human"
    with pytest.raises(ComputerUseError) as err:
        session.act("key", _ref(obs, "PIN"), key="7")
    assert err.value.code == "needs_human"
    assert not screen.calls
    session.act("key", None, key="Tab", window_id="cg:1")
    session.observe("Chrome", "cg:1")
    session.act("key", None, key="cmd+a", window_id="cg:1")
    session.observe("Chrome", "cg:1")
    session.act("key", _ref(obs, "Search"), key="backspace")
    # A chord is not typing: it goes through, on the hotkey route.
    assert [c[0] for c in screen.calls] == ["press_key", "hotkey", "press_key"]
    assert perception._key_text("shift+shift+1") == "1"
    assert perception._key_text(None) is None
    assert perception._key_text("+") == "+" and perception._key_text("shift++") == "+"
    assert (
        perception._key_text("cmd++") is None and perception._key_text("ctrl+a") is None
    )
    assert perception._split_key("cmd+shift+Return") == (
        frozenset({"cmd", "shift"}),
        "Return",
    )


def test_pasting_into_a_secret_field_is_the_users(session, screen):
    screen.show([E("pw", "AXTextField", "Password", states=("focused",))])
    session.observe("Chrome", "cg:1")
    for key in ("cmd+v", "shift+cmd+v", "cmd+option+shift+v"):
        with pytest.raises(ComputerUseError) as err:
            session.act("key", None, key=key, window_id="cg:1")
        assert err.value.code == "needs_human"
    assert not screen.calls


def test_values_the_user_typed_into_secret_fields_are_not_shown(session, screen):
    screen.show(
        [
            E("otp", "AXTextField", "Verification code", value="482913"),
            E("empty", "AXTextField", "PIN", value=""),
            E("q", "AXTextField", "Search", value="milk"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    text = obs.render()
    assert "482913" not in text and "482913" not in " ".join(obs.texts())
    assert "[entered by the user]" in text
    assert obs.by_ref()[_ref(obs, "PIN")].value == ""
    assert "'milk'" in text


def test_card_numbers_are_never_typed(session, screen):
    screen.show([E("n", "AXTextArea", "Message")])
    obs = session.observe("Chrome", "cg:1")
    for op in ("fill", "type"):
        with pytest.raises(ComputerUseError) as err:
            session.act(
                op, _ref(obs, "Message"), text="card 4242 4242 4242 4242 exp 12/30"
            )
        assert err.value.code == "sensitive_data"
    assert not screen.calls


def test_a_card_number_split_across_typing_is_still_refused(session, screen):
    screen.show([E("n", "AXTextArea", "Message", value="4242 4242 4242 ")])
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("type", _ref(obs, "Message"), text="4242")
    assert err.value.code == "sensitive_data"
    screen.show(
        [E("n", "AXTextArea", "Message", value="4242424242424", states=("focused",))]
    )
    session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("key", None, key="2", window_id="cg:1")
    assert err.value.code == "sensitive_data"
    # Naming a harmless ref does not hide the focused field the text lands in.
    screen.show(
        [
            E("n", "AXTextArea", "Message", value="4242424242424", states=("focused",)),
            E("s", "AXSearchField", "Search"),
        ]
    )
    other = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("type", _ref(other, "Search"), text="242")
    assert err.value.code == "sensitive_data"
    # A fill replaces the value, so only its own text counts.
    session.act("fill", _ref(obs, "Message"), text="4242")
    assert [c[0] for c in screen.calls] == ["set_value"]


def test_card_number_fields_are_the_users(session, screen):
    screen.show([E("cc", "AXTextField", "Card number")])
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("fill", _ref(obs, "Card number"), text="hello")
    assert err.value.code == "needs_human"


# -- guards: money commits ---------------------------------------------------


def _checkout(total="$48.14"):
    return [
        E("t1", "AXStaticText", "Order total"),
        E("t2", "AXStaticText", total),
        E("pay", "AXStaticText", "Visa ending 4242"),
        E("go", "AXButton", "Place order"),
    ]


def test_money_commit_needs_one_approval_of_this_exact_screen(session, screen):
    events = []
    session.on_event = lambda kind, payload: events.append((kind, payload))
    screen.show(_checkout())
    obs = session.observe("Chrome", "cg:1")
    place = _ref(obs, "Place order")
    with pytest.raises(ComputerUseError) as err:
        session.act("click", place)
    assert err.value.code == "needs_approval"
    assert (
        "approval a1" in err.value.message
        and "Order total: $48.14" in err.value.message
    )
    with pytest.raises(ComputerUseError):
        session.act("click", place)  # same question, same id
    pending = session.pending_approvals()
    assert list(pending) == ["a1"] and "key" not in pending["a1"]
    assert pending["a1"]["context"] == ["Order total: $48.14", "Visa ending 4242"]
    assert [kind for kind, _ in events] == ["approval_requested"]
    assert not screen.calls

    assert session.approve("a1") == {"approved": "a1", "label": "Place order"}
    screen.handlers["click"] = lambda app, **kw: {"effect": "confirmed"}
    assert session.act("click", place)["receipt"]["effect"] == "confirmed"
    assert ("approval_used", {"id": "a1", "label": "Place order"}) in events
    # Used once: the next press is a new question.
    with pytest.raises(ComputerUseError) as err:
        session.act("click", place)
    assert "approval a2" in err.value.message


def test_approval_of_a_partly_read_page_says_so(session, screen):
    screen.show(_checkout(), truncated=True)
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Place order"))
    assert "only partly read" in err.value.message
    assert "only partly read" in session.pending_approvals()["a1"]["context"][-1]


def test_approval_binds_to_every_amount_on_screen(session, screen):
    screen.show(_checkout("$48.14"))
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError):
        session.act("click", _ref(obs, "Place order"))
    session.approve("a1")
    screen.show(_checkout("$58.14"))
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Place order"))
    assert "approval a2" in err.value.message
    assert not screen.calls


def test_approval_binds_to_every_chosen_option(session, screen):
    def page(method):
        return [*_checkout(), E("m", "AXPopUpButton", "Pay with", value=method)]

    screen.show(page("Visa 4242"))
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError):
        session.act("click", _ref(obs, "Place order"))
    session.approve("a1")
    screen.show(page("Amex 1005"))  # same total, another payment method
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Place order"))
    assert "approval a2" in err.value.message
    assert not screen.calls


def test_approval_binds_to_what_the_text_fields_hold(session, screen):
    def page(to):
        return [*_checkout(), E("to", "AXTextField", "Recipient", value=to)]

    screen.show(page("ada@example.com"))
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError):
        session.act("click", _ref(obs, "Place order"))
    session.approve("a1")
    screen.show(page("eve@example.com"))  # same total, another recipient
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Place order"))
    assert "approval a2" in err.value.message
    assert not screen.calls


def test_refused_commit_click_keeps_the_approval(session, screen):
    events = []
    session.on_event = lambda kind, payload: events.append(kind)
    screen.show(_checkout())
    obs = session.observe("Chrome", "cg:1")
    place = _ref(obs, "Place order")
    with pytest.raises(ComputerUseError):
        session.act("click", place)
    session.approve("a1")
    screen.handlers["click"] = lambda app, **kw: (_ for _ in ()).throw(
        ComputerUseError("target_occluded", "x")
    )
    assert session.act("click", place)["receipt"]["effect"] == "refused"
    assert "approval_kept" in events
    screen.handlers["click"] = lambda app, **kw: {"effect": "confirmed"}
    assert session.act("click", place)["receipt"]["effect"] == "confirmed"


def test_shifted_retry_of_a_commit_needs_the_same_amounts(session, screen):
    screen.show(_checkout("$48.14"))
    obs = session.observe("Chrome", "cg:1")
    place = _ref(obs, "Place order")
    with pytest.raises(ComputerUseError):
        session.act("click", place)
    session.approve("a1")
    screen.show(_checkout("$99.00"))  # the page moved and the total changed
    clicks = []

    def click(app, **kw):
        clicks.append(kw)
        raise ComputerUseError("element_not_found", "moved")

    screen.handlers["click"] = click
    assert session.act("click", place)["receipt"]["effect"] == "refused"
    assert len(clicks) == 1  # not retried on a screen the user did not approve
    assert "a1" in session._approved  # and the approval still stands for $48.14


def test_commit_by_key_or_action_needs_approval_too(session, screen):
    screen.show(
        [*_checkout()[:3], E("go", "AXButton", "Place order", states=("focused",))]
    )
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("key", None, key="Return", window_id="cg:1")
    assert err.value.code == "needs_approval"
    with pytest.raises(ComputerUseError) as err:
        session.act("key", _ref(obs, "Place order"), key="space")
    assert err.value.code == "needs_approval"
    with pytest.raises(ComputerUseError) as err:
        session.act("action", _ref(obs, "Place order"), name="AXPress")
    assert err.value.code == "needs_approval"
    # A harmless ref does not hide the focused commit the key would press.
    harmless = next(r for r in obs.rows if r.label != "Place order")
    with pytest.raises(ComputerUseError) as err:
        session.act("key", harmless.ref, key="Return")
    assert err.value.code == "needs_approval" and "Place order" in str(err.value)
    # A modified Return or Space still presses the focused control.
    for chord in ("cmd+Return", "shift+return", "ctrl+space"):
        with pytest.raises(ComputerUseError) as err:
            session.act("key", None, key=chord, window_id="cg:1")
        assert err.value.code == "needs_approval"
    assert not screen.calls
    session.act("key", None, key="Tab", window_id="cg:1")
    assert [c[0] for c in screen.calls] == ["press_key"]


def test_deny_and_unknown_approvals(session, screen):
    events = []
    session.on_event = lambda kind, payload: events.append(kind)
    screen.show(_checkout())
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError):
        session.act("click", _ref(obs, "Place order"))
    assert session.deny("a1") == {"denied": "a1", "label": "Place order"}
    assert "denied" in events and session.pending_approvals() == {}
    for call in (session.approve, session.deny):
        with pytest.raises(ComputerUseError) as err:
            call("a1")
        assert err.value.code == "invalid_argument"


def test_event_hook_failures_never_break_a_task(session, screen):
    session.on_event = lambda kind, payload: 1 / 0
    screen.show(_checkout())
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Place order"))
    assert err.value.code == "needs_approval"


# -- the user's channel and handoff ------------------------------------------


def test_human_input_bypasses_the_agent_guards(session, screen):
    events = []
    session.on_event = lambda kind, payload: events.append((kind, payload))
    screen.show([*_checkout(), E("pw", "AXTextField", "Password")])
    obs = session.observe("Chrome", "cg:1")
    session.human_act("fill", _ref(obs, "Password"), text="hunter2")
    session.human_act("click", _ref(obs, "Place order"))
    assert [c[0] for c in screen.calls] == ["set_value", "click"]
    assert ("human_input", {"op": "fill", "label": "Password"}) in events
    assert session.pending_approvals() == {}


def test_handoff_blocks_the_agent_until_the_user_is_done(session, screen, monkeypatch):
    fronted, notified, events, refusals = [], [], [], []
    monkeypatch.setattr(
        session, "take_front", lambda app, wid: fronted.append((app, wid))
    )
    monkeypatch.setattr(
        perception, "_notify", lambda title, msg: notified.append((title, msg))
    )
    screen.show([E("pw", "AXTextField", "Password"), E("go", "AXButton", "Sign in")])
    obs = session.observe("Chrome", "cg:1")

    def on_event(kind, payload):
        events.append(kind)
        if kind == "handoff":
            try:
                session.act("click", _ref(obs, "Sign in"))
            except ComputerUseError as exc:
                refusals.append(exc.code)
            session.human_act("fill", _ref(obs, "Password"), text="hunter2")
            session.human_done(1)  # the CG number names the same window

    session.on_event = on_event
    out = session.handoff("cg:1", "sign in to your account", timeout=5)
    assert out["met"] is True and out["front"] is False
    assert fronted == [("Chrome", "cg:1")]
    assert notified == [("Your turn", "sign in to your account (Shop)")]
    assert refusals == ["with_human"]
    assert events[0] == "handoff" and events[-1] == "handoff_done"
    assert session._with_human == {}
    screen.handlers["click"] = lambda app, **kw: {"effect": "confirmed"}
    assert (
        session.act("click", _ref(obs, "Sign in"))["receipt"]["effect"] == "confirmed"
    )


def test_handoff_ends_on_its_page_condition_or_times_out(session, screen, monkeypatch):
    monkeypatch.setattr(session, "take_front", lambda app, wid: None)
    monkeypatch.setattr(perception, "_notify", lambda title, msg: None)
    screen.show([E("m", "AXStaticText", "Enter the code we sent")])
    session.observe("Chrome", "cg:1")
    assert (
        session.handoff("cg:1", "code", until_gone="the code we sent", timeout=0)["met"]
        is False
    )
    screen.show([E("m", "AXStaticText", "Welcome back")])
    assert (
        session.handoff("cg:1", "code", until_gone="the code we sent", timeout=5)["met"]
        is True
    )


# -- waiting -----------------------------------------------------------------


def test_wait_until_text_and_until_gone(session, screen):
    screen.show([E("s", "AXStaticText", "Processing payment")])
    session.observe("Chrome", "cg:1")
    assert session.wait("cg:1", until_gone="processing", timeout=0)["met"] is False
    start = session._latest["cg:1"]
    screen.show(
        [E("s", "AXStaticText", "Payment received"), E("n", "AXStaticText", "Ref 42")]
    )
    out = session.wait("cg:1", until_text="RECEIVED", timeout=1)
    assert out["met"] is True
    obs = out["observation"]
    assert obs.previous == start.obs_id  # changes are relative to where the wait began
    assert obs.change_counts == (1, 0, 1)
    assert [row.new for row in obs.rows] == [False, True]
    assert 'label "Processing payment" -> "Payment received"' in obs.changes[0]
    assert session.wait("cg:1", until_gone="processing", timeout=1)["met"] is True


def test_wait_holds_while_the_other_side_is_typing(session, screen, monkeypatch):
    monkeypatch.setattr(perception, "WAIT_QUIET_S", 0)
    screen.show([E("q", "AXStaticText", "Hi")])
    session.observe("Chrome", "cg:1")
    screen.show(
        [E("q", "AXStaticText", "Hi"), E("t", "AXStaticText", "Agent is typing…")]
    )
    assert session.wait("cg:1", timeout=0.05)["met"] is False
    screen.show([E("q", "AXStaticText", "Hi"), E("r", "AXStaticText", "Refund issued")])
    assert session.wait("cg:1", timeout=1)["met"] is True


def test_wait_without_any_change_times_out(session, screen):
    screen.show([E("q", "AXStaticText", "Hi")])
    session.observe("Chrome", "cg:1")
    assert session.wait("cg:1", timeout=0)["met"] is False


# -- open_url ------------------------------------------------------------------


def test_open_url_types_into_the_address_bar_and_drops_autocomplete(session, screen):
    screen.show(
        [E("bar", "AXTextField", "Address and search bar"), E("b", "AXButton", "Back")],
        title="New Tab",
    )
    session.observe("Chrome", "cg:1")

    def press_key(app, key, **kw):
        if key == "Return":
            screen.show(
                [
                    E(
                        "bar",
                        "AXTextField",
                        "Address and search bar",
                        value="mart.test",
                    ),
                    E("h", "AXHeading", "Mart"),
                ],
                title="Mart",
            )
        return {
            "effect": "unverifiable"
        }  # a press with no visible change must not gate the next step

    screen.handlers["press_key"] = press_key
    out = session.open_url("cg:1", "mart.test")
    assert [c[0] for c in screen.calls] == [
        "click",
        "set_value",
        "press_key",
        "press_key",
    ]
    assert [c[1]["args"][1] for c in screen.calls if c[0] == "press_key"] == [
        "forwarddelete",
        "Return",
    ]
    assert screen.calls[1][1]["args"][2] == "mart.test"
    receipt = out["receipt"]
    assert receipt["effect"] == "confirmed" and receipt["title"] == "Mart"
    assert receipt["action"] == "open_url mart.test"
    assert "cg:1" not in session._unresolved


def test_open_url_without_a_load_is_unverifiable_and_needs_a_bar(
    session, screen, monkeypatch
):
    monkeypatch.setattr(perception, "OPEN_URL_WAIT_S", 0)
    screen.show([E("bar", "AXComboBox", "Search or enter address")])
    session.observe("Chrome", "cg:1")
    receipt = session.open_url("cg:1", "x.test")["receipt"]
    assert receipt["effect"] == "unverifiable" and "unresolved" in receipt
    with pytest.raises(ComputerUseError) as err:
        session.act("key", None, key="Tab", window_id="cg:1")
    assert err.value.code == "unresolved_outcome"
    screen.show([E("b", "AXButton", "Back")])
    session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.open_url("cg:1", "x.test")
    assert err.value.code == "invalid_argument"


def test_open_url_polls_until_the_page_loads(session, screen, monkeypatch):
    screen.show([E("bar", "AXTextField", "Address and search bar")], title="New Tab")
    session.observe("Chrome", "cg:1")
    # The page shows only after open_url's load loop has slept once.
    monkeypatch.setattr(perception, "WAIT_POLL_S", 0.0001)
    slept = []
    real_time = perception.time

    def sleep(seconds):
        if seconds == perception.WAIT_POLL_S:
            slept.append(seconds)
            screen.show(
                [
                    E("bar", "AXTextField", "Address and search bar"),
                    E("h", "AXHeading", "Mart"),
                ],
                title="Mart",
            )

    monkeypatch.setattr(
        perception,
        "time",
        types.SimpleNamespace(
            sleep=sleep,
            monotonic=real_time.monotonic,
            perf_counter=real_time.perf_counter,
        ),
    )
    receipt = session.open_url("cg:1", "mart.test")["receipt"]
    assert receipt["effect"] == "confirmed" and receipt["title"] == "Mart"
    assert slept


def test_open_url_stops_when_the_bar_refuses_the_address(session, screen):
    screen.show([E("bar", "AXTextField", "Address and search bar")])
    session.observe("Chrome", "cg:1")
    screen.handlers["set_value"] = lambda *a, **k: (_ for _ in ()).throw(
        ComputerUseError("target_drift", "focus moved")
    )
    with pytest.raises(ComputerUseError) as err:
        session.open_url("cg:1", "x.test")
    assert (
        err.value.code == "target_drift"
        and "open_url stopped at fill" in err.value.message
    )
    assert [c[0] for c in screen.calls] == ["click", "set_value"]  # no Return


def test_open_url_will_not_type_a_card_number(session, screen):
    screen.show([E("bar", "AXTextField", "Address and search bar")])
    session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.open_url("cg:1", "x.test/?c=4242424242424242")
    assert err.value.code == "sensitive_data"


# -- windows -------------------------------------------------------------------


def test_window_resolution_never_swaps_a_named_window(session, screen, monkeypatch):
    with pytest.raises(ComputerUseError) as err:
        session._window_obs(None)
    assert err.value.code == "invalid_argument"
    screen.show([E("a", "AXLink", "A")])
    obs = session.observe("Chrome", "cg:1")
    assert session._window_obs(None) is obs
    assert session._window_obs(1) is obs and session._window_obs("cg:1") is obs
    monkeypatch.setattr(perception, "_window_owner", lambda wid: None)
    with pytest.raises(ComputerUseError) as err:
        session._window_obs("cg:5")
    assert err.value.code == "window_not_found"
    with pytest.raises(ComputerUseError):
        session._window_obs("not-a-window")
    # The owner's observation landed on another window: still not found.
    monkeypatch.setattr(perception, "_window_owner", lambda wid: "pid:7")
    screen.show([E("b", "AXLink", "B")], wid="cg:2")
    real = screen.get_app_state
    monkeypatch.setattr(
        perception.backend,
        "get_app_state",
        lambda app, **kw: real(app, window_id="cg:2"),
    )
    with pytest.raises(ComputerUseError) as err:
        session._window_obs("cg:5")
    assert err.value.code == "window_not_found"
    with pytest.raises(ComputerUseError) as err:
        session._window_obs(None)  # two windows observed now
    assert err.value.code == "invalid_argument"


def test_known_window_skips_observed_ids_that_are_not_window_ids(session, screen):
    screen.show([E("a", "AXLink", "A")])
    obs = session.observe("Chrome", "cg:1")
    session._latest["not-a-window"] = obs
    assert session._known_window(1) == "cg:1"
    assert session._known_window("cg:9") is None


def test_shows_is_false_for_a_ref_the_baseline_lacks(session, screen):
    screen.show([E("a", "AXTextField", "Qty", value="2")])
    obs = session.observe("Chrome", "cg:1")
    snapshot = screen.get_app_state("Chrome", window_id="cg:1")
    assert perception._shows(obs, snapshot, (_ref(obs, "Qty"), "2"))
    assert not perception._shows(obs, snapshot, ("e999", "2"))


def test_window_owner_reads_the_window_server(monkeypatch):
    quartz = types.ModuleType("Quartz")
    quartz.kCGNullWindowID = 0
    quartz.kCGWindowListOptionAll = 0
    quartz.CGWindowListCopyWindowInfo = lambda *a: [
        {"kCGWindowNumber": 3, "kCGWindowOwnerPID": 11},
        {"kCGWindowNumber": 9, "kCGWindowOwnerPID": 42},
    ]
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    assert perception._window_owner("cg:9") == "pid:42"
    assert perception._window_owner(4) is None
    assert perception._window_owner("junk") is None


# -- front, hand back, keep awake ------------------------------------------------


class _Popen:
    started: list = []

    def __init__(self, argv, **kw):
        self.argv = argv
        self.alive = True
        _Popen.started.append(self)

    def poll(self):
        return None if self.alive else 0

    stubborn = False

    def terminate(self):
        self.alive = self.stubborn

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        if self.alive:
            raise perception.subprocess.TimeoutExpired("caffeinate", timeout)
        self.reaped = True
        return 0


def test_release_awake_reaps_and_kills_a_stubborn_caffeinate(monkeypatch):
    _Popen.started = []
    monkeypatch.setattr(perception.subprocess, "Popen", _Popen)
    s = perception.PerceptionSession()
    s.release_awake()  # nothing held
    s.keep_awake()
    s.release_awake()
    assert _Popen.started[0].reaped
    s.keep_awake()
    _Popen.started[1].stubborn = True
    s.release_awake()
    assert _Popen.started[1].alive is False and _Popen.started[1].reaped
    assert s._awake is None


def test_keep_awake_holds_one_assertion_until_released(monkeypatch):
    _Popen.started = []
    monkeypatch.setattr(perception.subprocess, "Popen", _Popen)
    s = perception.PerceptionSession()
    s.keep_awake()
    s.keep_awake()
    assert len(_Popen.started) == 1
    argv = _Popen.started[0].argv
    assert argv[:4] == ["/usr/bin/caffeinate", "-d", "-i", "-w"]
    s.release_awake()
    assert _Popen.started[0].alive is False and s._awake is None
    s.keep_awake()
    assert len(_Popen.started) == 2

    def missing(*a, **k):
        raise FileNotFoundError("caffeinate")

    monkeypatch.setattr(perception.subprocess, "Popen", missing)
    s.release_awake()
    s.keep_awake()
    assert s._awake is None


def test_observe_and_act_hold_the_screen_on(monkeypatch):
    _Popen.started = []
    monkeypatch.setattr(perception.subprocess, "Popen", _Popen)
    monkeypatch.setattr(perception, "_require_display_awake", lambda: None)
    s = perception.PerceptionSession()
    Screen(monkeypatch).show([E("a", "AXLink", "Home")])
    s.observe("Chrome", "cg:1")
    assert len(_Popen.started) == 1 and _Popen.started[0].alive


def test_a_sleeping_display_is_reported(monkeypatch):
    quartz = types.ModuleType("Quartz")
    quartz.CGMainDisplayID = lambda: 1
    quartz.CGDisplayIsAsleep = lambda display: True
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    with pytest.raises(ComputerUseError) as err:
        perception._require_display_awake()
    assert err.value.code == "display_asleep"
    quartz.CGDisplayIsAsleep = lambda display: False
    perception._require_display_awake()


def test_actions_refuse_while_the_display_sleeps(session, screen, monkeypatch):
    screen.show([E("a", "AXLink", "Home")])
    obs = session.observe("Chrome", "cg:1")

    def asleep():
        raise ComputerUseError("display_asleep", "off")

    monkeypatch.setattr(perception, "_require_display_awake", asleep)
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Home"))
    assert err.value.code == "display_asleep" and not screen.calls


class _AX:
    def __init__(self, windows, minimized=False):
        self.windows = windows
        self.minimized = minimized
        self.sets = []
        self.actions = []
        self.discovered = []


@pytest.fixture
def ax(monkeypatch):
    from rapid_mlx.computer_use import ax_driver, background_input

    state = _AX(windows=["w9"])
    monkeypatch.setattr(
        perception.backend,
        "_resolve_app",
        lambda app, activate=False: ("app", {"pid": 42}),
    )
    monkeypatch.setattr(ax_driver, "_app_windows", lambda app: list(state.windows))
    monkeypatch.setattr(background_input, "ax_window_id", lambda w: int(w[1:]))
    monkeypatch.setattr(
        ax_driver, "_get", lambda w, a: state.minimized if a == "AXMinimized" else None
    )
    monkeypatch.setattr(
        ax_driver,
        "AXUIElementSetAttributeValue",
        lambda w, a, v: state.sets.append((w, a, v)),
    )
    monkeypatch.setattr(
        ax_driver, "AXUIElementPerformAction", lambda w, a: state.actions.append((w, a))
    )
    monkeypatch.setattr(
        ax_driver,
        "discover_remote_windows",
        lambda pid, ids: state.discovered.append((pid, ids)),
    )
    monkeypatch.setattr(ax_driver, "window_is_onscreen", lambda wid: True)
    front = iter([11, 42, 42, 42])
    monkeypatch.setattr(perception, "_frontmost_pid", lambda: next(front, 42))
    return state


def test_take_front_raises_the_exact_window_and_hand_back_restores(ax, monkeypatch):
    ax.minimized = True
    activated = []
    monkeypatch.setattr(
        perception.backend, "_activate_app", lambda pid: activated.append(pid) or True
    )
    s = perception.PerceptionSession()
    released = []
    monkeypatch.setattr(s, "release_awake", lambda: released.append(True))
    assert s.take_front("Chrome", "cg:9") == {"window_id": "cg:9", "front": True}
    assert ("w9", "AXMinimized", False) in ax.sets and ("w9", "AXMain", True) in ax.sets
    assert ax.actions == [("w9", "AXRaise")]
    s.take_front("Chrome", 9)  # a second take keeps the user's original app
    assert s.hand_back() == {"restored_pid": 11, "restored": True}
    # Each take activates the task's app (42); hand_back the user's (11).
    assert activated == [42, 42, 11] and released == [True]
    assert s.hand_back() == {"restored_pid": None, "restored": False}


def test_take_front_finds_a_window_on_another_space_or_refuses(ax, monkeypatch):
    ax.windows = []

    def discover(pid, ids):
        ax.discovered.append((pid, ids))
        ax.windows = ["w9"]

    from rapid_mlx.computer_use import ax_driver

    monkeypatch.setattr(ax_driver, "discover_remote_windows", discover)
    s = perception.PerceptionSession()
    assert s.take_front("Chrome", "cg:9")["front"] is True
    assert ax.discovered == [(42, {9})]
    ax.windows = ["w3"]
    monkeypatch.setattr(ax_driver, "discover_remote_windows", lambda pid, ids: None)
    with pytest.raises(ComputerUseError) as err:
        s.take_front("Chrome", "cg:9")
    assert err.value.code == "window_not_found"


def test_take_front_reports_a_window_that_did_not_come_forward(ax, monkeypatch):
    from rapid_mlx.computer_use import ax_driver

    monkeypatch.setattr(ax_driver, "window_is_onscreen", lambda wid: False)
    clock = iter(range(0, 100))
    monkeypatch.setattr(perception.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(perception.time, "sleep", lambda s: None)
    s = perception.PerceptionSession()
    assert s.take_front("Chrome", "cg:9") == {"window_id": "cg:9", "front": False}


def test_frontmost_app_comes_from_the_window_server(monkeypatch):
    from rapid_mlx.computer_use import background_input

    monkeypatch.setattr(background_input, "front_pid", lambda: 77)
    assert perception._frontmost_pid() == 77
    appkit = types.ModuleType("AppKit")
    app = types.SimpleNamespace(bundleIdentifier=lambda: "com.test.app")
    appkit.NSRunningApplication = types.SimpleNamespace(
        runningApplicationWithProcessIdentifier_=lambda pid: app if pid == 77 else None
    )
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    assert perception._frontmost_bundle() == "com.test.app"
    monkeypatch.setattr(background_input, "front_pid", lambda: None)
    appkit.NSWorkspace = types.SimpleNamespace(
        sharedWorkspace=lambda: types.SimpleNamespace(frontmostApplication=lambda: None)
    )
    assert perception._frontmost_pid() is None
    assert perception._frontmost_bundle() is None


def test_notify_quotes_the_message_for_applescript(monkeypatch):
    calls = []
    monkeypatch.setattr(
        perception.subprocess, "Popen", lambda argv, **kw: calls.append(argv)
    )
    perception._notify("Your turn", 'type the "code"\\now')
    script = calls[0][2]
    assert calls[0][:2] == ["/usr/bin/osascript", "-e"]
    assert (
        'display notification "type the \\"code\\"\\\\now" with title "Your turn"'
        in script
    )


# -- dispatch routes -----------------------------------------------------------


def test_dispatch_routes_each_op_to_the_backend(session, screen):
    row = perception.Row("e1", "AXScrollArea", "List", None, (), 5, (40, 60))
    session._dispatch(
        "scroll", "Chrome", {"s": 1}, "cg:1", row, {"direction": "up", "pages": 2}
    )
    session._dispatch("scroll", "Chrome", {"s": 1}, "cg:1", None, {})
    session._dispatch("action", "Chrome", {"s": 2}, "cg:1", row, {"name": "AXShowMenu"})
    session._dispatch("type", "Chrome", {}, "cg:1", None, {"text": "hi"})
    session._dispatch(
        "click",
        "Chrome",
        {},
        "cg:1",
        row,
        {"count": 2, "button": "right", "modifiers": ["cmd"]},
    )
    names = [c[0] for c in screen.calls]
    assert names == [
        "scroll",
        "scroll",
        "perform_secondary_action",
        "type_text",
        "click",
    ]
    assert screen.calls[0][1]["x"] == 40 and screen.calls[0][1]["pages"] == 2.0
    assert screen.calls[1][1]["x"] is None
    # An action names the element of the observation it was taken from.
    assert screen.calls[2][1]["expected_snapshot"] == {"s": 2}
    assert screen.calls[4][1]["click_count"] == 2 and screen.calls[4][1][
        "modifiers"
    ] == ["cmd"]
    for op in ("dance", "fill", "action"):
        with pytest.raises(ComputerUseError) as err:
            session._dispatch(
                op, "Chrome", {}, "cg:1", None, {"text": "x", "name": "AXPress"}
            )
        assert err.value.code == "invalid_argument"


# -- live-regression fixes: chords, secure fields, waits, closed windows ------


def _raise(code, message="x"):
    def fail(*args, **kwargs):
        raise ComputerUseError(code, message)

    return fail


def test_key_chords_take_the_hotkey_route_and_keep_every_guard(session, screen):
    screen.show(
        [
            E("n", "AXTextField", "Note", states=("focused",)),
            E("b", "AXButton", "Help"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    session.act("key", None, key="cmd+w", window_id="cg:1")
    assert screen.calls[-1][0] == "hotkey"
    assert screen.calls[-1][1]["args"] == ("Chrome", "cmd+w")
    assert screen.calls[-1][1]["window_id"] == "cg:1"
    # A chord named at a text field focuses it first; at anything else it is
    # refused (it would land on whatever has focus).
    session.observe("Chrome", "cg:1")
    session.act("key", _ref(obs, "Note"), key="cmd+a")
    assert [c[0] for c in screen.calls[-2:]] == ["click", "hotkey"]
    session.observe("Chrome", "cg:1")
    receipt = session.act("key", _ref(obs, "Help"), key="cmd+a")["receipt"]
    assert receipt["effect"] == "refused"
    assert receipt["error"]["code"] == "invalid_argument"
    # The backend's fail-closed refusal of a menu command stays a refusal.
    screen.handlers["hotkey"] = _raise("synthetic_input_blocked", "menu command")
    receipt = session.act("key", None, key="cmd+q", window_id="cg:1")["receipt"]
    assert receipt["error"]["code"] == "synthetic_input_blocked"
    # One key, "+" itself included, stays on the single-key route.
    session.act("key", None, key="+", window_id="cg:1")
    assert screen.calls[-1][0] == "press_key"
    assert perception._is_combo("Shift+Return") and not perception._is_combo("+")
    assert perception._activating("cmd+Return") and perception._activating(" ")


def test_chords_into_secret_fields_and_onto_commit_buttons_are_guarded(session, screen):
    screen.show(
        [
            E(
                "pw",
                "AXTextField",
                "[secure text redacted]",
                subrole="AXSecureTextField",
                states=("focused",),
                field_name="Password",
            )
        ]
    )
    session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("key", None, key="cmd+v", window_id="cg:1")
    assert err.value.code == "needs_human"
    screen.show([E("p", "AXButton", "Place order", states=("focused",))])
    session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("key", None, key="shift+Return", window_id="cg:1")
    assert err.value.code == "needs_approval"
    assert not screen.calls


def _secure(key, **extra):
    return E(
        key,
        "AXTextField",
        "[secure text redacted]",
        subrole="AXSecureTextField",
        **extra,
    )


def test_secure_fields_keep_their_name_and_say_only_whether_they_are_filled(
    session, screen
):
    screen.show(
        [
            _secure("pw", field_name="Password", filled=True),
            _secure("pin", field_name="PIN", filled=False),
            _secure("x"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    text = obs.render()
    password, pin = _ref(obs, "Password"), _ref(obs, "PIN")
    assert f"{password} securetextfield \"Password\" = '[entered by the user]'" in text
    assert f"{pin} securetextfield \"PIN\" = ''" in text
    # No name to show: it keeps the redaction marker, and no value at all.
    unnamed = obs.by_ref()[_ref(obs, "[secure text redacted]")]
    assert unnamed.value is None and unnamed.secure
    for label in ("Password", "PIN", "[secure text redacted]"):
        with pytest.raises(ComputerUseError) as err:
            session.act("fill", _ref(obs, label), text="hunter2")
        assert err.value.code == "needs_human"
    # Even under a harmless name a secure row is the user's.
    row = obs.by_ref()[password]
    row.subrole, row.label = "", "Notes"
    with pytest.raises(ComputerUseError) as err:
        session.act("fill", password, text="hunter2")
    assert err.value.code == "needs_human" and "password field" in err.value.message
    assert not screen.calls


def _recording_settle(session, monkeypatch):
    seen = []
    real = session._settle

    def settle(app, wid, **kw):
        seen.append(kw)
        return real(app, wid, **kw)

    monkeypatch.setattr(session, "_settle", settle)
    return seen


def test_secret_fills_and_refused_actions_do_not_wait_for_text(
    session, screen, monkeypatch
):
    seen = _recording_settle(session, monkeypatch)
    page = [
        E("otp", "AXTextField", "Verification code", value=""),
        E("p", "AXPopUpButton", "Account", value="Savings"),
    ]
    screen.show(page)
    obs = session.observe("Chrome", "cg:1")

    def set_value(app, index, text, **kw):
        screen.show([E("otp", "AXTextField", "Verification code", value=text), page[1]])
        return {"effect": "confirmed"}

    screen.handlers["set_value"] = set_value
    out = session.human_act("fill", _ref(obs, "Verification code"), text="397675")
    assert out["receipt"]["effect"] == "confirmed" and out["receipt"]["settled"]
    assert seen[-1]["want"] is None and seen[-1]["slow"] is False
    assert "397675" not in out["observation"].render()
    screen.handlers["click"] = _raise("element_not_found", "no enabled item 'Chk'")
    out = session.act("click", _ref(obs, "Account"), menu_item="Chk")
    assert out["receipt"]["effect"] == "refused"
    assert seen[-1]["want"] is None and seen[-1]["slow"] is False


def test_a_menu_choice_settles_on_the_item_the_backend_chose(
    session, screen, monkeypatch
):
    seen = _recording_settle(session, monkeypatch)
    screen.show([E("p", "AXPopUpButton", "Account", value="Savings ending 4242")])
    obs = session.observe("Chrome", "cg:1")
    account = _ref(obs, "Account")
    chosen = "Checking ending 6789 (no fee)"

    def click(app, **kw):
        screen.show([E("p", "AXPopUpButton", "Account", value=chosen)])
        return {"menu": {"closed": True, "chosen": chosen}}

    screen.handlers["click"] = click
    out = session.act("click", account, menu_item="checking")
    assert out["receipt"]["effect"] == "confirmed"
    assert seen[-1]["want"] == (account, chosen)
    assert out["receipt"]["target"] == f'{account} popupbutton "Account" = {chosen!r}'


def test_an_action_that_closes_its_window_returns_a_receipt(
    session, screen, monkeypatch
):
    screen.show([E("x", "AXButton", "Close"), E("t", "AXStaticText", "Page")])
    obs = session.observe("Chrome", "cg:1")
    real = screen.get_app_state

    def get_app_state(app, *, window_id=None, **kw):
        if "cg:1" not in screen.windows:
            raise ComputerUseError(
                "ax_unavailable",
                "selected CGWindow does not map to exactly one AX window",
            )
        return real(app, window_id=window_id, **kw)

    monkeypatch.setattr(perception.backend, "get_app_state", get_app_state)
    exists = iter([True, False])  # still closing, then gone
    monkeypatch.setattr(perception, "_window_exists", lambda wid: next(exists))
    screen.handlers["click"] = lambda app, **kw: screen.windows.pop("cg:1") and {}
    session._unresolved["cg:1"] = "stale"
    out = session.human_act("click", _ref(obs, "Close"))
    receipt = out["receipt"]
    assert receipt["effect"] == "window_closed" and receipt["window_closed"] is True
    assert "error" not in receipt and "cg:1" not in session._unresolved
    closed = out["observation"]
    assert closed.closed and closed.previous == obs.obs_id
    assert "closed (no elements; list windows to go on)" in closed.render()
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Close"))
    assert err.value.code == "stale_ref"


def test_a_refused_action_in_a_window_that_closed_stays_refused(
    session, screen, monkeypatch
):
    screen.show([E("x", "AXButton", "Close")])
    obs = session.observe("Chrome", "cg:1")
    monkeypatch.setattr(perception.backend, "get_app_state", _raise("ax_unavailable"))
    monkeypatch.setattr(perception, "_window_exists", lambda wid: False)
    screen.handlers["click"] = _raise("action_failed", "AXPress returned an error")
    receipt = session.act("click", _ref(obs, "Close"))["receipt"]
    assert receipt["effect"] == "refused" and receipt["window_closed"] is True
    assert receipt["error"]["code"] == "action_failed"


def test_an_observation_failure_with_the_window_still_there_is_raised(
    session, screen, monkeypatch
):
    screen.show([E("b", "AXButton", "Go")])
    obs = session.observe("Chrome", "cg:1")
    monkeypatch.setattr(perception.backend, "get_app_state", _raise("ax_unavailable"))
    clock = itertools.count()
    real_time = perception.time
    monkeypatch.setattr(
        perception,
        "time",
        types.SimpleNamespace(
            sleep=lambda s: None,
            monotonic=lambda: float(next(clock)),
            perf_counter=real_time.perf_counter,
        ),
    )
    for exists in (True, None):  # still there past the wait; cannot tell
        monkeypatch.setattr(perception, "_window_exists", lambda wid, e=exists: e)
        with pytest.raises(ComputerUseError) as err:
            session.act("click", _ref(obs, "Go"))
        assert err.value.code == "ax_unavailable"


def test_window_exists_reads_the_window_server(monkeypatch):
    quartz = types.ModuleType("Quartz")
    quartz.kCGNullWindowID = 0
    quartz.kCGWindowListOptionAll = 0
    listed = {"windows": [{"kCGWindowNumber": 9}]}
    quartz.CGWindowListCopyWindowInfo = lambda *a: listed["windows"]
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    assert perception._window_exists("cg:9") is True
    assert perception._window_exists("cg:4") is False
    assert perception._window_exists("junk") is None
    listed["windows"] = None
    assert perception._window_exists("cg:9") is None


def test_a_rerendered_page_keeps_its_refs_and_the_receipt_shows_the_new_state(
    session, screen
):
    def page(gen, state):
        return [
            E(f"{gen}h", "AXHeading", "Cart", path=[0, 0], web=True),
            E(f"{gen}c", "AXCheckBox", "Add membership", states=(state,), path=[0, 1]),
            E(f"{gen}a", "AXButton", "Remove", path=[0, 2, 0]),
            E(f"{gen}b", "AXButton", "Remove", path=[0, 3, 0]),
        ]

    screen.show(page("g1", "unchecked"))
    obs = session.observe("Chrome", "cg:1")
    box = _ref(obs, "Add membership")
    removes = {row.ref for row in obs.rows if row.label == "Remove"}
    # Toggling the box re-renders the list: every element is a new one.
    screen.handlers["click"] = lambda app, **kw: (
        screen.show(page("g2", "checked")) or {}
    )
    out = session.act("click", box)
    after = out["observation"]
    assert [row.ref for row in after.rows] == [row.ref for row in obs.rows]
    assert after.change_counts == (0, 0, 1)
    assert "[unchecked] -> [checked]" in after.changes[0]
    assert out["receipt"]["target"] == f'{box} checkbox "Add membership" [checked]'
    # Two rows with the same role, name and place are not guessed between.
    screen.show(
        [
            E("g3a", "AXButton", "Remove", path=[0, 2, 0]),
            E("g3b", "AXButton", "Remove", path=[0, 2, 0]),
        ]
    )
    again = session.observe("Chrome", "cg:1")
    assert not {row.ref for row in again.rows} & removes


def test_settled_observation_reports_its_walk_time(session, screen, monkeypatch):
    screen.show([E("b", "AXButton", "Go")])
    obs = session.observe("Chrome", "cg:1")
    real = screen.get_app_state

    def slow_walk(app, **kw):
        time.sleep(0.02)
        return real(app, **kw)

    monkeypatch.setattr(perception.backend, "get_app_state", slow_walk)
    after = session.act("click", _ref(obs, "Go"))["observation"]
    assert after.elapsed_ms >= 20 and " · 0 ms" not in after.render()


def test_open_url_steps_take_one_sample_each(session, screen, monkeypatch):
    seen = _recording_settle(session, monkeypatch)
    screen.show(
        [E("bar", "AXTextField", "Address and search bar"), E("b", "AXButton", "Back")],
        title="New Tab",
    )
    session.observe("Chrome", "cg:1")

    def set_value(app, index, text, **kw):
        # The omnibox shows the address reformatted, never the typed text.
        bar = E("bar", "AXTextField", "Address and search bar", value="mart.test/")
        screen.show([bar, E("b", "AXButton", "Back")], title="New Tab")
        return {"effect": "confirmed"}

    def press_key(app, key, **kw):
        if key == "Return":
            screen.show([E("h", "AXHeading", "Mart")], title="Mart")
        return {}

    screen.handlers["set_value"] = set_value
    screen.handlers["press_key"] = press_key
    receipt = session.open_url("cg:1", "http://mart.test")["receipt"]
    assert receipt["effect"] == "confirmed"
    assert [kw.get("cap") for kw in seen] == [0.0, 0.0, 0.0, 0.0, None]
    assert all(kw.get("want") is None for kw in seen)


# -- reading the page: diffs, chrome, dialogs, containers ---------------------


def test_diffs_skip_browser_noise_and_show_where_long_labels_differ(session, screen):
    long_old = "Organic Large Brown Eggs, 24 count, cage free, grade A — $7.49"
    long_new = "Organic Large Brown Eggs, 24 count, cage free, grade A — $8.99"
    screen.show(
        [
            E("tab", "AXRadioButton", "Shop", web=False),
            E("hover", "AXRadioButton", "Shop - Memory usage - 160 MB"),
            E("ext", "AXPopUpButton", "Grammarly has access to this site"),
            E("p", "AXStaticText", long_old, web=True),
            E("gone", "AXLink", "Deals"),
        ]
    )
    session.observe("Chrome", "cg:1")
    screen.show(
        [
            E("tab", "AXRadioButton", "Mart", web=False),
            E("hover", "AXRadioButton", "Shop - Memory usage - 171 MB"),
            E("ext", "AXPopUpButton", "Grammarly wants access to this site"),
            E("p", "AXStaticText", long_new, web=True),
            E("new", "AXLink", "Offers"),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    assert obs.change_counts == (1, 1, 1)
    joined = "\n".join(obs.changes)
    assert "Memory usage" not in joined and "Grammarly" not in joined
    assert "Mart" not in joined
    assert 'label "…rade A — $7.49" -> "…rade A — $8.99"' in joined
    # A row that was new when last seen is shown removed without the marker.
    screen.show([E("p", "AXStaticText", long_new, web=True)])
    gone = session.observe("Chrome", "cg:1")
    assert any(line.startswith("- e") and "Offers" in line for line in gone.changes)
    assert not any(line.startswith("- *") for line in gone.changes)
    # A native app's radio buttons and tabs are its own: their changes count.
    native = [
        E("r1", "AXRadioButton", "Light", states=("checked",), web=False),
        E("r2", "AXTab", "General", web=False),
        E("r3", "AXRadioButton", "Job - Memory usage - 1 GB"),
    ]
    screen.show(native, wid="cg:2")
    session.observe("Settings", "cg:2")
    native[0]["states"] = ["unchecked"]
    native[1]["label"] = "Advanced"
    native[2]["label"] = "Job - Memory usage - 2 GB"
    assert session.observe("Settings", "cg:2").change_counts == (0, 0, 3)
    assert perception._where_differ("abc", "abd") == ("abc", "abd")
    assert perception._where_differ("x" * 45, "x" * 46, 40) == (
        "…" + "x" * 10,
        "…" + "x" * 11,
    )


def test_render_puts_the_page_first_and_marks_browser_chrome_and_dialogs(
    session, screen
):
    def web(key, role, label, **kw):
        return E(key, role, label, web=True, **kw)

    screen.show(
        [
            E("back", "AXButton", "Back", web=False, path=[0, 0]),
            E("save", "AXButton", "Save your password", web=False, path=[0, 1]),
            web("dlg", "AXGroup", "Membership offer", subrole="AXApplicationDialog"),
            web("up", "AXButton", "Upgrade for $65"),
            web("pop", "AXPopUpButton", "Account", value="Savings"),
            web("hid", "AXButton", "Slide 2", width=0, height=0),
            web("txt", "AXStaticText", "Fine print"),
            web("foot", "AXLink", "Terms", width=0, height=0),
            E("tab", "AXRadioButton", "Shop", web=False, path=[2, 0]),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    head, body = obs.render().split("elements:")
    dialog = _ref(obs, "Membership offer")
    assert f'modal dialog open: {dialog} "Membership offer"; it covers' in head
    page, chrome = body.split("browser (outside the page):")
    assert '"Upgrade for $65"' in page and '"Back"' not in page
    assert '"Save your password"' in chrome and '"Shop"' in chrome
    assert "2 more elements off screen; 1 below, 1 hidden in place;" in page
    assert "popup buttons: click one to list its options" in page
    everything = obs.render(everything=True)
    assert '"Slide 2"' in everything.split("browser (outside")[0]
    # Without page rows nothing is split; rows above are named so.
    screen.show([E("a", "AXLink", "Top", width=0, height=0), E("b", "AXLink", "Here")])
    plain = session.observe("Chrome", "cg:1").render()
    assert "browser (outside" not in plain and "; 1 above;" in plain
    screen.show([E("a", "AXLink", "Top", width=0, height=0)])
    assert "; 1 below;" in session.observe("Chrome", "cg:1").render()
    # An unnamed dialog is still announced.
    screen.show([E("d", "AXSheet", "")])
    sheet = session.observe("Chrome", "cg:1")
    assert f"modal dialog open: {sheet.rows[0].ref}; it covers" in sheet.render()


def test_twin_controls_are_named_by_their_container(session, screen):
    screen.show(
        [
            E("t1", "AXGroup", "", path=[0, 0]),
            E("h1", "AXHeading", "Organic Large Brown Eggs 24ct", path=[0, 0, 0]),
            E("p1", "AXStaticText", "$7.49", path=[0, 0, 1]),
            E("b1", "AXButton", "Add to cart", path=[0, 0, 2]),
            E("t2", "AXGroup", "", path=[0, 1]),
            E("p2", "AXStaticText", "$8.99", path=[0, 1, 0]),
            E("n2", "AXStaticText", "Whole milk, 1 gal", path=[0, 1, 1]),
            E("b2", "AXButton", "Add to cart", path=[0, 1, 2]),
            E("t3", "AXGroup", "", path=[0, 2]),
            E("p3", "AXStaticText", "$1.00", path=[0, 2, 0]),
            E("b3", "AXButton", "Add to cart", path=[0, 2, 1]),
            E("t4", "AXGroup", "", path=[0, 3]),
            E("b4", "AXButton", "Add to cart", path=[0, 3, 0]),
            E("one", "AXButton", "Checkout", path=[0, 4]),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    lines = obs.render().splitlines()
    adds = [line for line in lines if '"Add to cart"' in line]
    assert adds[0].endswith("(in: Organic Large Brown Eggs 24ct)")
    assert adds[1].endswith("(in: Whole milk, 1 gal)")  # not the price
    assert adds[2].endswith("(in: $1.00)")  # only a price to go by
    assert "(in:" not in adds[3]  # nothing names its tile
    assert not any("(in:" in line for line in lines if '"Checkout"' in line)
    assert "(in: Whole milk, 1 gal)" in obs.find("add to cart")


def test_approval_lists_what_the_button_commits(session, screen, monkeypatch):
    screen.show(
        [
            E("x", "AXStaticText", "Bananas $0.29", path=[0, 0]),
            E(
                "dlg",
                "AXGroup",
                "Membership offer",
                subrole="AXApplicationDialog",
                path=[1],
            ),
            E("h", "AXHeading", "Upgrade to Executive", path=[1, 0]),
            E("l", "AXStaticText", "Annual fee", path=[1, 1]),
            E("a", "AXStaticText", "$65.00/yr", path=[1, 2]),
            E("l2", "AXStaticText", "Annual fee", path=[1, 3]),
            E("a2", "AXStaticText", "$65.00/yr", path=[1, 4]),
            E("go", "AXButton", "Upgrade for $65", path=[1, 5]),
        ]
    )
    obs = session.observe("Chrome", "cg:1")
    with pytest.raises(ComputerUseError) as err:
        session.act("click", _ref(obs, "Upgrade for $65"))
    assert err.value.code == "needs_approval"
    context = session.pending_approvals()["a1"]["context"]
    assert context[0] == "For: Upgrade to Executive"
    assert sum("Annual fee: $65.00/yr" in line for line in context) == 1
    go = obs.by_ref()[_ref(obs, "Upgrade for $65")]
    assert perception._near_button(obs, go) == (
        ["Annual fee: $65.00/yr", "Annual fee: $65.00/yr"],
        "Upgrade to Executive",
    )
    # A section as big as the page is not the button's.
    monkeypatch.setattr(perception, "MAX_NEAR_ROWS", 2)
    assert perception._near_button(obs, go) == ([], "")
    # A button whose sections show no other amount gets no section lines.
    screen.show([E("go", "AXButton", "Pay now $5", path=[0, 0, 0])])
    alone = session.observe("Chrome", "cg:1")
    assert perception._near_button(alone, alone.rows[0]) == ([], "")
