import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm")
pytest.importorskip("playwright")

MODULE_PATH = (
    Path(__file__).parents[1] / "tools" / "gui_verifier_cua_poc" / "run_poc.py"
)
SPEC = importlib.util.spec_from_file_location("gui_verifier_cua_poc", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_click_plan_requires_current_semantic_target():
    plan = MODULE._validate_plan(
        {
            "action": "click",
            "step_instruction": "Click the search field",
            "target_id": "t004",
        },
        {"t004", "t005"},
    )
    assert plan["target_id"] == "t004"


def test_plan_rejects_invented_semantic_target():
    with pytest.raises(ValueError, match="unknown target_id"):
        MODULE._validate_plan(
            {
                "action": "fill",
                "step_instruction": "Search for flashlight",
                "target_id": "t999",
                "text": "flashlight",
                "submit": True,
            },
            {"t004"},
        )


@pytest.mark.parametrize(
    "instruction",
    ["Add to cart", "现在购买", "Proceed to checkout", "Enter password"],
)
def test_plan_guard_rejects_purchase_and_account_actions(instruction):
    with pytest.raises(ValueError, match="forbidden"):
        MODULE._validate_plan(
            {
                "action": "click",
                "step_instruction": instruction,
                "target_id": "t004",
            }
        )


def test_execution_guard_uses_live_element_text_and_href():
    with pytest.raises(RuntimeError, match="safety guard"):
        MODULE._guard_element(
            {"text": "Buy now", "href": "https://example.test/checkout"}
        )


@pytest.mark.parametrize(
    "url",
    ["https://example.com/v1/chat/completions", "http://localhost:18730/v1"],
)
def test_planner_endpoint_rejects_nonliteral_loopback_hosts(url):
    with pytest.raises(ValueError, match="loopback"):
        MODULE._validate_loopback_url(url)


def test_planner_endpoint_accepts_literal_loopback_ip():
    assert (
        MODULE._validate_loopback_url("http://127.0.0.1:18730/v1/chat/completions")
        == "http://127.0.0.1:18730/v1/chat/completions"
    )


def test_target_candidates_are_derived_inside_viewport():
    target = MODULE.Target(
        target_id="t004",
        tag="a",
        role="",
        label="Flashlight",
        value="",
        href="https://example.test/item",
        input_type="",
        sponsored=False,
        context="",
        box={"x": 1200, "y": 760, "width": 200, "height": 100},
    )
    candidates = MODULE._target_candidates(target, {"width": 1280, "height": 800})
    assert len(candidates) == 3
    assert all(0 < candidate.x < 1 and 0 < candidate.y < 1 for candidate in candidates)


def test_protocol_recognizes_atomic_fill_submit_navigation():
    outcome = MODULE._protocol_outcome(
        {"action": "fill", "text": "flashlight", "submit": True},
        {
            "active_value_after": "",
            "url_changed": True,
            "dom_changed": True,
            "focus_matches_target": False,
            "scroll_changed": False,
        },
        "",
    )
    assert outcome == "success"


def _product(target_id, asin, sponsored=False):
    return MODULE.Target(
        target_id=target_id,
        tag="a",
        role="",
        label="Flashlight",
        value="",
        href=f"https://www.amazon.com/example/dp/{asin}",
        input_type="",
        sponsored=sponsored,
        context="Flashlight 4.6 out of 5 stars (50K)",
        box={"x": 10, "y": 10, "width": 100, "height": 50},
    )


def test_fast_controller_scrolls_until_three_organic_products_are_visible():
    targets = {
        "t001": _product("t001", "B000000001"),
        "t002": _product("t002", "B000000002"),
        "t003": _product("t003", "B000000003", sponsored=True),
    }
    plan = MODULE._fast_controller_plan(
        "https://www.amazon.com/s?k=flashlight", targets, [], 4
    )
    assert plan and plan["action"] == "scroll"

    targets["t004"] = _product("t004", "B000000004")
    assert (
        MODULE._fast_controller_plan(
            "https://www.amazon.com/s?k=flashlight", targets, [], 4
        )
        is None
    )


def test_navigation_guard_rejects_cross_origin_target():
    target = _product("t001", "B000000001")
    MODULE._guard_target_origin(target, "amazon.com")
    hostile = MODULE.Target(
        **{
            **target.__dict__,
            "href": "https://attacker.example/continue",
        }
    )
    with pytest.raises(RuntimeError, match="navigation guard"):
        MODULE._guard_target_origin(hostile, "amazon.com")
