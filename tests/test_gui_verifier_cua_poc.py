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


def test_click_plan_requires_bounded_normalized_candidates():
    plan = MODULE._validate_plan(
        {
            "action": "click",
            "step_instruction": "Click the search field",
            "candidates": [
                {"x": 0.4, "y": 0.1},
                {"x": 0.5, "y": 0.1},
                {"x": 0.6, "y": 0.1},
            ],
        }
    )
    assert len(plan["candidates"]) == 3


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
                "candidates": [{"x": 0.4, "y": 0.1}, {"x": 0.5, "y": 0.1}],
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
