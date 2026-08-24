import pytest

from scripts.classify_ci_changes import Lanes, classify


def test_docs_only_selects_no_product_lane():
    assert classify(["README.md", "docs/operations/ci.md"]) == Lanes(
        engine=False, desktop=False, docs_only=True
    )


def test_desktop_only_does_not_select_engine():
    assert classify(["apps/rapid-mac/Sources/App.swift"]) == Lanes(
        engine=False, desktop=True, docs_only=False
    )


def test_engine_only_does_not_select_desktop():
    assert classify(["vllm_mlx/server.py"]) == Lanes(
        engine=True, desktop=False, docs_only=False
    )


@pytest.mark.parametrize(
    "path",
    [
        "scripts/check_rapid_mac_ax_identifiers.py",
        "scripts/select_gui_flows.py",
        "tests/test_rapid_mac_ax_identifiers.py",
        "tests/test_rapid_mac_xcui_target.py",
        "tests/test_ax_baseline.py",
        "tests/test_ax_baseline_os_variance.py",
        "tests/test_gui_control_behavior_contract.py",
        "tests/test_gui_preflight_contract.py",
        "tests/test_gui_golden_ci_coverage.py",
        "tests/test_gui_flow_routing.py",
        "tests/test_gui_walk_completeness.py",
        "tests/test_fake_sidecar_image_catalog.py",
        "tests/fixtures/ax_baseline/macos.txt",
    ],
)
def test_desktop_support_path_stays_in_desktop_lane(path):
    assert classify([path]) == Lanes(engine=False, desktop=True, docs_only=False)


def test_cross_cutting_change_selects_both_lanes():
    assert classify([".github/workflows/ci.yml"]) == Lanes(
        engine=True, desktop=True, docs_only=False
    )


def test_unknown_product_area_fails_closed():
    assert classify(["new-product/config.toml", "new-product/README.md"]) == Lanes(
        engine=True, desktop=True, docs_only=False
    )


def test_cross_lane_rename_selects_removed_product_lane():
    # Workflows pass --no-renames, so a rename is represented by both paths.
    assert classify(["vllm_mlx/server.py", "docs/server.md"]) == Lanes(
        engine=True, desktop=False, docs_only=False
    )
    assert classify(["apps/rapid-mac/Sources/App.swift", "docs/App.swift.md"]) == Lanes(
        engine=False, desktop=True, docs_only=False
    )


def test_empty_diff_fails_closed():
    assert classify([]) == Lanes(engine=True, desktop=True, docs_only=False)
