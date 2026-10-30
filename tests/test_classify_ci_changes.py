import json
from pathlib import Path

import pytest

import scripts.classify_ci_changes as ci_changes
from scripts.classify_ci_changes import (
    Lanes,
    classify,
    classify_policy,
    linux_test_matrix,
)

ROOT = Path(__file__).resolve().parent.parent


def test_known_engine_roots_exist_in_the_repository():
    assert {
        root for root in ci_changes._ENGINE_ROOTS if not (ROOT / root).is_dir()
    } == set()


def test_docs_only_selects_no_product_lane():
    assert classify(["README.md", "docs/operations/ci.md"]) == Lanes(
        engine=False, desktop=False, docs_only=True
    )


def test_desktop_only_does_not_select_engine():
    assert classify(["apps/rapid-mac/Sources/App.swift"]) == Lanes(
        engine=False, desktop=True, docs_only=False
    )


def test_engine_only_does_not_select_desktop():
    assert classify(["rapid_mlx/server.py"]) == Lanes(
        engine=True, desktop=False, docs_only=False
    )


@pytest.mark.parametrize(
    "path",
    [
        "bench/bench_spec_decode_mtp.py",
        "community-benchmarks/schema.json",
        "config/mypy-error-baseline.txt",
        "config/mypy-requirements.txt",
        "config/pr-validate-deps.txt",
        "config/requirements-ci-linux.txt",
        "evals/coherence_gate.py",
        "examples/tool_calling.py",
        "harness/perf_floors.json",
        "Makefile",
        "reports/benchmarks/model.json",
        "scripts/l1_smoke.sh",
        "tests/test_coherence.py",
        "videox_fun_mlx/pipeline/scheduler.py",
        "rapid_mlx/server.py",
    ],
)
def test_known_engine_area_does_not_select_desktop(path):
    assert classify([path]) == Lanes(engine=True, desktop=False, docs_only=False)


def test_evaluation_report_change_does_not_select_desktop():
    assert classify(
        [
            "docs/engineering/performance/starter-model-bakeoff.md",
            "evals/starter_experience.py",
        ]
    ) == Lanes(engine=True, desktop=False, docs_only=False)


def test_engine_benchmark_evidence_change_does_not_select_desktop():
    assert classify(
        [
            "bench/bench_spec_decode_mtp.py",
            "reports/benchmarks/mtp/result.json",
            "tests/test_mtp_spec_decode.py",
            "rapid_mlx/spec_decode/mtp/generator.py",
        ]
    ) == Lanes(engine=True, desktop=False, docs_only=False)


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


def test_unknown_config_file_fails_closed():
    assert classify(["config/some-new-file.txt"]) == Lanes(
        engine=True, desktop=True, docs_only=False
    )


def test_pr_validate_dependency_and_test_diff_is_engine_only():
    paths = [
        "config/pr-validate-deps.txt",
        "config/requirements-ci-linux.txt",
        "pyproject.toml",
        "scripts/pr_validate/_xdist.py",
        "scripts/pr_validate/steps/full_unit.py",
        "tests/conftest.py",
        "tests/test_pr_validate_full_unit.py",
    ]
    assert classify(paths) == Lanes(engine=True, desktop=False, docs_only=False)
    assert classify_policy(paths).linux_matrix_mode == "py312"
    assert classify_policy(paths).linux_matrix_reason == "pr-single-python"


@pytest.mark.parametrize(
    "path",
    [
        "apps/rapid-mac/Sources/Rapid/App.swift",
        "apps/rapid-mac/Package.swift",
        "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml",
        ".github/workflows/rapid-mac-ci.yml",
    ],
)
def test_desktop_app_and_workflow_paths_select_desktop(path):
    assert classify([path]).desktop is True


def test_cross_lane_rename_selects_removed_product_lane():
    # Workflows pass --no-renames, so a rename is represented by both paths.
    assert classify(["rapid_mlx/server.py", "docs/server.md"]) == Lanes(
        engine=True, desktop=False, docs_only=False
    )
    assert classify(["apps/rapid-mac/Sources/App.swift", "docs/App.swift.md"]) == Lanes(
        engine=False, desktop=True, docs_only=False
    )


def test_empty_diff_fails_closed():
    assert classify([]) == Lanes(engine=True, desktop=True, docs_only=False)


@pytest.mark.parametrize(
    "paths",
    [
        ["rapid_mlx/telemetry/events.json", "tests/test_telemetry_registry.py"],
        ["rapid_mlx/telemetry/registry.py", "docs/telemetry.md"],
        ["tests/test_telemetry_registry.py"],
        ["tests/test_telemetry_registry_drift.py"],
    ],
)
def test_ordinary_engine_pr_changes_use_complete_py312_matrix(paths):
    policy = classify_policy(paths)

    assert policy.lanes.engine is True
    assert policy.linux_matrix_mode == "py312"
    assert policy.linux_matrix_reason == "pr-single-python"
    assert linux_test_matrix(policy.linux_matrix_mode) == {
        "include": [
            {"python-version": "3.12", "shard": 1},
            {"python-version": "3.12", "shard": 2},
            {"python-version": "3.12", "shard": 3},
        ]
    }


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ci.yml",
        "config/requirements-ci-linux.txt",
        "config/pr-validate-deps.txt",
        "pyproject.toml",
        ".coveragerc",
        "scripts/ci_test_shard.py",
        "videox_fun_mlx/pipeline/scheduler.py",
        "rapid_mlx/cli.py",
        "rapid_mlx/agent_runtime/server.py",
        "rapid_mlx/api/tool_calling.py",
        "rapid_mlx/audio/processor.py",
        "rapid_mlx/byom/preflight.py",
        "rapid_mlx/computer_use/backend.py",
        "rapid_mlx/cua/gates.py",
        "rapid_mlx/engine/batched.py",
        "rapid_mlx/headless_service/install.py",
        "rapid_mlx/image/engine.py",
        "rapid_mlx/kernels/qsa_stage1.py",
        "rapid_mlx/mcp/security.py",
        "rapid_mlx/models/mllm.py",
        "rapid_mlx/routes/chat.py",
        "rapid_mlx/runtime/primary_lifecycle.py",
        "rapid_mlx/spec_decode/mtp/continuous_engine.py",
        "rapid_mlx/agents/adapter.py",
        "rapid_mlx/bench/pflash_replication.py",
        "rapid_mlx/community_bench/local_runner.py",
        "rapid_mlx/share/cli.py",
        "rapid_mlx/telemetry/model_id.py",
        "rapid_mlx/video/ltx25.py",
        "tests/test_security_hardening.py",
        "tests/test_download_gate.py",
        "tests/test_scheduler_fairness.py",
        "tests/test_kv_cache.py",
        "tests/test_model_loader.py",
        "tests/test_tool_parser.py",
        "tests/test_kernel_perf.py",
        "tests/test_dflash_integration.py",
        "tests/test_mlx_inference.py",
        "tests/test_bench_dflash.py",
        "tests/test_tools_auto_disable_thinking.py",
        "tests/test_harmony_parsers.py",
        "tests/test_mirror_throughput_floor_2010.py",
        "tests/test_base_wheel_vlm_text_degrade.py",
        "tests/test_resident_models.py",
        "tests/test_benchmark_contract.py",
        "tests/test_release_golden.py",
        "tests/test_1256_forced_toolchoice_empty_args.py",
        "tests/test_streaming_pipeline_integration.py",
        "tests/test_cli_output.py",
        "tests/conftest.py",
        "tests/integrations/test_live.py",
        "tests/qwen3coder_stream_harness.py",
        "tests/fixtures/sillytavern_golden.json",
        "tests/fixtures/security/auth_cases.json",
        "new-product/module.py",
    ],
)
def test_shared_unknown_and_control_pr_changes_use_single_python(path):
    policy = classify_policy([path])

    assert policy.lanes.engine is True
    assert policy.linux_matrix_mode == "py312"
    assert policy.linux_matrix_reason == "pr-single-python"
    matrix = linux_test_matrix(policy.linux_matrix_mode)["include"]
    assert len(matrix) == 3
    assert {entry["python-version"] for entry in matrix} == {
        "3.12",
    }
    assert {entry["shard"] for entry in matrix} == {1, 2, 3}


def test_mixed_desktop_and_engine_pr_change_uses_single_python():
    policy = classify_policy(
        ["rapid_mlx/telemetry/events.json", "apps/rapid-mac/Sources/App.swift"]
    )

    assert policy.lanes == Lanes(engine=True, desktop=True, docs_only=False)
    assert policy.lanes.engine is True
    assert policy.linux_matrix_mode == "py312"
    assert policy.linux_matrix_reason == "pr-single-python"


@pytest.mark.parametrize(
    "path",
    ["../rapid_mlx/telemetry/events.json", "/tmp/test_x.py", ".", "tests/../test_x.py"],
)
def test_invalid_path_cannot_enter_reduced_matrix(path):
    policy = classify_policy([path, "rapid_mlx/telemetry/events.json"])

    assert policy.linux_matrix_mode == "full"
    assert policy.linux_matrix_reason == "invalid-path"


@pytest.mark.parametrize(
    "paths,lanes",
    [
        (["docs/guides/server.md", "README.md"], Lanes(False, False, True)),
        (["apps/rapid-mac/Sources/App.swift"], Lanes(False, True, False)),
    ],
)
def test_no_engine_lane_keeps_dormant_matrix_full(paths, lanes):
    policy = classify_policy(paths)

    assert policy.lanes == lanes
    assert policy.linux_matrix_mode == "full"
    assert policy.linux_matrix_reason == "no-engine-lane"


def test_promoted_head_forces_full_matrix_without_changing_lane_scope():
    policy = classify_policy(
        ["rapid_mlx/telemetry/events.json"],
        force_full=True,
        force_reason="promoted-head",
    )

    assert policy.lanes == Lanes(engine=True, desktop=False, docs_only=False)
    assert policy.linux_matrix_mode == "full"
    assert policy.linux_matrix_reason == "promoted-head"
    assert len(json.loads(policy.as_outputs()["test_matrix"])["include"]) == 6


@pytest.mark.parametrize("paths", [[], ["", "  "]])
def test_missing_diff_keeps_full_matrix(paths):
    policy = classify_policy(paths)
    assert policy.lanes == Lanes(True, True, False)
    assert policy.linux_matrix_mode == "full"
    assert policy.linux_matrix_reason == "missing-diff"
    assert len(json.loads(policy.as_outputs()["test_matrix"])["include"]) == 6


def test_normalized_pr_paths_use_single_python():
    policy = classify_policy([" ./rapid_mlx/server.py ", "", "README.md"])
    assert policy.linux_matrix_mode == "py312"
    assert policy.linux_matrix_reason == "pr-single-python"
    assert json.loads(policy.as_outputs()["test_matrix"]) == linux_test_matrix("py312")


def test_force_full_accepts_explicit_reason():
    policy = classify_policy(
        ["rapid_mlx/server.py"], force_full=True, force_reason="manual"
    )
    assert policy.linux_matrix_mode == "full"
    assert policy.linux_matrix_reason == "manual"
