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


def test_py311_allowlist_only_contains_existing_files():
    assert {
        path
        for path in ci_changes._PY311_ENGINE_PATHS | ci_changes._PY311_TEST_PATHS
        if not (ROOT / path).is_file()
    } == set()


@pytest.mark.parametrize(
    "paths",
    [
        ["rapid_mlx/telemetry/events.json", "tests/test_telemetry_registry.py"],
        ["rapid_mlx/telemetry/registry.py", "docs/telemetry.md"],
        ["tests/test_telemetry_registry.py"],
        ["tests/test_telemetry_registry_drift.py"],
    ],
)
def test_leaf_engine_changes_use_complete_py311_matrix(paths):
    policy = classify_policy(paths)

    assert policy.lanes.engine is True
    assert policy.linux_matrix_mode == "py311"
    assert policy.linux_matrix_reason == "leaf-engine-only"
    assert linux_test_matrix(policy.linux_matrix_mode) == {
        "include": [
            {"python-version": "3.11", "shard": 1},
            {"python-version": "3.11", "shard": 2},
            {"python-version": "3.11", "shard": 3},
        ]
    }


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        (".github/workflows/ci.yml", "ci-control"),
        ("config/requirements-ci-linux.txt", "dependency-or-policy"),
        ("pyproject.toml", "dependency-or-policy"),
        (".coveragerc", "dependency-or-policy"),
        ("scripts/ci_test_shard.py", "ci-or-build-control"),
        ("videox_fun_mlx/pipeline/scheduler.py", "shared-runtime"),
        ("rapid_mlx/cli.py", "shared-core-or-unmapped"),
        ("rapid_mlx/agent_runtime/server.py", "shared-core-or-unmapped"),
        ("rapid_mlx/api/tool_calling.py", "shared-core-or-unmapped"),
        ("rapid_mlx/audio/processor.py", "shared-core-or-unmapped"),
        ("rapid_mlx/byom/preflight.py", "shared-core-or-unmapped"),
        ("rapid_mlx/computer_use/backend.py", "shared-core-or-unmapped"),
        ("rapid_mlx/cua/gates.py", "shared-core-or-unmapped"),
        ("rapid_mlx/engine/batched.py", "shared-core-or-unmapped"),
        ("rapid_mlx/headless_service/install.py", "shared-core-or-unmapped"),
        ("rapid_mlx/image/engine.py", "shared-core-or-unmapped"),
        ("rapid_mlx/kernels/qsa_stage1.py", "shared-core-or-unmapped"),
        ("rapid_mlx/mcp/security.py", "shared-core-or-unmapped"),
        ("rapid_mlx/models/mllm.py", "shared-core-or-unmapped"),
        ("rapid_mlx/routes/chat.py", "shared-core-or-unmapped"),
        ("rapid_mlx/runtime/primary_lifecycle.py", "shared-core-or-unmapped"),
        ("rapid_mlx/spec_decode/mtp/continuous_engine.py", "shared-core-or-unmapped"),
        ("rapid_mlx/agents/adapter.py", "shared-core-or-unmapped"),
        ("rapid_mlx/bench/pflash_replication.py", "shared-core-or-unmapped"),
        ("rapid_mlx/community_bench/local_runner.py", "shared-core-or-unmapped"),
        ("rapid_mlx/share/cli.py", "shared-core-or-unmapped"),
        ("rapid_mlx/telemetry/model_id.py", "shared-core-or-unmapped"),
        ("rapid_mlx/video/ltx25.py", "shared-core-or-unmapped"),
        ("tests/test_security_hardening.py", "test-support-or-fixture"),
        ("tests/test_download_gate.py", "test-support-or-fixture"),
        ("tests/test_scheduler_fairness.py", "test-support-or-fixture"),
        ("tests/test_kv_cache.py", "test-support-or-fixture"),
        ("tests/test_model_loader.py", "test-support-or-fixture"),
        ("tests/test_tool_parser.py", "test-support-or-fixture"),
        ("tests/test_kernel_perf.py", "test-support-or-fixture"),
        ("tests/test_dflash_integration.py", "test-support-or-fixture"),
        ("tests/test_mlx_inference.py", "test-support-or-fixture"),
        ("tests/test_bench_dflash.py", "test-support-or-fixture"),
        ("tests/test_tools_auto_disable_thinking.py", "test-support-or-fixture"),
        ("tests/test_harmony_parsers.py", "test-support-or-fixture"),
        ("tests/test_mirror_throughput_floor_2010.py", "test-support-or-fixture"),
        ("tests/test_base_wheel_vlm_text_degrade.py", "test-support-or-fixture"),
        ("tests/test_resident_models.py", "test-support-or-fixture"),
        ("tests/test_benchmark_contract.py", "test-support-or-fixture"),
        ("tests/test_release_golden.py", "test-support-or-fixture"),
        ("tests/test_1256_forced_toolchoice_empty_args.py", "test-support-or-fixture"),
        ("tests/test_streaming_pipeline_integration.py", "test-support-or-fixture"),
        ("tests/test_cli_output.py", "test-support-or-fixture"),
        ("tests/conftest.py", "test-support-or-fixture"),
        ("tests/integrations/test_live.py", "test-support-or-fixture"),
        ("tests/qwen3coder_stream_harness.py", "test-support-or-fixture"),
        ("tests/fixtures/sillytavern_golden.json", "test-support-or-fixture"),
        ("tests/fixtures/security/auth_cases.json", "test-support-or-fixture"),
        ("new-product/module.py", "shared-core-or-unmapped"),
    ],
)
def test_shared_unknown_and_control_changes_keep_full_matrix(path, reason):
    policy = classify_policy([path])

    assert policy.linux_matrix_mode == "full"
    assert policy.linux_matrix_reason == reason
    matrix = linux_test_matrix(policy.linux_matrix_mode)["include"]
    assert len(matrix) == 9
    assert {entry["python-version"] for entry in matrix} == {
        "3.10",
        "3.11",
        "3.12",
    }
    assert {entry["shard"] for entry in matrix} == {1, 2, 3}


def test_mixed_desktop_and_leaf_engine_change_keeps_full_matrix():
    policy = classify_policy(
        ["rapid_mlx/telemetry/events.json", "apps/rapid-mac/Sources/App.swift"]
    )

    assert policy.lanes == Lanes(engine=True, desktop=True, docs_only=False)
    assert policy.linux_matrix_mode == "full"
    assert policy.linux_matrix_reason == "shared-core-or-unmapped"


@pytest.mark.parametrize(
    "path", ["../rapid_mlx/telemetry/events.json", "/tmp/test_x.py"]
)
def test_invalid_path_cannot_enter_reduced_matrix(path):
    policy = classify_policy([path, "rapid_mlx/telemetry/events.json"])

    assert policy.linux_matrix_mode == "full"
    assert policy.linux_matrix_reason == "invalid-path"


def test_no_engine_lane_keeps_dormant_matrix_full():
    policy = classify_policy(["docs/guides/server.md", "README.md"])

    assert policy.lanes == Lanes(engine=False, desktop=False, docs_only=True)
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
    assert len(json.loads(policy.as_outputs()["test_matrix"])["include"]) == 9
