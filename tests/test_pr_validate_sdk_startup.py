# SPDX-License-Identifier: Apache-2.0
"""SDK startup failures are validation failures, never baseline evidence."""

import sys
from contextlib import contextmanager
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired

import pytest

from scripts.pr_validate.base import Step, StepResult
from scripts.pr_validate.context import Context
from scripts.pr_validate.steps import sdk_env_check
from scripts.pr_validate.steps import stress_e2e_bench as matrix


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.chdir(tmp_path)
    context = Context(pr_number=123)
    context.work_dir = tmp_path / "artifacts"
    context.work_dir.mkdir()
    context.files_changed = ["rapid_mlx/server.py"]
    return context


@pytest.fixture
def agent(ctx):
    (ctx.repo_root / "sdk.py").touch()
    return {
        "name": "toy_sdk",
        "script": "sdk.py",
        "result_prefix": "Toy SDK",
        "prerequisite_imports": ["toy_sdk"],
    }


@pytest.mark.parametrize(
    "code,stdout,stderr,executed,status",
    [
        (1, "", "ModuleNotFoundError: No module named 'anthropic'", False, "fail"),
        (1, "", "ImportError: cannot import name 'Agent'", False, "fail"),
        (1, "", "RuntimeError: setup failed", False, "fail"),
        (0, "", "", False, "fail"),
        (0, "Toy SDK: 0/0 passed\n", "", False, "fail"),
        (0, "Toy SDK: 0/2 passed, 2 skipped\n", "", False, "fail"),
        (0, "Toy SDK: 3/2 passed\n", "", False, "fail"),
        (0, "Toy SDK: 1/2 passed\n", "", False, "fail"),
        (1, "Toy SDK: 2/2 passed\n", "", False, "fail"),
        (
            1,
            "Toy SDK: 1/2 passed\n",
            "Traceback (most recent call last)",
            False,
            "fail",
        ),
        (0, "Toy SDK: 2/2 passed\nToy SDK: 2/2 passed\n", "", False, "fail"),
        (0, "Toy SDK: 2/2 passed\n", "", True, "pass"),
        (0, "Toy SDK: 1/2 passed, 1 skipped\n", "", True, "pass"),
        (1, "Toy SDK: 1/2 passed\n", "", True, "fail"),
        (1, "Toy SDK: 0/2 passed\n", "", True, "fail"),
    ],
)
def test_sdk_requires_completed_nonempty_results(
    ctx, agent, monkeypatch, code, stdout, stderr, executed, status
):
    if stdout:
        stdout += "PR_VALIDATE_SDK_COMPLETE: Toy SDK\n"
    monkeypatch.setattr(
        matrix,
        "_supervise_sdk",
        lambda *a, **kw: CompletedProcess(a, code, stdout, stderr),
    )
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["executed"] is executed
    assert result["status"] == status
    assert Path(result["artifact"]).read_text() == stdout + stderr


@pytest.mark.parametrize("teardown_failure", [False, True])
def test_real_failed_battery_requires_clean_teardown_receipt(
    ctx, agent, monkeypatch, teardown_failure
):
    # Both children exit 1 after printing real failed-test counts. The second
    # fails cleanup without a traceback, so only the first supplies evidence
    # eligible for a baseline comparison.
    script = ctx.repo_root / agent["script"]
    script.write_text(
        "print('Toy SDK: 1/2 passed', flush=True)\n"
        + (
            "print('client cleanup failed', flush=True)\nraise SystemExit(1)\n"
            if teardown_failure
            else "print('PR_VALIDATE_SDK_READY: Toy SDK', flush=True)\nraise SystemExit(1)\n"
        )
    )
    real_run = matrix.subprocess.run

    def isolated_run(argv, **kwargs):
        return real_run([sys.executable, *argv[1:]], **kwargs)

    monkeypatch.setattr(matrix.subprocess, "run", isolated_run)
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["status"] == "fail"
    assert result["executed"] is (not teardown_failure)
    assert "Toy SDK: 1/2 passed" in Path(result["artifact"]).read_text()


@pytest.mark.parametrize(
    "failure", ["RuntimeError('late cleanup failed')", "SystemExit(1)"]
)
def test_real_atexit_failure_after_receipt_is_not_execution_evidence(
    ctx, agent, monkeypatch, failure
):
    script = ctx.repo_root / agent["script"]
    script.write_text(
        "import atexit\n"
        "def teardown():\n"
        f"    raise {failure}\n"
        "atexit.register(teardown)\n"
        "print('Toy SDK: 1/2 passed', flush=True)\n"
        "print('PR_VALIDATE_SDK_READY: Toy SDK', flush=True)\n"
        "raise SystemExit(1)\n"
    )
    real_run = matrix.subprocess.run

    def isolated_run(argv, **kwargs):
        return real_run([sys.executable, *argv[1:]], **kwargs)

    monkeypatch.setattr(matrix.subprocess, "run", isolated_run)
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["executed"] is False
    assert result["status"] == "fail"
    output = Path(result["artifact"]).read_text()
    assert "PR_VALIDATE_SDK_COMPLETE: Toy SDK" not in output
    assert "Traceback (most recent call last)" in output


@pytest.mark.parametrize("passed,total", [(2, 2), (1, 2), (0, 2)])
@pytest.mark.parametrize("late_exit", [None, 1, 2])
def test_supervisor_distinguishes_battery_outcome_from_silent_shutdown(
    ctx, agent, monkeypatch, passed, total, late_exit
):
    (ctx.repo_root / agent["script"]).write_text(
        "import atexit, os\n"
        + (
            f"atexit.register(lambda: os._exit({late_exit}))\n"
            if late_exit is not None
            else ""
        )
        + f"print('Toy SDK: {passed}/{total} passed', flush=True)\n"
        + "print('PR_VALIDATE_SDK_READY: Toy SDK', flush=True)\n"
        + f"raise SystemExit({int(passed != total)})\n"
    )
    real_run = matrix.subprocess.run
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda argv, **kw: real_run([sys.executable, *argv[1:]], **kw),
    )
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["executed"] is (late_exit is None)
    assert result["status"] == (
        "pass" if late_exit is None and passed == total else "fail"
    )
    output = Path(result["artifact"]).read_text()
    assert ("PR_VALIDATE_SDK_COMPLETE: Toy SDK" in output) is (late_exit is None)
    assert "Traceback" not in output


@pytest.mark.parametrize(
    "stdout",
    [
        "Toy SDK: 2/2 passed\n",
        "PR_VALIDATE_SDK_COMPLETE: Toy SDK\nToy SDK: 2/2 passed\n",
        "Toy SDK: 2/2 passed\nPR_VALIDATE_SDK_COMPLETE: Toy SDK\nPR_VALIDATE_SDK_COMPLETE: Toy SDK\n",
    ],
)
def test_incomplete_or_ambiguous_completion_receipt_blocks(
    ctx, agent, monkeypatch, stdout
):
    monkeypatch.setattr(
        matrix, "_supervise_sdk", lambda *a, **kw: CompletedProcess(a, 0, stdout, "")
    )
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["executed"] is False
    assert result["status"] == "fail"


def test_missing_registered_script_blocks(ctx, agent):
    (ctx.repo_root / "sdk.py").unlink()
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["status"] == "fail"
    assert result["executed"] is False


@pytest.mark.parametrize("prefix", [None, "", 42])
def test_sdk_without_valid_result_contract_cannot_pass(ctx, agent, monkeypatch, prefix):
    agent["result_prefix"] = prefix
    monkeypatch.setattr(
        matrix,
        "_supervise_sdk",
        lambda *a, **kw: CompletedProcess(a, 0, "Toy SDK: 2/2 passed\n", ""),
    )
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["status"] == "fail"
    assert result["executed"] is False


@pytest.mark.parametrize(
    "pr_executed,base_executed,expected",
    [
        (False, True, "fail"),
        (True, False, "fail"),
        (True, None, "fail"),
        (True, True, "pass"),
    ],
)
def test_only_real_test_failures_can_be_baseline_dispositioned(
    ctx, agent, monkeypatch, pr_executed, base_executed, expected
):
    registry = {
        "families": [
            {
                "family": "toy",
                "candidates": [
                    {"id": "toy/model", "ram_gb_required": 1, "quality_tier": "golden"}
                ],
            }
        ],
        "agents": [agent],
    }
    monkeypatch.setattr(matrix, "_load_registry", lambda: registry)
    monkeypatch.setattr(matrix, "_available_ram_gb", lambda: 64)

    @contextmanager
    def server(*args):
        yield "server.log"

    monkeypatch.setattr(matrix, "_server", server)
    monkeypatch.setattr(
        matrix, "_run_bench", lambda *a: {"status": "pass", "summary": "ok"}
    )
    monkeypatch.setattr(
        matrix, "_run_stress", lambda *a: {"status": "pass", "summary": "ok"}
    )
    monkeypatch.setattr(
        matrix,
        "_run_agent",
        lambda *a: {"status": "fail", "summary": "1/2 passed", "executed": pr_executed},
    )
    calls = []

    def base(*args):
        calls.append(args)
        result = {"status": "fail", "summary": "1/2 passed"}
        if base_executed is not None:
            result["executed"] = base_executed
        return result

    monkeypatch.setattr(matrix, "_run_base_check", base)
    result = matrix.StressE2EBenchStep().run(ctx)
    assert result.status == expected
    assert bool(calls) is pr_executed


@pytest.mark.parametrize("returncode", [0, 1])
def test_sdk_prerequisites_use_matrix_interpreter_without_running_scripts(
    ctx, agent, monkeypatch, returncode
):
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return CompletedProcess(
            argv, returncode, "missing toy_sdk" if returncode else "", ""
        )

    monkeypatch.setattr(sdk_env_check.subprocess, "run", run)
    result = sdk_env_check.SDKEnvCheckStep().run(ctx)
    assert result.status == ("pass" if returncode == 0 else "fail")
    argv, kwargs = calls[0]
    assert argv[:2] == ["python3.12", "-c"]
    assert argv[-1] == "toy_sdk"
    assert "sdk.py" not in argv
    assert kwargs["cwd"] == str(ctx.repo_root)
    assert kwargs["timeout"] == 60


@pytest.mark.parametrize(
    "registry",
    [None, [], "invalid", {}, {"agents": None}, {"agents": {}}, {"agents": []}],
)
def test_malformed_top_level_sdk_registry_fails_without_launching(
    ctx, monkeypatch, registry
):
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: registry)
    monkeypatch.setattr(
        sdk_env_check.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("malformed registry must not launch"),
    )
    result = sdk_env_check.SDKEnvCheckStep().run(ctx)
    assert result.status == "fail"
    assert "malformed SDK registry" in result.summary


def test_sdk_env_failure_stops_before_costly_tests_without_fail_fast(ctx):
    from scripts.pr_validate.runner import run_pipeline

    class MissingSDK(Step):
        name = "sdk_env_check"

        def run(self, ctx):
            return StepResult(self.name, "fail", "missing SDK")

    class Costly(Step):
        def run(self, ctx):
            pytest.fail("must not reach model/full-unit work")

    assert run_pipeline(123, steps=[MissingSDK(), Costly()], fail_fast=False) == 1


def test_sdk_preflight_follows_supply_chain_and_precedes_checks():
    from scripts.pr_validate.runner import STEPS

    names = [step.name for step in STEPS]
    assert (
        names.index("supply_chain")
        < names.index("sdk_env_check")
        < names.index("mac_only_tests")
    )


@pytest.mark.parametrize(
    "missing",
    ["name", "script", "script_file", "prerequisite_imports", "result_prefix"],
)
def test_incomplete_registration_blocks_without_running_scripts(
    ctx, agent, monkeypatch, missing
):
    if missing == "script_file":
        (ctx.repo_root / agent["script"]).unlink()
    else:
        del agent[missing]
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})
    monkeypatch.setattr(
        sdk_env_check.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("invalid registry must not launch"),
    )
    assert sdk_env_check.SDKEnvCheckStep().run(ctx).status == "fail"


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("python3.12 unavailable"),
        PermissionError("python3.12 not executable"),
        TimeoutExpired("python3.12", 60, output=b"import started", stderr=b"stalled"),
    ],
)
def test_sdk_probe_launch_failure_records_diagnostics_and_stops(
    ctx, agent, monkeypatch, error
):
    from scripts.pr_validate import runner

    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})

    def failed_probe(*args, **kwargs):
        raise error

    monkeypatch.setattr(sdk_env_check.subprocess, "run", failed_probe)
    result = sdk_env_check.SDKEnvCheckStep().run(ctx)
    assert result.status == "fail"
    log = Path(result.artifacts[0]).read_text()
    assert type(error).__name__ in log
    if isinstance(error, TimeoutExpired):
        assert "import started" in log
        assert "stalled" in log
    monkeypatch.setattr(runner, "Context", lambda **kwargs: ctx)

    class Costly(Step):
        name = "full_unit"

        def run(self, ctx):
            pytest.fail("failed SDK probe must stop before costly work")

    assert (
        runner.run_pipeline(123, steps=[sdk_env_check.SDKEnvCheckStep(), Costly()]) == 1
    )


def test_sdk_probe_missing_interpreter_fails_with_real_launch(ctx, agent, monkeypatch):
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})
    monkeypatch.setenv("PATH", str(ctx.repo_root / "no-interpreters"))
    result = sdk_env_check.SDKEnvCheckStep().run(ctx)
    assert result.status == "fail"
    log = Path(result.artifacts[0]).read_text()
    assert "FileNotFoundError" in log
    assert "python3.12" in log


@pytest.mark.parametrize(
    "missing,error",
    [
        (
            "_harbor_sdk_startup_negative_control_missing_module_0161",
            "ModuleNotFoundError",
        ),
        ("json:_harbor_missing_sdk_symbol_0161", "AttributeError"),
    ],
)
def test_prerequisite_probe_reports_real_import_failure_without_sdk_or_network(
    ctx, agent, monkeypatch, missing, error
):
    agent["prerequisite_imports"] = ["json", missing]
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})
    real_run = sdk_env_check.subprocess.run

    def isolated_run(argv, **kwargs):
        return real_run([sys.executable, "-I", *argv[1:]], **kwargs)

    monkeypatch.setattr(sdk_env_check.subprocess, "run", isolated_run)
    result = sdk_env_check.SDKEnvCheckStep().run(ctx)
    assert result.status == "fail"
    log = Path(result.artifacts[0]).read_text()
    assert f"{missing}: {error}" in log

    # Exercise the real failing probe through the pipeline too: a module may
    # exist while a required class is missing, and costly tests must not run.
    from scripts.pr_validate import runner

    monkeypatch.setattr(runner, "Context", lambda **kwargs: ctx)

    class Costly(Step):
        name = "full_unit"

        def run(self, ctx):
            pytest.fail("missing prerequisite must stop before costly work")

    assert (
        runner.run_pipeline(123, steps=[sdk_env_check.SDKEnvCheckStep(), Costly()]) == 1
    )


@pytest.mark.parametrize(
    "files,override,expected",
    [
        (["rapid_mlx/server.py"], False, True),
        (["README.md"], False, False),
        (["tests/test_example.py"], False, False),
        (["rapid_mlx/server.py"], True, False),
    ],
)
def test_sdk_preflight_matches_model_matrix_gating(
    ctx, monkeypatch, files, override, expected
):
    ctx.files_changed = files
    monkeypatch.setenv("PR_VALIDATE_NO_STRESS", "1" if override else "0")
    assert sdk_env_check.SDKEnvCheckStep().should_run(ctx) is expected


def test_cpu_only_partial_pipeline_omits_matrix_prerequisites(ctx):
    from scripts.pr_validate.runner import run_pipeline

    class SDKProbe(Step):
        name = "sdk_env_check"

        def run(self, ctx):
            pytest.fail("CPU-only pipeline must not require the omitted SDK matrix")

    class CPUCheck(Step):
        name = "cpu_check"

        def run(self, ctx):
            return StepResult(self.name, "pass", "CPU check ran")

    assert (
        run_pipeline(
            123,
            steps=[SDKProbe(), CPUCheck()],
            skip_steps=("stress_e2e_bench",),
        )
        == 0
    )


def test_sdk_failure_includes_install_command(ctx, agent, monkeypatch):
    agent["prerequisite_imports"] = ["anthropic", "langchain_core", "langchain_openai"]
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})
    output = "\n".join(
        f"{name}: ModuleNotFoundError: missing"
        for name in agent["prerequisite_imports"]
    )
    monkeypatch.setattr(
        sdk_env_check.subprocess,
        "run",
        lambda *a, **kw: CompletedProcess(a, 1, output, ""),
    )
    result = sdk_env_check.SDKEnvCheckStep().run(ctx)
    assert result.status == "fail"
    for name in agent["prerequisite_imports"]:
        assert name in result.summary
    assert 'python3.12 -m pip install -e ".[test]"' in result.summary


def test_validator_extra_contains_sdk_distributions():
    import tomllib

    from packaging.requirements import Requirement

    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    requirements = {
        Requirement(value).name: Requirement(value)
        for value in project["project"]["optional-dependencies"]["test"]
    }
    for distribution in ("anthropic", "langchain-core", "langchain-openai"):
        assert distribution in requirements
        assert str(requirements[distribution].specifier)
