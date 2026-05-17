# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the CLI ↔ Config fidelity story.

Two layers:
  1. The #400 fix itself — `--prefill-step-size` lands in SchedulerConfig.
  2. The audit script — proves it would catch the regression if it ever
     comes back, and proves it stays quiet when correct.

The audit script (scripts/audit_cli_config_fidelity.py) is pure AST and
imports no mlx/vllm_mlx runtime — it must work on plain Linux CI.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT = REPO_ROOT / "scripts" / "audit_cli_config_fidelity.py"
CLI_SOURCE = REPO_ROOT / "vllm_mlx" / "cli.py"


# ---------------------------------------------------------------------------
# Layer 1: the #400 fix
# ---------------------------------------------------------------------------


def _serve_command_scheduler_config_kwargs() -> set[str]:
    """Return kwargs passed to SchedulerConfig() inside serve_command."""
    tree = ast.parse(CLI_SOURCE.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "serve_command"):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "SchedulerConfig"
            ):
                return {kw.arg for kw in call.keywords if kw.arg}
    raise AssertionError("serve_command or its SchedulerConfig(...) call not found")


def test_prefill_step_size_is_plumbed_in_serve_command():
    """#400 regression — serve_command must pass --prefill-step-size to SchedulerConfig.

    Pre-fix, `serve_command` built `SchedulerConfig(...)` without
    `prefill_step_size=args.prefill_step_size`. The field kept its
    dataclass default (2048), which MLLMBatchGenerator then used as a
    hard per-batch cap → user's prompts >2048 tokens rejected with
    "exceeds safe limit (2048)".
    """
    kwargs = _serve_command_scheduler_config_kwargs()
    assert "prefill_step_size" in kwargs, (
        "regression: SchedulerConfig in serve_command no longer receives "
        "prefill_step_size — see #400. Add `prefill_step_size=args.prefill_step_size` "
        "to the SchedulerConfig(...) construction."
    )


# ---------------------------------------------------------------------------
# Layer 2: the audit script behavior
# ---------------------------------------------------------------------------


def test_audit_passes_on_current_main():
    """After the #400 fix, the audit must be clean (exit 0)."""
    result = subprocess.run(
        [sys.executable, str(AUDIT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"audit reported drift on current main:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_audit_detects_synthetic_drift(tmp_path):
    """Audit must fail (exit 1) when a CLI flag is dropped at the
    construction site. This is the structural invariant that catches #400."""
    fake_cli = tmp_path / "cli.py"
    fake_cli.write_text(
        textwrap.dedent(
            """
            from vllm_mlx.scheduler import SchedulerConfig

            def serve_command(args):
                # args.prefill_step_size is read here (would be from argparse)
                print(args.prefill_step_size)
                # ...but never plumbed into SchedulerConfig — the bug pattern.
                cfg = SchedulerConfig(max_num_seqs=args.max_num_seqs)
                return cfg
            """
        )
    )

    # Run the audit as a script and patch CLI_PATH via env-free monkey-import.
    # The simplest test: import the audit's `audit()` function directly with
    # our synthetic cli.py + the real SchedulerConfig source.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import importlib

        mod = importlib.import_module("audit_cli_config_fidelity")
        issues = mod.audit(
            cli_path=fake_cli,
            config_source=REPO_ROOT / "vllm_mlx" / "scheduler.py",
            config_cls_name="SchedulerConfig",
        )
    finally:
        sys.path.pop(0)

    assert issues, "audit did not flag obvious synthetic drift"
    assert any("prefill_step_size" in line for line in issues), (
        f"expected prefill_step_size flagged; got: {issues}"
    )


def test_load_model_prefill_step_size_back_compat_translation():
    """back-compat — load_model(prefill_step_size=X) must still be ACCEPTED
    (no TypeError) and must translate the value into scheduler_config so it
    actually takes effect this time. Pre-0.6.52 it was a silent no-op (#400).

    External callers written against the previous documented signature
    (``load_model(..., prefill_step_size=X)``) would otherwise hit a hard
    TypeError on upgrade.
    """
    import inspect
    import warnings

    from vllm_mlx import server
    from vllm_mlx.scheduler import SchedulerConfig

    # Signature must still accept the kwarg.
    sig = inspect.signature(server.load_model)
    assert "prefill_step_size" in sig.parameters, (
        "load_model must keep the prefill_step_size kwarg for back-compat — "
        "removing it in a patch release breaks external callers (#405 codex round 2)."
    )

    # Translation must work: passing the kwarg without scheduler_config
    # should produce a SchedulerConfig with the value set, AND emit a
    # DeprecationWarning. We exercise just the translation block — we do not
    # actually load a model (which would require an mlx download).
    captured_scheduler_config = {}

    def fake_engine_init(*args, **kwargs):
        captured_scheduler_config["value"] = kwargs.get("scheduler_config")
        raise RuntimeError("stop after translation — we only need the kwarg")

    # Monkeypatch BatchedEngine to halt after the translation step.
    import vllm_mlx.engine.batched as batched_mod

    original = batched_mod.BatchedEngine.__init__
    batched_mod.BatchedEngine.__init__ = fake_engine_init
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                server.load_model("dummy-model", prefill_step_size=9999)
            except RuntimeError as exc:
                assert "stop after translation" in str(exc)

        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, "expected DeprecationWarning for load_model(prefill_step_size=...)"
        assert "deprecated" in str(dep[0].message).lower()
    finally:
        batched_mod.BatchedEngine.__init__ = original

    cfg = captured_scheduler_config.get("value")
    assert isinstance(cfg, SchedulerConfig), (
        "load_model(prefill_step_size=X) without scheduler_config must "
        "synthesise a SchedulerConfig so the value actually takes effect."
    )
    assert cfg.prefill_step_size == 9999, (
        f"translation failed: cfg.prefill_step_size={cfg.prefill_step_size}, expected 9999"
    )


def test_audit_no_false_positive_when_field_unused(tmp_path):
    """Audit must NOT flag a SchedulerConfig site if the function doesn't
    even read args.X — that's correct subparser-scoped behavior (e.g. the
    bench command doesn't expose --prefill-step-size and so doesn't need
    to pass it). Without this guard the audit would over-flag and become
    noise."""
    fake_cli = tmp_path / "cli.py"
    fake_cli.write_text(
        textwrap.dedent(
            """
            from vllm_mlx.scheduler import SchedulerConfig

            def benchmark_command(args):
                # This subparser only exposes max_num_seqs — args.prefill_step_size
                # is not referenced. SchedulerConfig correctly uses the default.
                cfg = SchedulerConfig(max_num_seqs=args.max_num_seqs)
                return cfg
            """
        )
    )

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import importlib

        mod = importlib.import_module("audit_cli_config_fidelity")
        issues = mod.audit(
            cli_path=fake_cli,
            config_source=REPO_ROOT / "vllm_mlx" / "scheduler.py",
            config_cls_name="SchedulerConfig",
        )
    finally:
        sys.path.pop(0)

    assert not issues, f"audit produced false positive(s): {issues}"
