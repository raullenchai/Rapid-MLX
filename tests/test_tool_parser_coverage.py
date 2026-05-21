# SPDX-License-Identifier: Apache-2.0
"""
Regression test for ``scripts/audit_tool_parser_coverage.py``.

Two layers, same shape as ``tests/test_cli_config_fidelity.py``:

  1. **Coverage assertion** — every registered ``--tool-call-parser``
     value must be matrix-tested via ``golden_models.yaml`` or explicitly
     exempted with a documented reason.

  2. **Audit-script self-test** — synthetic injection: simulate a new
     parser registered without matrix coverage or exemption, and verify
     the audit reports it as a gap. Proves the audit would catch the
     #425-class regression if it ever recurred.

Bug class: see ``scripts/audit_tool_parser_coverage.py`` docstring and
#425 (jpcarranza94). ``qwen3_xml`` was registered but never matrix-
tested — the wire-format mismatch only surfaced in production.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT = REPO_ROOT / "scripts" / "audit_tool_parser_coverage.py"


def test_tool_parser_matrix_coverage_clean():
    """Every registered ``ToolParserManager`` parser must be either in
    ``golden_models.yaml`` overrides OR in the audit's ``MATRIX_EXEMPT``
    set with a reason.

    Failure means a new parser was registered without integration
    matrix coverage. Action: add a golden_models.yaml override that
    exercises the parser, OR add an entry to ``MATRIX_EXEMPT`` with
    a documented reason (alias / TODO with ticket).
    """
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(AUDIT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"audit_tool_parser_coverage.py failed (exit {proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def test_audit_catches_uncategorized_parser(monkeypatch):
    """Synthetic injection: register a parser without matrix coverage or
    exemption and verify the audit detects it.

    This proves the audit is load-bearing — that a real regression of
    the #425 class would actually fail CI. The companion test above
    asserts the current tree is clean; this one asserts the audit is
    correctly wired.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    audit_mod = importlib.import_module("audit_tool_parser_coverage")

    # Re-fetch the registered set with a synthetic addition. Patching
    # registered_parsers is the cleanest seam — keeps MATRIX_EXEMPT and
    # the YAML parser unchanged so we test the GAP detection specifically.
    real_registered = audit_mod.registered_parsers()
    synthetic = real_registered | {"fake_uncovered_parser_v2"}

    monkeypatch.setattr(audit_mod, "registered_parsers", lambda: synthetic)

    _registered, _matrix, gaps = audit_mod.audit()
    assert "fake_uncovered_parser_v2" in gaps, (
        f"synthetic uncovered parser not detected as a gap; got {gaps!r}. "
        f"The audit is not correctly identifying parsers missing both "
        f"matrix coverage AND MATRIX_EXEMPT entry."
    )


def test_audit_exempt_entries_carry_reasons():
    """Every ``MATRIX_EXEMPT`` entry must have a non-trivial reason string.

    Guards against future contributors silently padding the exempt dict
    to make the audit pass without thinking about why coverage is OK to
    skip. The convention: each entry is either an alias (≥1 word reason)
    or a TODO (must start with ``TODO`` and include a follow-up note).
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    audit_mod = importlib.import_module("audit_tool_parser_coverage")

    too_short = {
        name: reason
        for name, reason in audit_mod.MATRIX_EXEMPT.items()
        if not reason or len(reason.strip()) < 10
    }
    assert not too_short, (
        f"MATRIX_EXEMPT entries with empty or too-short reasons: {too_short!r}. "
        f"Each exemption must carry a real explanation (alias mapping or TODO "
        f"with follow-up note) — minimum 10 characters."
    )


@pytest.mark.parametrize(
    "synthetic_yaml,expected_parser",
    [
        # Override with --tool-call-parser flag → parser extracted
        (
            "overrides:\n"
            '  "mlx-community/Fake-Model":\n'
            '    args: ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]\n',
            "hermes",
        ),
        # Override without the flag → no parser extracted (e.g. some
        # overrides set only --max-model-len). Audit must not crash.
        (
            "overrides:\n"
            '  "mlx-community/Fake-Model":\n'
            '    args: ["--max-model-len", "8192"]\n',
            None,
        ),
    ],
)
def test_matrix_parser_extraction_handles_yaml_shapes(
    tmp_path, synthetic_yaml, expected_parser
):
    """``parsers_in_matrix`` must tolerate every shape golden_models.yaml
    overrides can legitimately take — with the flag, without the flag,
    with extra unrelated flags before/after, etc."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    audit_mod = importlib.import_module("audit_tool_parser_coverage")

    yaml_path = tmp_path / "synthetic.yaml"
    yaml_path.write_text(synthetic_yaml)
    found = audit_mod.parsers_in_matrix(yaml_path)
    if expected_parser is None:
        assert found == set()
    else:
        assert expected_parser in found
