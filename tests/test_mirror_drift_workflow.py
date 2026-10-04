"""Contracts for the daily audit and one-purpose release repair workflow."""

from pathlib import Path

import yaml


def test_daily_audit_keeps_bounded_parallelism_and_safety_contract() -> None:
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "mirror-drift-check.yml"
    ).read_text()
    assert "timeout-minutes: 30" in workflow
    assert "--only-used --workers 32" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "python -m pip install 'huggingface-hub>=1.1.0'" in workflow
    parsed = yaml.safe_load(workflow)
    audit = parsed["jobs"]["audit-used-models"]
    assert "github.event_name == 'schedule'" in audit["if"]
    audit_text = str(audit)
    assert "secrets." not in audit_text


def test_targeted_purge_is_manual_boolean_and_does_not_accept_urls() -> None:
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "mirror-drift-check.yml"
    ).read_text()
    parsed = yaml.safe_load(workflow)
    dispatch = parsed[True]["workflow_dispatch"]
    assert dispatch["inputs"] == {
        "purge_gemma_index": {
            "description": "Purge and verify the two fixed Gemma index URLs",
            "required": True,
            "default": False,
            "type": "boolean",
        }
    }
    job = parsed["jobs"]["targeted-gemma-index-purge"]
    assert job["name"] == "Targeted Gemma index CDN purge (not a full mirror audit)"
    assert job["if"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.purge_gemma_index }}"
    )
    assert job["steps"][-1]["run"] == "python scripts/purge_gemma_index_cdn.py"
