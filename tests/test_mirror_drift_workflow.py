"""Static contract for the credential-free daily mirror drift workflow."""

from pathlib import Path


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
    assert "secrets." not in workflow
    assert "python -m pip install 'huggingface-hub>=1.1.0'" in workflow


def test_pin_resolution_is_an_independent_read_only_scheduled_job():
    import yaml

    path = (
        Path(__file__).resolve().parents[1] / ".github/workflows/mirror-drift-check.yml"
    )
    workflow = yaml.safe_load(path.read_text())
    job = workflow["jobs"]["image-pin-resolution"]
    assert "needs" not in job
    assert job["timeout-minutes"] == 10
    assert job["steps"][-1]["run"] == "python scripts/check_image_pins.py"
    assert workflow["permissions"] == {"contents": "read"}
