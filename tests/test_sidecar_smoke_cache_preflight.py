from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SCRIPT = _ROOT / "apps/rapid-mac/scripts/check-sidecar-smoke-cache.py"
_MANIFEST = _ROOT / "apps/rapid-mac/scripts/sidecar-smoke-models.json"
_SPEC = importlib.util.spec_from_file_location("sidecar_cache_preflight", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _populate(cache: Path, repository: str, revision: str) -> None:
    files = next(
        files
        for pinned_repository, pinned_revision, files in _MODULE.load_pins(
            _MANIFEST
        ).values()
        if (pinned_repository, pinned_revision) == (repository, revision)
    )
    snapshot = _MODULE.snapshot_path(cache, repository, revision)
    for file in files:
        target = snapshot / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub")


def test_manifest_is_the_exact_three_pin_source_of_truth() -> None:
    pins = _MODULE.load_pins(_MANIFEST)
    assert pins["qwen"][:2] == (
        "mlx-community/Qwen3.5-9B-4bit",
        "8b2b98c00a6b4d291155e4890773ca8f769aee53",
    )
    assert pins["gemma"][:2] == (
        "mlx-community/gemma-4-e2b-it-8bit",
        "03dcf209f3f549b4075e7191e77cf69b3d48e1b2",
    )
    assert pins["flux"][:2] == (
        "Runpod/FLUX.2-klein-4B-mflux-4bit",
        "7ee1b3aa8178a1240050490072196a57da2bf2a9",
    )
    for key in ("qwen", "gemma"):
        files = pins[key][2]
        assert "config.json" in files
        assert "tokenizer.json" in files
        assert "model.safetensors.index.json" in files
        assert "model-00001-of-00002.safetensors" in files
        assert "model-00002-of-00002.safetensors" in files
    flux_files = pins["flux"][2]
    assert "config.json" in flux_files
    assert "tokenizer/tokenizer.json" in flux_files
    assert "text_encoder/model.safetensors.index.json" in flux_files
    assert "transformer/model.safetensors.index.json" in flux_files
    assert "vae/model.safetensors.index.json" in flux_files


def test_cache_root_matches_hugging_face_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_cache = tmp_path / "explicit-hub"
    hf_home = tmp_path / "hf-home"
    xdg_cache = tmp_path / "xdg-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))
    assert _MODULE.default_cache_root() == hub_cache

    monkeypatch.delenv("HF_HUB_CACHE")
    assert _MODULE.default_cache_root() == hf_home / "hub"

    monkeypatch.delenv("HF_HOME")
    assert _MODULE.default_cache_root() == xdg_cache / "huggingface" / "hub"


@pytest.mark.parametrize("missing_key", ["qwen", "gemma", "flux"])
def test_one_missing_pin_fails_and_names_exact_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing_key: str
) -> None:
    pins = _MODULE.load_pins(_MANIFEST)
    for key, (repository, revision, _) in pins.items():
        if key != missing_key:
            _populate(tmp_path, repository, revision)
    assert (
        _MODULE.main(["--manifest", str(_MANIFEST), "--cache-root", str(tmp_path)]) == 1
    )
    error = capsys.readouterr().err
    repository, revision, _ = pins[missing_key]
    assert f"{repository}@{revision}" in error
    assert f"revision='{revision}'" in error
    assert "No download was attempted" in error


def test_all_missing_are_reported_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _MODULE.main(["--manifest", str(_MANIFEST), "--cache-root", str(tmp_path)]) == 1
    )
    error = capsys.readouterr().err
    assert "3 immutable snapshot(s)" in error
    for repository, revision, _ in _MODULE.load_pins(_MANIFEST).values():
        assert f"{repository}@{revision}" in error


def test_all_present_pass_without_reading_model_files(tmp_path: Path) -> None:
    for repository, revision, _ in _MODULE.load_pins(_MANIFEST).values():
        _populate(tmp_path, repository, revision)
    assert (
        _MODULE.main(["--manifest", str(_MANIFEST), "--cache-root", str(tmp_path)]) == 0
    )


@pytest.mark.parametrize(
    ("damaged_file", "broken_link"),
    [
        ("model-00002-of-00002.safetensors", True),
        ("tokenizer.json", False),
    ],
)
def test_partial_snapshot_fails_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    damaged_file: str,
    broken_link: bool,
) -> None:
    pins = _MODULE.load_pins(_MANIFEST)
    for key, (repository, revision, _) in pins.items():
        if key == "qwen":
            _populate(tmp_path, repository, revision)
            snapshot = _MODULE.snapshot_path(tmp_path, repository, revision)
            damaged_path = snapshot / damaged_file
            damaged_path.unlink()
            if broken_link:
                damaged_path.symlink_to(tmp_path / "missing-blob")
        else:
            _populate(tmp_path, repository, revision)

    assert (
        _MODULE.main(["--manifest", str(_MANIFEST), "--cache-root", str(tmp_path)]) == 1
    )
    error = capsys.readouterr().err
    assert f"{pins['qwen'][0]}@{pins['qwen'][1]}" in error
    assert f"missing: {damaged_file}" in error


def test_present_pins_emit_workflow_outputs(tmp_path: Path) -> None:
    pins = _MODULE.load_pins(_MANIFEST)
    for repository, revision, _ in pins.values():
        _populate(tmp_path, repository, revision)
    output = tmp_path / "github-output"
    assert (
        _MODULE.main(
            [
                "--manifest",
                str(_MANIFEST),
                "--cache-root",
                str(tmp_path),
                "--github-output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text().splitlines() == [
        f"qwen_model={pins['qwen'][0]}",
        f"qwen_revision={pins['qwen'][1]}",
        f"gemma_model={pins['gemma'][0]}",
        f"gemma_revision={pins['gemma'][1]}",
        f"flux_model={pins['flux'][0]}",
        f"flux_revision={pins['flux'][1]}",
    ]


def test_malformed_manifest_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "pins.json"
    manifest.write_text(json.dumps({"schema": 1, "models": {}}))
    with pytest.raises(_MODULE.PreflightError, match="exactly qwen, gemma, and flux"):
        _MODULE.load_pins(manifest)


def test_repository_cannot_inject_a_workflow_output(tmp_path: Path) -> None:
    payload = json.loads(_MANIFEST.read_text())
    payload["models"]["qwen"]["repository"] = "owner/model\nevil=value"
    manifest = tmp_path / "pins.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(_MODULE.PreflightError, match="owner/name"):
        _MODULE.load_pins(manifest)


def test_manifest_file_paths_cannot_escape_the_snapshot(tmp_path: Path) -> None:
    payload = json.loads(_MANIFEST.read_text())
    payload["models"]["gemma"]["files"].append("../outside")
    manifest = tmp_path / "pins.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(_MODULE.PreflightError, match="unsafe path"):
        _MODULE.load_pins(manifest)


def test_workflow_preflight_precedes_every_expensive_command() -> None:
    workflow = (_ROOT / ".github/workflows/auto-release.yml").read_text()
    preflight = workflow.index("check-sidecar-smoke-cache.py")
    for expensive in (
        "/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv",
        "tests/integrations/agent_smoke.sh",
        "apps/rapid-mac/scripts/build-sidecar.sh",
    ):
        assert preflight < workflow.index(expensive)
    assert "steps.sidecar-pins.outputs.qwen_model" in workflow
    assert "steps.sidecar-pins.outputs.gemma_revision" in workflow
    assert "steps.sidecar-pins.outputs.flux_model" in workflow
    assert "steps.sidecar-pins.outputs.flux_revision" in workflow
    assert (
        "/opt/homebrew/opt/python@3.12/bin/python3.12 \\\n"
        "            apps/rapid-mac/scripts/check-sidecar-smoke-cache.py" in workflow
    )
