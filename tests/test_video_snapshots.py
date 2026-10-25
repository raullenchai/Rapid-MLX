# SPDX-License-Identifier: Apache-2.0
"""Video download boundaries exercised with local fixtures only."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx import model_aliases, model_downloads, model_integrity, video_snapshots

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("unexpected download")

    monkeypatch.setattr(model_downloads, "snapshot_download", refuse)


@pytest.mark.parametrize("revision", [None, "main", "abc", "A" * 40])
def test_unpinned_refused(monkeypatch, revision):
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda _: revision)
    with pytest.raises(ValueError, match="no approved immutable revision"):
        video_snapshots.resolve_model_dir("fixture/unpinned")


def test_verified_identity(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda _: SHA)
    monkeypatch.setattr(model_aliases, "resolve_hub_model", lambda _: "fixture/pinned")

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    def verify(repo, path, **kwargs):
        assert calls == [("fixture/pinned", {"revision": SHA})]
        assert repo == "fixture/pinned" and path == tmp_path
        assert kwargs["revision"] == SHA
        calls.append("verified")

    monkeypatch.setattr(model_downloads, "snapshot_download", download)
    monkeypatch.setattr(model_integrity, "verify_snapshot", verify)
    assert video_snapshots.resolve_model_dir("alias") == tmp_path
    assert calls[-1] == "verified"

    def corrupt(*args, **kwargs):
        raise model_integrity.ModelIntegrityError("corrupt")

    monkeypatch.setattr(model_integrity, "verify_snapshot", corrupt)
    with pytest.raises(model_integrity.ModelIntegrityError, match="corrupt"):
        video_snapshots.resolve_model_dir("alias")


def test_local_model_and_lora(tmp_path):
    assert video_snapshots.resolve_model_dir(tmp_path) == tmp_path
    with pytest.raises(FileNotFoundError):
        video_snapshots.resolve_lora_path(str(tmp_path))
    tensor = tmp_path / "one.safetensors"
    tensor.touch()
    assert video_snapshots.resolve_lora_path(str(tensor)) == str(tensor)
    assert video_snapshots.resolve_lora_path(str(tmp_path)) == str(tensor)
    (tmp_path / "two.safetensors").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        video_snapshots.resolve_lora_path(str(tmp_path))


def load_helper(relative, name):
    # Execute the real helper with its imports, without importing GPU pipelines.
    path = ROOT / "ltx_pipelines_mlx" / relative
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {"Path": Path, "argparse": SimpleNamespace(Namespace=object)}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


@pytest.mark.parametrize(
    "relative,name",
    [
        ("utils/blocks.py", "_resolve_model_dir"),
        ("utils/_orchestration.py", "resolve_model_dir"),
        ("utils/_orchestration.py", "resolve_lora_path"),
        ("ic_lora.py", "_resolve_lora_path"),
    ],
)
def test_helpers_refuse_unpinned_and_accept_local(relative, name, tmp_path):
    helper = load_helper(relative, name)
    with pytest.raises(ValueError, match="immutable revision"):
        helper("fixture/unpinned")
    (tmp_path / "one.safetensors").touch()
    result = helper(str(tmp_path))
    assert Path(result).exists()


def test_info_refuses_unpinned_and_accepts_local(tmp_path):
    helper = load_helper("cli.py", "_cmd_info")
    with pytest.raises(SystemExit, match="verified local model"):
        helper(SimpleNamespace(model="fixture/unpinned"))
    helper(SimpleNamespace(model=str(tmp_path)))


def test_vendored_packages_have_no_raw_downloads():
    for package in ("ltx_core_mlx", "ltx_pipelines_mlx"):
        for path in (ROOT / package).rglob("*.py"):
            assert "snapshot_download" not in path.read_text()
            assert "hf_hub_download" not in path.read_text()


def test_generated_provenance_is_importable():
    import runpy

    metadata = runpy.run_path(str(ROOT / "ltx_pipelines_mlx/_vendored.py"))
    assert len(metadata["TREE_SHA256"]) == 64


def test_failed_resolution_never_starts_child(monkeypatch, tmp_path):
    from rapid_mlx.video import ltx25

    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="vendored", executable="python"
        ),
    )

    def refuse(*args, **kwargs):
        pytest.fail("unverified model reached child")

    monkeypatch.setattr(ltx25.subprocess, "Popen", refuse)
    with pytest.raises(ValueError, match="immutable revision"):
        ltx25.LTX25VideoEngine("fixture/unpinned")._run_runtime(
            command=["generate", "--model", "fixture/unpinned"],
            prompt="fixture",
            output_path=tmp_path / "out.mp4",
            on_loaded=None,
        )


def test_run_runtime_resolves_repo_lora_before_offline_child(monkeypatch, tmp_path):
    from rapid_mlx.video import ltx25

    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="vendored", executable="python"
        ),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    lora_file = lora_dir / "adapter.safetensors"
    lora_file.touch()

    def resolve(value):
        if value == "fixture/model":
            return model_dir
        if value == "fixture/lora":
            return lora_dir
        raise AssertionError(value)

    monkeypatch.setattr(video_snapshots, "resolve_model_dir", resolve)
    calls = []

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            self.command = command
            calls.append((command, kwargs))

        def communicate(self, *, input, timeout):
            Path(self.command[self.command.index("--output") + 1]).write_bytes(b"mp4")
            return "", ""

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    ltx25.LTX25VideoEngine("fixture/model")._run_runtime(
        command=[
            "generate",
            "--model",
            "fixture/model",
            "--lora",
            "fixture/lora",
            "0.8",
            "--lora",
            str(lora_file),
            "1.0",
        ],
        prompt="fixture",
        output_path=tmp_path / "out.mp4",
        on_loaded=None,
    )

    command, run_kwargs = calls[0]
    assert run_kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert command[command.index("--model") + 1] == str(model_dir)
    lora_indexes = [i for i, value in enumerate(command) if value == "--lora"]
    assert command[lora_indexes[0] + 1] == str(lora_file)
    assert command[lora_indexes[0] + 2] == "0.8"
    assert command[lora_indexes[1] + 1] == str(lora_file)
    assert command[lora_indexes[1] + 2] == "1.0"
