"""CLI dispatch contracts with bounded local fixtures and mocked work engines."""

import json
import runpy
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from scripts import deepseek_v41_gptq as b
from scripts import deepseek_v41_gptq_amend as amend
from scripts import deepseek_v41_gptq_quality as quality
from scripts import deepseek_v41_gptq_scan as scan


@pytest.fixture
def cli(tmp_path, monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    for name in (
        "manifest",
        "recipe",
        "slices",
        "source-manifest",
        "reference",
        "checkpoint",
        "quality-config",
        "run",
    ):
        value = {}
        if name == "checkpoint":
            value = {"next_layer": 0}
        if name == "quality-config":
            value = {"reference": str(tmp_path)}
        if name == "run":
            value = {
                "recipe": b.recipe(),
                "calibration_corpus_hashes": [],
                "calibration_token_windows": [],
            }
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    source = SimpleNamespace(args=SimpleNamespace(n_layers=2))
    monkeypatch.setattr(b, "PartialSource", lambda *args: source)
    monkeypatch.setattr(b, "MemoryGuard", lambda *args: nullcontext(None))
    calls = {}
    for module, names in [
        (
            b,
            (
                "initialize",
                "run_layer",
                "assemble",
                "inventory",
                "select_recipe",
                "estimates",
            ),
        ),
        (
            quality,
            (
                "bind_quality",
                "validate_reference",
                "validate_separation",
                "planning_with_quality",
                "execute",
            ),
        ),
        (scan, ("execute",)),
        (amend, ("amend_recipe",)),
    ]:
        for name in names:
            mock = Mock(return_value={"identity": "result"})
            monkeypatch.setattr(module, name, mock)
            calls[f"{module.__name__}.{name}"] = mock
    monkeypatch.setattr(
        b, "calibration_windows", lambda *args: ([np.array([1, 2])], "calibration")
    )
    monkeypatch.setattr(
        b, "read_slices", lambda *args: {"one": [tmp_path / "slices.json"]}
    )

    def run(args):
        monkeypatch.setattr(sys, "argv", ["gptq", *map(str, args)])
        b.main()

    yield tmp_path, run, calls
    mx.set_default_device(old)


def test_inventory_and_default_dispatch(cli, capsys, monkeypatch):
    root, run, calls = cli
    run(["inventory", "--source", root, "--output", root / "out.json"])
    assert json.loads((root / "out.json").read_text()) == {"identity": "result"}
    run(["default-config"])
    assert json.loads(capsys.readouterr().out) == b.DEFAULTS
    monkeypatch.setattr(
        b, "required_shards", lambda _, component: [f"shard-{component}"]
    )
    run(["shards", "--manifest", root / "manifest.json", "--component", "2"])
    assert capsys.readouterr().out.strip() == "shard-2"
    run(["select-recipe", "--parts", root, "--output", root / "selected.json"])
    assert calls[b.__name__ + ".select_recipe"].call_args.kwargs == {"scan": False}
    run(
        [
            "amend-recipe",
            "--parts",
            root,
            "--scratch",
            root,
            "--recipe",
            root / "recipe.json",
            "--from-layer",
            "0",
            "--reason",
            "cap",
        ]
    )
    assert calls[amend.__name__ + ".amend_recipe"].called


@pytest.mark.parametrize("command", ["estimate", "init", "run", "assemble"])
def test_build_commands(cli, command):
    root, run, calls = cli
    args = [command, "--source", root, "--manifest", root / "manifest.json", "--cpu"]
    if command in ("estimate", "init"):
        args += ["--slices", root / "slices.json", "--tokens", "2", "--reference", root]
    if command != "estimate":
        args += ["--parts", root, "--scratch", root]
    if command == "run":
        args += ["--reference", root, "--through-layer", "1"]
    if command == "assemble":
        args += ["--destination", root / "build"]
    run(args)
    name = {
        "estimate": "estimates",
        "init": "initialize",
        "run": "run_layer",
        "assemble": "assemble",
    }[command]
    assert calls[b.__name__ + "." + name].call_count == (2 if command == "run" else 1)


@pytest.mark.parametrize("failure", ["quality", "separation", "range"])
def test_run_rejects_incomplete_or_invalid_state(cli, failure):
    root, run, calls = cli
    if failure == "quality":
        (root / "quality-config.json").unlink()
    if failure == "separation":
        (root / "run.json").write_text("{}")
    with pytest.raises(ValueError):
        run(
            [
                "run",
                "--source",
                root,
                "--manifest",
                root / "manifest.json",
                "--parts",
                root,
                "--scratch",
                root,
                "--layer",
                "8",
            ]
        )
    assert not calls[b.__name__ + ".run_layer"].called


def test_script_entrypoint_default(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gptq", "default-config"])
    runpy.run_path(b.__file__, run_name="__main__")
    assert json.loads(capsys.readouterr().out) == b.DEFAULTS


@pytest.mark.parametrize("command", ["reference", "drift", "milestone", "scan"])
def test_auxiliary_dispatch(cli, command):
    root, run, calls = cli
    args = [command, "--source", root, "--manifest", root / "manifest.json", "--cpu"]
    if command == "scan":
        args += ["--slices", root, "--heldout-slices", root, "--scan", root]
    else:
        args += ["--reference", root]
        if command != "reference":
            args += ["--build", root, "--scratch", root]
        if command == "drift":
            args += ["--layers", "0-1", "--metrics", root]
        if command == "milestone":
            args += ["--last-quantized-layer", "1", "--output", root]
    run(args)
    module = scan if command == "scan" else quality
    assert calls[module.__name__ + ".execute"].call_args.args[0].command == command


@pytest.mark.parametrize("command", ["reference", "drift", "milestone"])
def test_quality_execute(tmp_path, monkeypatch, command):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    source = SimpleNamespace(
        args=SimpleNamespace(n_layers=2),
        manifest={"source_revision": "source"},
        source=tmp_path,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    reference = tmp_path / "reference"
    reference.mkdir()
    args = SimpleNamespace(
        command=command,
        source=tmp_path,
        manifest=manifest,
        recipe=None,
        cpu=True,
        reference=reference,
        prefill_chunk=2,
        tokens_from_cache=None,
        slices=None,
        through_layer=None,
        write_logits=True,
        build=tmp_path,
        baseline=tmp_path,
        layers="0-1",
        scratch=tmp_path,
        metrics=tmp_path / "metrics",
        block_start=0,
        skip_projection_errors=False,
        update_parts_stats=True,
        last_quantized_layer=1,
        output=tmp_path / "output",
        logits_reference=None,
    )
    monkeypatch.setattr(quality, "PartialSource", lambda *a: source)
    monkeypatch.setattr(b, "MemoryGuard", lambda *a: nullcontext(None))
    native = object()
    monkeypatch.setattr(quality, "NativeBuild", lambda *a: native)
    mocks = {}
    for name in (
        "reference_init",
        "reference_run",
        "logit_cache",
        "drift",
        "milestone",
    ):
        mocks[name] = Mock(return_value={})
        monkeypatch.setattr(quality, name, mocks[name])
    try:
        if command == "reference":
            with pytest.raises(ValueError, match="needs"):
                quality.execute(args)
            cache = tmp_path / "cache"
            cache.mkdir()
            (cache / "metadata.json").write_text(
                json.dumps(
                    {"mode": "bad", "model_revision": "source", "slices": {"a": {}}}
                )
            )
            args.tokens_from_cache = cache
            with pytest.raises(ValueError, match="provenance"):
                quality.execute(args)
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "mode": "unpruned_reference",
                        "model_revision": "source",
                        "slices": {"a": {}},
                    }
                )
            )
            np.save(cache / "a-tokens.npy", np.array([1, 2]))
        quality.execute(args)
        assert mocks[
            {"reference": "reference_run", "drift": "drift", "milestone": "milestone"}[
                command
            ]
        ].called
    finally:
        mx.set_default_device(old)


def test_scan_execute(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq_io as io

    old = mx.default_device()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = SimpleNamespace(
        cpu=False,
        recipe=None,
        source=tmp_path,
        manifest=manifest,
        slices=manifest,
        heldout_slices=manifest,
        tokens=2,
        windows_per_slice=1,
        scan=tmp_path,
        through_layer=0,
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Metal"):
        scan.execute(args)
    args.cpu = True
    monkeypatch.setattr(io, "PartialSource", lambda *a: object())
    monkeypatch.setattr(
        scan, "calibration_subset", lambda *a: ([np.array([1, 2])], ["a"], {})
    )
    monkeypatch.setattr(b, "MemoryGuard", lambda *a: nullcontext(None))
    run = Mock()
    monkeypatch.setattr(scan, "run_scan", run)
    try:
        scan.execute(args)
        assert run.called
    finally:
        mx.set_default_device(old)
