# SPDX-License-Identifier: Apache-2.0
"""Mirror tooling scoped to a loader's fixed runtime file list (#4141).

A vendored image backend pulls an audited file list, not the whole repository.
SDXL's repository holds 51 runtime-relevant-looking files (~77 GB) but a pull
fetches 19 of them (~6.9 GB). The uploader must store exactly those, and the
drift audit must judge the mirror against exactly those — otherwise the audit
demands ~70 GB of weights no pull ever requests.
"""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from rapid_mlx._download_gate import IMAGE_MODEL_DATA_FILES, SDXL_REPO

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load(
    "scripts.mirror_runtime_files", ROOT / "scripts" / "mirror_runtime_files.py"
)
drift = _load("scripts.mirror_drift_check", ROOT / "scripts" / "mirror_drift_check.py")
mirror = _load("scripts.mirror_to_r2", ROOT / "scripts" / "mirror_to_r2.py")

PINNED = "org/pinned"
RUNTIME = frozenset({"model_index.json", "unet/model.fp16.safetensors"})


def test_runtime_files_come_from_the_download_gate():
    assert runtime.runtime_files(SDXL_REPO) == frozenset(
        IMAGE_MODEL_DATA_FILES[SDXL_REPO]
    )
    assert runtime.runtime_files("mlx-community/Qwen3-0.6B-4bit") is None


def test_direct_script_execution_resolves_the_checkout(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    namespace = runpy.run_path(str(ROOT / "scripts" / "mirror_runtime_files.py"))
    assert sys.path[0] == str(ROOT)
    assert namespace["runtime_files"](SDXL_REPO) is not None


# --------------------------------------------------------------------------
# Drift audit
# --------------------------------------------------------------------------


def _audit_pinned(monkeypatch, tmp_path, present: set[str]):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"pinned": {"hf_path": PINNED}}))
    audio.write_text("{}")
    listing = [
        drift.HfFile("model_index.json", 2, None, "a" * 40),
        drift.HfFile("unet/model.fp16.safetensors", 3, "b" * 64),
        drift.HfFile("unet/model.safetensors", 6, "c" * 64),
        drift.HfFile("vae_decoder/model.onnx", 5, "d" * 64),
    ]
    probed = []

    def public_probe(repo, item):
        probed.append(item.path)
        if item.path not in present:
            return drift.MirrorProbe(404, None, None, None)
        oid = item.oid if item.sha256 is None else None
        etag = f'"{item.sha256}"' if item.sha256 else '"etag"'
        return drift.MirrorProbe(200, item.size, etag, oid)

    monkeypatch.setattr(drift, "_hf_repo", lambda _repo: drift.HfRepo("rev", listing))
    monkeypatch.setattr(drift, "_public_probe", public_probe)
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "pinned", "hf_path": PINNED, "status": "mirrored"}],
    )
    monkeypatch.setattr(
        drift.mirror_runtime_files,
        "runtime_files",
        lambda repo: RUNTIME if repo == PINNED else None,
    )
    [report] = drift.audit(main, audio, unmirrored_path=None)
    return report, probed


def test_audit_judges_a_pinned_repo_by_its_runtime_files_only(monkeypatch, tmp_path):
    report, probed = _audit_pinned(monkeypatch, tmp_path, set(RUNTIME))
    assert sorted(probed) == sorted(RUNTIME)
    assert report.checked_files == len(RUNTIME)
    assert report.findings == []


def test_audit_still_fails_a_missing_runtime_file(monkeypatch, tmp_path):
    report, _probed = _audit_pinned(monkeypatch, tmp_path, {"model_index.json"})
    assert [(f.kind, f.severity, f.path) for f in report.findings] == [
        ("missing_file", "error", "unet/model.fp16.safetensors")
    ]


# --------------------------------------------------------------------------
# Uploader
# --------------------------------------------------------------------------


def _f(relpath: str, size: int = 10) -> Any:
    return mirror.FileMeta(relpath=relpath, size=size, key=f"{PINNED}/{relpath}")


LISTING = [
    _f(".gitattributes"),
    _f("LICENSE.md"),
    _f("README.md"),
    _f("model_index.json"),
    _f("unet/model.fp16.safetensors", 3_000),
    _f("unet/model.safetensors", 6_000),
    _f("vae_decoder/model.onnx", 5_000),
]


def test_select_runtime_files_keeps_the_list_and_root_terms():
    selected = mirror._select_runtime_files(LISTING, RUNTIME)
    assert [f.relpath for f in selected] == [
        "LICENSE.md",
        "README.md",
        "model_index.json",
        "unet/model.fp16.safetensors",
    ]


def test_select_runtime_files_refuses_an_incomplete_upstream():
    with pytest.raises(ValueError, match="does not have"):
        mirror._select_runtime_files(LISTING, RUNTIME | {"vae/config.json"})


def test_mirror_repo_uploads_only_the_runtime_set(monkeypatch, capsys):
    monkeypatch.setattr(mirror, "_hf_files", lambda _repo: list(LISTING))
    monkeypatch.setattr(
        mirror.mirror_runtime_files,
        "runtime_files",
        lambda repo: RUNTIME if repo == PINNED else None,
    )
    monkeypatch.setattr(mirror, "_r2_client", lambda *_args: object())
    sizes = {f.key: f.size for f in LISTING}
    checked = []
    monkeypatch.setattr(
        mirror,
        "_r2_head_size",
        lambda _client, _bucket, key: checked.append(key) or sizes[key],
    )
    monkeypatch.setattr(mirror, "_http_range_get_status", lambda _url: 200)
    monkeypatch.setattr(mirror, "_http_head_status", lambda _url: 200)

    assert mirror.mirror_repo(PINNED, verify_only=True) == 0
    assert sorted(checked) == sorted(
        f"{PINNED}/{name}" for name in ("LICENSE.md", "README.md", *RUNTIME)
    )
    assert "runtime:  4/7 files the loader fetches" in capsys.readouterr().out


def test_mirror_repo_subfolder_takes_precedence_over_runtime_list(monkeypatch):
    """An explicit ``--subfolder`` is the operator's scope; no list lookup."""
    lookups = []
    monkeypatch.setattr(
        mirror.mirror_runtime_files,
        "runtime_files",
        lambda repo: lookups.append(repo) or RUNTIME,
    )
    monkeypatch.setattr(
        mirror,
        "_hf_files",
        lambda _repo: [_f("4bit/config.json"), _f("4bit/model.safetensors")],
    )
    monkeypatch.setattr(mirror, "_r2_client", lambda *_args: object())
    monkeypatch.setattr(mirror, "_r2_head_size", lambda *_args: 10)
    monkeypatch.setattr(mirror, "_http_range_get_status", lambda _url: 200)
    monkeypatch.setattr(mirror, "_http_head_status", lambda _url: 200)
    assert mirror.mirror_repo(PINNED, verify_only=True, subfolder="4bit") == 0
    assert lookups == []
