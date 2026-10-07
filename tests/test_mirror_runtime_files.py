# SPDX-License-Identifier: Apache-2.0
"""Mirror tooling for repositories downloaded at a pinned file list.

SDXL's runtime pulls 19 files (6.9 GB) of a 77 GB repository at an exact
commit. The uploader must be able to mirror exactly those bytes, and the drift
audit must judge the mirror against exactly that list at exactly that commit.
"""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from rapid_mlx._download_gate import (
    IMAGE_MODEL_DATA_FILES,
    IMAGE_MODEL_REVISIONS,
    SDXL_REPO,
)

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

REPO = "org/pinned"
PIN = "a" * 40
LIST = frozenset({"model_index.json", "unet/model.fp16.safetensors"})


def test_pinned_runtime_files_come_from_the_download_gate():
    assert runtime.pinned_runtime_files(SDXL_REPO) == (
        IMAGE_MODEL_REVISIONS[SDXL_REPO],
        frozenset(IMAGE_MODEL_DATA_FILES[SDXL_REPO]),
    )
    assert runtime.pinned_runtime_files("mlx-community/Qwen3-0.6B-4bit") is None


def test_direct_script_execution_resolves_the_checkout(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    namespace = runpy.run_path(str(ROOT / "scripts" / "mirror_runtime_files.py"))
    assert sys.path[0] == str(ROOT)
    assert namespace["pinned_runtime_files"](SDXL_REPO) is not None


# --------------------------------------------------------------------------
# Drift audit
# --------------------------------------------------------------------------


def _audit(monkeypatch, tmp_path, listing, present):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"pinned": {"hf_path": REPO}}))
    audio.write_text("{}")
    listed_at = []
    probed = []

    def hf_repo(repo, revision=None):
        listed_at.append((repo, revision))
        return drift.HfRepo(revision, listing)

    def public_probe(_repo, item):
        probed.append(item.path)
        if item.path not in present:
            return drift.MirrorProbe(404, None, None, None)
        return drift.MirrorProbe(200, item.size, f'"{item.sha256}"', item.oid)

    monkeypatch.setattr(drift, "_hf_repo", hf_repo)
    monkeypatch.setattr(drift, "_public_probe", public_probe)
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "pinned", "hf_path": REPO, "status": "mirrored"}],
    )
    monkeypatch.setattr(
        drift.mirror_runtime_files,
        "pinned_runtime_files",
        lambda repo: (PIN, LIST) if repo == REPO else None,
    )
    [report] = drift.audit(main, audio, unmirrored_path=None)
    return report, listed_at, probed


LISTING = [
    drift.HfFile("model_index.json", 2, None, "x" * 40),
    drift.HfFile("unet/model.fp16.safetensors", 3, "b" * 64),
    drift.HfFile("unet/model.safetensors", 6, "c" * 64),
    drift.HfFile("vae_decoder/model.onnx", 5, "d" * 64),
]


def test_audit_reads_the_pin_and_probes_only_the_runtime_list(monkeypatch, tmp_path):
    report, listed_at, probed = _audit(monkeypatch, tmp_path, LISTING, set(LIST))
    assert listed_at == [(REPO, PIN)]
    assert sorted(probed) == sorted(LIST)
    assert report.findings == []


def test_audit_still_fails_a_missing_runtime_object(monkeypatch, tmp_path):
    report, _, _ = _audit(monkeypatch, tmp_path, LISTING, {"model_index.json"})
    assert [(f.kind, f.severity, f.path) for f in report.findings] == [
        ("missing_file", "error", "unet/model.fp16.safetensors")
    ]


def test_audit_fails_a_runtime_file_absent_upstream_at_the_pin(monkeypatch, tmp_path):
    report, _, _ = _audit(monkeypatch, tmp_path, LISTING[:1], {"model_index.json"})
    assert [(f.kind, f.severity, f.path) for f in report.findings] == [
        ("hf_missing_runtime_file", "error", "unet/model.fp16.safetensors")
    ]


def test_hf_repo_lists_a_pinned_revision(monkeypatch):
    seen = []
    monkeypatch.setattr(
        drift,
        "_model_info",
        lambda repo, revision=None: (
            seen.append((repo, revision))
            or type("I", (), {"siblings": [], "sha": PIN})()
        ),
    )
    monkeypatch.setattr(drift, "_throttle_hf", lambda: None)
    assert drift._hf_repo(REPO, PIN).revision == PIN
    assert drift._hf_repo(REPO).revision == PIN
    assert seen == [(REPO, PIN), (REPO, None)]


# --------------------------------------------------------------------------
# Uploader
# --------------------------------------------------------------------------


def _f(relpath: str, size: int = 10) -> Any:
    return mirror.FileMeta(relpath=relpath, size=size, key=f"{REPO}/{relpath}")


FILES = [
    _f(".gitattributes"),
    _f("LICENSE.md"),
    _f("README.md"),
    _f("model_index.json"),
    _f("unet/model.fp16.safetensors"),
    _f("unet/model.safetensors"),
]


def test_include_keeps_matches_and_root_terms():
    selected = mirror._select_included(FILES, ["model_index.json", "unet/*.fp16.*"])
    assert [f.relpath for f in selected] == [
        "LICENSE.md",
        "README.md",
        "model_index.json",
        "unet/model.fp16.safetensors",
    ]


@pytest.mark.parametrize(
    ("include", "message"),
    [
        (["vae/config.json"], "matched no repository files"),
        (["nothing/*"], "matched no repository files"),
        (["model_index.json", "unet/typo*"], r"\['unet/typo\*'\]"),
    ],
)
def test_include_refuses_an_incomplete_or_empty_selection(include, message):
    with pytest.raises(ValueError, match=message):
        mirror._select_included(FILES, include)


def test_mirror_repo_lists_and_downloads_at_the_revision(monkeypatch, tmp_path):
    listed = []
    downloaded = []
    local = tmp_path / "blob"
    local.write_bytes(b"x" * 10)
    monkeypatch.setattr(
        mirror, "_hf_files", lambda repo, *rest: listed.append(rest) or list(FILES)
    )
    monkeypatch.setattr(
        mirror,
        "_download_one_hf",
        lambda repo, relpath, tmp, *rest: downloaded.append((relpath, rest)) or local,
    )
    monkeypatch.setattr(mirror, "_r2_client", lambda *_a: object())
    monkeypatch.setattr(mirror, "_r2_head", lambda *_a: None)
    monkeypatch.setattr(mirror, "_upload_one", lambda *_a, **_k: None)
    monkeypatch.setattr(mirror, "_r2_head_size", lambda *_a: 10)
    monkeypatch.setattr(mirror, "_http_range_get_status", lambda _url: 200)
    monkeypatch.setattr(mirror, "_http_head_status", lambda _url: 200)

    assert (
        mirror.mirror_repo(
            REPO, tmp_dir=tmp_path, revision=PIN, include=["model_index.json"]
        )
        == 0
    )
    assert listed == [(PIN,)]
    assert downloaded == [
        ("LICENSE.md", (PIN,)),
        ("README.md", (PIN,)),
        ("model_index.json", (PIN,)),
    ]
    args = mirror._build_parser().parse_args(
        [REPO, "--revision", PIN, "--include", "a", "--include", "b"]
    )
    assert (args.revision, args.include) == (PIN, ["a", "b"])


def test_hf_helpers_forward_the_revision(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []

    class Api:
        def model_info(self, repo, **kwargs):
            calls.append(("info", kwargs))
            return type("I", (), {"siblings": []})()

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: calls.append(("dl", kwargs.get("revision"))) or str(tmp_path),
    )
    mirror._hf_files(REPO)
    mirror._hf_files(REPO, PIN)
    mirror._download_one_hf(REPO, "a", tmp_path)
    mirror._download_one_hf(REPO, "a", tmp_path, PIN)
    assert calls == [
        ("info", {"files_metadata": True}),
        ("info", {"revision": PIN, "files_metadata": True}),
        ("dl", None),
        ("dl", PIN),
    ]


def test_revision_must_be_an_immutable_commit(monkeypatch):
    monkeypatch.setattr(mirror, "_hf_files", lambda *_a: pytest.fail("listed"))
    with pytest.raises(ValueError, match="40-hex commit"):
        mirror.mirror_repo(REPO, revision="main")


def test_pinned_runtime_list_overrides_an_alias_subfolder(monkeypatch, tmp_path):
    """A pinned pull fetches its declared list whatever the catalog subfolder
    says, so the audit probes every listed file and checks upstream existence
    against the whole pinned listing."""
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"pinned": {"hf_path": REPO, "subfolder": "unet"}}))
    audio.write_text("{}")
    monkeypatch.setattr(
        drift, "_hf_repo", lambda repo, revision=None: drift.HfRepo(revision, LISTING)
    )
    monkeypatch.setattr(
        drift,
        "_public_probe",
        lambda _r, item: drift.MirrorProbe(
            200, item.size, f'"{item.sha256}"', item.oid
        ),
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "pinned", "hf_path": REPO, "status": "mirrored"}],
    )
    monkeypatch.setattr(
        drift.mirror_runtime_files,
        "pinned_runtime_files",
        lambda repo: (PIN, LIST) if repo == REPO else None,
    )
    [report] = drift.audit(main, audio, unmirrored_path=None)
    assert report.findings == []
    assert report.checked_files == len(LIST)
