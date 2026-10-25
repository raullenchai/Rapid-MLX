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
from rapid_mlx.video.wan import WAN_REVISIONS

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


@pytest.fixture(autouse=True)
def _no_live_mirror_state(monkeypatch):
    monkeypatch.setattr(
        mirror.mirror_unmirrored, "load_published_unmirrored", lambda: {}
    )


REPO = "org/pinned"
PIN = "a" * 40
LIST = frozenset({"model_index.json", "unet/model.fp16.safetensors"})


def test_pinned_runtime_files_come_from_the_download_gate():
    assert runtime.pinned_runtime_files(SDXL_REPO) == (
        IMAGE_MODEL_REVISIONS[SDXL_REPO],
        frozenset(IMAGE_MODEL_DATA_FILES[SDXL_REPO]),
    )
    assert runtime.pinned_runtime_files("mlx-community/Qwen3-0.6B-4bit") is None


def test_wan_audit_uses_pinned_revision_and_entire_repository():
    for repo_id, revision in WAN_REVISIONS.items():
        assert runtime.pinned_runtime_files(repo_id) == (revision, None)


def test_direct_script_execution_resolves_the_checkout(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    namespace = runpy.run_path(str(ROOT / "scripts" / "mirror_runtime_files.py"))
    assert sys.path[0] == str(ROOT)
    assert namespace["pinned_runtime_files"](SDXL_REPO) is not None


# --------------------------------------------------------------------------
# Drift audit
# --------------------------------------------------------------------------


NO_RUNTIME_PIN = object()


def _audit(
    monkeypatch,
    tmp_path,
    listing,
    present,
    runtime_list=LIST,
    subfolder=None,
    *,
    hf_error=None,
):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"pinned": {"hf_path": REPO, "subfolder": subfolder}}))
    audio.write_text("{}")
    listed_at = []
    probed = []

    def hf_repo(repo, revision=None):
        listed_at.append((repo, revision))
        if hf_error is not None:
            raise hf_error
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
        lambda repo: (
            (PIN, runtime_list)
            if repo == REPO and runtime_list is not NO_RUNTIME_PIN
            else None
        ),
    )
    monkeypatch.setattr(
        drift.mirror_runtime_files,
        "pinned_revision",
        lambda repo, **kw: PIN if repo == REPO else None,
    )
    [report] = drift.audit(main, audio, unmirrored_path=None)
    return report, listed_at, probed


def test_pinned_whole_repo_audit_checks_every_file(monkeypatch, tmp_path):
    report, listed_at, probed = _audit(
        monkeypatch, tmp_path, LISTING, {item.path for item in LISTING}, None
    )
    assert listed_at == [(REPO, PIN)]
    assert set(probed) == {item.path for item in LISTING}
    assert report.checked_files == len(LISTING)
    assert report.findings == []


def test_pinned_whole_repo_ignores_alias_subfolder(monkeypatch, tmp_path):
    report, listed_at, probed = _audit(
        monkeypatch,
        tmp_path,
        LISTING,
        {"unet/model.fp16.safetensors"},
        None,
        "unet",
    )
    assert listed_at == [(REPO, PIN)]
    assert set(probed) == {item.path for item in LISTING}
    assert {
        finding.path for finding in report.findings if finding.kind == "missing_file"
    } == {
        "model_index.json",
        "unet/model.safetensors",
        "vae_decoder/model.onnx",
    }


def test_pinned_whole_repo_missing_readme_fails(monkeypatch, tmp_path):
    readme = drift.HfFile("README.md", 3, None, "a" * 40)
    report, _, _ = _audit(
        monkeypatch,
        tmp_path,
        [*LISTING, readme],
        {item.path for item in LISTING},
        None,
    )
    assert ("missing_file", "error", "README.md") in {
        (finding.kind, finding.severity, finding.path) for finding in report.findings
    }


@pytest.mark.parametrize(
    ("present", "false_mirrored"),
    [({"model.safetensors"}, False), (set(), True)],
)
def test_pinned_whole_repo_false_mirrored_counts_all_required_files(
    monkeypatch, tmp_path, present, false_mirrored
):
    listing = [
        drift.HfFile("model.safetensors", 3, "b" * 64),
        drift.HfFile("README.md", 3, None, "a" * 40),
    ]
    report, _, _ = _audit(monkeypatch, tmp_path, listing, present, None)
    assert any(f.kind == "false_mirrored" for f in report.findings) == false_mirrored
    assert {f.path for f in report.findings if f.kind == "missing_file"} == (
        {item.path for item in listing} - present
    )


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


@pytest.mark.parametrize("runtime_list", [frozenset({"model_index.json"}), frozenset()])
def test_audit_uses_the_supplied_runtime_list(monkeypatch, tmp_path, runtime_list):
    report, listed_at, probed = _audit(
        monkeypatch, tmp_path, LISTING, set(runtime_list), runtime_list=runtime_list
    )
    assert listed_at == [(REPO, PIN)]
    assert set(probed) == runtime_list
    assert report.checked_files == len(runtime_list)
    assert not any(f.kind == "missing_file" for f in report.findings)


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


def test_pinned_revision_covers_checkpoints_pinned_without_a_file_list():
    whole_repo_pins = set(IMAGE_MODEL_REVISIONS) - set(IMAGE_MODEL_DATA_FILES)
    assert "mflux-community/qwen-image-mflux-q6" in whole_repo_pins
    for repo_id in whole_repo_pins:
        assert runtime.pinned_runtime_files(repo_id) is None
        assert runtime.pinned_revision(repo_id) == IMAGE_MODEL_REVISIONS[repo_id]
    assert runtime.pinned_revision("example/uncataloged-model") is None


def test_audit_lists_a_whole_repo_pin_at_its_commit(monkeypatch, tmp_path):
    # Regression: a checkpoint pinned without a file list was compared with
    # the moving default branch, so an upstream re-upload looked like 20
    # mirror size mismatches although clients never pull that branch.
    paths = {item.path for item in LISTING}
    report, listed_at, probed = _audit(
        monkeypatch, tmp_path, LISTING, paths, runtime_list=NO_RUNTIME_PIN
    )
    assert listed_at == [(REPO, PIN)]
    assert sorted(probed) == sorted(paths)
    assert report.findings == []


def test_audit_checks_an_image_alias_at_its_loader_pin(monkeypatch, tmp_path):
    """A catalog alias without ``revision`` still resolves to the loader pin."""
    repo = "mlx-community/Qwen-Image-2.1-mflux-q4"
    pin = IMAGE_MODEL_REVISIONS[repo]
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"qwen-image": {"hf_path": repo}}))
    audio.write_text("{}")
    listed = []
    probed = []

    def hf_repo(repo_id, revision=None):
        listed.append((repo_id, revision))
        return drift.HfRepo(revision, LISTING)

    def public_probe(_repo, item):
        probed.append(item.path)
        return drift.MirrorProbe(200, item.size, f'"{item.sha256}"', item.oid)

    monkeypatch.setattr(drift, "_hf_repo", hf_repo)
    monkeypatch.setattr(drift, "_public_probe", public_probe)
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "qwen-image", "hf_path": repo, "status": "mirrored"}],
    )
    [report] = drift.audit(main, audio, unmirrored_path=None)
    assert listed == [(repo, pin)]
    assert sorted(probed) == sorted(item.path for item in LISTING)
    assert report.findings == []


def _revision_not_found(message: str) -> Exception:
    import httpx
    from huggingface_hub.errors import RevisionNotFoundError

    request = httpx.Request("GET", "https://huggingface.co/api/models/org/pinned")
    return RevisionNotFoundError(message, response=httpx.Response(404, request=request))


def test_audit_keeps_other_pinned_listing_errors_generic(monkeypatch, tmp_path):
    report, _, probed = _audit(
        monkeypatch,
        tmp_path,
        LISTING,
        set(),
        runtime_list=NO_RUNTIME_PIN,
        hf_error=TimeoutError("read timed out"),
    )
    assert probed == []
    assert [(f.kind, f.severity, f.detail) for f in report.findings] == [
        ("hf_unavailable", "error", "read timed out")
    ]


def test_audit_fails_when_the_pinned_commit_vanishes_upstream(monkeypatch, tmp_path):
    report, listed_at, probed = _audit(
        monkeypatch,
        tmp_path,
        LISTING,
        set(),
        runtime_list=NO_RUNTIME_PIN,
        hf_error=_revision_not_found("404 Client Error: Invalid rev id"),
    )
    assert listed_at == [(REPO, PIN)]
    assert probed == []
    assert [(f.kind, f.severity, f.detail) for f in report.findings] == [
        (
            "hf_unavailable",
            "error",
            f"pinned runtime revision {PIN} unavailable upstream: "
            "404 Client Error: Invalid rev id",
        )
    ]


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


def test_generic_catalog_pin_is_shared_by_tools():
    assert (
        runtime.pinned_revision("RecursiveSI/flint-int4")
        == "3b93f9053b530488405ae9fca51d4900d6ce21f0"
    )


@pytest.mark.parametrize("revision", ["main", "A" * 40, "abc", 12])
def test_audit_rejects_invalid_catalog_pin(tmp_path, revision):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"example": {"hf_path": REPO, "revision": revision}}))
    audio.write_text("{}")
    with pytest.raises(ValueError, match="invalid revision"):
        drift._load_aliases(main, audio)


@pytest.mark.parametrize("other", [None, "b" * 40])
def test_audit_rejects_disagreeing_catalog_pins(tmp_path, other):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(
        json.dumps(
            {
                "one": {"hf_path": REPO, "revision": PIN},
                "two": {"hf_path": REPO.upper(), "revision": other},
            }
        )
    )
    audio.write_text("{}")
    with pytest.raises(ValueError, match="different revisions"):
        drift._load_aliases(main, audio)


def test_audit_uses_supplied_catalog_pin_instead_of_head(monkeypatch, tmp_path):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"example": {"hf_path": REPO, "revision": PIN}}))
    audio.write_text("{}")
    assert drift._load_aliases(main, audio)[0].revision == PIN
    listed = []
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "example", "hf_path": REPO, "status": "mirrored"}],
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift.mirror_runtime_files, "pinned_revision", lambda _, **kw: None
    )
    monkeypatch.setattr(
        drift.mirror_runtime_files, "pinned_runtime_files", lambda _: None
    )
    monkeypatch.setattr(
        drift,
        "_hf_repo",
        lambda repo, revision: (
            listed.append((repo, revision))
            or drift.HfRepo(revision, [drift.HfFile("config.json", 3, "c" * 64)])
        ),
    )
    monkeypatch.setattr(
        drift,
        "_public_probe",
        lambda *a: drift.MirrorProbe(200, 3, '"' + "c" * 64 + '"', None),
    )
    [report] = drift.audit(main, audio, unmirrored_path=None)
    assert listed == [(REPO, PIN)]
    assert report.findings == []


@pytest.mark.parametrize("explicit", [None, PIN, "b" * 40])
def test_uploader_defaults_to_catalog_and_rejects_conflict(
    monkeypatch, tmp_path, explicit
):
    monkeypatch.setattr(mirror.mirror_runtime_files, "catalog_revision", lambda _: PIN)
    listed = []
    monkeypatch.setattr(mirror, "_r2_client", lambda *a: object())
    monkeypatch.setattr(
        mirror, "_hf_files", lambda repo, revision: listed.append(revision) or []
    )
    if explicit and explicit != PIN:
        with pytest.raises(ValueError, match="conflicts with pinned revision"):
            mirror.mirror_repo(REPO, tmp_dir=tmp_path, revision=explicit, dry_run=True)
        assert listed == []
    else:
        mirror.mirror_repo(REPO, tmp_dir=tmp_path, revision=explicit, dry_run=True)
        assert listed == [PIN]


IMAGE_PIN_REPO = "mlx-community/Qwen-Image-2.1-mflux-q4"
IMAGE_PIN = IMAGE_MODEL_REVISIONS[IMAGE_PIN_REPO]


def _run_uploader(monkeypatch, tmp_path, repo, *, revision=None, files=FILES):
    """Run the uploader against mocked HF and R2; return listings/downloads."""
    listed: list[tuple] = []
    downloaded: list[tuple[str, tuple]] = []
    local = tmp_path / "blob"
    local.write_bytes(b"x" * 10)
    monkeypatch.setattr(
        mirror,
        "_hf_files",
        lambda repo_id, *rest: listed.append(rest) or list(files),
    )
    monkeypatch.setattr(
        mirror,
        "_download_one_hf",
        lambda repo_id, relpath, tmp, *rest: (
            downloaded.append((relpath, rest)) or local
        ),
    )
    monkeypatch.setattr(mirror, "_r2_client", lambda *_a: object())
    monkeypatch.setattr(mirror, "_r2_head", lambda *_a: None)
    monkeypatch.setattr(mirror, "_upload_one", lambda *_a, **_k: None)
    monkeypatch.setattr(mirror, "_r2_head_size", lambda *_a: 10)
    monkeypatch.setattr(mirror, "_http_range_get_status", lambda _url: 200)
    monkeypatch.setattr(mirror, "_http_head_status", lambda _url: 200)
    code = mirror.mirror_repo(repo, tmp_dir=tmp_path, revision=revision)
    return code, listed, downloaded


def test_uploader_mirrors_an_image_repo_at_its_loader_pin(monkeypatch, tmp_path):
    """Only the loader pins this repository; ``main`` may move under it."""
    assert runtime.catalog_revision(IMAGE_PIN_REPO) == IMAGE_PIN
    assert runtime.pinned_revision(IMAGE_PIN_REPO) == IMAGE_PIN
    code, listed, downloaded = _run_uploader(monkeypatch, tmp_path, IMAGE_PIN_REPO)
    assert code == 0
    assert listed == [(IMAGE_PIN,)]
    assert downloaded == [(f.relpath, (IMAGE_PIN,)) for f in FILES]


def test_uploader_accepts_an_explicit_revision_at_the_loader_pin(monkeypatch, tmp_path):
    code, listed, downloaded = _run_uploader(
        monkeypatch, tmp_path, IMAGE_PIN_REPO, revision=IMAGE_PIN
    )
    assert code == 0
    assert listed == [(IMAGE_PIN,)]
    assert {rest for _, rest in downloaded} == {(IMAGE_PIN,)}


def test_uploader_rejects_an_explicit_revision_off_the_loader_pin(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(mirror, "_hf_files", lambda *_a: pytest.fail("listed"))
    with pytest.raises(ValueError, match="conflicts with pinned revision"):
        mirror.mirror_repo(IMAGE_PIN_REPO, tmp_dir=tmp_path, revision="b" * 40)


def test_uploader_keeps_a_revision_pinned_alias_at_its_catalog_pin(
    monkeypatch, tmp_path
):
    repo = "RecursiveSI/flint-int4"
    pin = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
    assert runtime.pinned_revision(repo) == pin
    code, listed, downloaded = _run_uploader(monkeypatch, tmp_path, repo)
    assert code == 0
    assert listed == [(pin,)]
    assert {rest for _, rest in downloaded} == {(pin,)}


def test_uploader_explicit_revision_still_selects_an_unpinned_repo(
    monkeypatch, tmp_path
):
    repo = "example/uncataloged-model"
    assert runtime.pinned_revision(repo) is None
    code, listed, downloaded = _run_uploader(monkeypatch, tmp_path, repo, revision=PIN)
    assert code == 0
    assert listed == [(PIN,)]
    assert {rest for _, rest in downloaded} == {(PIN,)}


@pytest.mark.parametrize("catalog_revision", [None, PIN])
def test_alternate_audit_catalog_is_authoritative(
    monkeypatch, tmp_path, catalog_revision
):
    repo = "RecursiveSI/flint-int4"
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(
        json.dumps({"example": {"hf_path": repo, "revision": catalog_revision}})
    )
    audio.write_text("{}")
    monkeypatch.setattr(drift, "_catalog_entries", lambda: [])
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    listed = []
    monkeypatch.setattr(
        drift,
        "_hf_repo",
        lambda repo, revision=None: (
            listed.append(revision) or drift.HfRepo(revision, [])
        ),
    )
    drift.audit(main, audio, unmirrored_path=None)
    assert listed == [catalog_revision]


def test_audit_rejects_a_catalog_runtime_pin_conflict(monkeypatch, tmp_path):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"example": {"hf_path": REPO, "revision": PIN}}))
    audio.write_text("{}")
    monkeypatch.setattr(drift, "_catalog_entries", lambda: [])
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift.mirror_runtime_files, "pinned_revision", lambda repo, **kw: "b" * 40
    )
    with pytest.raises(ValueError, match="conflicts with runtime pin"):
        drift.audit(main, audio, unmirrored_path=None)
