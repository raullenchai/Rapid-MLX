# SPDX-License-Identifier: Apache-2.0
"""Offline coverage and selection contract for packaged model digests."""

import json
import runpy
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_model_digests import (
    OUTPUT,
    check,
    check_r2_coverage,
    fetch,
    inventory_hash,
    selections,
)


def test_packaged_digests_cover_every_mirrored_pin_offline():
    check(json.loads(OUTPUT.read_text()), selections())


def test_digest_validation_rejects_incomplete_or_unselected_files():
    selection = {
        "repo": "owner/repo",
        "revision": "a" * 40,
        "allow_patterns": ["runtime/*"],
    }
    file = {"path": "runtime/config.json", "size": 1, "git_oid": "b" * 40}
    payload = {"schema_version": 1, "aliases": {"one": {**selection, "files": [file]}}}
    expected = {"one": selection}
    inventory = {"one": inventory_hash([file])}
    check(payload, expected, inventory)
    with pytest.raises(ValueError, match="lock alias drift"):
        check(payload, expected, {})
    payload["aliases"]["one"]["files"] = []
    with pytest.raises(ValueError, match="empty digest"):
        check(payload, expected, inventory)
    payload["aliases"]["one"]["files"] = [{**file, "git_oid": "wrong"}]
    with pytest.raises(ValueError, match="invalid digest"):
        check(payload, expected, inventory)
    payload["aliases"]["one"]["files"] = [{**file, "path": "docs/README.md"}]
    with pytest.raises(ValueError, match="unselected digest"):
        check(payload, expected, inventory)


def test_fetch_pins_metadata_and_validates_every_digest(monkeypatch):
    siblings = [
        SimpleNamespace(rfilename=".gitattributes"),
        SimpleNamespace(rfilename="README.md"),
        SimpleNamespace(
            rfilename="weights.safetensors",
            size=2,
            lfs=SimpleNamespace(sha256="a" * 64),
            blob_id=None,
        ),
        SimpleNamespace(rfilename="config.json", size=1, lfs=None, blob_id="b" * 40),
    ]
    calls = []

    def metadata(repo, **kwargs):
        calls.append((repo, kwargs))
        return SimpleNamespace(sha="c" * 40, siblings=siblings)

    monkeypatch.setattr("huggingface_hub.model_info", metadata)
    assert [file["path"] for file in fetch("owner/repo", "c" * 40)] == [
        "config.json",
        "weights.safetensors",
    ]
    assert calls == [("owner/repo", {"revision": "c" * 40, "files_metadata": True})]
    with pytest.raises(ValueError, match="revision mismatch"):
        fetch("owner/repo", "d" * 40)
    siblings[2].size = None
    with pytest.raises(ValueError, match="missing size/digest"):
        fetch("owner/repo", "c" * 40)


def test_r2_coverage_checks_unique_selected_objects(monkeypatch, capsys):
    from scripts import build_model_digests

    paths = [f"file-{index}.json" for index in range(100)]
    payload = {
        "aliases": {
            "first": {
                "repo": "owner/repo",
                "files": [{"path": path} for path in paths],
            },
            "duplicate": {"repo": "owner/repo", "files": [{"path": paths[0]}]},
        }
    }
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def urlopen(request, timeout):
        assert request.get_method() == "HEAD"
        assert timeout == 30
        assert request.get_header("User-agent") == "rapid-mlx-r2-coverage/1.0"
        requests.append(request.full_url)
        if request.full_url.endswith("file-10.json"):
            raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        if request.full_url.endswith("file-11.json"):
            raise urllib.error.URLError("unreachable")
        return Response()

    monkeypatch.setattr(build_model_digests.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(build_model_digests.time, "sleep", lambda *_args: None)
    missing = check_r2_coverage(payload)
    assert len(requests) == 100
    assert missing == [
        ("owner/repo", "file-10.json", 404),
        ("owner/repo", "file-11.json", 0),
    ]
    assert "checked 100/100" in capsys.readouterr().out


def test_r2_coverage_retries_rate_limit(monkeypatch):
    from scripts import build_model_digests

    attempts = 0

    def urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "rate limit", {}, None)

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        return Response()

    monkeypatch.setattr(build_model_digests.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(build_model_digests.time, "sleep", lambda *_args: None)
    payload = {"aliases": {"one": {"repo": "a/b", "files": [{"path": "x"}]}}}
    assert check_r2_coverage(payload) == []
    assert attempts == 3


def test_r2_coverage_mode_exits_on_gaps(monkeypatch, capsys):
    from scripts import build_model_digests

    monkeypatch.setattr(build_model_digests.sys, "argv", ["script", "--check-r2"])
    monkeypatch.setattr(build_model_digests, "check_r2_coverage", lambda _: [])
    build_model_digests.main()
    assert "R2 coverage: 0" in capsys.readouterr().out
    monkeypatch.setattr(
        build_model_digests,
        "check_r2_coverage",
        lambda _: [("a/b", "x", 404)],
    )
    with pytest.raises(SystemExit, match="1"):
        build_model_digests.main()


def test_digest_generation_writes_selected_inventory(monkeypatch, tmp_path):
    from scripts import build_model_digests

    output = tmp_path / "digests.json"
    inventory = tmp_path / "inventory.json"
    selection = {
        "repo": "owner/repo",
        "revision": "a" * 40,
        "allow_patterns": ["data/*"],
    }
    files = [
        {"path": "data/model.safetensors", "size": 1, "sha256": "b" * 64},
        {"path": "README.md", "size": 1, "git_oid": "c" * 40},
    ]
    monkeypatch.setattr(build_model_digests, "OUTPUT", output)
    monkeypatch.setattr(build_model_digests, "INVENTORY_LOCK", inventory)
    monkeypatch.setattr(build_model_digests, "selections", lambda: {"one": selection})
    monkeypatch.setattr(build_model_digests, "fetch", lambda *_: files)
    monkeypatch.setattr(sys, "argv", ["script"])
    build_model_digests.main()
    assert [
        item["path"]
        for item in json.loads(output.read_text())["aliases"]["one"]["files"]
    ] == ["data/model.safetensors"]
    assert json.loads(inventory.read_text()) == {"one": inventory_hash(files[:1])}


def test_digest_script_entrypoint_checks_offline(monkeypatch):
    from scripts import build_model_digests

    monkeypatch.setattr(sys, "argv", ["build_model_digests.py", "--check"])
    runpy.run_path(str(Path(build_model_digests.__file__)), run_name="__main__")


def test_runtime_asset_selections_with_pins(monkeypatch, tmp_path):
    from scripts import build_model_digests

    (tmp_path / "rapid_mlx/audio").mkdir(parents=True)
    (tmp_path / "rapid_mlx/models/deepseek_v41_native").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "rapid_mlx/aliases.json").write_text("{}")
    (tmp_path / "rapid_mlx/audio/aliases.json").write_text(
        json.dumps(
            {
                "_runtime_assets": {
                    "voice": [
                        {
                            "repo_id": "owner/voice",
                            "revision": "a" * 40,
                            "allow_patterns": ["*.pt"],
                        }
                    ]
                }
            }
        )
    )
    (tmp_path / "scripts/mirror_unmirrored.json").write_text('{"entries": []}')
    (tmp_path / "rapid_mlx/models/deepseek_v41_native/artifacts.py").write_text(
        "MTP_REPO = 'owner/mtp'\nMTP_REVISION = '"
        + "b" * 40
        + "'\nMTP_ALLOW_PATTERNS = ('*.safetensors',)\n"
    )
    monkeypatch.setattr(build_model_digests, "ROOT", tmp_path)
    selections = build_model_digests.selections()
    assert selections["asset/audio/owner/voice"]["allow_patterns"] == ["*.pt"]
    assert selections["asset/mtp/owner/mtp"]["allow_patterns"] == ["*.safetensors"]


def test_missing_or_changed_alias_pin_fails_offline():
    expected = selections()
    payload = json.loads(OUTPUT.read_text())
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="digest schema"):
        check(payload, expected)
    payload["schema_version"] = 1
    alias = "qwen3-0.6b-4bit"
    payload["aliases"][alias]["revision"] = "f" * 40
    with pytest.raises(ValueError, match="revision or selection drift"):
        check(payload, expected)
    del payload["aliases"][alias]
    with pytest.raises(ValueError, match="alias drift"):
        check(payload, expected)


def test_missing_selected_file_fails_offline():
    payload = json.loads(OUTPUT.read_text())
    files = payload["aliases"]["qwen3-0.6b-4bit"]["files"]
    files.pop()
    with pytest.raises(ValueError, match="incomplete digest inventory"):
        check(payload, selections())


def test_image_runtime_pin_must_match_alias_pin(monkeypatch):
    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS, SDXL_REPO

    monkeypatch.setitem(IMAGE_MODEL_REVISIONS, SDXL_REPO, "f" * 40)
    with pytest.raises(ValueError, match="image runtime pin differs"):
        selections()


def test_subfolder_inventory_contains_only_selected_files():
    payload = json.loads(Path(OUTPUT).read_text())
    entry = payload["aliases"]["lfm2.5-2.6b-4bit"]
    assert entry["allow_patterns"] == ["4bit/*"]
    assert entry["files"]
    assert all(file["path"].startswith("4bit/") for file in entry["files"])


def test_packaged_pin_size_preflight_never_queries_hub(monkeypatch, tmp_path):
    from rapid_mlx import _download_gate, cli
    from rapid_mlx.model_aliases import resolve_revision

    repo = "mlx-community/Qwen3-0.6B-4bit"
    pin = resolve_revision(repo)
    assert pin is not None
    monkeypatch.setattr(
        _download_gate,
        "_model_info_with_timeout",
        lambda *args: pytest.fail("Hub size metadata queried"),
    )
    assert _download_gate.estimate_repo_size_bytes(repo) > 0
    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda *args, **kwargs: pytest.fail("Hub disk metadata queried"),
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    cli._check_disk_space(repo, revision_override=pin)


def test_image_file_selection_disk_preflight_uses_package(monkeypatch, tmp_path):
    from rapid_mlx import cli
    from rapid_mlx._download_gate import (
        IMAGE_MODEL_DATA_FILES,
        IMAGE_MODEL_REVISIONS,
        SDXL_REPO,
    )

    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda *args, **kwargs: pytest.fail("Hub image metadata queried"),
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    cli._check_disk_space(
        SDXL_REPO,
        force=True,
        revision_override=IMAGE_MODEL_REVISIONS[SDXL_REPO],
        allow_patterns=list(IMAGE_MODEL_DATA_FILES[SDXL_REPO]),
    )


def test_explicit_packaged_subfolder_selector_skips_hub(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_revision

    repo = "LiquidAI/LFM2.5-2.6B-MLX"
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        lambda: pytest.fail("Hub variant listing queried"),
    )
    assert cli._resolve_variant_allow_patterns(
        repo, "4", None, resolve_revision(repo)
    ) == ["4bit/*"]
