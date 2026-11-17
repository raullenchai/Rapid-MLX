# SPDX-License-Identifier: Apache-2.0
"""Pinned DFlash drafters must not need Hub metadata while offline."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

REPO = "z-lab/Qwen3.8-27B-DFlash2"
REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"


@pytest.fixture
def drafter_cache(tmp_path, monkeypatch):
    from huggingface_hub import constants

    cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    # Hub constants may already be imported by another test.
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    snapshot = cache / ("models--" + REPO.replace("/", "--")) / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "dflash2"}')
    (snapshot / "model.safetensors").write_bytes(b"cached weights")
    return snapshot


@pytest.fixture
def runtime_stubs(monkeypatch):
    from rapid_mlx.speculative.dflash import runtime

    resolve = Mock(side_effect=AssertionError("unexpected Hub resolution"))
    load = Mock(
        return_value=(SimpleNamespace(config={"model_type": "dflash2"}), "dflash")
    )
    for name in (
        "mlx_vlm",
        "mlx_vlm.speculative",
        "mlx_vlm.speculative.drafters",
        "mlx_vlm.utils",
        "rapid_mlx.models.mlx_vlm_vendored.speculative.drafters",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        sys.modules["mlx_vlm.utils"], "get_model_path", resolve, raising=False
    )
    monkeypatch.setattr(
        sys.modules["rapid_mlx.models.mlx_vlm_vendored.speculative.drafters"],
        "load_drafter",
        load,
        raising=False,
    )
    monkeypatch.setattr(runtime, "have_runtime", lambda: True)
    # Any accidental Hub entry is a test failure; no real network is used.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", resolve)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", resolve)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", resolve)
    return runtime, resolve, load


@pytest.mark.parametrize("refs", ["absent", "stale"])
@pytest.mark.parametrize("layout", ["copied", "symlinks"])
def test_complete_pinned_drafter_offline(drafter_cache, runtime_stubs, refs, layout):
    runtime, resolve, load = runtime_stubs
    if layout == "symlinks":
        blobs = drafter_cache.parent.parent / "blobs"
        blobs.mkdir()
        for name in ("config.json", "model.safetensors"):
            leaf = drafter_cache / name
            leaf.rename(blobs / name)
            leaf.symlink_to("../../blobs/" + name)
    if refs == "stale":
        ref = drafter_cache.parent.parent / "refs" / "main"
        ref.parent.mkdir()
        ref.write_text("a" * 40)
        other = drafter_cache.parent / ("a" * 40)
        other.mkdir()
        (other / "config.json").write_text("{}")
        (other / "model.safetensors").write_bytes(b"other weights")
    loaded = runtime.load_runtime(
        REPO, drafter_revision=REVISION, expected_algorithm="dflash2"
    )
    assert loaded.drafter_path == str(drafter_cache)
    assert loaded.drafter_revision == REVISION
    load.assert_called_once_with(str(drafter_cache), kind="dflash")
    resolve.assert_not_called()


@pytest.mark.parametrize("damage", ["missing", "config", "weights", "broken", "shard"])
def test_incomplete_pinned_drafter_offline(drafter_cache, runtime_stubs, damage):
    from rapid_mlx.speculative.dflash.eligibility import DFlashUnavailable

    runtime, resolve, load = runtime_stubs
    if damage == "config":
        (drafter_cache / "config.json").unlink()
    else:
        (drafter_cache / "model.safetensors").unlink()
        if damage == "missing":
            (drafter_cache / "config.json").unlink()
            drafter_cache.rmdir()
        elif damage == "broken":
            (drafter_cache / "model.safetensors").symlink_to("missing-blob")
        elif damage == "shard":
            (drafter_cache / "model-00001-of-00002.safetensors").write_bytes(b"shard")
            (drafter_cache / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                )
            )
    with pytest.raises(DFlashUnavailable) as exc:
        runtime.load_runtime(REPO, drafter_revision=REVISION)
    message = str(exc.value)
    assert REPO in message and REVISION in message
    assert "offline" in message.lower()
    assert "download" in message.lower() and "HF_HUB_OFFLINE" in message
    resolve.assert_not_called()
    load.assert_not_called()


def test_online_drafter_resolution_unchanged(drafter_cache, runtime_stubs, monkeypatch):
    runtime, resolve, load = runtime_stubs
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    resolve.side_effect = None
    resolve.return_value = "/resolved/drafter"
    runtime.load_runtime(REPO, drafter_revision=REVISION)
    resolve.assert_called_once_with(REPO, revision=REVISION)
    load.assert_called_once_with("/resolved/drafter", kind="dflash")


def test_offline_startup_error_has_no_traceback(
    drafter_cache, runtime_stubs, monkeypatch, caplog
):
    from rapid_mlx.speculative.dflash import server

    runtime, resolve, load = runtime_stubs
    (drafter_cache / "model.safetensors").unlink()

    def load_runtime(**kwargs):
        return runtime.load_runtime(REPO, drafter_revision=REVISION)

    monkeypatch.setattr(server, "load_dflash_runtime", load_runtime)
    with pytest.raises(SystemExit) as exc:
        server.run_dflash_server(
            main_model_repo="test/target",
            drafter_repo=REPO,
            drafter_revision=REVISION,
            host="127.0.0.1",
            port=8000,
            served_model_name="target",
            default_max_tokens=32,
            cors_origins=[],
            uvicorn_log_level="error",
        )
    assert exc.value.code == 1
    assert REPO in caplog.text and REVISION in caplog.text
    assert "Download" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert "Traceback" not in caplog.text
    resolve.assert_not_called()
    load.assert_not_called()
