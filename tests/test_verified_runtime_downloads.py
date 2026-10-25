"""Runtime downloads must not expose unverified or mutable model assets."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from rapid_mlx import model_downloads
from rapid_mlx.model_integrity import ModelIntegrityError

SHA = "a" * 40


@pytest.fixture
def asset(monkeypatch, tmp_path):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    root = tmp_path / "snapshots" / SHA
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"ok":true}')
    data = (root / "config.json").read_bytes()
    sibling = SimpleNamespace(
        rfilename="config.json",
        size=len(data),
        blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
        lfs=None,
    )
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        lambda *a, **k: SimpleNamespace(sha=SHA, siblings=[sibling]),
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(root))
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: str(root / a[1])
    )
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_revision", lambda _: SHA)
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_hub_model", lambda x: x)
    model_downloads.configure(False)
    yield root
    model_downloads.configure(False)


@pytest.mark.parametrize("file", [False, True])
def test_download_verifies_bytes_and_rejects_tampered_cache(asset, file):
    def download():
        if file:
            return model_downloads.hf_hub_download("fixture/model", "config.json")
        return model_downloads.snapshot_download("fixture/model")

    assert Path(download()).exists()
    (asset / "config.json").write_text("corrupt")
    with pytest.raises(ModelIntegrityError):
        download()


@pytest.mark.parametrize("revision", [None, "main", "v1", "abc", "A" * 40])
def test_unknown_or_mutable_revision_refused_before_transfer(monkeypatch, revision):
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_revision", lambda _: None)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: pytest.fail("network called"),
    )
    with pytest.raises(ModelIntegrityError, match="immutable revision"):
        model_downloads.snapshot_download("fixture/unknown", revision=revision)


def test_conflicting_catalog_pin_refused(asset):
    with pytest.raises(ValueError, match="conflicting revision"):
        model_downloads.snapshot_download("fixture/model", revision="b" * 40)


def test_explicit_and_auxiliary_pins(asset, monkeypatch):
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_revision", lambda _: None)
    assert model_downloads.snapshot_download("fixture/model", revision=SHA) == str(
        asset
    )
    monkeypatch.setitem(
        __import__(
            "rapid_mlx.runtime_model_pins", fromlist=["AUXILIARY_REVISIONS"]
        ).AUXILIARY_REVISIONS,
        "fixture/aux",
        SHA,
    )
    assert model_downloads.snapshot_download("fixture/aux") == str(asset)


def test_disabled_uses_receipt_without_network(asset, monkeypatch):
    model_downloads.snapshot_download("fixture/model")
    model_downloads.configure(True)

    def local(repo, **kwargs):
        assert kwargs["local_files_only"] is True
        return str(asset)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", local)
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        lambda *a, **k: pytest.fail("network called"),
    )
    assert model_downloads.snapshot_download("fixture/model") == str(asset)


@pytest.mark.parametrize("file", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_missing_cache_preserves_policy(asset, monkeypatch, file, disabled):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def missing(*args, **kwargs):
        raise LocalEntryNotFoundError("missing")

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download" if file else "snapshot_download", missing
    )
    model_downloads.configure(disabled)
    error = (
        model_downloads.ModelDownloadsDisabledError
        if disabled
        else LocalEntryNotFoundError
    )
    with pytest.raises(error):
        if file:
            model_downloads.hf_hub_download("fixture/model", "config.json")
        else:
            model_downloads.snapshot_download("fixture/model")


def test_unverified_offline_cache_refused(asset):
    with pytest.raises(ModelIntegrityError, match="Offline integrity"):
        model_downloads.snapshot_download("fixture/model", local_files_only=True)


def test_single_file_subfolder_and_local_dir(asset, monkeypatch):
    captured = []
    monkeypatch.setattr(
        "rapid_mlx.model_integrity.verify_snapshot",
        lambda *a, **k: captured.append((a, k)),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: str(asset / "nested/config.json"),
    )
    model_downloads.hf_hub_download(
        "fixture/model",
        "config.json",
        subfolder="nested",
        local_dir=str(asset),
        token="fixture",
    )
    args, kwargs = captured.pop()
    assert args == ("fixture/model", asset)
    assert kwargs["filenames"] == ["nested/config.json"]
    assert kwargs["resolved_revision"] == SHA
    assert kwargs["token"] == "fixture"


# Raw Hub imports are deliberately restricted to these transport boundaries.
# Explicit import/preflight are not runtime loads; CLI pull verifies itself.
RAW_HUB_ALLOWLIST = {
    ("rapid_mlx/model_downloads.py", "snapshot_download"),
    ("rapid_mlx/model_downloads.py", "hf_hub_download"),
    ("rapid_mlx/cli.py", "_pull_repository"),
    ("rapid_mlx/cli.py", "pull_command"),  # pinned, already verified sidecar probe
    ("rapid_mlx/_mirror.py", "_hf_supports_tqdm_class"),  # signature inspection only
    ("rapid_mlx/_mirror.py", "_hf_fallback_one"),  # transport, callers verify selection
    ("rapid_mlx/byom/imports.py", "_download_source"),  # explicit conversion input
    ("rapid_mlx/byom/preflight.py", "_fetch_hub_config"),  # explicit metadata preflight
}


def test_no_new_raw_hub_downloads():
    root = Path(__file__).resolve().parents[1]
    found = set()
    for package in ("rapid_mlx", "ltx_pipelines_mlx", "ltx_core_mlx"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            parents = {
                child: node
                for node in ast.walk(tree)
                for child in ast.iter_child_nodes(node)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[0] == "huggingface_hub"
                ):
                    if not any(
                        a.name in {"snapshot_download", "hf_hub_download"}
                        for a in node.names
                    ):
                        continue
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"snapshot_download", "hf_hub_download"}
                ):
                    # Calls through the owned module are already verified.
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "model_downloads"
                    ):
                        continue
                else:
                    continue
                owner = node
                while owner in parents and not isinstance(
                    owner, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    owner = parents[owner]
                location = (
                    path.relative_to(root).as_posix(),
                    getattr(owner, "name", "<module>"),
                )
                assert location in RAW_HUB_ALLOWLIST, (location, node.lineno)
                found.add(location)
    assert found == RAW_HUB_ALLOWLIST


def test_mirror_success_verifies_before_return(asset, monkeypatch):
    from rapid_mlx import _mirror

    monkeypatch.setattr(_mirror, "_hf_cache_root", lambda: asset.parents[2])
    monkeypatch.setattr(_mirror, "download_with_mirror_fallback", lambda *a, **k: True)
    # Use the actual mirror cache layout and corrupt bytes after pin selection.
    root = asset.parents[2] / "models--fixture--model" / "snapshots" / SHA
    root.mkdir(parents=True)
    (root / "config.json").write_text("corrupt")
    with pytest.raises(ModelIntegrityError):
        _mirror.pinned_snapshot_download("fixture/model", SHA)


def test_auxiliary_pins_do_not_duplicate_catalog():
    import re

    from rapid_mlx.model_aliases import resolve_revision
    from rapid_mlx.runtime_model_pins import AUXILIARY_REVISIONS

    for repo, revision in AUXILIARY_REVISIONS.items():
        assert re.fullmatch("[0-9a-f]{40}", revision)
        assert resolve_revision(repo) is None, repo


def test_nested_f5_vocoder_cannot_load_unverified_bytes(asset, monkeypatch):
    from rapid_mlx.audio.tts import TTSEngine

    vocos = pytest.importorskip("vocos_mlx.model")
    utils = pytest.importorskip("f5_tts_mlx.utils")
    cfm = pytest.importorskip("f5_tts_mlx.cfm")
    monkeypatch.setattr(utils, "snapshot_download", utils.snapshot_download)
    monkeypatch.setattr(vocos, "snapshot_download", vocos.snapshot_download)
    monkeypatch.setattr(
        cfm.F5TTS,
        "from_pretrained",
        lambda name: vocos.Vocos.from_pretrained("lucasnewman/vocos-mel-24khz"),
    )
    monkeypatch.setattr(
        vocos.mx, "load", lambda *a, **k: pytest.fail("unverified tensor load")
    )
    with pytest.raises(ModelIntegrityError):
        TTSEngine("lucasnewman/f5-tts-mlx").load()


@pytest.mark.parametrize(
    "module,name",
    [
        ("ltx_pipelines_mlx.utils.blocks", "_resolve_model_dir"),
        ("ltx_pipelines_mlx.utils._orchestration", "resolve_model_dir"),
        ("ltx_pipelines_mlx.utils._orchestration", "resolve_lora_path"),
        ("ltx_pipelines_mlx.ic_lora", "_resolve_lora_path"),
    ],
)
def test_nested_video_downloads_reject_corruption(asset, module, name):
    import importlib

    pytest.importorskip("mlx.core")
    loader = getattr(importlib.import_module(module), name)
    (asset / "config.json").write_text("corrupt")
    with pytest.raises(ModelIntegrityError):
        loader("fixture/model")


def test_video_text_encoder_checks_before_upstream_load(asset, monkeypatch):
    pytest.importorskip("mlx.core")
    import mlx_lm

    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import (
        GemmaLanguageModel,
    )

    monkeypatch.setattr(
        mlx_lm, "load", lambda *a, **k: pytest.fail("unverified model load")
    )
    (asset / "config.json").write_text("corrupt")
    with pytest.raises(ModelIntegrityError):
        GemmaLanguageModel("fixture/model").load()


def test_video_info_verifies_downloads(asset):
    pytest.importorskip("mlx.core")
    from ltx_pipelines_mlx.cli import _cmd_info

    (asset / "config.json").write_text("corrupt")
    with pytest.raises(SystemExit) as exc:
        _cmd_info(SimpleNamespace(model="fixture/model"))
    assert exc.value.code == 1


def test_download_adapter_reproducible_and_preserves_upstream(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "vendor_adapter", root / "scripts/vendor_ltx25.py"
    )
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    (tmp_path / "download.py").write_text(
        "from huggingface_hub import snapshot_download\n"
    )
    (tmp_path / "base_encoder.py").write_text(
        "        self._model, self._tokenizer = mlx_lm_load(path)\n"
    )
    (tmp_path / "untouched.py").write_text("UNCHANGED = 1\n")
    vendor._apply_download_adapter(tmp_path)
    assert (
        tmp_path / "download.py"
    ).read_text() == "from rapid_mlx.model_downloads import snapshot_download\n"
    assert (
        "path = snapshot_download(path)" in (tmp_path / "base_encoder.py").read_text()
    )
    assert (tmp_path / "untouched.py").read_text() == "UNCHANGED = 1\n"


def test_clef_public_loader_verifies_before_backbone(asset, monkeypatch):
    import sys

    pytest.importorskip("torch")
    from rapid_mlx.clef.vendor.joint_schema_model import load_release_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=object(), Qwen3_5ForConditionalGeneration=object()
        ),
    )
    (asset / "config.json").write_text("corrupt")
    with pytest.raises(ModelIntegrityError):
        load_release_model("fixture/model", device="cpu")


def test_generated_video_provenance_imports():
    pytest.importorskip("mlx.core")
    from ltx_core_mlx import _vendored as core
    from ltx_pipelines_mlx import _vendored as pipelines

    assert len(core.TREE_SHA256) == len(pipelines.TREE_SHA256) == 64


def test_chatterbox_nested_hook_cannot_load_unverified_bytes(asset, monkeypatch):
    from rapid_mlx.audio import tts

    chatterbox = pytest.importorskip("mlx_audio.tts.models.chatterbox.chatterbox")
    tokenizer = pytest.importorskip("mlx_audio.tts.models.chatterbox.tokenizer")
    generate = pytest.importorskip("mlx_audio.tts.generate")
    monkeypatch.setattr(
        chatterbox.Model,
        "post_load_hook",
        staticmethod(chatterbox.Model.post_load_hook),
    )
    monkeypatch.setattr(tokenizer, "hf_hub_download", tokenizer.hf_hub_download)
    monkeypatch.setattr(tts, "_resolve_safe_tts_checkpoint", lambda name: str(asset))
    monkeypatch.setattr(
        generate,
        "load_model",
        lambda path: chatterbox.Model.post_load_hook(
            SimpleNamespace(_init_text_tokenizers=lambda p: None), path
        ),
    )
    monkeypatch.setattr(
        chatterbox.mx, "load", lambda *a, **k: pytest.fail("unverified tensor load")
    )
    (asset / "config.json").write_text("corrupt")
    with pytest.raises(ModelIntegrityError):
        tts.TTSEngine("mlx-community/chatterbox-fp16").load()


def test_scoped_hub_import_leaves_other_imports_and_hub_unchanged():
    original = huggingface_hub.snapshot_download

    def hook(*, keyword=True):
        import json

        from huggingface_hub import HfApi, snapshot_download

        return json.loads("true"), HfApi, snapshot_download, keyword

    adapted = model_downloads.bind_hub_downloads(hook)
    assert adapted(keyword=False) == (
        True,
        huggingface_hub.HfApi,
        model_downloads.snapshot_download,
        False,
    )
    assert huggingface_hub.snapshot_download is original
    assert hook()[2] is original


def test_image_prompt_tokenizer_uses_verified_local_data(asset, monkeypatch):
    import sys

    from rapid_mlx.image.engine import ImageGenerationEngine

    runtime = SimpleNamespace(
        MAX_PROMPT_TOKENS=1024,
        _encode_prompt_ids=lambda *a: SimpleNamespace(shape=(1, 2)),
    )
    monkeypatch.setitem(sys.modules, "rapid_mlx.image.hidream_runtime.runtime", runtime)
    calls = []
    tokenizer = SimpleNamespace(
        from_pretrained=lambda source, **kw: calls.append((source, kw)) or object()
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer)
    )
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_local_snapshot", lambda name: None
    )
    engine = ImageGenerationEngine("HiDream-O1-Image-Dev")
    engine._validate_hidream_prompt_tokens("fixture")
    assert calls == [
        (str(asset), {"trust_remote_code": False, "local_files_only": True})
    ]
    (asset / "config.json").write_text("corrupt")
    engine._prompt_tokenizer = None
    with pytest.raises(ModelIntegrityError):
        engine._validate_hidream_prompt_tokens("fixture")
    assert len(calls) == 1
