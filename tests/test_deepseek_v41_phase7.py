"""Registry and unpinned-local qualification contracts; no real model loads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rapid_mlx.models.deepseek_v41_native import artifacts as a


def _local_build(root):
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "engram_token_map.json",
        "dspark/config.json",
        "dspark/model.safetensors.index.json",
        "dspark/model-mtp.safetensors",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (root / "model-layer-0.safetensors").write_text("{}")
    for index, shard in (
        ("model.safetensors.index.json", "model-layer-0.safetensors"),
        ("dspark/model.safetensors.index.json", "model-mtp.safetensors"),
    ):
        (root / index).write_text(json.dumps({"weight_map": {"w": shard}}))
    return root


def _pinned_build(root):
    _local_build(root)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (root / "dspark/model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.0.weight": "model-mtp.safetensors"}})
    )

    def manifest(directory):
        return tuple(
            a.ArtifactFile(
                p.name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()
            )
            for p in directory.iterdir()
            if p.is_file()
        )

    return replace(
        a.FLASH_1017,
        target_revision="a" * 40,
        mtp_revision="a" * 40,
        target_files=manifest(root),
        mtp_files=manifest(root / "dspark"),
    )


@pytest.mark.parametrize("artifact", [a.REAP, a.FLASH_1017])
def test_registry_selects_alias_and_repo(artifact):
    assert a.PRODUCT_ARTIFACTS[artifact.alias] == artifact
    for name in (artifact.alias, artifact.target_repo):
        assert a.product_artifact(name) == artifact
        assert a.is_product_target(name)
    assert a.product_artifact("/local/unregistered") is None


def test_original_registry_contract_and_identity_unchanged():
    assert a.REAP.target_revision == "a25fec277b9e7cedc0e9f3f15da874a5cf9d491b"
    assert a.REAP.mtp_revision == "9530d6d2bf59e0d05177bd538095d5704ded1488"
    assert a.REAP.mtp_files == a.MTP_FILES
    assert a.REAP.draft_subfolder == ""
    assert a.mtp_model_identity_digest() == a.mtp_model_identity_digest(a.REAP)
    assert a.mtp_model_identity_digest() == (
        "sha256:eb94ba9766612111d89b89676a7485d612ad534b2d8b0d37b78c137ff8994d75"
    )


def _unpinned(monkeypatch):
    unpinned = replace(
        a.FLASH_1017,
        target_revision=None,
        mtp_revision=None,
        target_files=None,
        mtp_files=None,
    )
    monkeypatch.setitem(a.PRODUCT_ARTIFACTS, unpinned.alias, unpinned)
    return unpinned


def test_shipped_pins_are_complete_and_immutable():
    a.require_pins(a.FLASH_1017)
    assert a.FLASH_1017.target_revision == a.FLASH_1017.mtp_revision
    names = {item.name for item in a.FLASH_1017.target_files or ()}
    assert {f"model-layer-{i}.safetensors" for i in range(40)} <= names
    assert {
        f"model-engram-{layer}-{part}.safetensors"
        for layer in (1, 14)
        for part in ("weight", "affine")
    } <= names
    # Hub per-file limit is 50 GB.
    assert max(item.size for item in a.FLASH_1017.target_files or ()) < 50e9


@pytest.mark.parametrize("operation", ["resolve", "identity", "pull"])
def test_unset_pins_fail_before_network(monkeypatch, operation):
    monkeypatch.delenv(a.LOCAL_MODEL_DIR_ENV, raising=False)
    unpinned = _unpinned(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download",
        lambda *_a, **_k: pytest.fail("network reached"),
    )
    with pytest.raises(RuntimeError, match="pins are unset"):
        if operation == "identity":
            a.mtp_model_identity_digest(unpinned)
        elif operation == "pull":
            from rapid_mlx import cli

            cli.pull_command(
                SimpleNamespace(
                    model=a.FLASH_1017.target_repo,
                    _original_alias=a.FLASH_1017.alias,
                    bits=None,
                    format=None,
                )
            )
        else:
            a.resolve_product_paths(unpinned)


@pytest.mark.parametrize(
    "change",
    [
        {"target_revision": "main"},
        {"mtp_revision": "b" * 40},
        {"mtp_files": ()},
        {"target_files": ()},
        {"mtp_files": (a.ArtifactFile("model-mtp.safetensors", 1, "TODO"),)},
    ],
)
def test_partial_pins_fail_closed(tmp_path, change):
    artifact = replace(_pinned_build(tmp_path), **change)
    with pytest.raises(RuntimeError, match="pins are unset"):
        a.require_pins(artifact)


def test_published_colocated_snapshot_verifies_both_manifests(monkeypatch, tmp_path):
    artifact = _pinned_build(tmp_path)
    monkeypatch.delenv(a.LOCAL_MODEL_DIR_ENV, raising=False)
    calls = []
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download",
        lambda *args: calls.append(args) or str(tmp_path),
    )
    assert a.resolve_product_paths(artifact) == (tmp_path, tmp_path / "dspark")
    assert calls == [(artifact.target_repo, artifact.target_revision)]
    (tmp_path / "dspark/model-mtp.safetensors").write_bytes(b"xx")
    with pytest.raises(ValueError, match="hash mismatch"):
        a.resolve_product_paths(artifact)


def test_published_snapshot_requires_all_indexed_shards(monkeypatch, tmp_path):
    artifact = _pinned_build(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.weight": "unlisted.safetensors"}})
    )
    monkeypatch.delenv(a.LOCAL_MODEL_DIR_ENV, raising=False)
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download", lambda *_: str(tmp_path)
    )
    with pytest.raises(ValueError, match="missing from the pinned"):
        a.resolve_product_paths(artifact)


@pytest.mark.parametrize("subdir", ["", "dspark"])
def test_published_snapshot_rejects_unpinned_shards(monkeypatch, tmp_path, subdir):
    artifact = _pinned_build(tmp_path)
    monkeypatch.delenv(a.LOCAL_MODEL_DIR_ENV, raising=False)
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download", lambda *_: str(tmp_path)
    )
    (tmp_path / subdir / "zzz.safetensors").write_bytes(b"override")
    with pytest.raises(ValueError, match="unpinned safetensors"):
        a.resolve_product_paths(artifact)


def test_local_override_is_explicit_logged_and_bypasses_pins(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.delenv(a.LOCAL_MODEL_DIR_ENV, raising=False)
    assert a.local_product_path(a.FLASH_1017) is None
    root = _local_build(tmp_path)
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, str(root))
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download",
        lambda *_a, **_k: pytest.fail("download"),
    )
    monkeypatch.setattr(a, "_sha256", lambda *_: pytest.fail("hash"))
    assert a.resolve_product_paths(a.FLASH_1017) == (root, root / "dspark")
    assert "UNPINNED LOCAL QUALIFICATION" in caplog.text
    assert "bypassing downloads and revision/hash" in caplog.text
    with pytest.raises(ValueError, match="only supported by serve"):
        a.resolve_product_paths(a.FLASH_1017, allow_local=False)


@pytest.mark.parametrize("value", ["/does/not/exist", "/"])
def test_bad_local_override_does_not_fall_back(monkeypatch, value):
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, value)
    with pytest.raises(ValueError, match="incomplete local"):
        a.resolve_product_paths(a.FLASH_1017)


@pytest.mark.parametrize("subdir", ["", "dspark"])
def test_local_override_requires_every_indexed_shard(monkeypatch, tmp_path, subdir):
    root = _local_build(tmp_path)
    weight_map = {p.name: p.name for p in (root / subdir).glob("*.safetensors")}
    (root / subdir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {**weight_map, "w": "missing.safetensors"}})
    )
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, str(root))
    with pytest.raises(ValueError, match="incomplete local"):
        a.local_product_path(a.FLASH_1017)
    (root / subdir / "missing.safetensors").write_bytes(b"x")
    assert a.local_product_path(a.FLASH_1017) == root


@pytest.mark.parametrize("index", [{}, {"weight_map": {}}, []])
def test_local_override_rejects_empty_or_malformed_index(monkeypatch, tmp_path, index):
    root = _local_build(tmp_path)
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, str(root))
    with pytest.raises(ValueError, match="incomplete local"):
        a.local_product_path(a.FLASH_1017)


def test_local_override_rejects_shards_outside_the_build(monkeypatch, tmp_path):
    root = _local_build(tmp_path / "build")
    (tmp_path / "outside.safetensors").write_bytes(b"x")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "../outside.safetensors"}})
    )
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, str(root))
    with pytest.raises(ValueError, match="incomplete local"):
        a.local_product_path(a.FLASH_1017)


@pytest.mark.parametrize("artifact", [None, a.REAP])
def test_local_override_cannot_select_unrelated_or_original_lane(monkeypatch, artifact):
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, "/anything")
    with pytest.raises(ValueError, match="requires deepseek-v41-flash-1017"):
        a.local_product_path(artifact)


def test_new_alias_keeps_limits_and_memory_guard(monkeypatch):
    from rapid_mlx.model_aliases import resolve_profile

    profile = resolve_profile(a.FLASH_1017.alias)
    assert profile.hf_path == a.FLASH_1017.target_repo
    assert profile.min_memory_gb == a.FLASH_1017.min_memory_gb == 224
    assert profile.enforce_min_memory and profile.is_text_only
    assert profile.default_max_tokens == 4096
    assert profile.tool_call_parser is None
    assert dict(profile.recommended_sampling) == {"temperature": 0.0, "top_p": 1.0}
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=192 * 1024**3)
    )
    with pytest.raises(RuntimeError, match="224 GiB"):
        a.require_product_memory(a.FLASH_1017)


@pytest.mark.requires_mlx
def test_tiny_colocated_draft_loads_from_target_or_draft(tmp_path):
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.dspark import (
        DSparkWeights,
        _required_tensors,
    )
    from tests.test_deepseek_v41_dspark import _config

    config = _config()
    config["quantization"] = {
        "bits": 8,
        "group_size": 64,
        "modules": {"mtp.0.ffn.experts.0.w1": {"bits": 4, "group_size": 64}},
    }
    root = tmp_path / "dspark"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    names = _required_tensors(config)
    arrays = {
        n: mx.array(
            [1],
            dtype=mx.uint32
            if n.endswith(".weight") and n.removesuffix(".weight") + ".scales" in names
            else mx.bfloat16,
        )
        for n in names
    }
    projection = "mtp.0.ffn.experts.0.w1"
    triple = mx.quantize(mx.ones((2, 64), dtype=mx.bfloat16), group_size=64, bits=4)
    arrays.update(
        {
            f"{projection}.{suffix}": value
            for suffix, value in zip(("weight", "scales", "biases"), triple)
        }
    )
    mx.save_safetensors(str(root / "model-mtp.safetensors"), arrays)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "model-mtp.safetensors" for n in names}})
    )
    for path in (tmp_path, root):
        weights = DSparkWeights(path)
        assert weights.path == root
        assert set(weights.entries) == names
        assert weights.quant_config("mtp.0.ffn.experts.0.w1")["bits"] == 4
        assert weights.quant_config("mtp.0.attn.wkv")["bits"] == 8
        output = weights.linear(projection, mx.ones((1, 64), dtype=mx.bfloat16))
        assert output.shape == (1, 2)
        assert output.tolist() == [[64.0, 64.0]]


@pytest.mark.requires_mlx
def test_colocated_hub_shard_accepts_only_its_repo_blobs(tmp_path):
    from rapid_mlx.models.deepseek_v41_native.dspark import _safe_shard

    repo = tmp_path / "models--owner--1017"
    root = repo / "snapshots" / ("a" * 40) / "dspark"
    root.mkdir(parents=True)
    blobs = repo / "blobs"
    blobs.mkdir()
    blob = blobs / ("b" * 64)
    blob.write_bytes(b"x")
    (root / "model-mtp.safetensors").symlink_to(blob)
    assert _safe_shard(root, "model-mtp.safetensors").resolve() == blob
    outside = tmp_path / "outside"
    outside.touch()
    (root / "escape.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="leaves its Hub repository"):
        _safe_shard(root, "escape.safetensors")


@pytest.mark.requires_mlx
@pytest.mark.parametrize("prefetched", [True, False])
def test_local_server_uses_same_guarded_lane_and_unpinned_identity(
    monkeypatch, tmp_path, prefetched
):
    from rapid_mlx.models.deepseek_v41_native import server
    from rapid_mlx.speculative.dflash import server as serial
    from tests.test_deepseek_v41_artifacts import _Tokenizer

    root = _local_build(tmp_path)
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, str(root))
    calls = {}

    class Executor:
        def submit(self, fn):
            return SimpleNamespace(result=fn)

    monkeypatch.setattr(serial, "_dflash_executor", Executor())
    monkeypatch.setattr(
        server, "require_product_memory", lambda artifact: calls.update(memory=artifact)
    )

    def load(*args, **kwargs):
        calls["load"] = args, kwargs
        return object(), _Tokenizer(), object()

    monkeypatch.setattr(server, "load_product_runtime", load)
    monkeypatch.setattr(serial, "_build_app", lambda **kw: calls.update(app=kw))
    monkeypatch.setattr(
        "rapid_mlx._uvicorn.run_uvicorn", lambda *args, **kw: calls.update(listener=kw)
    )
    monkeypatch.setattr(a, "_RESOLVED", {})
    if prefetched:
        # Serve reuses the CLI's verified prefetch instead of resolving again.
        a.resolve_product_paths(a.FLASH_1017)
        monkeypatch.setattr(
            server,
            "resolve_product_paths",
            lambda *_a, **_k: pytest.fail("re-resolved"),
        )
    server.run_server(
        model_name=a.FLASH_1017.alias,
        host="127.0.0.1",
        port=8000,
        served_model_name="dsv41-t256",
        default_max_tokens=4096,
        cors_origins=[],
        uvicorn_log_level="warning",
    )
    assert calls["memory"] == a.FLASH_1017
    assert calls["load"] == (
        (str(root), str(root / "dspark")),
        {
            "target_revision": "local-unpinned",
            "mtp_revision": "local-unpinned",
            "mtp_identity": str(root / "dspark"),
            "mixed_target_precision": True,
        },
    )
    assert calls["app"]["served_model_name"] == "dsv41-t256"
    assert calls["app"]["tool_call_parser"] is None
    assert calls["app"]["render_prompt_fn"] is server.render_prompt
    assert calls["app"]["validate_request_fn"] is server.validate_request
    assert calls["listener"]["host"] == "127.0.0.1"
    assert calls["listener"]["port"] == 8000


@pytest.mark.requires_mlx
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_local_cli_skips_generic_and_pinned_downloads(monkeypatch, tmp_path, lanes):
    from rapid_mlx import _version_check, cli
    from rapid_mlx import server as generic_server
    from rapid_mlx.models.deepseek_v41_native import server
    from tests.test_deepseek_v41_artifacts import _product_serve_args

    args = _product_serve_args(
        "--served-model-name",
        "dsv41-t256",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dsv41-lanes",
        str(lanes),
    )
    args.model = a.FLASH_1017.target_repo
    args._original_alias = a.FLASH_1017.alias
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, str(_local_build(tmp_path)))
    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_: None
    )
    memory = []
    monkeypatch.setattr(cli, "_check_alias_min_memory", memory.append)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *args, **kw: None)
    monkeypatch.setattr(
        cli, "_check_disk_space", lambda *args, **kw: pytest.fail("disk/network probe")
    )
    monkeypatch.setattr(
        cli,
        "_ensure_model_downloaded",
        lambda *args, **kw: pytest.fail("generic download"),
    )
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download",
        lambda *args, **kw: pytest.fail("pinned download"),
    )
    monkeypatch.setattr(generic_server, "configure_logging", lambda _: "warning")
    calls = []
    monkeypatch.setattr(server, "run_server", lambda **kw: calls.append(kw))
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    cli.serve_command(args)
    assert memory == [a.FLASH_1017.alias]
    assert calls[0]["lanes"] == lanes
    assert calls[0]["model_name"] == a.FLASH_1017.target_repo
    assert calls[0]["served_model_name"] == "dsv41-t256"
    assert calls[0]["default_max_tokens"] == 4096


def test_benchmark_new_lane_fails_closed_without_pins(monkeypatch):
    from rapid_mlx.community_bench.local_runner import _text_speculative_execution

    _unpinned(monkeypatch)
    with pytest.raises(RuntimeError, match="pins are unset"):
        _text_speculative_execution(a.FLASH_1017.target_repo)


@pytest.mark.requires_mlx
def test_mixed_target_runtime_accepts_partial_qmv_and_colocated_draft(monkeypatch):
    from rapid_mlx.models.deepseek_v41_native import serving

    model = SimpleNamespace(layers=[object(), object()], eval_interval=0)
    monkeypatch.setattr(serving, "load", lambda *args, **kw: (model, None))
    monkeypatch.setattr(serving, "install_target_qmv", lambda _: 1)
    monkeypatch.setattr(serving, "supports_engram_ssd_offload", lambda _: True)
    paths = []
    weights = SimpleNamespace(
        config={}, attach_target=lambda _: None, pin_mtp=lambda: None
    )
    monkeypatch.setattr(
        serving, "DSparkWeights", lambda path: paths.append(path) or weights
    )
    monkeypatch.setattr(serving, "DSpark", lambda *args, **kw: object())
    monkeypatch.setattr(serving, "_install_vectorized_mtp_attention", lambda _: None)
    monkeypatch.setattr(serving, "_install_packed_mtp_moe", lambda _: None)
    monkeypatch.setattr(
        serving.PreTrainedTokenizerFast, "from_pretrained", lambda *args, **kw: object()
    )
    loaded, _, _ = serving.load_product_runtime(
        "/target",
        "/target/dspark",
        target_revision="local-unpinned",
        mtp_revision="local-unpinned",
        mixed_target_precision=True,
    )
    assert loaded is model
    assert paths == ["/target/dspark"]


def test_resolution_is_taken_once(monkeypatch, tmp_path):
    root = _local_build(tmp_path)
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, str(root))
    monkeypatch.setattr(a, "_RESOLVED", {})
    a.resolve_product_paths(a.FLASH_1017)
    assert a.take_resolved_product_paths(a.FLASH_1017) == (root, root / "dspark", True)
    assert a.take_resolved_product_paths(a.FLASH_1017) is None


def test_pull_uses_pinned_product_resolution(monkeypatch):
    from rapid_mlx import cli

    monkeypatch.delenv(a.LOCAL_MODEL_DIR_ENV, raising=False)
    disk, resolved = [], []
    monkeypatch.setattr(cli, "_check_disk_space", lambda *args, **kw: disk.append(kw))
    monkeypatch.setattr(
        a, "resolve_product_paths", lambda *args, **kw: resolved.append((args, kw))
    )
    cli.pull_command(
        SimpleNamespace(
            model=a.FLASH_1017.target_repo,
            _original_alias=a.FLASH_1017.alias,
            bits=None,
            format=None,
        )
    )
    assert disk == [{"revision_override": a.FLASH_1017.target_revision}]
    assert resolved == [((a.FLASH_1017,), {"allow_local": False})]


@pytest.mark.requires_mlx
def test_pinned_cli_checks_disk_before_resolution(monkeypatch):
    from rapid_mlx import _version_check, cli
    from rapid_mlx import server as generic_server
    from rapid_mlx.models.deepseek_v41_native import server
    from tests.test_deepseek_v41_artifacts import _product_serve_args

    args = _product_serve_args("--served-model-name", "dsv41-t256")
    args.model = a.FLASH_1017.target_repo
    args._original_alias = a.FLASH_1017.alias
    monkeypatch.delenv(a.LOCAL_MODEL_DIR_ENV, raising=False)
    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_: None
    )
    monkeypatch.setattr(cli, "_check_alias_min_memory", lambda *_: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *args, **kw: None)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    order = []
    monkeypatch.setattr(
        cli, "_check_disk_space", lambda repo, **kw: order.append(("disk", repo, kw))
    )
    monkeypatch.setattr(
        a, "resolve_product_paths", lambda artifact: order.append(("resolve", artifact))
    )
    monkeypatch.setattr(generic_server, "configure_logging", lambda _: "warning")
    monkeypatch.setattr(server, "run_server", lambda **kw: None)
    cli.serve_command(args)
    assert order == [
        (
            "disk",
            a.FLASH_1017.target_repo,
            {"force": False, "revision_override": a.FLASH_1017.target_revision},
        ),
        ("resolve", a.FLASH_1017),
    ]


def test_original_lane_resolution_uses_its_own_snapshots(monkeypatch, tmp_path):
    monkeypatch.setattr(a, "download_target_snapshot", lambda: tmp_path / "t")
    monkeypatch.setattr(a, "download_mtp_snapshot", lambda: tmp_path / "m")
    assert a.resolve_product_paths(a.REAP, allow_local=False) == (
        tmp_path / "t",
        tmp_path / "m",
    )


def test_server_rejects_unknown_product():
    pytest.importorskip("mlx")
    from rapid_mlx.models.deepseek_v41_native import server

    with pytest.raises(ValueError, match="unknown V4.1 product artifact"):
        server.run_server(
            host="127.0.0.1",
            port=0,
            served_model_name="x",
            default_max_tokens=1,
            cors_origins=[],
            uvicorn_log_level="warning",
            model_name="not-a-product",
        )


@pytest.mark.parametrize("experts", [(4,), (4, 1), (4, 9)])
def test_per_layer_expert_counts_must_cover_layers_and_top_k(experts):
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs

    with pytest.raises(ValueError, match="experts_per_layer"):
        ModelArgs(
            n_layers=2,
            n_routed_experts=8,
            n_activated_experts=2,
            experts_per_layer=experts,
        )


@pytest.mark.requires_mlx
def test_block_sizes_routed_experts_per_layer():
    pytest.importorskip("mlx")
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.model import Block

    args = ModelArgs(
        dim=16,
        n_heads=8,
        head_dim=8,
        rope_head_dim=2,
        q_lora_rank=8,
        o_lora_rank=3,
        o_groups=2,
        moe_inter_dim=8,
        n_layers=2,
        compress_ratios=(0, 0),
        n_routed_experts=8,
        n_activated_experts=2,
        experts_per_layer=(4, 8),
    )
    assert [Block(i, args).ffn.gate.weight.shape[0] for i in range(2)] == [4, 8]


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_local_override_is_unset(monkeypatch, value):
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, value)
    assert a.local_product_path(a.FLASH_1017) is None
    assert a.local_product_path(None) is None


@pytest.mark.requires_mlx
@pytest.mark.parametrize("model", ["deepseek-v41-flash-reap-2bit", "qwen3-0.6b-4bit"])
def test_local_override_never_breaks_other_models(monkeypatch, model):
    from rapid_mlx import _version_check, cli
    from tests.test_deepseek_v41_artifacts import _product_serve_args

    args = _product_serve_args()
    args.model = model
    args._original_alias = model
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, "/anything")
    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_: None
    )
    # Host-independent: the memory guards run first and differ per runner.
    monkeypatch.setattr(cli, "_check_alias_min_memory", lambda *_: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *args, **kw: None)
    reached = []

    def stop(name):
        # Runs right after the override lookup; reaching it proves no ValueError.
        reached.append(name)
        raise SystemExit(0)

    monkeypatch.setattr(a, "is_product_target", stop)
    with pytest.raises(SystemExit):
        cli.serve_command(args)
    assert reached == [model]


@pytest.mark.requires_mlx
def test_original_lane_server_ignores_local_override(monkeypatch, tmp_path):
    from rapid_mlx.models.deepseek_v41_native import server
    from rapid_mlx.speculative.dflash import server as serial
    from tests.test_deepseek_v41_artifacts import _Tokenizer

    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, "/anything")
    calls = {}

    class Executor:
        def submit(self, fn):
            return SimpleNamespace(result=fn)

    monkeypatch.setattr(serial, "_dflash_executor", Executor())
    monkeypatch.setattr(server, "require_product_memory", lambda *_: None)
    monkeypatch.setattr(server, "download_target_snapshot", lambda: tmp_path / "t")
    monkeypatch.setattr(server, "download_mtp_snapshot", lambda: tmp_path / "m")

    def load(*args, **kwargs):
        calls["load"] = args, kwargs
        return object(), _Tokenizer(), object()

    monkeypatch.setattr(server, "load_product_runtime", load)
    monkeypatch.setattr(serial, "_build_app", lambda **kw: calls.update(app=kw))
    monkeypatch.setattr("rapid_mlx._uvicorn.run_uvicorn", lambda *args, **kw: None)
    server.run_server(
        model_name=a.REAP.alias,
        host="127.0.0.1",
        port=8000,
        served_model_name="reap",
        default_max_tokens=4096,
        cors_origins=[],
        uvicorn_log_level="warning",
    )
    assert calls["load"] == (
        (str(tmp_path / "t"), str(tmp_path / "m")),
        {
            "target_revision": a.REAP.target_revision,
            "mtp_revision": a.REAP.mtp_revision,
            "mtp_identity": a.REAP.mtp_repo,
        },
    )


def test_original_lane_resolution_ignores_local_override(monkeypatch, tmp_path):
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, "/anything")
    monkeypatch.setattr(a, "download_target_snapshot", lambda: tmp_path / "t")
    monkeypatch.setattr(a, "download_mtp_snapshot", lambda: tmp_path / "m")
    assert a.resolve_product_paths(a.REAP) == (tmp_path / "t", tmp_path / "m")


@pytest.mark.parametrize("subdir", ["", "dspark"])
def test_local_override_rejects_unindexed_shards(monkeypatch, tmp_path, subdir):
    root = _local_build(tmp_path)
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, str(root))
    assert a.local_product_path(a.FLASH_1017) == root
    (root / subdir / "stale.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="incomplete local"):
        a.local_product_path(a.FLASH_1017)


def test_empty_local_override_does_not_block_pinned_resolution(monkeypatch):
    monkeypatch.setenv(a.LOCAL_MODEL_DIR_ENV, " ")
    seen = []
    monkeypatch.setattr(a, "require_pins", lambda artifact: seen.append(artifact))
    monkeypatch.setattr(a, "download_target_snapshot", lambda: "t")
    monkeypatch.setattr(a, "download_mtp_snapshot", lambda: "m")
    assert a.resolve_product_paths(a.REAP, allow_local=False) == ("t", "m")
    assert seen == [a.REAP]


@pytest.mark.parametrize("artifact", [a.REAP, a.FLASH_1017])
def test_benchmark_readiness_checks_each_lane_memory_guard(monkeypatch, artifact):
    from rapid_mlx.community_bench.workspace import benchmark_runtime_readiness

    seen = []
    monkeypatch.setattr(a, "require_product_memory", lambda *args: seen.append(args))
    status = benchmark_runtime_readiness(artifact.alias, "text_generation")
    assert status == {"status": "ready", "message": None}
    assert seen == [()] if artifact == a.REAP else [(artifact,)]
