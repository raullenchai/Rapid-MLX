# SPDX-License-Identifier: Apache-2.0
"""Startup adapter validation, dispatch and identity regressions."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from rapid_mlx.cli_parser import build_parser
from rapid_mlx.lora_adapter import (
    Adapter,
    prepare_adapter_cli,
    reject_adapter_options,
    validate_adapter,
    validate_adapter_checkpoint,
)


@pytest.fixture(autouse=True)
def isolated_adapter_cache_state(monkeypatch):
    """Restore process cache state even when a loader/lifecycle assertion fails."""
    from rapid_mlx import disk_caches, response_cache

    monkeypatch.setattr(disk_caches, "_cli_disabled", disk_caches._cli_disabled)
    monkeypatch.setattr(disk_caches, "_adapter_users", disk_caches._adapter_users)
    monkeypatch.setattr(response_cache, "_adapter_users", response_cache._adapter_users)
    monkeypatch.setattr(
        response_cache, "_response_cache", response_cache.ResponseCache(capacity=0)
    )
    monkeypatch.setattr(
        response_cache,
        "_adapter_response_cache",
        response_cache.ResponseCache(capacity=0),
    )


@pytest.fixture
def adapter(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen3"}')
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "model": str(base),
                "num_layers": 1,
                "lora_parameters": {"rank": 2, "scale": 2},
            }
        )
    )
    (directory / "adapters.safetensors").write_bytes(b"test weights")
    return directory, base


def test_parser_adapter_opt_in():
    parser = build_parser()
    assert parser.parse_args(["serve", "base"]).adapter is None
    assert (
        parser.parse_args(["serve", "base", "--adapter", "~/adapter"]).adapter
        == "~/adapter"
    )


@pytest.mark.parametrize(
    "missing", ["directory", "adapter_config.json", "adapters.safetensors"]
)
def test_required_files(adapter, missing):
    directory, base = adapter
    if missing == "directory":
        directory = directory / "missing"
    else:
        (directory / missing).unlink()
    with pytest.raises(ValueError, match="--adapter.*(directory|required file)"):
        validate_adapter(str(directory), str(base))


@pytest.mark.parametrize("content", ["{", "[]", "{}", '{"model": 42}'])
def test_invalid_config(adapter, content):
    directory, base = adapter
    (directory / "adapter_config.json").write_text(content)
    with pytest.raises(ValueError, match="adapter_config.json"):
        validate_adapter(str(directory), str(base))


def test_base_mismatch(adapter):
    directory, _ = adapter
    with pytest.raises(ValueError, match="base mismatch"):
        validate_adapter(str(directory), "another/base")


def test_identity_content_and_location(adapter, tmp_path):
    import shutil

    from rapid_mlx.response_cache import make_cache_key

    directory, base = adapter
    first = validate_adapter(str(directory), str(base))
    copy = tmp_path / "copy"
    shutil.copytree(directory, copy)
    assert validate_adapter(str(copy), str(base)).digest == first.digest
    (copy / "adapters.safetensors").write_bytes(b"different weights")
    second = validate_adapter(str(copy), str(base))
    assert second.digest != first.digest
    config = copy / "adapter_config.json"
    config.write_text(config.read_text() + "\n")
    assert validate_adapter(str(copy), str(base)).digest != second.digest
    identities = [
        str(base),
        first.cache_identity(str(base)),
        second.cache_identity(str(base)),
    ]
    assert len(set(identities)) == 3
    assert first.model_identity(str(base)) == f"{base}:lora:{first.digest[:12]}"
    assert len(first.digest) == 64
    assert (
        len(
            {
                make_cache_key(
                    model=name, prompt=[], sampling_kwargs={"temperature": 0}
                )
                for name in identities
            }
        )
        == 3
    )


def test_public_digest_collision_keeps_internal_namespaces_distinct():
    from rapid_mlx.engine.batched import BatchedEngine

    first = Adapter("/adapter", "0123456789ab" + "0" * 52)
    second = Adapter("/adapter", "0123456789ab" + "1" * 52)
    assert first.model_identity("base") == second.model_identity("base")
    assert first != second  # Startup stability compares the full digest.
    engine = object.__new__(BatchedEngine)
    engine._model_name = "base"
    engine._adapter = first
    first_namespace = engine.model_name
    engine._adapter = second
    assert first_namespace != engine.model_name
    assert first_namespace.endswith(first.digest)
    assert engine.model_name.endswith(second.digest)


@pytest.mark.parametrize("option", ["mllm", "disk_stream", "speculative", "lazy"])
def test_unsupported_options(option):
    with pytest.raises(ValueError, match="--adapter is unsupported"):
        reject_adapter_options(**{option: True})


@pytest.mark.parametrize(
    "config",
    [
        {"vision_config": {"layers": 2}},
        {"audio_config": {"layers": 2}},
        {"num_nextn_predict_layers": 1},
        {"mtp_num_hidden_layers": 1},
        {"text_config": {"mtp_num_hidden_layers": 1}},
        {"text_config": {"num_nextn_predict_layers": 1}},
        {"quantization_config": {"quant_method": "fp8"}},
    ],
)
def test_checkpoint_paths(adapter, config):
    _, base = adapter
    (base / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="--adapter is unsupported"):
        validate_adapter_checkpoint(str(base))


@pytest.mark.parametrize(
    "flag",
    [
        "--disk-stream",
        "--mllm",
        "--lazy-load",
        "--enable-ddtree",
        "--force-spec-decode",
    ],
)
def test_cli_rejects_before_alternate_dispatch(adapter, flag):
    directory, base = adapter
    args = build_parser().parse_args(
        ["serve", str(base), "--adapter", str(directory), flag]
    )
    with pytest.raises(ValueError, match="--adapter is unsupported"):
        prepare_adapter_cli(args)


def test_cli_disables_automatic_draft_selection(adapter):
    directory, base = adapter
    args = build_parser().parse_args(["serve", str(base), "--adapter", str(directory)])
    prepare_adapter_cli(args)
    assert args.no_spec_decode is True


def test_without_adapter_does_not_mutate_options():
    args = SimpleNamespace(adapter=None, no_spec_decode=False)
    prepare_adapter_cli(args)
    assert vars(args) == {"adapter": None, "no_spec_decode": False}


@pytest.mark.parametrize(
    "method",
    [
        "save_cache_to_disk",
        "load_cache_from_disk",
        "save_cache_with_outcome",
        "load_cache_with_result",
    ],
)
def test_explicit_cache_persistence_rejected(adapter, method):
    from rapid_mlx.cache.protocol import CachePersistenceUnsupportedError
    from rapid_mlx.engine.batched import BatchedEngine

    directory, base = adapter
    engine = object.__new__(BatchedEngine)
    engine._adapter = validate_adapter(str(directory), str(base))
    with pytest.raises(
        CachePersistenceUnsupportedError, match="--adapter disables persisted"
    ):
        getattr(engine, method)("unused")


def test_batched_engine_guards_and_identity(adapter):
    from rapid_mlx import disk_caches
    from rapid_mlx.engine.batched import BatchedEngine

    directory, base = adapter
    before = disk_caches.disabled()
    engine = BatchedEngine(str(base), adapter_path=str(directory), force_text=True)
    assert engine.model_name == engine._adapter.cache_identity(str(base))
    assert engine.get_stats()["adapter_sha256"] == engine._adapter.digest
    assert engine._no_spec_decode
    assert disk_caches.disabled() == before
    with pytest.raises(ValueError, match="disk streaming"):
        BatchedEngine(str(base), adapter_path=str(directory), enable_disk_stream=True)
    assert disk_caches.disabled() == before


def test_text_loader_passes_adapter_and_never_falls_back(adapter, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    mlx_lm = pytest.importorskip("mlx_lm")
    tuner = pytest.importorskip("mlx_lm.tuner.utils")
    from rapid_mlx.utils import chat_template_registry
    from rapid_mlx.utils import tokenizer as loader

    directory, base = adapter
    mx.save_safetensors(
        str(directory / "adapters.safetensors"), {"layer.lora_a": mx.ones((2, 2))}
    )
    model = SimpleNamespace(
        parameters=lambda: {"layer": {"lora_a": mx.ones((2, 2))}},
        load_weights=lambda *_a, **_k: None,
        eval=lambda: None,
    )
    tokenizer = object()
    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return model, tokenizer

    monkeypatch.setattr(mlx_lm, "load", load)
    monkeypatch.setattr(tuner, "linear_to_lora_layers", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "_register_vendored_archs", lambda: None)
    monkeypatch.setattr(loader, "_post_load_ubc_evict", lambda _: None)
    monkeypatch.setattr(
        loader, "augment_eos_token_ids_from_generation_config", lambda *_: None
    )
    monkeypatch.setattr(
        chat_template_registry, "resolve_chat_template", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        loader,
        "_load_model_with_fallback_impl",
        lambda *_a, **_k: pytest.fail("adapter load must never fall back"),
    )
    result = loader.load_model_with_fallback(
        str(base), adapter_path=str(directory), return_source=True
    )
    assert result == (model, tokenizer, str(base))
    assert "adapter_path" not in calls[0][1]

    mx.save_safetensors(
        str(directory / "adapters.safetensors"), {"unrelated.lora_a": mx.ones((2, 2))}
    )
    with pytest.raises(ValueError, match="--adapter could not load.*missing LoRA"):
        loader.load_model_with_fallback(str(base), adapter_path=str(directory))

    def fail(*_a, **_k):
        raise RuntimeError("incompatible tensor")

    monkeypatch.setattr(mlx_lm, "load", fail)
    with pytest.raises(
        ValueError, match="--adapter could not load.*incompatible tensor"
    ):
        loader.load_model_with_fallback(str(base), adapter_path=str(directory))


def test_text_loader_without_adapter_preserves_call(adapter, monkeypatch):
    from rapid_mlx.utils import chat_template_registry
    from rapid_mlx.utils import tokenizer as loader

    _, base = adapter
    calls = []
    expected = object(), object()

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(loader, "_register_vendored_archs", lambda: None)
    monkeypatch.setattr(loader, "_post_load_ubc_evict", lambda _: None)
    monkeypatch.setattr(
        chat_template_registry, "resolve_chat_template", lambda *_a, **_k: None
    )
    monkeypatch.setattr(loader, "_load_model_with_fallback_impl", load)
    assert loader.load_model_with_fallback(str(base)) == expected
    assert calls == [((str(base), {"trust_remote_code": False}), {})]


@pytest.mark.parametrize("bad_tensor", [None, "lora_a", "lora_b"])
def test_real_text_loader_validates_shapes_before_loading(
    adapter, monkeypatch, bad_tensor
):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    utils = pytest.importorskip("mlx_lm.utils")
    from rapid_mlx.utils import chat_template_registry
    from rapid_mlx.utils import tokenizer as loader

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 6, bias=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block()]
            self.loads = []

        def load_weights(self, weights, **kwargs):
            self.loads.append(weights)
            return super().load_weights(weights, **kwargs)

    directory, base = adapter
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["lora_parameters"]["dropout"] = 0.0
    config["lora_parameters"]["keys"] = ["proj"]
    config_path.write_text(json.dumps(config))
    weights = {
        "layers.0.proj.lora_a": mx.ones((4, 2)),
        "layers.0.proj.lora_b": mx.ones((2, 6)),
    }
    if bad_tensor:
        weights[f"layers.0.proj.{bad_tensor}"] = mx.ones((3, 3))
    mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
    model = Model()
    tokenizer = SimpleNamespace(chat_template="test")
    # Exercise mlx_lm.load and its real LoRA construction; only base checkpoint
    # and tokenizer loading are stubbed so no downloaded weights are needed.
    monkeypatch.setattr(utils, "load_model", lambda *_a, **_k: (model, {}))
    monkeypatch.setattr(utils, "load_tokenizer", lambda *_a, **_k: tokenizer)
    monkeypatch.setattr(loader, "_register_vendored_archs", lambda: None)
    monkeypatch.setattr(loader, "_post_load_ubc_evict", lambda _: None)
    monkeypatch.setattr(
        chat_template_registry, "resolve_chat_template", lambda *_a, **_k: None
    )
    if bad_tensor:
        with pytest.raises(
            ValueError,
            match=f"adapter tensor does not match base model: layers.0.proj.{bad_tensor}",
        ):
            loader.load_model_with_fallback(str(base), adapter_path=str(directory))
        assert model.loads == []
    else:
        result = loader.load_model_with_fallback(str(base), adapter_path=str(directory))
        assert result == (model, tokenizer)
        assert len(model.loads) == 1
        assert model.layers[0].proj.lora_a.shape == (4, 2)
        assert model.layers[0].proj.lora_b.shape == (2, 6)


@pytest.mark.usefixtures("isolated_server_model_state")
@pytest.mark.parametrize("served_name", [None, "custom", "custom:lora:verbatim"])
def test_server_propagates_adapter_and_publishes_identity(
    adapter, monkeypatch, served_name
):
    from rapid_mlx import disk_caches, response_cache, server
    from rapid_mlx.runtime.model_registry import ModelRegistry
    from rapid_mlx.scheduler import SchedulerConfig

    directory, base = adapter
    expected = validate_adapter(str(directory), str(base))
    constructed = []

    class Engine:
        is_mllm = False
        _tokenizer = None

        def __init__(self, **kwargs):
            constructed.append(kwargs)

    (base / "model.safetensors").write_bytes(b"stub weights")
    monkeypatch.setattr(server, "BatchedEngine", Engine)
    monkeypatch.setattr(server, "_model_alias", None)
    monkeypatch.setattr(server, "_model_registry", ModelRegistry())
    monkeypatch.setattr(server, "_primary_lazy_load", False)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 0)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _: None)
    monkeypatch.setattr(server, "_prefetch_routing_metadata", lambda name: name)
    before = disk_caches.disabled()
    server.load_model(
        str(base),
        adapter_path=str(directory),
        served_model_name=served_name,
        scheduler_config=SchedulerConfig(response_cache_entries=8),
        force_text=True,
    )
    assert constructed[0]["adapter_path"] == str(directory)
    assert constructed[0]["no_spec_decode"] is True
    public_name = (
        served_name if served_name is not None else expected.model_identity(str(base))
    )
    assert server.get_config().model_name == public_name
    assert server._model_registry.list_entries()[0].model_name == public_name
    assert server._model_registry.list_entries()[0].is_adapter
    assert server._model_registry.is_adapter_model(public_name)
    assert response_cache.get_response_cache().enabled
    assert disk_caches.disabled() == before


@pytest.mark.parametrize("unloaded", [False, True])
@pytest.mark.parametrize("name", ["base:lora:abc", "custom", "alias"])
async def test_resident_reload_rejects_adapter_metadata(monkeypatch, name, unloaded):
    from rapid_mlx import server
    from rapid_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from rapid_mlx.runtime.resident_models import ResidentModelError

    registry = ModelRegistry()
    entry = ModelEntry(
        engine=SimpleNamespace(_adapter=object()),
        model_name=name if name != "alias" else "custom",
        model_path="base",
        aliases={"alias"},
    )
    registry.add(entry)
    if unloaded:
        registry.remove(entry.model_name)
        entry.engine = None
    monkeypatch.setattr(server, "_model_registry", registry)
    with pytest.raises(ValueError, match="--adapter identities cannot be reloaded"):
        await server._load_dynamic_resident_model(name, "base")
    with pytest.raises(ResidentModelError, match="--adapter is startup-only"):
        server._handoff_resident_primary_audio_worker(entry)


@pytest.mark.usefixtures("isolated_server_model_state")
async def test_resident_load_allows_lora_substring_without_adapter(
    monkeypatch, adapter
):
    from rapid_mlx import server
    from rapid_mlx.runtime.model_registry import ModelEntry, ModelRegistry

    _, base = adapter
    (base / "model.safetensors").write_bytes(b"stub weights")
    name = "custom:lora:verbatim"
    constructed = []

    class Engine:
        is_mllm = False

        def __init__(self, **kwargs):
            constructed.append(kwargs)

        async def start(self):
            pass

        def generate_warmup(self):
            pass

    registry = ModelRegistry()
    registry.add(ModelEntry(engine=Engine(), model_name=name, model_path=str(base)))
    monkeypatch.setattr(server, "_model_registry", registry)
    monkeypatch.setattr(server, "BatchedEngine", Engine)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _: None)
    monkeypatch.setattr(server, "resolve_serving_lane", lambda *_a, **_k: (False, True))
    for unloaded in (False, True):
        if unloaded:
            registry.remove(name)
        entry = await server._load_dynamic_resident_model(name, str(base))
        assert entry.model_name == name
        assert entry.model_path == str(base)
        assert isinstance(entry.engine, Engine)
    assert len(constructed) == 3


@pytest.mark.parametrize("disk_setting", ["enabled", "cli", "env"])
async def test_adapter_cache_scope_lasts_until_last_engine_stops(
    adapter, monkeypatch, disk_setting
):
    from unittest.mock import AsyncMock

    from rapid_mlx import disk_caches, response_cache
    from rapid_mlx.engine import batched

    directory, base = adapter
    disk_caches.configure(disk_setting == "cli")
    monkeypatch.setenv(disk_caches.ENV_VAR, "1" if disk_setting == "env" else "0")
    original_source = disk_caches.source()
    response_cache.configure_response_cache(8)
    configured_cache = response_cache.get_response_cache()
    monkeypatch.setattr(batched, "prepare_execution_host", AsyncMock())

    async def load():
        assert disk_caches.source() == "--adapter"
        assert not response_cache.get_response_cache().enabled

    engines = [
        batched.BatchedEngine(str(base), adapter_path=str(directory), force_text=True)
        for _ in range(2)
    ]
    for engine in engines:
        monkeypatch.setattr(engine, "_start_llm", load)
    assert disk_caches.source() == original_source
    assert response_cache.get_response_cache() is configured_cache
    try:
        for engine in engines:
            await engine.start()
            await engine.start()  # idempotent acquisition
        # Release out of acquisition order: the other engine still owns its scope.
        await engines[0].stop()
        await engines[0].stop()  # idempotent release
        assert disk_caches.source() == "--adapter"
        assert not response_cache.get_response_cache().enabled
        await engines[1].stop()
        assert disk_caches.source() == original_source
        assert response_cache.get_response_cache() is configured_cache
        assert configured_cache.capacity == 8
    finally:
        for engine in engines:
            await engine.stop()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_failed_adapter_start_releases_cache_overrides(
    adapter, monkeypatch, failure
):
    from unittest.mock import AsyncMock

    from rapid_mlx import disk_caches, response_cache
    from rapid_mlx.engine import batched

    directory, base = adapter
    before = disk_caches.source()
    response_cache.configure_response_cache(8)
    configured_cache = response_cache.get_response_cache()
    engine = batched.BatchedEngine(
        str(base), adapter_path=str(directory), force_text=True
    )
    monkeypatch.setattr(batched, "prepare_execution_host", AsyncMock())

    async def fail():
        assert disk_caches.source() == "--adapter"
        assert not response_cache.get_response_cache().enabled
        raise failure()

    monkeypatch.setattr(engine, "_start_llm", fail)
    try:
        with pytest.raises(failure):
            await engine.start()
        assert disk_caches.source() == before
        assert response_cache.get_response_cache() is configured_cache
        assert engine._adapter_cache_scope is None
    finally:
        await engine.stop()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("preserve_model_worker", [False, True])
async def test_failed_adapter_stop_keeps_cache_isolation_until_retry(
    adapter, monkeypatch, caplog, failure, preserve_model_worker
):
    from unittest.mock import AsyncMock, Mock

    from rapid_mlx import disk_caches, response_cache
    from rapid_mlx.engine import batched

    directory, base = adapter
    disk_caches.configure(False)
    monkeypatch.setenv(disk_caches.ENV_VAR, "0")
    response_cache.configure_response_cache(8)
    configured_cache = response_cache.get_response_cache()
    engine = batched.BatchedEngine(
        str(base), adapter_path=str(directory), force_text=True
    )
    monkeypatch.setattr(batched, "prepare_execution_host", AsyncMock())
    monkeypatch.setattr(engine, "_start_llm", AsyncMock())
    await engine.start()
    scope = engine._adapter_cache_scope
    stop_started = asyncio.Event()

    async def fail_stop():
        stop_started.set()
        if failure is asyncio.CancelledError:
            await asyncio.Future()
        raise failure("stop")

    # Exercise the real shutdown boundary with a core that is still serving.
    core = SimpleNamespace(stop=AsyncMock(side_effect=fail_stop), engine=Mock())
    engine._engine = core
    try:
        stopping = asyncio.create_task(
            engine.stop(preserve_model_worker=preserve_model_worker)
        )
        await stop_started.wait()
        if failure is asyncio.CancelledError:
            stopping.cancel()
        with pytest.raises(failure):
            await stopping
        assert engine._engine is core
        assert engine._loaded
        core.engine.close.assert_not_called()
        assert engine._adapter_cache_scope is scope
        assert disk_caches.source() == "--adapter"
        assert not response_cache.get_response_cache().enabled
        assert "retaining adapter cache isolation" in caplog.text
    finally:
        core.stop.side_effect = None
        await engine.stop(preserve_model_worker=preserve_model_worker)
    core.engine.close.assert_called_once()
    assert engine._adapter_cache_scope is None
    assert not disk_caches.disabled()
    assert response_cache.get_response_cache() is configured_cache


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize(
    "cleanup_failure", [None, RuntimeError, asyncio.CancelledError]
)
async def test_partial_adapter_start_releases_isolation_only_after_cleanup(
    adapter, monkeypatch, caplog, failure, cleanup_failure
):
    from unittest.mock import AsyncMock, Mock

    from rapid_mlx import disk_caches, response_cache
    from rapid_mlx.engine import batched

    directory, base = adapter
    disk_caches.configure(False)
    monkeypatch.setenv(disk_caches.ENV_VAR, "0")
    response_cache.configure_response_cache(8)
    configured_cache = response_cache.get_response_cache()
    engine = batched.BatchedEngine(
        str(base), adapter_path=str(directory), force_text=True
    )
    monkeypatch.setattr(batched, "prepare_execution_host", AsyncMock())
    start_error = failure("start")

    async def cleanup():
        assert disk_caches.source() == "--adapter"
        assert not response_cache.get_response_cache().enabled
        if cleanup_failure is not None:
            raise cleanup_failure("cleanup")

    core = SimpleNamespace(stop=AsyncMock(side_effect=cleanup), engine=Mock())

    async def partial_start():
        # A scheduler may exist before _loaded/_engine_started become true.
        engine._engine = core
        raise start_error

    monkeypatch.setattr(engine, "_start_llm", partial_start)
    try:
        with pytest.raises(failure) as caught:
            await engine.start()
        assert caught.value is start_error
        core.stop.assert_awaited_once()
        if cleanup_failure is None:
            core.engine.close.assert_called_once()
            assert engine._adapter_cache_scope is None
            assert not disk_caches.disabled()
            assert response_cache.get_response_cache() is configured_cache
        else:
            core.engine.close.assert_not_called()
            assert engine._adapter_cache_scope is not None
            assert disk_caches.source() == "--adapter"
            assert not response_cache.get_response_cache().enabled
            assert "retaining adapter cache isolation" in caplog.text
    finally:
        core.stop.side_effect = None
        await engine.stop()
    assert engine._adapter_cache_scope is None
    assert not disk_caches.disabled()
    assert response_cache.get_response_cache() is configured_cache


async def test_engine_without_adapter_preserves_cache_settings(adapter, monkeypatch):
    from unittest.mock import AsyncMock

    from rapid_mlx import disk_caches, response_cache
    from rapid_mlx.engine import batched

    _, base = adapter
    disk_caches.configure(False)
    monkeypatch.setenv(disk_caches.ENV_VAR, "0")
    response_cache.configure_response_cache(8)
    configured_cache = response_cache.get_response_cache()
    engine = batched.BatchedEngine(str(base), force_text=True)
    monkeypatch.setattr(batched, "prepare_execution_host", AsyncMock())

    async def load():
        assert not disk_caches.disabled()
        assert response_cache.get_response_cache() is configured_cache

    monkeypatch.setattr(engine, "_start_llm", load)
    try:
        await engine.start()
    finally:
        await engine.stop()
    assert not disk_caches.disabled()
    assert response_cache.get_response_cache() is configured_cache
