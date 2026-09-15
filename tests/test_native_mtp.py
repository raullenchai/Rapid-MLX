from __future__ import annotations

import builtins
import concurrent.futures
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException

from rapid_mlx.speculative.native_mtp.eligibility import (
    GLM53_FLASH_4BIT,
    QWEN36_35B_4BIT,
    NativeMTPUnavailableError,
    resolve_native_mtp_pair,
)
from rapid_mlx.speculative.native_mtp.server import _validate_greedy_request

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_native_mtp_resolves_only_qualified_pair() -> None:
    pair = resolve_native_mtp_pair(
        alias="qwen3.6-35b-4bit",
        target_repo=QWEN36_35B_4BIT.target_repo,
        drafter_repo=QWEN36_35B_4BIT.drafter_repo,
        draft_tokens=2,
    )
    assert pair == QWEN36_35B_4BIT
    assert pair.block_size == pair.draft_tokens + 1


def test_glm53_native_mtp_resolves_immutable_public_pair() -> None:
    pair = resolve_native_mtp_pair(
        alias="glm5.3-flash-4bit",
        target_repo=GLM53_FLASH_4BIT.target_repo,
        drafter_repo=GLM53_FLASH_4BIT.drafter_repo,
        draft_tokens=1,
    )
    assert pair == GLM53_FLASH_4BIT
    assert pair.block_size == pair.draft_tokens + 1
    assert pair.drafter_model_type == "glm5_next_mtp"

    full_repo_pair = resolve_native_mtp_pair(
        alias=GLM53_FLASH_4BIT.target_repo,
        target_repo=GLM53_FLASH_4BIT.target_repo,
        drafter_repo=GLM53_FLASH_4BIT.drafter_repo,
        draft_tokens=1,
    )
    assert full_repo_pair == GLM53_FLASH_4BIT


@pytest.mark.parametrize(
    "overrides",
    [
        {"alias": "qwen3.6-35b-8bit"},
        {"target_repo": "mlx-community/Qwen3.6-35B-A3B-8bit"},
        {"drafter_repo": "other/model"},
        {"draft_tokens": 3},
    ],
)
def test_native_mtp_rejects_unqualified_pair(overrides: dict[str, object]) -> None:
    kwargs = {
        "alias": "qwen3.6-35b-4bit",
        "target_repo": QWEN36_35B_4BIT.target_repo,
        "drafter_repo": QWEN36_35B_4BIT.drafter_repo,
        "draft_tokens": 2,
    }
    kwargs.update(overrides)
    with pytest.raises(NativeMTPUnavailableError):
        resolve_native_mtp_pair(**kwargs)


def test_native_mtp_accepts_greedy_without_penalties() -> None:
    _validate_greedy_request(
        SimpleNamespace(
            temperature=0,
            repetition_penalty=None,
            presence_penalty=0,
            frequency_penalty=0,
        )
    )


def test_native_mtp_accepts_sampling_for_ar_fallback() -> None:
    _validate_greedy_request(SimpleNamespace(temperature=0.7))


@pytest.mark.parametrize(
    "penalty",
    [
        {"repetition_penalty": 1.1},
        {"presence_penalty": 0.2},
        {"frequency_penalty": -0.1},
    ],
)
def test_native_mtp_rejects_sampling_penalties(penalty: dict[str, float]) -> None:
    request = SimpleNamespace(temperature=0, **penalty)
    with pytest.raises(HTTPException, match="penalties") as exc_info:
        _validate_greedy_request(request)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "field,value",
    [("logprobs", True), ("top_logprobs", 2)],
)
def test_native_mtp_rejects_logprobs(field: str, value: object) -> None:
    request = SimpleNamespace(temperature=0, **{field: value})
    with pytest.raises(HTTPException, match="logprobs") as exc_info:
        _validate_greedy_request(request)
    assert exc_info.value.status_code == 400


def test_native_mtp_runtime_probe_is_exact_version(monkeypatch) -> None:
    from rapid_mlx.speculative.native_mtp import runtime

    monkeypatch.setattr(runtime, "find_spec", lambda _name: object())
    monkeypatch.setattr(runtime, "version", lambda _name: "0.7.1")
    assert runtime.have_runtime() is True

    monkeypatch.setattr(runtime, "version", lambda _name: "0.7.0")
    assert runtime.have_runtime() is False

    monkeypatch.setattr(
        runtime,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(RuntimeError("broken optional stack")),
    )
    assert runtime.have_runtime() is False


def test_load_native_mtp_runtime_reports_missing_optional_runtime(monkeypatch) -> None:
    from rapid_mlx.speculative.native_mtp.runtime import load_runtime

    real_import = builtins.__import__

    def fail_drafter_import(name, *args, **kwargs):
        if name == "mlx_vlm.speculative.drafters":
            raise ImportError("missing native runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_drafter_import)
    with pytest.raises(RuntimeError, match=r"rapid-mlx\[mtp\]"):
        load_runtime(
            "org/drafter",
            target_revision="a" * 40,
            drafter_revision="b" * 40,
            block_size=3,
        )


def test_mtp_extra_carries_the_qualified_native_runtime() -> None:
    from rapid_mlx.speculative.native_mtp.runtime import QUALIFIED_MLX_VLM_VERSION

    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]

    assert f"mlx-vlm=={QUALIFIED_MLX_VLM_VERSION}" in extras["mtp"]


def _fake_mlx_vlm_modules(monkeypatch, drafter, kind: str = "mtp") -> None:
    root = ModuleType("mlx_vlm")
    root.__path__ = []
    speculative = ModuleType("mlx_vlm.speculative")
    speculative.__path__ = []
    drafters = ModuleType("mlx_vlm.speculative.drafters")
    drafters.load_drafter = lambda source, kind: (drafter, kind)
    utils = ModuleType("mlx_vlm.utils")
    utils.get_model_path = lambda repo, revision: f"/{repo}@{revision}"
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", speculative)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", drafters)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)


def test_load_native_mtp_runtime_validates_architecture_and_block(monkeypatch) -> None:
    from rapid_mlx.speculative.native_mtp.runtime import load_runtime

    drafter = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_mtp", block_size=3),
        accept_lens=[1],
        draft_lens=[2],
    )
    _fake_mlx_vlm_modules(monkeypatch, drafter)
    runtime = load_runtime(
        "org/drafter",
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        block_size=3,
    )
    assert runtime.kind == "mtp"
    assert runtime.algorithm == "mtp"
    assert runtime.accept_lens_snapshot() == [1]
    runtime.reset_accept_lens()
    assert drafter.accept_lens == []
    assert drafter.draft_lens == []

    drafter.config.model_type = "wrong_architecture"
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_runtime(
            "org/drafter",
            target_revision="a" * 40,
            drafter_revision="b" * 40,
            block_size=3,
        )

    drafter.config.model_type = "qwen3_5_mtp"
    drafter.config.block_size = 4
    with pytest.raises(RuntimeError, match="block-size mismatch"):
        load_runtime(
            "org/drafter",
            target_revision="a" * 40,
            drafter_revision="b" * 40,
            block_size=3,
        )


def test_glm_runtime_fails_before_resolving_sidecar(monkeypatch) -> None:
    from rapid_mlx.speculative.native_mtp import runtime

    drafter = SimpleNamespace(
        config=SimpleNamespace(model_type="glm5_next_mtp", block_size=2)
    )
    _fake_mlx_vlm_modules(monkeypatch, drafter)
    monkeypatch.setattr(runtime, "have_glm_cache_runtime", lambda: False)
    sys.modules["mlx_vlm.utils"].get_model_path = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("must not resolve or download sidecar"))

    with pytest.raises(RuntimeError, match="cache-owned GLM runtime"):
        runtime.load_runtime(
            "org/glm-drafter",
            target_revision="a" * 40,
            drafter_revision="b" * 40,
            block_size=2,
            expected_model_type="glm5_next_mtp",
        )


def test_glm_runtime_structural_probe_and_fail_closed(monkeypatch) -> None:
    from rapid_mlx.patches import glm5_next_runtime as glm_patch
    from rapid_mlx.speculative.native_mtp import runtime

    root = ModuleType("mlx_vlm")
    root.__path__ = []
    generate = ModuleType("mlx_vlm.generate")
    generate.__path__ = []
    ar = ModuleType("mlx_vlm.generate.ar")
    for name in (
        "generate_step",
        "SpeculativePrefill",
        "run_speculative_rounds",
        "speculative_prefill_kwargs",
    ):
        setattr(ar, name, object())
    generate.ar = ar
    models = ModuleType("mlx_vlm.models")
    models.__path__ = []
    cache = ModuleType("mlx_vlm.models.cache")
    methods = (
        "start_speculation",
        "validate_speculation",
        "commit_speculation",
        "abort_speculation",
    )
    arrays = type("ArraysCache", (), {name: lambda self: None for name in methods})
    pooling = type("PoolingCache", (), {name: lambda self: None for name in methods})
    cache.ArraysCache = arrays
    cache.PoolingCache = pooling
    glm = ModuleType("mlx_vlm.models.glm5_next")
    glm.__path__ = []
    language = ModuleType("mlx_vlm.models.glm5_next.language")
    glm.language = language
    speculative = ModuleType("mlx_vlm.speculative")
    speculative.__path__ = []
    drafters = ModuleType("mlx_vlm.speculative.drafters")
    drafters.__path__ = []
    drafters.load_drafter = object()
    mtp = ModuleType("mlx_vlm.speculative.drafters.glm5_next_mtp")
    mtp.Glm5NextMTPDraftModel = type(
        "Glm5NextMTPDraftModel", (), {"_RAPID_STATELESS_GLM_MTP": True}
    )
    compat = ModuleType("rapid_mlx.speculative.native_mtp.glm5_compat")
    compat.install_glm5_mtp_compatibility = lambda: False
    compat._is_stateless_drafter = lambda drafter_type: bool(
        getattr(drafter_type, "_RAPID_STATELESS_GLM_MTP", False)
    )
    for name, module in {
        "mlx_vlm": root,
        "mlx_vlm.generate": generate,
        "mlx_vlm.generate.ar": ar,
        "mlx_vlm.models": models,
        "mlx_vlm.models.cache": cache,
        "mlx_vlm.models.glm5_next": glm,
        "mlx_vlm.models.glm5_next.language": language,
        "mlx_vlm.speculative": speculative,
        "mlx_vlm.speculative.drafters": drafters,
        "mlx_vlm.speculative.drafters.glm5_next_mtp": mtp,
        "rapid_mlx.speculative.native_mtp.glm5_compat": compat,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(glm_patch, "_has_native_glm5_next_runtime", lambda _mod: True)

    assert runtime.have_glm_cache_runtime() is True
    delattr(ar, "run_speculative_rounds")
    assert runtime.have_glm_cache_runtime() is False

    real_import = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == "mlx_vlm.generate":
            raise RuntimeError("broken runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    assert runtime.have_glm_cache_runtime() is False


def test_qualified_glm_release_gets_rapid_stateless_adapter() -> None:
    from importlib.metadata import PackageNotFoundError, version

    from rapid_mlx.speculative.native_mtp import runtime

    try:
        installed = version("mlx-vlm")
    except PackageNotFoundError:
        pytest.skip("mlx-vlm optional runtime is not installed")
    if installed != runtime.QUALIFIED_MLX_VLM_VERSION:
        pytest.skip(f"test requires mlx-vlm {runtime.QUALIFIED_MLX_VLM_VERSION}")

    assert runtime.have_glm_cache_runtime() is True

    from mlx_vlm.speculative.drafters.glm5_next_mtp import (
        Glm5NextMTPDraftModel,
        Model,
    )

    assert Glm5NextMTPDraftModel is Model
    assert Glm5NextMTPDraftModel._RAPID_STATELESS_GLM_MTP is True

    from mlx_vlm.generate import ar, dispatch

    assert dispatch.generate_step is ar.generate_step


def test_load_glm_runtime_installs_rapid_hooks(monkeypatch) -> None:
    from rapid_mlx.speculative.native_mtp import runtime, transaction

    drafter = SimpleNamespace(
        config=SimpleNamespace(model_type="glm5_next_mtp", block_size=2)
    )
    _fake_mlx_vlm_modules(monkeypatch, drafter)
    monkeypatch.setattr(runtime, "have_glm_cache_runtime", lambda: True)
    installed = []
    monkeypatch.setattr(
        transaction, "install_generation_hooks", lambda: installed.append(True)
    )
    loaded = runtime.load_runtime(
        "org/glm-drafter",
        target_revision="a" * 40,
        drafter_revision="b" * 40,
        block_size=2,
        expected_model_type="glm5_next_mtp",
    )
    assert loaded.model_type == "glm5_next_mtp"
    assert installed == [True]


def test_native_mtp_stats_ignore_non_list_counters() -> None:
    from rapid_mlx.speculative.native_mtp.runtime import NativeMTPRuntime

    drafter = SimpleNamespace(accept_lens=None, draft_lens=(1, 2))
    runtime = NativeMTPRuntime(drafter, "repo", "target", "draft", 3)
    runtime.reset_accept_lens()
    assert runtime.accept_lens_snapshot() == []


def test_serve_native_mtp_helper_routes_exact_pair(monkeypatch) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative.native_mtp import server as native_server

    disk_checks = []
    capacity_checks = []
    captured = {}
    monkeypatch.setattr(
        cli, "_check_disk_space", lambda model, force=False: disk_checks.append(model)
    )
    monkeypatch.setattr(
        cli,
        "_check_memory_capacity",
        lambda model, alias=None: capacity_checks.append((model, alias)),
    )
    monkeypatch.setattr(
        native_server,
        "run_native_mtp_server",
        lambda **kwargs: captured.update(kwargs),
    )
    preflight_calls = []
    monkeypatch.setattr(
        cli,
        "_preflight_native_mtp_or_exit",
        lambda args: preflight_calls.append(args) or QWEN36_35B_4BIT,
    )
    sync_calls = []
    server_stub = SimpleNamespace(
        _api_key="secret",
        _max_request_bytes=123,
        _body_receive_timeout_seconds=4.0,
        _default_timeout=5.0,
        _sync_config=lambda: sync_calls.append(True),
        get_resolved_cors_policy=lambda: "cors-policy",
    )
    args = SimpleNamespace(
        mtp_backend="native",
        mcp_config=None,
        embedding_model=None,
        enable_disk_stream=False,
        mllm=False,
        mtp_continuous_batching=False,
        _original_alias="qwen3.6-35b-4bit",
        model=QWEN36_35B_4BIT.target_repo,
        mtp_sidecar=QWEN36_35B_4BIT.drafter_repo,
        mtp_max_k=2,
        force_disk_check=False,
        host="127.0.0.1",
        port=8766,
        served_model_name=None,
        no_thinking=True,
        rate_limit=7,
        max_concurrent_requests=8,
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder_xml",
        reasoning_parser="qwen3",
    )

    assert cli._serve_native_mtp_if_requested(
        args,
        server_module=server_stub,
        effective_max_tokens=192,
        cors_origins=["http://localhost"],
        uvicorn_log_level="warning",
    )
    assert disk_checks == [
        QWEN36_35B_4BIT.target_repo,
        QWEN36_35B_4BIT.drafter_repo,
    ]
    assert capacity_checks == [(QWEN36_35B_4BIT.target_repo, "qwen3.6-35b-4bit")]
    assert sync_calls == [True]
    assert preflight_calls == [args]
    assert captured["pair"] == QWEN36_35B_4BIT
    assert captured["served_model_name"] == "qwen3.6-35b-4bit"


def test_native_mtp_preflight_rejects_wrong_alias_before_runtime_probe(
    monkeypatch,
) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative.native_mtp import runtime

    runtime_probes = []
    monkeypatch.setattr(
        runtime, "have_runtime", lambda: runtime_probes.append(True) or True
    )
    args = SimpleNamespace(
        mtp_backend="native",
        model="mlx-community/Qwen3.6-35B-A3B-4bit",
        _original_alias="another-model",
        mtp_sidecar=QWEN36_35B_4BIT.drafter_repo,
        mtp_max_k=2,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli._preflight_native_mtp_or_exit(args)

    assert exc_info.value.code == 2
    assert runtime_probes == []


def test_native_mtp_preflight_caches_pair(monkeypatch) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative.native_mtp import runtime

    monkeypatch.setattr(runtime, "have_runtime", lambda: True)
    args = SimpleNamespace(
        mtp_backend="native",
        model=QWEN36_35B_4BIT.target_repo,
        _original_alias="qwen3.6-35b-4bit",
        mtp_sidecar=QWEN36_35B_4BIT.drafter_repo,
        mtp_max_k=2,
    )

    assert cli._preflight_native_mtp_or_exit(args) == QWEN36_35B_4BIT
    assert args._native_mtp_pair == QWEN36_35B_4BIT


def test_native_mtp_preflight_is_noop_for_standard_backend() -> None:
    from rapid_mlx.cli import _preflight_native_mtp_or_exit

    assert _preflight_native_mtp_or_exit(SimpleNamespace(mtp_backend=None)) is None


def test_native_mtp_preflight_rejects_all_unsupported_features(capsys) -> None:
    from rapid_mlx.cli import _preflight_native_mtp_or_exit

    args = SimpleNamespace(
        mtp_backend="native",
        mcp_config="mcp.json",
        embedding_model="org/embed",
        enable_disk_stream=True,
        mllm=True,
        mtp_continuous_batching=True,
    )

    with pytest.raises(SystemExit) as exc_info:
        _preflight_native_mtp_or_exit(args)

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    for expected in (
        "--mcp-config",
        "--embedding-model",
        "--disk-stream",
        "--mllm",
        "continuous MTP",
    ):
        assert expected in stderr


def test_native_mtp_preflight_reports_missing_runtime(monkeypatch, capsys) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative.native_mtp import runtime

    monkeypatch.setattr(runtime, "have_runtime", lambda: False)
    args = SimpleNamespace(
        mtp_backend="native",
        model=QWEN36_35B_4BIT.target_repo,
        _original_alias="qwen3.6-35b-4bit",
        mtp_sidecar=QWEN36_35B_4BIT.drafter_repo,
        mtp_max_k=2,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli._preflight_native_mtp_or_exit(args)

    assert exc_info.value.code == 1
    assert "rapid-mlx[mtp]" in capsys.readouterr().err


def test_glm_preflight_rejects_old_runtime_before_weight_load(
    monkeypatch, capsys
) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative.native_mtp import runtime

    monkeypatch.setattr(runtime, "have_runtime", lambda: True)
    monkeypatch.setattr(runtime, "have_glm_cache_runtime", lambda: False)
    args = SimpleNamespace(
        mtp_backend="native",
        model=GLM53_FLASH_4BIT.target_repo,
        _original_alias="glm5.3-flash-4bit",
        mtp_sidecar=GLM53_FLASH_4BIT.drafter_repo,
        mtp_max_k=1,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli._preflight_native_mtp_or_exit(args)

    assert exc_info.value.code == 1
    assert "structurally incompatible" in capsys.readouterr().err
    assert not hasattr(args, "_native_mtp_pair")


def test_serve_native_mtp_helper_is_noop_for_standard_backend() -> None:
    from rapid_mlx.cli import _serve_native_mtp_if_requested

    assert (
        _serve_native_mtp_if_requested(
            SimpleNamespace(mtp_backend=None),
            server_module=None,
            effective_max_tokens=1,
            cors_origins=[],
            uvicorn_log_level="warning",
        )
        is False
    )


def test_native_mtp_server_builds_qualified_serial_app(monkeypatch) -> None:
    from rapid_mlx.speculative.native_mtp import server as native_server

    class ImmediateExecutor:
        def submit(self, fn, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(*args, **kwargs))
            return future

    loaded = []
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.load = lambda repo, revision: (
        loaded.append((repo, revision)) or "model",
        "processor",
    )
    uvicorn = ModuleType("uvicorn")
    run_calls = []
    uvicorn.run = lambda app, **kwargs: run_calls.append((app, kwargs))
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    bound = []
    drafter = SimpleNamespace(bind=lambda model: bound.append(model))
    runtime = SimpleNamespace(
        drafter=drafter,
        kind="mtp",
        block_size=3,
        algorithm="mtp",
        drafter_repo=QWEN36_35B_4BIT.drafter_repo,
        target_revision=QWEN36_35B_4BIT.target_revision,
        drafter_revision=QWEN36_35B_4BIT.drafter_revision,
    )
    monkeypatch.setattr(native_server, "load_runtime", lambda *args, **kwargs: runtime)

    app_kwargs = {}
    dflash_server = ModuleType("rapid_mlx.speculative.dflash.server")
    dflash_server._dflash_executor = ImmediateExecutor()
    dflash_server._build_app = lambda **kwargs: app_kwargs.update(kwargs) or "app"
    monkeypatch.setitem(
        sys.modules, "rapid_mlx.speculative.dflash.server", dflash_server
    )

    native_server.run_native_mtp_server(
        pair=QWEN36_35B_4BIT,
        host="127.0.0.1",
        port=8766,
        served_model_name="qwen3.6-35b-4bit",
        default_max_tokens=192,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    assert loaded == [(QWEN36_35B_4BIT.target_repo, QWEN36_35B_4BIT.target_revision)]
    assert bound == ["model"]
    assert app_kwargs["backend_name"] == "Native MTP"
    assert app_kwargs["runtime"] is runtime
    assert app_kwargs["generation_kwargs_fn"](
        max_tokens=9, temperature=0.0, top_p=1.0
    ) == {
        "max_tokens": 9,
        "temperature": 0.0,
        "top_p": 1.0,
        "draft_model": drafter,
        "draft_kind": "mtp",
        "draft_block_size": 3,
    }
    assert app_kwargs["generation_kwargs_fn"](
        max_tokens=9, temperature=0.7, top_p=0.9
    ) == {
        "max_tokens": 9,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    assert run_calls == [
        (
            "app",
            {
                "host": "127.0.0.1",
                "port": 8766,
                "log_level": "warning",
                "timeout_keep_alive": 30,
            },
        )
    ]


def test_native_mtp_server_sanitizes_glm_target_before_load(monkeypatch) -> None:
    from rapid_mlx.speculative.native_mtp import server as native_server

    class ImmediateExecutor:
        def submit(self, fn, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(*args, **kwargs))
            return future

    events = []
    runtime_patch = ModuleType("rapid_mlx.patches.glm5_next_runtime")
    runtime_patch.install_glm5_next_runtime_fix = lambda: events.append("sanitize")
    monkeypatch.setitem(
        sys.modules, "rapid_mlx.patches.glm5_next_runtime", runtime_patch
    )

    mlx_vlm = ModuleType("mlx_vlm")

    def _load(repo, revision):
        events.append(("load", repo, revision))
        return "model", "processor"

    mlx_vlm.load = _load
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    drafter = SimpleNamespace(bind=lambda _model: None)
    runtime = SimpleNamespace(
        drafter=drafter,
        kind="mtp",
        block_size=2,
        algorithm="mtp",
        drafter_repo=GLM53_FLASH_4BIT.drafter_repo,
        target_revision=GLM53_FLASH_4BIT.target_revision,
        drafter_revision=GLM53_FLASH_4BIT.drafter_revision,
    )
    monkeypatch.setattr(native_server, "load_runtime", lambda *args, **kwargs: runtime)

    dflash_server = ModuleType("rapid_mlx.speculative.dflash.server")
    dflash_server._dflash_executor = ImmediateExecutor()
    dflash_server._build_app = lambda **_kwargs: "app"
    monkeypatch.setitem(
        sys.modules, "rapid_mlx.speculative.dflash.server", dflash_server
    )

    native_server.run_native_mtp_server(
        pair=GLM53_FLASH_4BIT,
        host="127.0.0.1",
        port=8766,
        served_model_name="glm5.3-flash-4bit",
        default_max_tokens=192,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    assert events == [
        "sanitize",
        (
            "load",
            GLM53_FLASH_4BIT.target_repo,
            GLM53_FLASH_4BIT.target_revision,
        ),
    ]


def test_native_mtp_server_reports_missing_optional_runtime(monkeypatch) -> None:
    from rapid_mlx.speculative.native_mtp import server as native_server

    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(RuntimeError, match=r"rapid-mlx\[mtp\]"):
        native_server.run_native_mtp_server(
            pair=QWEN36_35B_4BIT,
            host="127.0.0.1",
            port=8766,
            served_model_name="qwen3.6-35b-4bit",
            default_max_tokens=192,
            cors_origins=[],
            uvicorn_log_level="warning",
        )
