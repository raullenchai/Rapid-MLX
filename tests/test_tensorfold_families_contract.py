# SPDX-License-Identifier: Apache-2.0
"""Contracts for the data-driven TensorFold family profiles."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.speculative import tensorfold_families as families
from rapid_mlx.speculative.tensorfold_qwen27 import TensorFoldUnavailable

PROFILE_IDS = sorted(families.PROFILES)
REPO_ROOT = Path(__file__).parents[1]
DRAFT_REVISION = "a" * 40


def _paired(profile):
    return dataclasses.replace(
        profile,
        drafter="example/drafter",
        drafter_revision=DRAFT_REVISION,
        drafter_bits=8,
        drafter_architecture="DFlashDraftModel",
    )


def _snapshot(root: Path, name: str, revision: str, config: object) -> Path:
    path = root / name / "snapshots" / revision
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    return path


def _target(root: Path, profile, **overrides) -> Path:
    bits, group = profile.quantization
    config = {
        "model_type": profile.model_type,
        "quantization": {"bits": bits, "group_size": group},
    }
    config.update(overrides)
    return _snapshot(root, "target", profile.target_revision, config)


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_catalog_alias_matches_the_registered_profile(profile_id: str) -> None:
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile
    from rapid_mlx.speculative.tensorfold_runtime import SUPPORTED_REVISION

    profile = families.PROFILES[profile_id]
    alias = resolve_profile(profile_id)
    assert families.profile_for(profile_id) is profile
    assert alias.hf_path == profile.target
    assert alias.tensorfold_mtp is True
    assert alias.tensorfold_target_revision == profile.target_revision
    assert alias.tensorfold_runtime_revision == SUPPORTED_REVISION
    assert alias.min_memory_gb == profile.min_memory_gb
    assert resolve_profile(profile.fallback_model) is not None

    args = SimpleNamespace(
        model=profile_id, speculative_config=None, no_spec_decode=False, mllm=False
    )
    cli._normalize_speculative_config_or_exit(args)
    assert args.speculative_config == '{"method":"mtp","backend":"tensorfold"}'


def test_profile_lookup_ignores_unregistered_aliases() -> None:
    assert families.profile_for(None) is None
    assert families.profile_for("glm5.3-flash-tensorfold") is None


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_target_gates_fail_closed(profile_id: str, tmp_path: Path) -> None:
    profile = families.PROFILES[profile_id]
    families.validate_artifacts(profile, _target(tmp_path / "ok", profile), None)

    with pytest.raises(TensorFoldUnavailable, match="pinned Hugging Face snapshot"):
        families.validate_artifacts(profile, tmp_path, None)
    wrong_revision = _snapshot(tmp_path, "moved", "b" * 40, {})
    with pytest.raises(TensorFoldUnavailable, match="target revision"):
        families.validate_artifacts(profile, wrong_revision, None)
    wrong_type = _target(tmp_path / "type", profile, model_type="other")
    with pytest.raises(TensorFoldUnavailable, match="checkpoint layout"):
        families.validate_artifacts(profile, wrong_type, None)
    wrong_quant = _target(
        tmp_path / "quant", profile, quantization={"bits": 8, "group_size": 64}
    )
    with pytest.raises(TensorFoldUnavailable, match="checkpoint layout"):
        families.validate_artifacts(profile, wrong_quant, None)

    malformed = _target(tmp_path / "malformed", profile, quantization=["4bit"])
    with pytest.raises(TensorFoldUnavailable, match="checkpoint layout"):
        families.validate_artifacts(profile, malformed, None)

    unreadable = tmp_path / "bad" / "snapshots" / profile.target_revision
    unreadable.mkdir(parents=True)
    with pytest.raises(TensorFoldUnavailable, match="readable config.json"):
        families.validate_artifacts(profile, unreadable, None)
    (unreadable / "config.json").write_text("[]")
    with pytest.raises(TensorFoldUnavailable, match="not an object"):
        families.validate_artifacts(profile, unreadable, None)


def test_paired_drafter_gates_fail_closed(tmp_path: Path) -> None:
    profile = _paired(families.PROFILES[PROFILE_IDS[0]])
    target = _target(tmp_path, profile)
    drafter = _snapshot(
        tmp_path, "drafter", DRAFT_REVISION, {"architectures": ["DFlashDraftModel"]}
    )
    families.validate_artifacts(profile, target, drafter)

    with pytest.raises(TensorFoldUnavailable, match="requires its qualified drafter"):
        families.validate_artifacts(profile, target, None)
    moved = _snapshot(tmp_path, "moved", "c" * 40, {})
    with pytest.raises(TensorFoldUnavailable, match="drafter revision"):
        families.validate_artifacts(profile, target, moved)
    other = _snapshot(
        tmp_path / "other", "drafter", DRAFT_REVISION, {"architectures": ["Other"]}
    )
    with pytest.raises(TensorFoldUnavailable, match="draft model"):
        families.validate_artifacts(profile, target, other)


def test_download_resolves_only_pinned_snapshots(monkeypatch, tmp_path: Path) -> None:
    from rapid_mlx import _mirror

    profile = families.PROFILES[PROFILE_IDS[0]]
    paired = _paired(profile)
    target = _target(tmp_path, profile)
    drafter = _snapshot(
        tmp_path, "drafter", DRAFT_REVISION, {"architectures": ["DFlashDraftModel"]}
    )
    requested = []

    def fake_download(repo: str, revision: str) -> str:
        requested.append((repo, revision))
        return str(target if repo == profile.target else drafter)

    monkeypatch.setattr(_mirror, "pinned_snapshot_download", fake_download)
    alone = families.download_qualified_artifacts(profile)
    assert alone == families.FamilyArtifacts(str(target), "")
    both = families.download_qualified_artifacts(paired)
    assert both.drafter_path == str(drafter)
    assert requested == [
        (profile.target, profile.target_revision),
        (profile.target, profile.target_revision),
        ("example/drafter", DRAFT_REVISION),
    ]


def test_memory_gate_uses_the_profile_floor(monkeypatch) -> None:
    profile = families.PROFILES[PROFILE_IDS[0]]
    families.require_memory(profile, memory_gb=profile.min_memory_gb)
    with pytest.raises(TensorFoldUnavailable, match=f"{profile.min_memory_gb} GB Mac"):
        families.require_memory(profile, memory_gb=profile.min_memory_gb - 1)
    monkeypatch.setattr(families.os, "sysconf", lambda _name: 1)
    with pytest.raises(TensorFoldUnavailable):
        families.require_memory(profile)


def _install_fake_tensorfold(monkeypatch, family, seen: dict) -> None:
    def module(name: str, **attrs):
        mod = ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    class LaneEngine:
        prefill_step = 2048

        def __init__(self, model, **kwargs):
            seen["probe_engine"] = (model, kwargs)

    class PrefillPlan:
        def __init__(self, *args):
            self.args = args

    class ChatApp:
        def __init__(self, model, tokenizer, **kwargs):
            seen.update(app_model=model, app_tokenizer=tokenizer, app=kwargs)
            self.scheduler = SimpleNamespace(stop=lambda: None)

    def choose(make_engine, steps, budget, tokens, window):
        make_engine(steps[0])
        seen.update(steps=steps, budget=budget, tokens=tokens, window=window)
        return max(steps)

    mlx_core = module("mlx.core")
    module("mlx", core=mlx_core)
    module("tensorfold.families", detect=lambda _path: family)
    module(
        "tensorfold.engine",
        prefill_step=module("tensorfold.engine.prefill_step", choose=choose),
    )
    module("tensorfold.engine.lane_engine", LaneEngine=LaneEngine)
    module(
        "tensorfold.engine.prefill_plan",
        PrefillPlan=PrefillPlan,
        message_markers=lambda _tokenizer: ((1,), (2,)),
    )
    module("tensorfold.server.app", ChatApp=ChatApp)
    module(
        "tensorfold.server.memory_budget",
        PROCESS_BYTES=3 * 1024**3,
        configure_mlx=lambda _mx, cache, **kw: (
            seen.update(cache=cache, fraction=kw["fraction"]) or 40 * 1024**3
        ),
        model_fraction=lambda _package: 0.7,
    )
    module("tensorfold.server.prompt_memory", probe_tokens=lambda _tokenizer: [1, 2])
    module(
        "tensorfold.server.residency",
        wire_resident=lambda _mx, limit: seen.update(wired=limit),
    )


@pytest.mark.parametrize("paired", [False, True])
def test_loader_follows_the_upstream_serve_construction(
    monkeypatch, tmp_path: Path, paired: bool
) -> None:
    profile = families.PROFILES[PROFILE_IDS[0]]
    if paired:
        profile = _paired(profile)
    seen: dict = {}

    class Model:
        def release_rounds(self):
            seen["released"] = True

        def resolve_prefill_identity(self):
            seen["prefill_identity"] = True

    model, tokenizer = Model(), object()

    class Package:
        MLX_ENV = {"RAPID_TEST_TF_FAMILY": "1"}

        @staticmethod
        def load(path, **kwargs):
            seen.update(load_path=path, load_kwargs=kwargs)
            return model, tokenizer

        @staticmethod
        def engine_settings(_model):
            return {"max_rows": 12, "max_draft": 11, "prefill_steps": (4096, 2048)}

        @staticmethod
        def setup(app, loaded, **options):
            seen["setup"] = (app, loaded, options)

    family = SimpleNamespace(model_type=profile.model_type, package=Package)
    _install_fake_tensorfold(monkeypatch, family, seen)
    monkeypatch.delenv("RAPID_TEST_TF_FAMILY", raising=False)
    monkeypatch.setattr(families, "require_runtime", lambda: None)
    monkeypatch.setattr(families, "require_environment", lambda: None)
    monkeypatch.setattr(families, "require_memory", lambda _profile: None)
    monkeypatch.setattr(
        families,
        "validate_artifacts",
        lambda *args: seen.update(validated=args),
    )

    backend_class = families.backend_class_for(profile)
    backend = backend_class.load(
        str(tmp_path),
        str(tmp_path / "drafter") if paired else "",
        served_name="served",
        context_window=4096,
        max_tokens=256,
    )
    expected = {"lane_kernels": "auto", "parallel": 1}
    if paired:
        expected.update(drafter=str(tmp_path / "drafter"), drafter_bits=8)
    assert seen["load_kwargs"] == expected
    assert seen["validated"][0] is profile
    assert families.os.environ["RAPID_TEST_TF_FAMILY"] == "1"
    assert seen["released"] is True and seen["prefill_identity"] is True
    assert seen["steps"] == (4096, 2048)
    assert seen["window"] == 4096
    assert seen["wired"] == seen["budget"] == 37 * 1024**3
    assert (seen["cache"], seen["fraction"]) == (8 * 1024**3, 0.7)
    app = seen["app"]
    assert (app["max_rows"], app["max_draft"], app["lanes"]) == (12, 11, 1)
    assert app["context_window"] == 4096 and app["fit_context"] is False
    assert app["default_max_tokens"] == 256
    assert app["enable_thinking"] is True and app["checkpoint_slots"] == 0
    assert app["engine_factory"].keywords["prefill_plan"].args[0] == 4096
    assert app["engine_factory"].keywords["prefill_pass"] == 8
    assert seen["setup"][1] is model
    backend.close()

    # A failing setup hook closes the already-running app before it surfaces.
    stopped: list[bool] = []

    def failing_setup(app, loaded, **options):
        app.scheduler = SimpleNamespace(stop=lambda: stopped.append(True))
        raise RuntimeError("setup failed")

    Package.setup = staticmethod(failing_setup)
    with pytest.raises(RuntimeError, match="setup failed"):
        backend_class.load(str(tmp_path), "", served_name="served")
    assert stopped == [True]

    # A family without optional hooks still loads, and fits its own context.
    del Package.setup, Package.MLX_ENV, Model.resolve_prefill_identity
    Package.engine_settings = staticmethod(lambda _model: {})
    backend_class.load(str(tmp_path), "", served_name="served").close()
    assert seen["steps"] == (2048,)
    assert seen["app"]["fit_context"] is True
    assert (seen["app"]["max_rows"], seen["app"]["max_draft"]) == (16, 32)

    family.model_type = "wrong"
    with pytest.raises(TensorFoldUnavailable, match="did not select"):
        backend_class.load(str(tmp_path), "", served_name="served")


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_server_wrapper_declares_product_metadata(monkeypatch, profile_id: str) -> None:
    from rapid_mlx.speculative import tensorfold_qwen27_server

    profile = families.PROFILES[profile_id]
    captured = {}
    monkeypatch.setattr(
        tensorfold_qwen27_server,
        "run_tensorfold_qwen27_server",
        lambda **kwargs: captured.update(kwargs),
    )
    families.run_tensorfold_family_server(profile, main_model_repo="/target")
    assert captured["backend_class"].profile is profile
    assert issubclass(captured["backend_class"], families.TensorFoldFamilyBackend)
    assert captured["profile_id"] == profile_id
    assert captured["method"] == profile.method
    assert captured["target_revision"] == profile.target_revision
    assert captured["paired_repository"] == profile.drafter
    assert captured["min_memory_gb"] == profile.min_memory_gb
    assert captured["supports_reasoning_budget"] is True
    assert captured["main_model_repo"] == "/target"


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_cli_dispatches_family_profile(monkeypatch, profile_id: str) -> None:
    from rapid_mlx import cli

    captured = {}
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_resolved_serve_port", lambda _args: 8123)
    monkeypatch.setattr(cli, "port_explicit_for", lambda _args: True)
    monkeypatch.setattr(
        families,
        "run_tensorfold_family_server",
        lambda profile, **kwargs: captured.update(kwargs, profile=profile),
    )
    server = SimpleNamespace(
        _sync_config=lambda: None,
        _api_key=None,
        _max_request_bytes=1024,
        _body_receive_timeout_seconds=1.0,
        _default_timeout=2.0,
        get_resolved_cors_policy=lambda: None,
    )
    args = SimpleNamespace(
        mtp_backend="tensorfold",
        _original_alias=profile_id,
        model="/pinned/target",
        force_disk_check=False,
        host="127.0.0.1",
        served_model_name=None,
        no_thinking=False,
        rate_limit=0,
        max_concurrent_requests=8,
        reasoning_parser="qwen3",
        default_reasoning_effort=None,
    )
    assert cli._serve_tensorfold_mtp_if_requested(
        args,
        server_module=server,
        effective_max_tokens=512,
        cors_origins=[],
        uvicorn_log_level="info",
    )
    assert captured["profile"] is families.PROFILES[profile_id]
    assert captured["main_model_repo"] == "/pinned/target"
    assert captured["served_model_name"] == profile_id
    assert captured["reasoning_parser_name"] == "qwen3"


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_cli_preflight_reports_the_family(monkeypatch, capsys, profile_id: str) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative import tensorfold_qwen27

    profile = families.PROFILES[profile_id]
    args = SimpleNamespace(model=profile_id, _original_alias=profile_id)
    monkeypatch.setattr(tensorfold_qwen27, "require_runtime", lambda: None)
    monkeypatch.setattr(tensorfold_qwen27, "require_environment", lambda: None)
    monkeypatch.setattr(families, "require_memory", lambda _profile: None)
    cli._preflight_tensorfold_qwen27_or_exit(args)

    def too_small(_profile):
        raise TensorFoldUnavailable("needs more memory")

    monkeypatch.setattr(families, "require_memory", too_small)
    with pytest.raises(SystemExit, match="1"):
        cli._preflight_tensorfold_qwen27_or_exit(args)
    err = capsys.readouterr().err
    assert profile.label.removeprefix("TensorFold ") in err
    assert "needs more memory" in err


@pytest.mark.parametrize("flag", ["no_spec_decode", "mllm"])
@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_opt_out_needs_a_checkpoint_the_ordinary_engine_loads(
    capsys, profile_id: str, flag: str
) -> None:
    from rapid_mlx import cli

    profile = families.PROFILES[profile_id]
    args = SimpleNamespace(
        model=profile.target,
        _original_alias=profile_id,
        speculative_config=None,
        no_spec_decode=flag == "no_spec_decode",
        mllm=flag == "mllm",
    )
    if profile.ordinary_engine:
        cli._normalize_speculative_config_or_exit(args)
        assert args.speculative_config is None
        return
    with pytest.raises(SystemExit, match="2"):
        cli._normalize_speculative_config_or_exit(args)
    assert profile.fallback_model in capsys.readouterr().err


def test_only_the_measured_fallback_is_declared() -> None:
    ordinary = {p.profile_id for p in families.PROFILES.values() if p.ordinary_engine}
    assert ordinary == {"nemotron-3.5-lightning-tensorfold"}


def test_models_reference_documents_every_family_profile() -> None:
    reference = (REPO_ROOT / "docs" / "reference" / "models.md").read_text()
    for profile_id in PROFILE_IDS:
        assert f"`{profile_id}`" in reference


def test_preload_probes_leave_mlx_unimported_for_family_env() -> None:
    """The family's MLX settings are read once, at MLX import.

    The probes that run ahead of them must therefore never import MLX.
    """
    script = (
        "import sys\n"
        "import rapid_mlx.cli\n"
        "import rapid_mlx.speculative.tensorfold_qwen27_server\n"
        "from rapid_mlx.speculative import tensorfold_families as families\n"
        "from rapid_mlx.speculative.tensorfold_qwen27 import SUPPORTED_MLX_VERSION\n"
        "families.require_environment(\n"
        "    mlx_version=SUPPORTED_MLX_VERSION, machine='arm64'\n"
        ")\n"
        "families.require_memory(\n"
        "    families.PROFILES['nemotron-3.5-lightning-tensorfold'], memory_gb=512\n"
        ")\n"
        "sys.exit(1 if 'mlx.core' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
