# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from rapid_mlx.cli import _resolve_system_one_backend, build_parser, system_one_command


def test_system_one_cli_defaults_to_laya_service():
    args = build_parser().parse_args(["system-one"])
    assert args.model == "convaiinnovations/laya"
    assert args.backend == "auto"
    assert args.host == "127.0.0.1"
    assert args.port == 8700
    assert args.max_concurrent_requests == 8


def test_system_one_cli_accepts_clm_runtime_inputs():
    args = build_parser().parse_args(
        [
            "system-one",
            "clm-latest",
            "--backend",
            "clm",
            "--head",
            "/tmp/head",
            "--encoder",
            "Qwen/Qwen3-8B",
        ]
    )
    assert args.head == "/tmp/head"
    assert args.encoder == "Qwen/Qwen3-8B"


def test_system_one_cli_rejects_non_positive_laya_batch_size():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["system-one", "--batch-size", "0"])


def test_system_one_auto_backend_does_not_substring_match_clm():
    assert _resolve_system_one_backend("org/aclmish-laya", "auto") == "laya"
    assert _resolve_system_one_backend("Contrastive-LM/CLM-v0.1-8B", "auto") == "clm"


def test_system_one_extra_is_optional_and_included_in_all():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert all(not item.startswith("laya-mlx") for item in project["dependencies"])
    extras = project["optional-dependencies"]
    assert any(item.startswith("laya-mlx") for item in extras["system-one"])
    assert any(item.startswith("laya-mlx") for item in extras["all"])


def test_system_one_checks_port_before_backend_initialization(monkeypatch):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.cli as cli_module
    import rapid_mlx.system_one.backends as backend_module

    events = []

    class Backend:
        default_model = "laya"

        def __init__(self, *args, **kwargs):
            events.append("backend")

        def models(self):
            return []

    monkeypatch.setattr(
        cli_module,
        "_port_preflight_or_die",
        lambda *args, **kwargs: events.append("port"),
    )
    monkeypatch.setattr(backend_module, "LayaBackend", Backend)
    monkeypatch.setattr(
        uvicorn_module, "run_uvicorn", lambda *args, **kwargs: events.append("serve")
    )
    system_one_command(build_parser().parse_args(["system-one"]))
    assert events == ["port", "backend", "serve"]


def test_system_one_passes_public_clm_model_name(monkeypatch):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.cli as cli_module
    import rapid_mlx.system_one.backends as backend_module

    captured = {}

    class Backend:
        default_model = "public-clm"

        def __init__(self, encoder, head, **kwargs):
            captured.update(encoder=encoder, head=head, **kwargs)

        def models(self):
            return []

    monkeypatch.setattr(
        cli_module, "_port_preflight_or_die", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(backend_module, "CLMBackend", Backend)
    monkeypatch.setattr(uvicorn_module, "run_uvicorn", lambda *args, **kwargs: None)
    args = build_parser().parse_args(
        ["system-one", "public-clm", "--backend", "clm", "--head", "/tmp/head"]
    )
    system_one_command(args)
    assert captured["model_name"] == "public-clm"
