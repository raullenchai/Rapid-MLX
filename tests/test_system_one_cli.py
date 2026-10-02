# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from rapid_mlx.cli import (
    _resolve_system_one_backend,
    _stamp_port_explicit,
    build_parser,
    system_one_command,
)


def test_system_one_cli_defaults_to_laya_service():
    args = build_parser().parse_args(["system-one"])
    assert args.model == "convaiinnovations/laya"
    assert args.backend == "auto"
    assert args.host == "127.0.0.1"
    assert args.port is None
    assert args._port_explicit is False
    assert args.max_concurrent_requests == 8


@pytest.mark.parametrize(
    ("argv", "expected_port", "expected_explicit"),
    [
        (["system-one"], 8700, False),
        (["system-one", "--port", "8123"], 8123, True),
    ],
)
def test_system_one_forwards_stamped_port_context_to_preflight_and_uvicorn(
    monkeypatch, argv, expected_port, expected_explicit
):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.system_one.backends as backend_module
    import rapid_mlx.system_one.server as server_module

    captured = {}

    class Backend:
        default_model = "fake-system-one"

        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setitem(
        system_one_command.__globals__,
        "_port_preflight_or_die",
        lambda host, port, **kwargs: captured.update(
            preflight=(host, port, kwargs["port_explicit"])
        ),
    )
    monkeypatch.setattr(backend_module, "LayaBackend", Backend)
    monkeypatch.setattr(server_module, "create_app", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        uvicorn_module,
        "run_uvicorn",
        lambda _app, **kwargs: captured.update(uvicorn=kwargs),
    )

    args = build_parser().parse_args(argv)
    assert args._port_explicit is expected_explicit
    system_one_command(args)

    assert captured["preflight"] == (
        "127.0.0.1",
        expected_port,
        expected_explicit,
    )
    assert captured["uvicorn"]["port"] == expected_port
    assert captured["uvicorn"]["port_explicit"] is expected_explicit


def test_system_one_default_collision_forwards_nonexplicit_context(monkeypatch):
    captured = {}

    class CollisionError(Exception):
        pass

    def fake_exit(port, collision_host, *, model, port_explicit):
        captured.update(
            port=port,
            collision_host=collision_host,
            model=model,
            port_explicit=port_explicit,
        )
        raise CollisionError

    monkeypatch.setitem(
        system_one_command.__globals__,
        "_port_collision_host",
        lambda host, _port: host,
    )
    monkeypatch.setitem(
        system_one_command.__globals__, "_exit_for_port_collision", fake_exit
    )
    args = build_parser().parse_args(["system-one"])

    with pytest.raises(CollisionError):
        system_one_command(args)

    assert captured == {
        "port": 8700,
        "collision_host": "127.0.0.1",
        "model": "convaiinnovations/laya",
        "port_explicit": False,
    }


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


def test_server_parsers_stamp_port_context_at_parse_time():
    from rapid_mlx import server

    parser = build_parser()
    implicit = parser.parse_args(["serve", "qwen3.5-4b-4bit"])
    explicit = parser.parse_args(["serve", "qwen3.5-4b-4bit", "--port", "8123"])
    inherited = parser.parse_args(["serve", "qwen3.5-4b-4bit", "--listen-fd", "7"])
    standalone = server._build_parser()

    assert implicit._port_explicit is False
    assert explicit._port_explicit is True
    assert inherited._port_explicit is None
    assert standalone.parse_args([])._port_explicit is False
    assert standalone.parse_args(["--port=8123"])._port_explicit is True
    assert standalone.parse_args(["--por", "8123"])._port_explicit is True


def test_port_context_parser_preserves_caller_namespace_and_non_server_args():
    non_server = argparse.Namespace(command="models")
    assert _stamp_port_explicit(non_server) is non_server

    namespace = argparse.Namespace(caller_seed="kept")
    parsed = build_parser().parse_args(["system-one"], namespace=namespace)
    assert parsed is namespace
    assert parsed.caller_seed == "kept"
    assert parsed._port_explicit is False


def test_system_one_cli_rejects_non_positive_laya_batch_size():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["system-one", "--batch-size", "0"])


def test_system_one_cli_rejects_backend_specific_argument_mismatches():
    with pytest.raises(SystemExit, match="CLM requires --head"):
        system_one_command(
            build_parser().parse_args(["system-one", "clm", "--backend", "clm"])
        )
    with pytest.raises(SystemExit, match="only valid with --backend clm"):
        system_one_command(
            build_parser().parse_args(
                ["system-one", "--backend", "laya", "--head", "x"]
            )
        )


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
    import rapid_mlx.system_one.backends as backend_module

    events = []

    class Backend:
        default_model = "laya"

        def __init__(self, *args, **kwargs):
            events.append("backend")

        def models(self):
            return []

    # Patch the exact globals mapping used by the imported command. Some of
    # the full-suite CLI tests reload ``rapid_mlx.cli`` during collection,
    # so patching the current sys.modules entry can target a newer module
    # object than this function references.
    monkeypatch.setitem(
        system_one_command.__globals__,
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
    import rapid_mlx.system_one.backends as backend_module

    captured = {}

    class Backend:
        default_model = "public-clm"

        def __init__(self, encoder, head, **kwargs):
            captured.update(encoder=encoder, head=head, **kwargs)

        def models(self):
            return []

    monkeypatch.setitem(
        system_one_command.__globals__,
        "_port_preflight_or_die",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(backend_module, "CLMBackend", Backend)
    monkeypatch.setattr(uvicorn_module, "run_uvicorn", lambda *args, **kwargs: None)
    args = build_parser().parse_args(
        [
            "system-one",
            "public-clm",
            "--backend",
            "clm",
            "--head",
            "/tmp/head",
            "--device",
            "cpu",
        ]
    )
    system_one_command(args)
    assert captured["model_name"] == "public-clm"
    assert captured["device"] == "cpu"


def test_main_dispatches_system_one(monkeypatch):
    import rapid_mlx.cli as cli_module

    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "system-one",
            "clm-latest",
            "--backend",
            "clm",
            "--head",
            "/tmp/head",
        ],
    )
    monkeypatch.setattr(
        cli_module, "system_one_command", lambda args: captured.append(args.model)
    )
    cli_module.main()
    assert captured == ["clm-latest"]


def test_main_still_dispatches_serve_after_system_one_branch(monkeypatch):
    import rapid_mlx.cli as cli_module

    captured = []
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "qwen3.5-4b-4bit"])
    monkeypatch.setattr(
        cli_module, "serve_command", lambda args: captured.append(args.command)
    )
    cli_module.main()
    assert captured == ["serve"]
