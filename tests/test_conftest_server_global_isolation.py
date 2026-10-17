# SPDX-License-Identifier: Apache-2.0
"""Guard launch-setting coverage and exercise real pytest teardown boundaries."""

import ast
import copy
import importlib.machinery
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.conftest import _SERVER_SERVE_CONFIG_GLOBALS, _isolate_server_serve_config

_ROOT = Path(__file__).resolve().parents[1]


def _collect_server_writes(source, aliases):
    mutated = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and ast.unparse(node.value) in aliases
        ):
            mutated.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 3
            and ast.unparse(node.args[0]) in aliases
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            mutated.add(node.args[1].value)
    return mutated


def test_collect_server_writes_recognizes_setattr():
    assert _collect_server_writes('setattr(server, "_x", 1)', {"server"}) == {"_x"}


def test_serve_config_tuple_covers_launch_writers():
    cli = ast.parse((_ROOT / "rapid_mlx/cli.py").read_text())
    server = ast.parse((_ROOT / "rapid_mlx/server.py").read_text())
    aliases = {
        "server",
        "server_mod",
        "server_module",
    }  # Helper parameters receiving the module.
    for node in ast.walk(cli):
        if isinstance(node, ast.ImportFrom) and node.module in {None, "rapid_mlx"}:
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "server"
            )
        elif isinstance(node, ast.Import):
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "rapid_mlx.server"
            )
    mutated = _collect_server_writes(ast.unparse(cli), aliases)
    assert "_default_timeout" in mutated
    # Lifecycle fixtures own engines, managers, locks, and telemetry latches.
    runtime_globals = {
        "_engine",
        "_embedding_engine",
        "_residency_manager",
        "_primary_model_lifecycle",
        "_mcp_manager",
        "_mcp_executor",
        "_mcp_init_error",
        "_mcp_rejected",
        "_telemetry_model_served_state",
        "_telemetry_audio_model_served_state",
        "_telemetry_embedding_model_served_state",
    }
    original_helpers = {
        "load_model",
        "load_embedding_model",
        "configure_model_residency",
        "configure_primary_model_lifecycle",
        "configure_cors_from_env",
        "init_mcp",
    }
    functions = {
        node.name: node
        for node in server.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    cli_helpers = {
        node.func.attr
        for node in ast.walk(cli)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in aliases
        and node.func.attr in functions
    }
    # Preserve the original coverage while following both launch entry points.
    roots = {"main"} | cli_helpers | original_helpers
    helpers = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in helpers:
            continue
        helpers.add(name)
        pending.extend(
            node.func.id
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and node.func.id not in helpers
        )
    assert original_helpers <= helpers, original_helpers - helpers
    for name in helpers:
        function = functions[name]
        names = {
            name
            for node in ast.walk(function)
            if isinstance(node, ast.Global)
            for name in node.names
        }
        names.update(_collect_server_writes(ast.unparse(function), aliases))
        # No exclusions for main: every declared global must be covered.
        mutated.update(names if function.name == "main" else names - runtime_globals)
    assert len(_SERVER_SERVE_CONFIG_GLOBALS) == len(set(_SERVER_SERVE_CONFIG_GLOBALS))
    assert not (mutated - set(_SERVER_SERVE_CONFIG_GLOBALS))


def test_preloaded_settings_restored_even_on_failure(monkeypatch):
    module = ModuleType("rapid_mlx.server")
    module._default_timeout = 1800.0
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", module)
    original_finders = list(sys.meta_path)
    isolate = contextmanager(_isolate_server_serve_config.__wrapped__)
    with pytest.raises(ValueError, match="test failed"), isolate():
        module._default_timeout = 60
        module._api_key = "test-only-key"
        raise ValueError("test failed")
    assert module._default_timeout == 1800.0
    assert not hasattr(module, "_api_key")
    assert sys.meta_path == original_finders


def test_preloaded_container_restored_after_in_place_mutation(monkeypatch):
    module = ModuleType("rapid_mlx.server")
    module._alias_recommended_sampling = {"temperature": 0.7}
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", module)
    isolate = contextmanager(_isolate_server_serve_config.__wrapped__)
    with isolate():
        module._alias_recommended_sampling["temperature"] = 0.1
        module._alias_recommended_sampling["top_p"] = 0.9
    assert module._alias_recommended_sampling == {"temperature": 0.7}


def test_preloaded_nested_container_restored_after_in_place_mutation(monkeypatch):
    module = ModuleType("rapid_mlx.server")
    module._alias_recommended_sampling = {"model": {"temperature": 0.7}}
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", module)
    isolate = contextmanager(_isolate_server_serve_config.__wrapped__)
    with isolate():
        module._alias_recommended_sampling["model"]["temperature"] = 0.1
        module._alias_recommended_sampling["model"]["top_p"] = 0.9
    assert module._alias_recommended_sampling == {"model": {"temperature": 0.7}}


@pytest.mark.parametrize("change", ["mutate", "rebind", "delete"])
@pytest.mark.parametrize("container", [dict, list, set])
def test_preloaded_container_alias_restored(monkeypatch, container, change):
    module = ModuleType("rapid_mlx.server")
    original = (
        {"model": {"temperature": 0.7}} if container is dict else container(["model"])
    )
    module._alias_recommended_sampling = alias = copy.deepcopy(original)
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", module)
    isolate = contextmanager(_isolate_server_serve_config.__wrapped__)
    with isolate():
        if container is dict:
            alias["model"]["temperature"] = 0.1
            alias["model"]["top_p"] = 0.9
        elif container is list:
            alias.append("changed")
        else:
            alias.add("changed")
        if change == "rebind":
            module._alias_recommended_sampling = None
        elif change == "delete":
            del module._alias_recommended_sampling
    assert module._alias_recommended_sampling is alias
    assert alias == original


def test_server_import_observer_delegates_to_remaining_finders(monkeypatch):
    calls = []
    module = ModuleType("rapid_mlx.server")
    path = []
    target = module

    def execute(module):
        assert loader.exec_module is execute
        module._default_timeout = 1800.0

    loader = SimpleNamespace(exec_module=execute)
    spec = importlib.machinery.ModuleSpec("rapid_mlx.server", loader)

    def find_spec(fullname, path, target):
        calls.append((fullname, path, target))
        return spec

    def unexpected_find_spec(*args):
        pytest.fail("finder outside the remaining search order was consulted")

    with monkeypatch.context() as patch:
        patch.setattr(sys, "meta_path", [])
        isolate = contextmanager(_isolate_server_serve_config.__wrapped__)
        with isolate():
            observer = sys.meta_path[0]
            sys.meta_path.insert(0, SimpleNamespace(find_spec=unexpected_find_spec))
            sys.meta_path.extend(
                [
                    object(),  # Legacy finders without find_spec are skipped.
                    SimpleNamespace(find_spec=lambda *args: None),
                    SimpleNamespace(find_spec=find_spec),
                    SimpleNamespace(find_spec=unexpected_find_spec),
                ]
            )
            assert observer.find_spec("rapid_mlx.server", path, target) is spec
            assert calls == [("rapid_mlx.server", path, target)]
            assert loader.exec_module is not execute
            loader.exec_module(module)
            module._default_timeout = 60
    assert module._default_timeout == 1800.0
    assert loader.exec_module is execute


@pytest.mark.parametrize("change", ["remove", "clear", "replace"])
def test_server_import_observer_teardown_tolerates_removal(monkeypatch, change):
    module = ModuleType("rapid_mlx.server")
    module._default_timeout = 1800.0
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", module)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "meta_path", [])
        isolate = contextmanager(_isolate_server_serve_config.__wrapped__)
        with isolate():
            module._default_timeout = 60
            if change == "remove":
                sys.meta_path.remove(sys.meta_path[0])
            elif change == "clear":
                sys.meta_path.clear()
            else:
                sys.meta_path = []
    assert module._default_timeout == 1800.0


# Run this pair in one fresh worker so xdist/random ordering cannot separate it.
# The first import occurs inside a test, exercising the late-import branch too.
_ORDERED_PAIR = """\
import sys

_expected = {}
_limiter_config = {}


def test_mutate_launch_settings():
    assert "rapid_mlx.server" not in sys.modules
    from rapid_mlx import server
    from conftest import _SERVER_SERVE_CONFIG_GLOBALS

    assert server._default_timeout == 1800.0
    _limiter_config.update(
        requests_per_minute=server._rate_limiter.requests_per_minute,
        enabled=server._rate_limiter.enabled,
    )
    server.configure_rate_limiter(7, enabled=True)
    for name in _SERVER_SERVE_CONFIG_GLOBALS:
        _expected[name] = getattr(server, name)
        setattr(server, name, 60 if name == "_default_timeout" else object())
    assert server._default_timeout == 60


def test_next_test_sees_defaults():
    from rapid_mlx import server

    assert server._default_timeout == 1800.0
    for name, value in _expected.items():
        # Containers are restored from deep copies; others by reference.
        if isinstance(value, (dict, list, set, tuple)):
            assert getattr(server, name) == value, name
        else:
            assert getattr(server, name) is value, name
    for name, value in _limiter_config.items():
        assert getattr(server._rate_limiter, name) == value, name
"""


def test_serve_config_restored_between_tests(tmp_path):
    (tmp_path / "conftest.py").write_text(
        "import runpy\n"
        f"_source = runpy.run_path({str(_ROOT / 'tests/conftest.py')!r})\n"
        "_SERVER_SERVE_CONFIG_GLOBALS = _source['_SERVER_SERVE_CONFIG_GLOBALS']\n"
        "_isolate_server_serve_config = _source['_isolate_server_serve_config']\n"
    )
    (tmp_path / "test_ordered_pair.py").write_text(_ORDERED_PAIR)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env["PYTHONPATH"] = str(_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-q",
            f"--confcutdir={tmp_path}",
            str(tmp_path / "test_ordered_pair.py"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
