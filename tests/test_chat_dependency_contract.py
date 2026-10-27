"""Keep the base harness installable without raising the Python floor."""

import sys
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement


def test_base_harness_respects_python_floor():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    from pathlib import Path

    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())[
        "project"
    ]
    requirements = {
        Requirement(dep).name: Requirement(dep) for dep in config["dependencies"]
    }
    agent = requirements["pydantic-ai-slim"]
    assert "openai" in agent.extras
    assert str(agent.specifier) == "==2.61.0"
    assert agent.marker is not None
    assert not agent.marker.evaluate({"python_version": "3.10"})
    assert agent.marker.evaluate({"python_version": "3.11"})
    assert {"httpx", "prompt-toolkit"} <= requirements.keys()
    extras = config["optional-dependencies"]
    for extra in ("chat", "all"):
        assert not any(Requirement(dep).name == "pyte" for dep in extras[extra])


@pytest.mark.parametrize(
    "name",
    [
        "test_chat_runtime.py",
        "test_chat_harness_client.py",
        "test_chat_context_budget.py",
    ],
)
def test_python_310_harness_collection_skips_before_import(
    name, monkeypatch, tmp_path, request
):
    from tests import conftest

    module = tmp_path / name
    module.write_text("raise AssertionError('unsupported dependency imported')")
    with monkeypatch.context() as patch:
        patch.setattr(conftest, "sys", SimpleNamespace(version_info=(3, 10)))
        collector = conftest.pytest_pycollect_makemodule(module, request.session)
    with pytest.raises(pytest.skip.Exception, match="Python 3.11"):
        collector.collect()


@pytest.mark.parametrize(
    "version,name",
    [((3, 11), "test_chat_runtime.py"), ((3, 10), "test_cli_chat.py")],
)
def test_supported_harness_and_legacy_chat_are_not_hidden(
    version, name, monkeypatch, tmp_path
):
    from tests import conftest

    with monkeypatch.context() as patch:
        patch.setattr(conftest, "sys", SimpleNamespace(version_info=version))
        assert conftest.pytest_pycollect_makemodule(tmp_path / name, None) is None
