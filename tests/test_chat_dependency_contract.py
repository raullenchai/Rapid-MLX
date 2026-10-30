"""Keep the harness dependencies in the base installation."""

from packaging.requirements import Requirement


def test_harness_dependencies_are_in_base_install():
    import tomllib
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
    assert agent.marker is None
    assert {"httpx", "prompt-toolkit"} <= requirements.keys()
    extras = config["optional-dependencies"]
    for extra in ("chat", "all"):
        assert not any(Requirement(dep).name == "pyte" for dep in extras[extra])
