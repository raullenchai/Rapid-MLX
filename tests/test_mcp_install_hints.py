"""Missing MCP dependencies point to the running Python environment."""

import asyncio
import sys

import pytest

from rapid_mlx.connect import package_install_command
from rapid_mlx.mcp.client import MCPClient
from rapid_mlx.mcp.config import load_mcp_config


def test_yaml_config_missing_dependency_hint(tmp_path, monkeypatch):
    path = tmp_path / "mcp.yaml"
    path.write_text("mcpServers: {}")
    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(ImportError) as error:
        load_mcp_config(str(path))

    assert error.value.name == "yaml"
    assert package_install_command("pyyaml") in str(error.value)


@pytest.mark.parametrize("transport", ["stdio", "sse"])
def test_mcp_client_missing_sdk_hint(monkeypatch, transport):
    monkeypatch.setitem(sys.modules, "mcp", None)
    client = object.__new__(MCPClient)

    with pytest.raises(ImportError) as error:
        asyncio.run(getattr(client, f"_connect_{transport}")())

    assert package_install_command("mcp") in str(error.value)
