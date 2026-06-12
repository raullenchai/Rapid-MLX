# SPDX-License-Identifier: Apache-2.0
"""
MCP configuration loading and validation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from .types import MCPConfig, MCPServerConfig

logger = logging.getLogger(__name__)

# Default config search paths.
# Intentionally excludes ./mcp.json and ./mcp.yaml: an attacker who can plant
# a file in a victim's CWD (shared dirs, /tmp, downloaded archives) could
# inject arbitrary MCP server commands/args. Use --mcp-config <path> or the
# RAPID_MLX_MCP_CONFIG env var for explicit project-local configs.
#
# The ~/.config/vllm-mlx/ entries are the pre-rename location (Rapid-MLX was
# formerly vllm-mlx). They are kept as a back-compat fallback so existing user
# configs keep working; new installs should use ~/.config/rapid-mlx/.
CONFIG_SEARCH_PATHS = [
    "~/.config/rapid-mlx/mcp.json",
    "~/.config/rapid-mlx/mcp.yaml",
    "~/.config/vllm-mlx/mcp.json",
    "~/.config/vllm-mlx/mcp.yaml",
]

# Environment variable for config path. VLLM_MLX_MCP_CONFIG is the deprecated
# pre-rename alias, still honored for back-compat.
CONFIG_ENV_VAR = "RAPID_MLX_MCP_CONFIG"
CONFIG_ENV_VAR_LEGACY = "VLLM_MLX_MCP_CONFIG"


def load_mcp_config(path: str | Path | None = None) -> MCPConfig:
    """
    Load MCP configuration from file.

    Search order:
    1. Explicit path argument
    2. RAPID_MLX_MCP_CONFIG environment variable (or the deprecated
       VLLM_MLX_MCP_CONFIG alias)
    3. ~/.config/rapid-mlx/mcp.json or mcp.yaml (falling back to the
       pre-rename ~/.config/vllm-mlx/ location)

    CWD discovery (./mcp.json, ./mcp.yaml) is intentionally NOT searched —
    see CONFIG_SEARCH_PATHS for rationale.

    Args:
        path: Optional explicit path to config file

    Returns:
        MCPConfig object

    Raises:
        FileNotFoundError: If no config file found
        ValueError: If config is invalid
    """
    config_path = _find_config_file(path)

    if config_path is None:
        logger.info("No MCP config file found, using empty config")
        return MCPConfig()

    logger.info(f"Loading MCP config from: {config_path}")

    # Load file content
    config_path = Path(config_path).expanduser()
    content = config_path.read_text()

    # Parse based on extension
    if config_path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(content)
        except ImportError:
            raise ImportError(
                "PyYAML required for .yaml config files: pip install pyyaml"
            )
    else:
        data = json.loads(content)

    return validate_config(data)


def _find_config_file(
    explicit_path: str | Path | None = None,
) -> Path | None:
    """Find the config file to use."""
    # 1. Explicit path
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"MCP config file not found: {explicit_path}")

    # 2. Environment variable (with deprecated pre-rename alias). Require a
    # real file, not just an existing path: a directory value (e.g. the
    # config *dir*) must fall through to the next var rather than be returned
    # and then blow up with IsADirectoryError in read_text().
    for env_var in (CONFIG_ENV_VAR, CONFIG_ENV_VAR_LEGACY):
        env_path = os.environ.get(env_var)
        if env_path:
            path = Path(env_path).expanduser()
            if path.is_file():
                return path
            logger.warning(f"MCP config from {env_var} not found: {env_path}")

    # 3. Search paths
    for search_path in CONFIG_SEARCH_PATHS:
        path = Path(search_path).expanduser()
        if path.exists():
            return path

    return None


def validate_config(data: dict[str, Any]) -> MCPConfig:
    """
    Validate and parse configuration dictionary.

    Args:
        data: Raw configuration dictionary

    Returns:
        Validated MCPConfig object

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("MCP config must be a dictionary")

    # Validate servers section
    servers_data = data.get("servers", {})
    if not isinstance(servers_data, dict):
        raise ValueError("'servers' must be a dictionary")

    servers = {}
    for name, server_data in servers_data.items():
        try:
            # Ensure name is set
            if isinstance(server_data, dict):
                server_data = server_data.copy()
                server_data["name"] = name
                servers[name] = MCPServerConfig(**server_data)
            else:
                raise ValueError(f"Server '{name}' config must be a dictionary")
        except TypeError as e:
            raise ValueError(f"Invalid config for server '{name}': {e}")

    # Validate other fields
    max_tool_calls = data.get("max_tool_calls", 10)
    if not isinstance(max_tool_calls, int) or max_tool_calls < 1:
        raise ValueError("'max_tool_calls' must be a positive integer")

    default_timeout = data.get("default_timeout", 30.0)
    if not isinstance(default_timeout, (int, float)) or default_timeout <= 0:
        raise ValueError("'default_timeout' must be a positive number")

    allowed_high_risk_tools = data.get("allowed_high_risk_tools", [])
    if not isinstance(allowed_high_risk_tools, list) or not all(
        isinstance(t, str) for t in allowed_high_risk_tools
    ):
        raise ValueError("'allowed_high_risk_tools' must be a list of strings")

    return MCPConfig(
        servers=servers,
        max_tool_calls=max_tool_calls,
        default_timeout=default_timeout,
        allowed_high_risk_tools=allowed_high_risk_tools,
    )


def create_example_config() -> str:
    """
    Create an example MCP configuration.

    Returns:
        JSON string with example configuration
    """
    example = {
        "servers": {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "enabled": True,
                "timeout": 30,
            },
            "web-search": {
                "transport": "sse",
                "url": "http://localhost:3001/sse",
                "enabled": True,
                "timeout": 60,
            },
            "sqlite": {
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-sqlite", "--db-path", "data.db"],
                "enabled": True,
            },
        },
        "max_tool_calls": 10,
        "default_timeout": 30.0,
    }
    return json.dumps(example, indent=2)
