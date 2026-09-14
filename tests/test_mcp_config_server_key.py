# SPDX-License-Identifier: Apache-2.0
"""The MCP config loader must accept the ecosystem-standard ``mcpServers`` key.

Claude Desktop, ``.mcp.json``, VS Code — and our own
``docs/guides/mcp-tools.md`` — all key the server map under ``mcpServers``.
The loader historically read only ``servers``, so a user who pasted a standard
config (or copied our own documented example) got ZERO servers loaded with no
error: a mysteriously tool-less agent. These tests pin the both-keys contract:
``mcpServers`` is accepted and preferred, ``servers`` still works for
back-compat, and a non-empty config with neither key warns instead of silently
loading nothing.
"""

from __future__ import annotations

import json
import logging

import pytest

from vllm_mlx.mcp.config import create_example_config, validate_config
from vllm_mlx.mcp.types import MCPConfig, MCPTransport

_SERVER = {"command": "python3", "args": ["-m", "some_mcp_server"]}


@pytest.fixture(autouse=True)
def _stub_command_path_lookup(monkeypatch):
    """Make config validation independent of what's installed on PATH.

    These tests exercise server-KEY handling, not the security PATH-presence
    check. Stub ``shutil.which`` so an allowlisted command (python3 / npx /
    uvx) always resolves — otherwise a runner exposing only ``python`` (not
    ``python3``), or one without Node/uv, would fail validation for reasons
    unrelated to what's under test.
    """
    monkeypatch.setattr(
        "vllm_mlx.mcp.security.shutil.which", lambda cmd: f"/usr/bin/{cmd}"
    )


def test_mcpservers_standard_key_loads_servers():
    cfg = validate_config({"mcpServers": {"filesystem": _SERVER}})
    assert list(cfg.servers) == ["filesystem"]
    assert cfg.servers["filesystem"].command == "python3"


def test_legacy_servers_key_still_loads():
    # Back-compat: existing rapid-mlx configs used the "servers" key.
    cfg = validate_config({"servers": {"filesystem": _SERVER}})
    assert list(cfg.servers) == ["filesystem"]


def test_mcpservers_preferred_when_both_present(caplog):
    # Standard key wins; the legacy key is ignored (and a warning explains it)
    # consistently on BOTH load paths (shared select_server_map).
    with caplog.at_level(logging.WARNING):
        cfg = validate_config(
            {"mcpServers": {"std": dict(_SERVER)}, "servers": {"legacy": dict(_SERVER)}}
        )
        MCPConfig.from_dict(
            {"mcpServers": {"std": dict(_SERVER)}, "servers": {"legacy": dict(_SERVER)}}
        )
    assert list(cfg.servers) == ["std"]
    assert "legacy" not in cfg.servers
    both_warnings = [
        r for r in caplog.records if "both 'mcpServers' and 'servers'" in r.getMessage()
    ]
    # One warning per path — from_dict no longer diverges from validate_config.
    assert len(both_warnings) == 2


def test_typoed_server_key_warns(caplog):
    # A config whose server map is under a mistyped key ("mcp_server") must
    # NOT silently load nothing — warn, naming the unrecognized key.
    with caplog.at_level(logging.WARNING):
        cfg = validate_config({"mcp_server": {"filesystem": dict(_SERVER)}})
    assert cfg.servers == {}
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "neither 'mcpServers' nor 'servers'" in msgs
    assert "mcp_server" in msgs


def test_globals_only_config_does_not_warn(caplog):
    # A config with only recognized global settings and no server map is an
    # intentional "no servers yet" config — not a typo. It must NOT warn.
    with caplog.at_level(logging.WARNING):
        cfg = validate_config({"default_timeout": 45.0})
    assert cfg.servers == {}
    assert not any("neither" in r.getMessage() for r in caplog.records)


def test_flat_legacy_agent_map_keeps_sibling_servers():
    # ``agent`` disambiguates this historical flat form, but it must not
    # cause the loader to discard the other servers in the same map.
    data = {
        "agent": dict(_SERVER),
        "filesystem": dict(_SERVER),
        "default_timeout": 45.0,
    }

    validated = validate_config(data)
    parsed = MCPConfig.from_dict(data)

    assert list(validated.servers) == ["agent", "filesystem"]
    assert list(parsed.servers) == ["agent", "filesystem"]
    assert validated.default_timeout == 45.0
    assert parsed.default_timeout == 45.0


def test_empty_mcpservers_does_not_warn(caplog):
    # ``{"mcpServers": {}}`` is a legitimate "no servers yet" config (the
    # discovery-test fixture writes exactly this) — it must not trip the
    # neither-key warning.
    with caplog.at_level(logging.WARNING):
        cfg = validate_config({"mcpServers": {}})
    assert cfg.servers == {}
    assert not any("neither" in r.getMessage() for r in caplog.records)


def test_standard_stdio_schema_omits_transport():
    # The shape our mcp-tools.md guide documents: command + args, no explicit
    # "transport". It must parse as stdio via the field default. Uses python3
    # (PATH-guaranteed) — a real npx/uvx command would trip the loader's
    # command-in-PATH check on a Node/uv-less runner.
    cfg = validate_config(
        {"mcpServers": {"filesystem": {"command": "python3", "args": ["-m", "srv"]}}}
    )
    assert cfg.servers["filesystem"].transport is MCPTransport.STDIO


def test_from_dict_accepts_mcpservers():
    # The secondary MCPConfig.from_dict path mirrors validate_config.
    cfg = MCPConfig.from_dict({"mcpServers": {"filesystem": dict(_SERVER)}})
    assert list(cfg.servers) == ["filesystem"]


def test_present_but_null_mcpservers_rejected_consistently():
    # A present-but-invalid standard key (JSON ``null``) must RAISE on BOTH
    # paths — never silently fall back to a legacy ``servers`` value. Keying
    # on presence keeps from_dict and validate_config consistent.
    bad = {"mcpServers": None, "servers": {"legacy": dict(_SERVER)}}
    with pytest.raises(ValueError, match="mcpServers"):
        validate_config(dict(bad))
    with pytest.raises(ValueError, match="mcpServers"):
        MCPConfig.from_dict(dict(bad))


def test_example_config_uses_mcpservers_and_roundtrips():
    data = json.loads(create_example_config())
    # The emitted example must key under the standard name...
    assert "mcpServers" in data
    assert "servers" not in data
    # ...and must itself be a valid config the loader accepts. (The example's
    # npx/uvx commands need not be installed — see the autouse PATH stub.)
    cfg = validate_config(data)
    assert set(cfg.servers) == set(data["mcpServers"])
