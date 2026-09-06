# SPDX-License-Identifier: Apache-2.0
"""The connector config store: validation, the file round trip, consent."""

from __future__ import annotations

import json
import os
import stat

import pytest

from rmlx_web import connectors
from rmlx_web.connectors import ConnectorError, ConnectorStore, ServerConfig


@pytest.fixture
def store(tmp_path):
    return ConnectorStore(
        config_path=tmp_path / "mcp.json",
        settings_path=tmp_path / "rmlx-web.json",
    )


def stdio(name="fs", command="npx", args=None, **kwargs):
    return ServerConfig(name=name, command=command, args=list(args or []), **kwargs)


class TestNameValidation:
    def test_a_name_becomes_half_a_tool_name(self):
        assert connectors.is_valid_name("filesystem")
        assert connectors.is_valid_name("web-search_2")

    def test_a_space_is_refused(self):
        # `my server__read_file` is not a legal OpenAI function name, so the
        # model could never call it — the engine would forward it verbatim.
        assert not connectors.is_valid_name("my server")

    def test_a_double_underscore_is_refused(self):
        # Both sides split `server__tool` on the FIRST `__`, so `my__server`
        # dispatches as server `my` and never resolves. A single one is fine.
        assert not connectors.is_valid_name("my__server")
        assert connectors.is_valid_name("my_server")

    def test_the_length_cap_leaves_room_for_the_tool_half(self):
        assert connectors.is_valid_name("x" * connectors.MAX_NAME_LENGTH)
        assert not connectors.is_valid_name("x" * (connectors.MAX_NAME_LENGTH + 1))


class TestValidationErrors:
    def test_a_command_connector_needs_a_command(self):
        assert "command to run" in connectors.validation_error(
            ServerConfig(name="fs", command="")
        )

    def test_a_url_connector_needs_an_http_url(self):
        entry = ServerConfig(name="remote", transport="sse", url="ftp://x/y")
        assert "http://" in connectors.validation_error(entry)

    def test_a_valid_entry_has_no_error(self):
        assert connectors.validation_error(stdio()) is None

    def test_a_zero_timeout_is_refused(self):
        assert connectors.validation_error(stdio(timeout=0)) is not None


class TestFileRoundTrip:
    def test_a_saved_connector_reads_back(self, store, tmp_path):
        store.upsert(stdio(args=["-y", "server-filesystem", "/tmp"]))

        reopened = ConnectorStore(
            config_path=tmp_path / "mcp.json",
            settings_path=tmp_path / "rmlx-web.json",
        )
        assert [s.name for s in reopened.servers] == ["fs"]
        assert reopened.servers[0].args == ["-y", "server-filesystem", "/tmp"]

    def test_the_file_uses_the_ecosystem_standard_key(self, store, tmp_path):
        # `mcpServers` is what Claude Desktop and VS Code read, so a config
        # authored here can be lifted out and one the user has drops in.
        store.upsert(stdio())
        written = json.loads((tmp_path / "mcp.json").read_text())

        assert list(written) == ["mcpServers"]
        assert "fs" in written["mcpServers"]
        # The name is the map key, never a field beside it.
        assert "name" not in written["mcpServers"]["fs"]

    def test_the_historical_servers_key_is_still_read(self):
        # A config written against an older guide must not read as empty.
        parsed = connectors.decode_servers({"servers": {"fs": {"command": "npx"}}})
        assert [s.name for s in parsed] == ["fs"]

    def test_an_existing_servers_key_is_not_migrated(self, tmp_path):
        # This file is read by other tools on this Mac too. Rewriting the
        # key it already uses changes a document the user did not ask to have
        # restructured.
        (tmp_path / "mcp.json").write_text(
            json.dumps({"servers": {"fs": {"command": "npx"}}})
        )
        store = ConnectorStore(
            config_path=tmp_path / "mcp.json",
            settings_path=tmp_path / "rmlx-web.json",
        )
        store.set_server_enabled("fs", False)

        written = json.loads((tmp_path / "mcp.json").read_text())
        assert "servers" in written
        assert "mcpServers" not in written

    def test_engine_settings_around_the_servers_survive_a_write(self, tmp_path):
        # The root carries engine behaviour this package has no UI for.
        # Dropping it would change how the engine runs because a switch was
        # toggled. Shape taken from a real config.
        (tmp_path / "mcp.json").write_text(
            json.dumps(
                {
                    "servers": {"fs": {"command": "npx"}},
                    "default_timeout": 30,
                    "max_tool_calls": 10,
                    "allowed_high_risk_tools": ["fs__exec"],
                }
            )
        )
        store = ConnectorStore(
            config_path=tmp_path / "mcp.json",
            settings_path=tmp_path / "rmlx-web.json",
        )
        store.upsert(stdio(name="time", command="uvx"))

        written = json.loads((tmp_path / "mcp.json").read_text())
        assert written["max_tool_calls"] == 10
        assert written["allowed_high_risk_tools"] == ["fs__exec"]
        assert sorted(written["servers"]) == ["fs", "time"]

    def test_a_transportless_entry_defaults_to_stdio(self):
        # Hand-written and pasted configs routinely give only a command.
        parsed = connectors.decode_servers({"mcpServers": {"fs": {"command": "npx"}}})
        assert parsed[0].transport == "stdio"

    def test_only_the_transports_fields_are_written(self, store, tmp_path):
        # A stdio entry carrying a stale url reads as ambiguous in a file the
        # user may open by hand.
        store.upsert(stdio(url="http://left-over"))
        entry = json.loads((tmp_path / "mcp.json").read_text())["mcpServers"]["fs"]

        assert "url" not in entry
        assert entry["command"] == "npx"

    def test_rows_are_name_sorted(self, store):
        # A JSON object is unordered, so without this the rows reshuffle on
        # every load.
        store.upsert(stdio(name="zebra"))
        store.upsert(stdio(name="alpha"))
        assert [s.name for s in store.servers] == ["alpha", "zebra"]

    def test_a_malformed_entry_does_not_take_the_others_down(self):
        # "My connectors vanished" is the failure this avoids: one broken
        # entry must not blank the whole list.
        parsed = connectors.decode_servers(
            {"mcpServers": {"fs": {"command": "npx"}, "bad": ["not", "a", "dict"]}}
        )
        assert [s.name for s in parsed] == ["fs"]

    def test_an_unreadable_file_is_reported_not_swallowed(self, tmp_path):
        (tmp_path / "mcp.json").write_text("{ not json")
        store = ConnectorStore(
            config_path=tmp_path / "mcp.json",
            settings_path=tmp_path / "rmlx-web.json",
        )
        assert store.servers == []
        assert store.load_error is not None

    def test_an_imported_illegal_name_is_surfaced(self, tmp_path):
        # Validation runs on the write path, but an imported config never
        # went through it, and the engine forwards the name verbatim — the
        # tool then silently cannot be called.
        (tmp_path / "mcp.json").write_text(
            json.dumps({"mcpServers": {"my server": {"command": "npx"}}})
        )
        store = ConnectorStore(
            config_path=tmp_path / "mcp.json",
            settings_path=tmp_path / "rmlx-web.json",
        )
        assert "my server" in store.load_error


class TestFilePermissions:
    def test_the_file_and_its_directory_are_private(self, tmp_path):
        # The file names local commands the engine will run, so it is kept
        # out of reach of other accounts on a shared Mac.
        directory = tmp_path / "nested"
        store = ConnectorStore(
            config_path=directory / "mcp.json",
            settings_path=directory / "rmlx-web.json",
        )
        store.upsert(stdio())

        assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(directory / "mcp.json").st_mode) == 0o600

    def test_an_existing_loose_directory_is_tightened(self, tmp_path):
        # mkdir does NOT tighten a directory that already exists, so a
        # `~/.config/rapid-mlx` created earlier at the umask default would
        # otherwise stay world-readable.
        directory = tmp_path / "loose"
        directory.mkdir(mode=0o755)
        store = ConnectorStore(
            config_path=directory / "mcp.json",
            settings_path=directory / "rmlx-web.json",
        )
        store.upsert(stdio())

        assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700


class TestUpsert:
    def test_a_duplicate_name_is_refused(self, store):
        store.upsert(stdio())
        with pytest.raises(ConnectorError, match="already exists"):
            store.upsert(stdio(command="uvx"))

    def test_a_rename_does_not_collide_with_its_own_old_name(self, store):
        store.upsert(stdio())
        store.upsert(stdio(name="files"), replacing="fs")
        assert [s.name for s in store.servers] == ["files"]

    def test_an_invalid_entry_never_reaches_the_file(self, store, tmp_path):
        with pytest.raises(ConnectorError):
            store.upsert(ServerConfig(name="bad name", command="npx"))
        assert not (tmp_path / "mcp.json").exists()

    def test_removing_an_unknown_connector_is_refused(self, store):
        with pytest.raises(ConnectorError, match="No connector"):
            store.remove("ghost")


class TestExecutionIdentity:
    def test_changing_the_command_is_a_reconfiguration(self, store):
        store.upsert(stdio())
        assert store.upsert(stdio(command="uvx"), replacing="fs") is True

    def test_a_rename_is_a_reconfiguration(self, store):
        # The grant is keyed on `server__tool`, so a rename hands the old
        # name's consent to whatever takes it.
        store.upsert(stdio())
        assert store.upsert(stdio(name="files"), replacing="fs") is True

    def test_toggling_enabled_is_not(self, store):
        # Enable and timeout do not change what code runs, so they must not
        # cost the user their consent.
        store.upsert(stdio())
        assert store.upsert(stdio(enabled=False), replacing="fs") is False

    def test_argument_order_is_part_of_the_identity(self):
        assert stdio(args=["a", "b"]).runs_different_code(stdio(args=["b", "a"]))


class TestConsent:
    def test_a_grant_survives_a_reopen(self, store, tmp_path):
        store.grant_tool("fs__read_file")
        reopened = ConnectorStore(
            config_path=tmp_path / "mcp.json",
            settings_path=tmp_path / "rmlx-web.json",
        )
        assert "fs__read_file" in reopened.granted_tools

    def test_revoking_one_server_leaves_the_others(self, store):
        store.grant_tool("fs__read_file")
        store.grant_tool("time__now")
        store.revoke_grants_for_server("fs")

        assert store.granted_tools == {"time__now"}

    def test_reset_leaves_the_blanket_switch_alone(self, store):
        # Resetting individual grants is not a request to change the global
        # posture.
        store.set_auto_approve_all(True)
        store.grant_tool("fs__read_file")
        store.reset_grants()

        assert store.granted_tools == set()
        assert store.auto_approve_all is True

    def test_a_hand_edited_command_revokes_the_grant(self, store, tmp_path):
        # The config file is hand-editable by design, and a direct edit never
        # passes through upsert. Reconciling at load is what closes that gap.
        store.upsert(stdio())
        store.grant_tool("fs__read_file")
        store.reconcile_grants()

        (tmp_path / "mcp.json").write_text(
            json.dumps({"mcpServers": {"fs": {"command": "uvx"}}})
        )
        store.reload_from_disk()
        store.reconcile_grants()

        assert store.granted_tools == set()

    def test_an_unchanged_command_keeps_the_grant(self, store):
        store.upsert(stdio())
        store.grant_tool("fs__read_file")
        store.reconcile_grants()

        store.reload_from_disk()
        store.reconcile_grants()

        assert store.granted_tools == {"fs__read_file"}


class TestLaunchPath:
    def test_off_means_no_flag_at_all(self, store):
        # Not "zero servers": the engine stands up no MCP subsystem, because
        # connectors run arbitrary local commands.
        store.upsert(stdio())
        assert store.launch_config_path() is None

    def test_on_with_an_enabled_server_arms_the_flag(self, store, tmp_path):
        store.set_enabled(True)
        store.upsert(stdio())
        assert store.launch_config_path() == str(tmp_path / "mcp.json")

    def test_on_with_nothing_enabled_stays_none(self, store):
        # Handing the engine a config with no enabled server makes it connect
        # to nothing and report a subsystem there is no reason to show.
        store.set_enabled(True)
        store.upsert(stdio(enabled=False))
        assert store.launch_config_path() is None

    def test_it_is_re_read_rather_than_snapshotted(self, store):
        store.set_enabled(True)
        store.upsert(stdio())
        assert store.launch_config_path() is not None

        store.set_enabled(False)
        assert store.launch_config_path() is None


class TestPayloadParsing:
    def test_a_well_formed_payload_becomes_an_entry(self):
        server = connectors.server_from_payload(
            {"name": "fs", "command": "npx", "args": ["-y", "pkg"]}
        )
        assert server.name == "fs"
        assert server.args == ["-y", "pkg"]

    def test_non_string_args_are_refused(self):
        # These reach a file the engine spawns from; a number in argv would
        # be written verbatim and fail much further away.
        with pytest.raises(ConnectorError, match="array of strings"):
            connectors.server_from_payload(
                {"name": "fs", "command": "npx", "args": [1]}
            )

    def test_an_invalid_entry_is_refused_at_the_boundary(self):
        with pytest.raises(ConnectorError, match="command to run"):
            connectors.server_from_payload({"name": "fs"})
