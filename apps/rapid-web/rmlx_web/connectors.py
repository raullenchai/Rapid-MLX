# SPDX-License-Identifier: Apache-2.0
"""MCP connectors: the file the engine reads, and the switch that arms it.

The engine spawns the connector processes and validates them
(``vllm_mlx/mcp/security.py``); this module only authors the config file it
reads and decides whether to point the child at it.

The file is ``~/.config/rapid-mlx/mcp.json`` — the first entry of the
engine's own search path. Sharing it is deliberate: one connector set per
machine, so a server added from the phone is the one ``rapid-mlx serve``
picks up too.

The shape is the ecosystem-standard ``mcpServers`` map (Claude Desktop, VS
Code), so a config the user already has drops straight in.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

# The engine namespaces every tool as ``server__tool`` and that composite
# travels as an OpenAI function name — ``[A-Za-z0-9_-]``, at most 64 chars.
# Capping the server half here leaves room for a real tool name, and means
# the user is told in the editor rather than by a model that mysteriously
# never calls one server's tools.
MAX_NAME_LENGTH = 32

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

DEFAULT_TIMEOUT = 30.0

TRANSPORTS = ("stdio", "sse")


class ConnectorError(ValueError):
    """A connector entry cannot be saved. The message is shown verbatim."""


def is_valid_name(name: str) -> bool:
    """Whether ``name`` can be the namespace half of a tool name.

    Stricter than the engine, which accepts any dict key: a server called
    ``my server`` produces ``my server__read_file``, and a space is not a
    legal function-name character. ``__`` is rejected outright because both
    sides split the composite on the FIRST one, so ``my__server`` dispatches
    as server ``my`` and never resolves.
    """
    if not name or len(name) > MAX_NAME_LENGTH:
        return False
    if "__" in name:
        return False
    return bool(_NAME_RE.match(name))


@dataclass
class ServerConfig:
    """One entry, as it round-trips through the config file."""

    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    enabled: bool = True
    timeout: float = DEFAULT_TIMEOUT

    @property
    def summary_line(self) -> str:
        """One-line "what is this" — the command that runs, or the URL."""
        if self.transport == "sse":
            return self.url or ""
        return " ".join(part for part in [self.command or "", *self.args] if part)

    @property
    def execution_fingerprint(self) -> str:
        """A stable string of what this connector actually runs.

        Two entries with the same fingerprint run the same code; a change
        means consent must be re-established. Deliberately excludes name,
        enabled and timeout — none of them changes what executes.
        """
        env_part = ",".join(f"{k}={v}" for k, v in sorted(self.env.items()))
        return "\x02".join(
            [
                self.transport,
                self.command or "",
                "\x01".join(self.args),
                env_part,
                self.url or "",
            ]
        )

    def runs_different_code(self, other: ServerConfig) -> bool:
        return self.execution_fingerprint != other.execution_fingerprint

    def to_dict(self) -> dict:
        """The page's view. Carries ``name``, which the file does not."""
        return {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "url": self.url,
            "enabled": self.enabled,
            "timeout": self.timeout,
            "summary": self.summary_line,
        }

    def to_file_entry(self) -> dict:
        """The engine's JSON shape. ``name`` is the map key, not a field.

        Only the fields this transport uses are written: a stdio entry
        carrying a leftover ``url`` reads as ambiguous in a file the user may
        open by hand.
        """
        entry: dict = {"transport": self.transport}
        if self.transport == "stdio":
            if self.command is not None:
                entry["command"] = self.command
            if self.args:
                entry["args"] = list(self.args)
            if self.env:
                entry["env"] = dict(self.env)
        else:
            if self.url is not None:
                entry["url"] = self.url
        entry["enabled"] = self.enabled
        entry["timeout"] = self.timeout
        return entry


def validation_error(server: ServerConfig) -> str | None:
    """Why this entry cannot be saved, in the user's language, or None."""
    if not server.name:
        return "Give this connector a name."
    if not is_valid_name(server.name):
        return (
            f"Use up to {MAX_NAME_LENGTH} letters, numbers, dashes or "
            "underscores — the name becomes part of every tool name."
        )
    if server.transport not in TRANSPORTS:
        return "Choose a command or a URL connector."
    if server.transport == "stdio":
        if not (server.command or "").strip():
            return "A command connector needs a command to run."
    else:
        raw = (server.url or "").strip()
        if not raw:
            return "A URL connector needs a URL."
        parsed = urlparse(raw)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            return "Enter an http:// or https:// URL."
    if server.timeout <= 0:
        return "Timeout must be greater than zero."
    return None


def server_from_payload(payload: object) -> ServerConfig:
    """Build an entry from the page's JSON, rejecting a malformed body.

    Type-checked field by field rather than splatted into the dataclass: the
    values reach a file the engine will spawn processes from, and a list of
    non-strings would be written verbatim and fail much further away.
    """
    if not isinstance(payload, dict):
        raise ConnectorError("connector must be a JSON object")

    name = payload.get("name")
    if not isinstance(name, str):
        raise ConnectorError("`name` must be a string")

    transport = payload.get("transport", "stdio")
    if transport not in TRANSPORTS:
        raise ConnectorError("`transport` must be 'stdio' or 'sse'")

    command = payload.get("command")
    if command is not None and not isinstance(command, str):
        raise ConnectorError("`command` must be a string")

    args = payload.get("args") or []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ConnectorError("`args` must be an array of strings")

    env = payload.get("env") or {}
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ConnectorError("`env` must be an object of strings")

    url = payload.get("url")
    if url is not None and not isinstance(url, str):
        raise ConnectorError("`url` must be a string")

    timeout = payload.get("timeout", DEFAULT_TIMEOUT)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ConnectorError("`timeout` must be a number")

    server = ServerConfig(
        name=name.strip(),
        transport=transport,
        command=command.strip() if isinstance(command, str) else None,
        args=list(args),
        env=dict(env),
        url=url.strip() if isinstance(url, str) else None,
        enabled=payload.get("enabled", True) is not False,
        timeout=float(timeout),
    )
    why = validation_error(server)
    if why is not None:
        raise ConnectorError(why)
    return server


def decode_servers(data: object) -> list[ServerConfig]:
    """Parse a config file's server map into a name-sorted list.

    Both keys are read: ``mcpServers`` is the ecosystem standard and what is
    written, ``servers`` is the engine's historical spelling and appears in
    configs written against an older guide.

    A malformed entry is DROPPED rather than raising. The file is
    hand-editable by design, and refusing to show any connector because one
    is broken is the "my connectors vanished" failure this exists to avoid —
    the engine is tolerant here for the same reason.
    """
    if not isinstance(data, dict):
        return []
    raw = data.get("mcpServers")
    if not isinstance(raw, dict):
        raw = data.get("servers")
    if not isinstance(raw, dict):
        return []

    servers: list[ServerConfig] = []
    for name, entry in raw.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        args = entry.get("args")
        env = entry.get("env")
        timeout = entry.get("timeout")
        transport = entry.get("transport")
        servers.append(
            ServerConfig(
                name=name,
                # A hand-written config routinely omits `transport` and just
                # gives a command. Default rather than skip the entry.
                transport=transport if transport in TRANSPORTS else "stdio",
                command=entry.get("command")
                if isinstance(entry.get("command"), str)
                else None,
                args=[a for a in args if isinstance(a, str)]
                if isinstance(args, list)
                else [],
                env={
                    k: v
                    for k, v in env.items()
                    if isinstance(k, str) and isinstance(v, str)
                }
                if isinstance(env, dict)
                else {},
                url=entry.get("url") if isinstance(entry.get("url"), str) else None,
                enabled=entry.get("enabled", True) is not False,
                timeout=float(timeout)
                if isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
                else DEFAULT_TIMEOUT,
            )
        )
    # The JSON object is unordered, so sort by name — without this the rows
    # reshuffle on every load.
    servers.sort(key=lambda server: server.name.casefold())
    return servers


def server_map_key(data: object) -> str:
    """Which key this document already keeps its servers under.

    ``mcpServers`` is the ecosystem standard and what a new file gets, but the
    engine's historical ``servers`` is what a config written against an older
    guide uses — and silently migrating it would rewrite a file the user
    shares with the CLI.
    """
    if (
        isinstance(data, dict)
        and not isinstance(data.get("mcpServers"), dict)
        and isinstance(data.get("servers"), dict)
    ):
        return "servers"
    return "mcpServers"


def encode_servers(servers: list[ServerConfig], document: dict | None = None) -> str:
    """Render the file, preserving everything this package does not own.

    ``document`` is what was read from disk. The root carries engine settings
    with no UI here — ``default_timeout``, ``allowed_high_risk_tools``,
    ``max_tool_calls`` — and writing only the server map would delete them,
    changing how the engine behaves because a switch was toggled. Verified
    against a real config that carried all three.
    """
    root = dict(document) if isinstance(document, dict) else {}
    key = server_map_key(root)
    root[key] = {s.name: s.to_file_entry() for s in servers}
    return json.dumps(root, indent=2, sort_keys=True)


def default_config_path() -> Path:
    return Path.home() / ".config" / "rapid-mlx" / "mcp.json"


def default_settings_path() -> Path:
    """Where the master switch and the per-tool switches live.

    Beside the config rather than inside it: the config file is the engine's,
    in a shape other tools read, and adding private keys to it would travel
    into every config the user exports.
    """
    return Path.home() / ".config" / "rapid-mlx" / "rmlx-web.json"


class ConnectorStore:
    """Owns the connector file and the switches around it."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        settings_path: Path | None = None,
    ) -> None:
        self._config_path = config_path or default_config_path()
        self._settings_path = settings_path or default_settings_path()
        self._servers: list[ServerConfig] = []
        #: The config file as read, so a write keeps the root keys this
        #: package does not own.
        self._document: dict = {}
        #: Non-None when the file exists but could not be read. Showing an
        #: empty list over a broken file is the failure this reports instead.
        self.load_error: str | None = None
        self._settings: dict = {}
        self.reload_from_disk()

    # -------------------------------------------------------------- reading

    @property
    def path(self) -> Path:
        return self._config_path

    @property
    def servers(self) -> list[ServerConfig]:
        return list(self._servers)

    @property
    def is_enabled(self) -> bool:
        """Master opt-in. Off means the child is spawned with no
        ``--mcp-config`` at all, so the engine has no MCP subsystem — not
        merely zero servers. Connectors run arbitrary local commands, so this
        is an explicit choice and defaults off."""
        return self._settings.get("connectorsEnabled") is True

    @property
    def disabled_tools(self) -> set[str]:
        """Per-tool off switches, by namespaced ``server__tool`` name."""
        raw = self._settings.get("disabledTools")
        if not isinstance(raw, list):
            return set()
        return {name for name in raw if isinstance(name, str)}

    @property
    def granted_tools(self) -> set[str]:
        """Tools with a remembered "always allow"."""
        raw = self._settings.get("grantedTools")
        if not isinstance(raw, list):
            return set()
        return {name for name in raw if isinstance(name, str)}

    @property
    def auto_approve_all(self) -> bool:
        return self._settings.get("autoApproveAllTools") is True

    def launch_config_path(self) -> str | None:
        """The ``--mcp-config`` value for the next spawn, or None.

        None for "enabled but nothing to connect": handing the engine a
        config with no enabled server makes it stand up a manager, connect to
        nothing, and report an MCP subsystem there is no reason to show.
        """
        if not self.is_enabled:
            return None
        if not any(server.enabled for server in self._servers):
            return None
        if not self._config_path.exists():
            return None
        return str(self._config_path)

    def reload_from_disk(self) -> None:
        self.load_error = None
        self._settings = self._read_settings()
        if not self._config_path.exists():
            self._servers = []
            self._document = {}
            return
        try:
            data = json.loads(self._config_path.read_text())
        except (OSError, ValueError) as exc:
            self._servers = []
            self._document = {}
            self.load_error = f"Couldn't read {self._config_path}: {exc}"
            return
        # Kept so a write preserves the root keys this package has no UI for.
        self._document = data if isinstance(data, dict) else {}
        self._servers = decode_servers(data)
        # Validation runs on the write path, but an IMPORTED config never went
        # through it — and the engine forwards an illegal name verbatim, so
        # the tool becomes `bad name__tool` and the model silently cannot call
        # it. Say so here rather than let it fail invisibly downstream.
        invalid = [s.name for s in self._servers if not is_valid_name(s.name)]
        if invalid:
            names = ", ".join(f"“{name}”" for name in invalid)
            plural = len(invalid) > 1
            self.load_error = (
                f"{'Connectors' if plural else 'Connector'} {names} "
                f"{'have' if plural else 'has'} an invalid name — use up to "
                f"{MAX_NAME_LENGTH} letters, numbers, dashes or underscores. "
                f"{'Their' if plural else 'Its'} tools won't be callable "
                "until renamed."
            )

    def fingerprints(self) -> dict[str, str]:
        return {s.name: s.execution_fingerprint for s in self._servers}

    # -------------------------------------------------------------- writing

    def upsert(self, server: ServerConfig, *, replacing: str | None = None) -> bool:
        """Insert or replace one entry.

        ``replacing`` is the name being edited, so a rename does not trip the
        duplicate check against its own old name.

        Returns True when the connector's execution identity changed, which
        is the caller's cue to revoke its remembered grants: a command swapped
        under a name would otherwise inherit an "always allow" the user gave
        to different code. Decided here but acted on only after the write is
        durable — a failed save must not strand a connector with its grants
        deleted.
        """
        why = validation_error(server)
        if why is not None:
            raise ConnectorError(why)
        if any(s.name == server.name and s.name != replacing for s in self._servers):
            raise ConnectorError(f"A connector named “{server.name}” already exists.")

        nxt = list(self._servers)
        index = (
            next((i for i, s in enumerate(nxt) if s.name == replacing), None)
            if replacing
            else None
        )
        reconfigured = False
        if index is not None:
            reconfigured = (
                nxt[index].runs_different_code(server) or replacing != server.name
            )
            nxt[index] = server
        else:
            nxt.append(server)
        self._persist(nxt)
        return reconfigured

    def remove(self, name: str) -> None:
        if not any(s.name == name for s in self._servers):
            raise ConnectorError(f"No connector named “{name}”.")
        self._persist([s for s in self._servers if s.name != name])

    def set_server_enabled(self, name: str, enabled: bool) -> None:
        nxt = list(self._servers)
        index = next((i for i, s in enumerate(nxt) if s.name == name), None)
        if index is None:
            raise ConnectorError(f"No connector named “{name}”.")
        # Replaced, not mutated in place: the list is a shallow copy, so
        # assigning to the entry's field would change live state before the
        # write that is supposed to make it true.
        nxt[index] = replace(nxt[index], enabled=enabled)
        self._persist(nxt)

    def _persist(self, servers: list[ServerConfig]) -> None:
        ordered = sorted(servers, key=lambda server: server.name.casefold())
        try:
            _write_private(self._config_path, encode_servers(ordered, self._document))
        except OSError as exc:
            raise ConnectorError(f"Couldn't save {self._config_path}: {exc}") from exc
        self._servers = ordered
        self.load_error = None

    # ------------------------------------------------------------- switches

    def set_enabled(self, enabled: bool) -> None:
        self._write_settings({"connectorsEnabled": bool(enabled)})

    def set_auto_approve_all(self, enabled: bool) -> None:
        self._write_settings({"autoApproveAllTools": bool(enabled)})

    def set_tool_enabled(self, tool: str, enabled: bool) -> None:
        disabled = self.disabled_tools
        if enabled:
            disabled.discard(tool)
        else:
            disabled.add(tool)
        self._write_settings({"disabledTools": sorted(disabled)})

    def grant_tool(self, tool: str) -> None:
        self._write_settings({"grantedTools": sorted(self.granted_tools | {tool})})

    def reset_grants(self) -> None:
        """Forget every remembered grant. The blanket auto-approve switch is
        left alone — resetting individual grants is not a request to change
        the global posture."""
        self._write_settings({"grantedTools": []})

    def revoke_grants_for_server(self, name: str) -> None:
        """Drop every grant belonging to one connector.

        A grant is keyed on ``server__tool``, which survives a
        reconfiguration: point ``fs`` at a different command and "always allow
        fs__read_file" would authorise the new program. The identity changed,
        so the grants must not carry over.
        """
        prefix = f"{name}__"
        remaining = {t for t in self.granted_tools if not t.startswith(prefix)}
        self._write_settings({"grantedTools": sorted(remaining)})

    def reconcile_grants(self) -> None:
        """Revoke grants for any connector whose command changed on disk.

        The in-app editor already revokes on an edit, but the config file is
        hand-editable — that is the point of the standard shape — and a direct
        edit never passes through that path. Run at load so a command swapped
        while nothing was watching drops the grant before the tool can run.
        """
        stored = self._settings.get("fingerprints")
        stored = stored if isinstance(stored, dict) else {}
        current = self.fingerprints()
        granted = self.granted_tools
        for name, fingerprint in current.items():
            previous = stored.get(name)
            if isinstance(previous, str) and previous != fingerprint:
                granted = {t for t in granted if not t.startswith(f"{name}__")}
        self._write_settings({"fingerprints": current, "grantedTools": sorted(granted)})

    def _read_settings(self) -> dict:
        try:
            data = json.loads(self._settings_path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_settings(self, patch: dict) -> None:
        merged = {**self._settings, **patch}
        try:
            _write_private(
                self._settings_path, json.dumps(merged, indent=2, sort_keys=True)
            )
        except OSError as exc:
            raise ConnectorError(f"Couldn't save {self._settings_path}: {exc}") from exc
        self._settings = merged


def _write_private(path: Path, text: str) -> None:
    """Write atomically, 0600, into a 0700 directory.

    The directory holds a file naming local commands to run, so it is kept out
    of reach of other accounts. ``mkdir`` does NOT tighten a directory that
    already exists, so the mode is set explicitly every time.

    The temp file is created INSIDE that directory, which is what makes the
    window between the rename and the chmod unreachable: no other account can
    traverse in to read the file whatever its own mode is during that window.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    handle, temp_path = tempfile.mkstemp(dir=directory, prefix=".rmlx-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        raise
