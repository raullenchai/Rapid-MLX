"""Versioned plugin manifests."""

import re
import tomllib
from pathlib import Path


def load(path):
    path = Path(path)
    try:
        value = tomllib.loads(path.read_text())
        for key in ("id", "name", "version", "description", "hub_api"):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError(f"{key} must be nonempty text")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value["id"]):
            raise ValueError("invalid plugin id")
        if value["id"] in ("hub", "admin"):
            raise ValueError("plugin id is reserved for Hub administration")
        if value["hub_api"] not in ("^1", "1", "1.0"):
            raise ValueError("incompatible hub_api; supported major is 1")
        command = value["run"]["command"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(s, str) and s for s in command)
        ):
            raise ValueError("run.command must be a nonempty argv list")
        for arg in command:
            arg.format(python="python", plugin_dir="plugin")
        ui = value["ui"]
        if ui["kind"] != "web" or not re.fullmatch(r"/[A-Za-z0-9/_-]*", ui["health"]):
            raise ValueError("ui requires kind=web and an absolute health path")
        requires = value.get("requires", [])
        if not isinstance(requires, list) or any(
            r
            not in (
                "models.chat",
                "models.embeddings",
                "models.chat?",
                "models.embeddings?",
            )
            for r in requires
        ):
            raise ValueError("unknown required capability")
        if "permissions" in value:
            raise ValueError(
                "permissions is unsupported; plugins are trusted local code"
            )
        return dict(value, directory=str(path.parent.resolve()))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def discover(directory):
    plugins, errors = {}, []
    for path in sorted(Path(directory).glob("*/rapid-plugin.toml")):
        try:
            item = load(path)
            if item["id"] in plugins:
                raise ValueError(f"duplicate plugin id: {item['id']}")
            plugins[item["id"]] = item
        except ValueError as exc:
            errors.append(str(exc))
    return plugins, errors
