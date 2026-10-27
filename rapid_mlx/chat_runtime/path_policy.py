# SPDX-License-Identifier: Apache-2.0
"""Conversation workspace authority and shared secrets-path exclusions."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from rapid_mlx.local_search.extraction import SECRET_PATTERNS


def workspace_path(value: str | None) -> str | None:
    """Accept only explicit client directories, never the owner's cwd."""
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("workspace must be an absolute directory")
    try:
        path = path.resolve(strict=True)
        is_directory = path.is_dir()
    except (OSError, RuntimeError) as error:
        raise ValueError("workspace must be an accessible directory") from error
    if not is_directory:
        raise ValueError("workspace must be a directory")
    return str(path)


SECRET_DIRS = {
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".azure",
    ".docker",
    "keychains",
    "mcp-tokens",
    "browser-profile",
}
SECRET_NAMES = {
    ".netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "auth.json",
    "token.json",
    "tokens.json",
    "login data",
    "cookies",
    "cookies.sqlite",
    "credentials",
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    ".anthropic_oauth.json",
    "bws_cache.json",
    "bws_cache.enc.json",
    "stored_tokens",
    "hosts.yml",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "shadow",
    "keychain",
    "gcloud",
    "passwd",
}


class FileError(ValueError):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)


def sensitive(path: Path | str) -> bool:
    path = Path(path)
    parts = [part.lower() for part in path.parts]
    return (
        any(
            re.search(r"(?:^|[-_.])(?:secrets?|credentials?|tokens?)(?:[-_.]|$)", p)
            for p in parts
        )
        or ("huggingface" in parts and parts[-1] == "token")
        or any(part in SECRET_DIRS or part in SECRET_NAMES for part in parts)
        or any(
            fnmatch.fnmatchcase(part, pattern)
            for part in parts
            for pattern in SECRET_PATTERNS
        )
        or any(
            part.startswith(".env")
            or part.endswith(".token")
            or "oauth" in part.lower()
            and part.endswith(".json")
            for part in parts
        )
        or any(
            parts[i : i + 2] in [[".config", "gh"], [".config", "gcloud"]]
            for i in range(len(parts))
        )
        or ("library" in parts and any("cookies" in part for part in parts))
    )


def workspace_root(workspace: str | None) -> Path:
    if not workspace or not Path(workspace).is_absolute():
        raise FileError(
            "workspace_unavailable",
            "Ask the user to restart chat in a workspace folder or /attach the file.",
        )
    root = Path(workspace)
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise FileError(
            "workspace_changed",
            "Workspace location changed; restart chat in the intended folder.",
        )
    return root


def resolve_candidate(
    workspace: str | None, value: str = "."
) -> tuple[Path, Path, Path]:
    root = workspace_root(workspace)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return root, candidate, candidate.resolve()


def resolve_path(workspace: str | None, value: str = ".") -> tuple[Path, Path]:
    """File reads and file mutations share the same non-overridable boundary."""
    root, candidate, resolved = resolve_candidate(workspace, value)
    if sensitive(candidate) or sensitive(resolved):
        raise FileError(
            "sensitive_path",
            "Secrets-like paths are denied; ask the user to explicitly /attach the file.",
        )
    if not resolved.is_relative_to(root):
        raise FileError(
            "outside_workspace",
            "Outside the workspace. Paths may be absolute, workspace-relative, or ~/path (search path defaults to '.'); otherwise ask the user to /attach the file or restart chat in that folder.",
        )
    return root, resolved
