# SPDX-License-Identifier: Apache-2.0
"""Deterministic approval heuristics, not an OS sandbox."""

from __future__ import annotations

import ast
import os
import re
import shlex
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..path_policy import resolve_candidate
from ..path_policy import sensitive as secret_path


@dataclass(frozen=True)
class Policy:
    prompt: bool = False
    risk: str = "file"
    reason: str = "Workspace file change with an undo checkpoint."
    allow_conversation: bool = True


def actions_enabled():
    return os.environ.get("RAPID_MLX_CHAT_ACTION_TOOLS", "1").lower() not in {
        "0",
        "false",
        "off",
    }


def resolve_path(workspace, path):
    return resolve_candidate(workspace, path)[2]


def file_policy(workspace, path):
    root, candidate, target = resolve_candidate(workspace, path)
    if secret_path(candidate) or secret_path(target):
        return Policy(
            True,
            "secret",
            "Secrets-like file paths are denied.",
            False,
        )
    if not target.is_relative_to(root):
        return Policy(
            True,
            "outside_workspace",
            "File is outside this conversation's workspace.",
            False,
        )
    if any(
        p in {".git", ".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"}
        for p in target.parts
    ):
        return Policy(
            True, "dangerous", "Repository metadata or shell startup file.", False
        )
    return Policy()


# Independently authored patterns. Broad matches deliberately prefer a prompt
# when shell text is ambiguous. No auxiliary model decides authorization.
RULES = (
    (r"\b(?:rm|rmdir)\b", "File or directory deletion"),
    (r"\b(?:sudo|doas|su)\b", "Privilege escalation"),
    (
        r"\bgit\b[^\n;]*\b(?:push|reset|clean|checkout|restore)\b",
        "Git remote or destructive worktree operation",
    ),
    (r"\b(?:chmod|chown|chgrp)\b", "Permission or ownership change"),
    (
        r"\b(?:mkfs\w*|fdisk|diskutil|dd|mount|umount|shutdown|reboot|halt|launchctl|systemctl)\b",
        "Disk or system mutation",
    ),
    (r"\b(?:kill|pkill|killall)\b", "Process termination"),
    (
        r"\bfind\b[^\n]*-(?:delete|exec|execdir)\b",
        "Find can delete files or execute programs",
    ),
    (
        r"\b(?:curl|wget)\b[^\n]*\|[^\n]*\b(?:sh|bash|zsh|python\d*)\b",
        "Remote code execution",
    ),
    (
        r"\b(?:curl|wget)\b[^\n]*(?:--data|--form|--upload|--post|--body|--json|\s-[a-zA-Z]*[dFT]\b)",
        "Network upload",
    ),
    (
        r"\b(?:scp|sftp|rsync|nc|ncat|netcat|socat|ssh)\b",
        "Network transfer or remote execution",
    ),
    (
        r"\b(?:eval|exec|source)\b|\b(?:sh|bash|zsh|dash|perl|ruby|node)\s+.*",
        "Indirect code execution",
    ),
    (
        r"\b(?:pip\d*|npm|brew|apt|apt-get|yum)\b.*\b(?:install|uninstall|remove|upgrade)\b",
        "Package installation or removal",
    ),
    (
        r"\b(?:drop\s+(?:table|database)|truncate\s+|delete\s+from)\b",
        "Database mutation",
    ),
    (
        r"169\.254\.169\.254|metadata\.google\.internal|100\.100\.100\.200|fd00:ec2::254",
        "Cloud credential endpoint",
    ),
    (
        r"\b(?:rmtree|unlink|rmdir|remove|system|popen|subprocess|killpg|chmod|chown)\b",
        "Code may delete, spawn processes, or change system state",
    ),
    (
        r"\b(?:socket|requests|httpx|http|aiohttp|urllib3?|ftplib|smtplib|paramiko|websockets)\b",
        "Code network access",
    ),
    (
        r"\b(?:eval|exec|compile|__import__|getattr)\s*\(|\b(?:base64|b64decode|ctypes|importlib)\b",
        "Dynamic or obfuscated execution",
    ),
    (r"(?:\$\(|`|\$\{|\$[A-Za-z_]|<\(|>\()", "Dynamic shell expansion"),
    (
        r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]",
        "Invisible command characters",
    ),
)
COMPILED_RULES = [
    (re.compile(pattern, re.I | re.S), reason) for pattern, reason in RULES
]


def scan_code(text, workspace, *, python=False):
    if len(text.encode()) > 65536:
        return "Code exceeds the inspection limit"
    normalized = unicodedata.normalize("NFKC", text).replace("\\\n", "")
    if not python:
        # Discarding stdout/stderr is not an outside-workspace file mutation.
        normalized = re.sub(
            r"(?<![\w])(?:[012])?>>?\s*/dev/null(?=[\s;&|]|$)", "", normalized
        )
    for pattern, reason in COMPILED_RULES:
        if pattern.search(normalized):
            return reason
    if python:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return "Python could not be inspected"
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if secret_path(value):
                    return "Secrets-like path or credential access"
                if "\n" not in value and "\x00" not in value:
                    try:
                        if secret_path(resolve_path(workspace, value)):
                            return "Secrets-like path or credential access"
                    except (OSError, ValueError):
                        return "Code path could not be inspected"
                if (
                    value == ".." or value.startswith(("/", "~", "../")) or "/" in value
                ) and "\n" not in value:
                    try:
                        if not resolve_path(workspace, value).is_relative_to(
                            Path(workspace)
                        ):
                            return "Code references a path outside the workspace"
                    except (OSError, ValueError):
                        return "Code path could not be inspected"
        return None
    try:
        lexer = shlex.shlex(normalized, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return "Shell quoting could not be inspected"
    for token in tokens:
        if secret_path(token):
            return "Secrets-like path or credential access"
        try:
            if secret_path(resolve_path(workspace, token)):
                return "Secrets-like path or credential access"
        except (OSError, ValueError):
            return "Command path could not be inspected"
    # Scan normalized shell words as well (r'm' / escaped spellings).
    words = " ".join(tokens).replace("\\", "")
    for pattern, reason in COMPILED_RULES:
        if pattern.search(words):
            return reason
    if re.search(r"(?:^|[;&|\n])\s*(?:cd|pushd)\b", normalized):
        return "Command changes the workspace directory"
    # Resolve every path token for commands with writes, not only redirections.
    writes = re.search(
        r"(?:>|\b(?:tee|cp|mv|install|touch|mkdir|truncate|sed|perl)\b)", normalized
    )
    if writes:
        for token in re.findall(r"(?:[^\s'\"<>;&|]+)", normalized):
            if token.startswith(("/", "~", "../")) or "/../" in token:
                if not resolve_path(workspace, token).is_relative_to(Path(workspace)):
                    return "Command may write outside the workspace"
        for token in tokens:
            if not token.startswith("-") and file_policy(workspace, token).risk in {
                "secret",
                "dangerous",
            }:
                return "Command may change a protected path"
        # Relative symlinks also cross the boundary.
        for token in tokens:
            if (
                token
                and not token.startswith("-")
                and not resolve_path(workspace, token).is_relative_to(Path(workspace))
            ):
                return "Command may write through a path outside the workspace"
    if tokens and re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(tokens[0]).name):
        if "-c" in tokens:
            index = tokens.index("-c")
            return (
                scan_code(tokens[index + 1], workspace, python=True)
                if index + 1 < len(tokens)
                else "Missing Python code"
            )
        scripts = [t for t in tokens[1:] if t.endswith(".py")]
        if scripts:
            path = resolve_path(workspace, scripts[0])
            if not path.is_relative_to(Path(workspace)) or secret_path(path):
                return "Python script is outside the workspace or protected"
            try:
                if path.stat().st_size > 65536:
                    return "Python script exceeds the inspection limit"
                return scan_code(path.read_text(), workspace, python=True)
            except (OSError, UnicodeError):
                return "Python script could not be inspected"
    return None


def execution_policy(text, workspace, *, python=False):
    reason = scan_code(text, workspace, python=python)
    if reason:
        return Policy(
            True, "dangerous", reason + "; approval is required every time.", False
        )
    return Policy(False, "execution", "No dangerous patterns detected.")


def configured_policy(policy):
    if os.environ.get("RAPID_MLX_CHAT_APPROVALS", "").lower() == "always":
        return Policy(
            True,
            policy.risk,
            "All side-effect calls require approval. " + policy.reason,
            False,
        )
    return policy
