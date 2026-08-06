# SPDX-License-Identifier: Apache-2.0
"""The wire adapters must work on a machine with no MLX.

``anthropic_to_openai`` is a pure message-shape translation: Anthropic
JSON in, OpenAI JSON out. No weights, no Metal, no engine. The whole
``TestAnthropicToOpenai`` suite has always run on Linux CI for exactly
that reason.

Then this PR taught the adapter to consult ``get_config()`` for the
mid-conversation-system flag, and ``vllm_mlx.config`` transitively did::

    config/__init__ -> config.server_config -> engine.base
                    -> engine_core -> import mlx.core

which is fatal off Apple Silicon. Every test in that class died with
``ModuleNotFoundError: No module named 'mlx'`` — on Linux only, so the
macOS dev loop stayed green and CI caught it instead.

The fix is not to swallow the ImportError: that would silently ignore an
operator's explicit flag. It is for ``config`` not to reach into the
engine at runtime in the first place (the annotation is type-only, and
``server_config`` already has ``from __future__ import annotations``).

These tests run in a subprocess with MLX blocked at the meta-path, which
is the only honest way to assert it here — this suite's own process has
MLX loaded long before any test starts. Mutation check: restore the
module-scope ``from ..engine.base import BaseEngine`` and both fail.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Installed at sys.meta_path[0] so it is consulted before any real
# finder, and raises rather than returning None — returning None would
# merely fall through to the next finder and find the real MLX.
_BLOCK_MLX = """
import sys


class _NoMLX:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise ImportError("mlx is unavailable in this environment")
        return None


sys.meta_path.insert(0, _NoMLX())
"""


def _run_without_mlx(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_MLX + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=180,
    )


def test_the_blocker_actually_blocks() -> None:
    """Guard the guard: a blocker that silently no-ops proves nothing."""
    proc = _run_without_mlx(
        """
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            print("BLOCKED")
        else:
            print("NOT-BLOCKED")
        """
    )
    assert "BLOCKED" in proc.stdout, proc.stderr
    assert "NOT-BLOCKED" not in proc.stdout


def test_config_is_importable_without_mlx() -> None:
    proc = _run_without_mlx(
        """
        from vllm_mlx.config import ServerConfig, get_config

        cfg = get_config()
        assert isinstance(cfg, ServerConfig)
        # The type-only annotation must survive as an unevaluated string.
        assert "BaseEngine" in ServerConfig.__annotations__["engine"]
        assert "mlx" not in __import__("sys").modules
        print("CONFIG-OK")
        """
    )
    assert "CONFIG-OK" in proc.stdout, proc.stderr + proc.stdout


@pytest.mark.parametrize("flag", [False, True])
def test_anthropic_adapter_translates_without_mlx(flag: bool) -> None:
    """The flag lookup itself must not need an engine, either way."""
    proc = _run_without_mlx(
        f"""
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
        from vllm_mlx.api.anthropic_models import AnthropicMessage, AnthropicRequest
        from vllm_mlx.config import get_config

        get_config().relocate_mid_conversation_system = {flag}

        req = AnthropicRequest(
            model="test-model",
            max_tokens=64,
            system="be brief",
            messages=[
                AnthropicMessage(role="user", content="hello"),
                AnthropicMessage(role="assistant", content="hi"),
                AnthropicMessage(role="system", content="stay on topic"),
                AnthropicMessage(role="user", content="continue"),
            ],
        )
        result = anthropic_to_openai(req)
        roles = [m.role for m in result.messages]
        assert roles[0] == "system", roles
        assert roles.count("system") == 1, roles
        assert "mlx" not in __import__("sys").modules
        print("ADAPTER-OK", roles)
        """
    )
    assert "ADAPTER-OK" in proc.stdout, proc.stderr + proc.stdout
