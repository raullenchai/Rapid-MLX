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


def test_annotations_resolve_to_the_real_engine_type_without_mlx() -> None:
    """Introspection must return ``BaseEngine`` itself, not a stand-in.

    ``typing.get_type_hints`` is how dataclass-driven serializers and DI
    containers read annotations, and it resolves names in the defining
    module's globals. Two tempting shortcuts both fail here:

    * hiding the import behind ``TYPE_CHECKING`` leaves no runtime name at
      all, so this raises ``NameError``;
    * binding a placeholder such as ``Any`` makes it resolve — to the wrong
      type, which is worse, because a container keyed on the annotation
      silently stops matching instead of failing.

    So assert the identity of what comes back, not merely that the key is
    present. The weaker form claimed introspection was preserved while the
    placeholder was in place.
    """
    proc = _run_without_mlx(
        """
        import typing

        from vllm_mlx.config import ServerConfig
        from vllm_mlx.engine.base import BaseEngine

        hints = typing.get_type_hints(ServerConfig)
        assert "engine" in hints, sorted(hints)
        assert BaseEngine in typing.get_args(hints["engine"]), hints["engine"]
        assert "mlx" not in __import__("sys").modules
        print("HINTS-OK")
        """
    )
    assert "HINTS-OK" in proc.stdout, proc.stderr + proc.stdout


def test_engine_package_defers_its_mlx_dependent_members() -> None:
    """``import vllm_mlx.engine`` must not pull MLX in on its own.

    This is the property the config fix rests on. ``base`` is stdlib-pure
    and stays eager; ``engine_core`` and ``batched`` are PEP 562 lazies.
    """
    proc = _run_without_mlx(
        """
        import sys

        import vllm_mlx.engine as engine

        # The property the config fix rests on: the package alone is cheap.
        assert "mlx" not in sys.modules, "importing the package pulled MLX in"
        assert "vllm_mlx.engine_core" not in sys.modules
        assert engine.BaseEngine.__name__ == "BaseEngine"

        # The deferred members are still reachable by name...
        assert "BatchedEngine" in dir(engine)
        assert engine.BatchedEngine.__name__ == "BatchedEngine"
        # ...and resolving one caches it, so the hook runs once.
        assert engine.__dict__["BatchedEngine"] is engine.BatchedEngine

        # An unknown name must still be an AttributeError, not a KeyError
        # escaping the lookup table.
        try:
            engine.NoSuchMember
        except AttributeError:
            print("LAZY-OK")
        """
    )
    assert "LAZY-OK" in proc.stdout, proc.stderr + proc.stdout


def test_anthropic_streaming_reasoning_uses_the_reasoning_sanitizer() -> None:
    """`</tool_call>` must not survive into a streamed `thinking_delta`.

    The non-streaming path removed it while both streaming sites applied
    only `strip_special_tokens`, which does not. Agents stream, so the leak
    the fix was written for was still live on the path that matters.

    Asserted structurally against the two call sites plus a behavioural
    check of the sanitizer itself: driving the full Anthropic SSE loop needs
    an engine, and this is the property that loop depends on.
    """
    import inspect

    from vllm_mlx.api.utils import sanitize_reasoning_for_stream
    from vllm_mlx.routes import anthropic as route

    # The reasoning channel strips the closer; whitespace is preserved
    # because streaming clients concatenate deltas verbatim.
    assert sanitize_reasoning_for_stream(" x</tool_call>y ") == " xy "
    assert sanitize_reasoning_for_stream(None) == ""

    src = inspect.getsource(route)
    for anchor in (
        'if output_channel == "reasoning":',
        "if delta_msg.reasoning:",
    ):
        after = src[src.index(anchor) : src.index(anchor) + 1400]
        assert "sanitize_reasoning_for_stream(" in after, (
            f"streaming reasoning site after {anchor!r} does not use the "
            f"reasoning sanitizer"
        )
        head = after[: after.index("sanitize_reasoning_for_stream(")]
        assert "strip_special_tokens(" not in head, (
            f"streaming reasoning site after {anchor!r} still reaches "
            f"strip_special_tokens first"
        )
