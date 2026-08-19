"""Regression: the `rapid-mlx agents` roster must keep its columns aligned
even when an agent name is as wide as (or wider than) the name column's old
hardcoded 15-char width.

`deepseek-harness` is 16 chars. With the pre-fix `{p.name:<15}` the name
overflowed its field, consumed the single separator space, and shifted the
client / GitHub / tools columns one character right *on that row only* — a
ragged table. The column is now sized to the widest alias, so every row lines
up under the header.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx import cli


def _profile(name: str, display: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        display_name=display,
        stars=1000,
        needs_function_calling=True,
        recommended_models=["qwen3.5-9b-4bit"],
        kind="agent",
    )


def test_agents_list_client_column_aligns_for_max_width_name(monkeypatch, capsys):
    # A short name and a name exactly at the old 15-char boundary + 1.
    profiles = [
        _profile("claude-code", "Claude Code"),
        _profile("deepseek-harness", "DeepSeek Harness"),
    ]
    monkeypatch.setattr("vllm_mlx.agents.list_profiles", lambda: profiles)

    cli.agents_command(
        SimpleNamespace(agent_name=None, base_url="http://localhost:8000")
    )
    out = capsys.readouterr().out

    header = next(line for line in out.splitlines() if "client" in line)
    short_row = next(line for line in out.splitlines() if "Claude Code" in line)
    long_row = next(line for line in out.splitlines() if "DeepSeek Harness" in line)

    # The client column must start at the same offset on the header, a short
    # name row, and the widest name row — otherwise the table is ragged.
    assert (
        header.index("client")
        == short_row.index("Claude Code")
        == long_row.index("DeepSeek Harness")
    )
