# SPDX-License-Identifier: Apache-2.0
"""Cache settings the legacy prefix cache does not apply are reported.

``--hybrid-cache-entries`` and ``--prefix-cache-index`` configure the
prefix-cache backend through ``MemoryAwarePrefixCache``. The legacy branch of
``Scheduler.__init__`` builds a bare ``PrefixCacheManager(model, max_entries)``
that applies neither, so a serve that typed them under
``--no-memory-aware-cache`` came up healthy with the hybrid quota replaced by
the global entry limit and no radix index built, and nothing said so.

Only explicitly typed values are reported: ``--prefix-cache-index`` defaults
to ``radix`` and ``--hybrid-cache-entries`` is auto-derived for hybrid models,
and the auto-derived value still gates message-boundary snapshots in the
batched engine, so neither default is a dropped expectation.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from vllm_mlx import cli
from vllm_mlx.cli import _legacy_prefix_cache_dropped_flags


def _args(**overrides):
    base = {
        "no_memory_aware_cache": True,
        "use_paged_cache": False,
        "enable_prefix_cache": True,
        "disable_prefix_cache": False,
        "prefix_cache_index": "radix",
        "prefix_cache_size": 100,
        "hybrid_cache_entries": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


BASE_ARGV = ["rapid-mlx", "serve", "m", "--no-memory-aware-cache"]


def test_reports_explicit_hybrid_entries():
    argv = BASE_ARGV + ["--hybrid-cache-entries", "8"]
    lines = _legacy_prefix_cache_dropped_flags(_args(hybrid_cache_entries=8), argv)
    assert len(lines) == 1
    assert lines[0].startswith("--hybrid-cache-entries=8")
    assert "--prefix-cache-size=100" in lines[0]


def test_reports_explicit_hybrid_entries_equals_form():
    argv = BASE_ARGV + ["--hybrid-cache-entries=4"]
    lines = _legacy_prefix_cache_dropped_flags(_args(hybrid_cache_entries=4), argv)
    assert len(lines) == 1
    assert lines[0].startswith("--hybrid-cache-entries=4")


def test_silent_for_auto_derived_hybrid_entries():
    """The CLI auto-derives 8 for hybrid models; that value still gates
    message-boundary snapshots and was never typed, so it is not reported."""
    assert (
        _legacy_prefix_cache_dropped_flags(_args(hybrid_cache_entries=8), BASE_ARGV)
        == []
    )


def test_silent_for_explicit_zero_hybrid_entries():
    argv = BASE_ARGV + ["--hybrid-cache-entries", "0"]
    assert _legacy_prefix_cache_dropped_flags(_args(hybrid_cache_entries=0), argv) == []


@pytest.mark.parametrize("value", ["radix", "hash"])
def test_reports_explicit_index(value):
    argv = BASE_ARGV + ["--prefix-cache-index", value]
    lines = _legacy_prefix_cache_dropped_flags(_args(prefix_cache_index=value), argv)
    assert len(lines) == 1
    assert lines[0].startswith(f"--prefix-cache-index={value}")
    assert "RadixPrefixIndex" in lines[0]


def test_reports_explicit_index_equals_form():
    argv = BASE_ARGV + ["--prefix-cache-index=hash"]
    lines = _legacy_prefix_cache_dropped_flags(_args(prefix_cache_index="hash"), argv)
    assert len(lines) == 1


def test_reports_both_flags_in_order():
    argv = BASE_ARGV + ["--prefix-cache-index=hash", "--hybrid-cache-entries=8"]
    lines = _legacy_prefix_cache_dropped_flags(
        _args(prefix_cache_index="hash", hybrid_cache_entries=8), argv
    )
    assert [line.split("=")[0] for line in lines] == [
        "--hybrid-cache-entries",
        "--prefix-cache-index",
    ]


def test_silent_for_defaulted_index():
    assert _legacy_prefix_cache_dropped_flags(_args(), BASE_ARGV) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"no_memory_aware_cache": False},
        {"use_paged_cache": True},
        {"disable_prefix_cache": True},
    ],
)
def test_silent_when_legacy_branch_not_taken(overrides):
    argv = [
        "rapid-mlx",
        "serve",
        "m",
        "--prefix-cache-index",
        "hash",
        "--hybrid-cache-entries",
        "8",
    ]
    args = _args(prefix_cache_index="hash", hybrid_cache_entries=8, **overrides)
    assert _legacy_prefix_cache_dropped_flags(args, argv) == []


def test_serve_command_consults_the_helper_with_sys_argv():
    """The helper is only useful if ``serve_command`` calls it against the
    real command line; pin the call site so deleting the loop fails here."""
    tree = ast.parse(inspect.getsource(cli.serve_command))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_legacy_prefix_cache_dropped_flags"
    ]
    assert len(calls) == 1
    argv_arg = calls[0].args[1]
    assert isinstance(argv_arg, ast.Attribute)
    assert isinstance(argv_arg.value, ast.Name) and argv_arg.value.id == "sys"
    assert argv_arg.attr == "argv"
