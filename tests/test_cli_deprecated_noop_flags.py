# SPDX-License-Identifier: Apache-2.0
"""Guard: deprecated ``serve`` CLI flags stay accepted-but-ignored no-ops.

PR #1101 removed the implementations behind several long-deprecated
``rapid-mlx serve`` flags (the single BatchedEngine, legacy KV-bit quant,
the ``--draft-model`` / ``--num-draft-tokens`` speculation frontend, the
``--specprefill`` prototype, and the legacy chunked-prefill monkey-patch
that mlx-lm 0.31+ made unreachable). The *implementations* stay removed —
but the launcher must still PARSE these flags without an argparse hard-fail,
so an existing user launch script (or older docs) that still passes them
keeps booting instead of dying with ``error: unrecognized arguments``.

These tests pin that back-compat contract: each flag parses to a namespace
(no ``SystemExit`` from argparse) and is hidden from ``--help``. They do NOT
assert any functional effect — the flags are genuine no-ops.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from rapid_mlx import cli

# Mirror the ``test_serve_listen_fd.py`` style: drive ``cli.main()`` with a
# stubbed ``serve_command`` so we capture the resolved ``argparse.Namespace``
# without booting a server.


def _capture_serve_args(argv: list[str]) -> list:
    captured: list = []
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured


# (flag, extra argv tokens, namespace attribute the value lands on)
_DEPRECATED_NOOP_FLAGS: tuple[tuple[str, list[str], str], ...] = (
    ("--continuous-batching", ["--continuous-batching"], "continuous_batching"),
    ("--simple-engine", ["--simple-engine"], "simple_engine"),
    ("--kv-bits", ["--kv-bits", "8"], "kv_bits"),
    ("--kv-group-size", ["--kv-group-size", "32"], "kv_group_size"),
    ("--draft-model", ["--draft-model", "some/drafter"], "draft_model"),
    ("--num-draft-tokens", ["--num-draft-tokens", "4"], "num_draft_tokens"),
    ("--specprefill", ["--specprefill"], "specprefill"),
    (
        "--specprefill-threshold",
        ["--specprefill-threshold", "4096"],
        "specprefill_threshold",
    ),
    (
        "--specprefill-keep-pct",
        ["--specprefill-keep-pct", "0.5"],
        "specprefill_keep_pct",
    ),
    (
        "--specprefill-draft-model",
        ["--specprefill-draft-model", "some/drafter"],
        "specprefill_draft_model",
    ),
    (
        "--chunked-prefill-tokens",
        ["--chunked-prefill-tokens", "2048"],
        "chunked_prefill_tokens",
    ),
)


@pytest.mark.parametrize(
    "flag,extra_argv,attr",
    _DEPRECATED_NOOP_FLAGS,
    ids=[f[0] for f in _DEPRECATED_NOOP_FLAGS],
)
def test_deprecated_flag_parses_without_argparse_error(flag, extra_argv, attr):
    """Each deprecated flag is accepted (no ``SystemExit``) and lands on the
    namespace — proving the launcher won't hard-fail on old user scripts."""
    captured = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", *extra_argv]
    )
    assert len(captured) == 1, (
        f"{flag} did not reach serve_command (argparse rejected it?)"
    )
    ns = captured[0]
    assert hasattr(ns, attr), f"{flag} did not populate args.{attr}"


def test_all_deprecated_flags_together_parse():
    """The whole deprecated bundle passed at once still parses — mirrors a
    real legacy launch script that stacked several of these flags."""
    argv = ["rapid-mlx", "serve", "qwen3.5-4b-4bit"]
    for _flag, extra_argv, _attr in _DEPRECATED_NOOP_FLAGS:
        argv.extend(extra_argv)
    captured = _capture_serve_args(argv)
    assert len(captured) == 1
    ns = captured[0]
    for _flag, _extra_argv, attr in _DEPRECATED_NOOP_FLAGS:
        assert hasattr(ns, attr)


# Names the deprecated flags write onto the namespace — the ONLY attributes
# allowed to differ when a deprecated flag is present. Everything else in the
# parsed namespace must be byte-identical with vs. without the flag; that is
# what "no-op" means at the config layer.
_DEPRECATED_ATTRS = frozenset(attr for _flag, _argv, attr in _DEPRECATED_NOOP_FLAGS)


@pytest.mark.parametrize(
    "flag,extra_argv,attr",
    _DEPRECATED_NOOP_FLAGS,
    ids=[f[0] for f in _DEPRECATED_NOOP_FLAGS],
)
def test_deprecated_flag_is_behaviorally_inert(flag, extra_argv, attr):
    """Directly answer the "mocked serve_command hides a live flag" concern:
    the deprecated flag must change NOTHING in the parsed configuration except
    (harmlessly) its own unread attribute.

    We diff the resolved ``argparse.Namespace`` with vs. without the flag. The
    launcher reads its entire server/engine configuration off this namespace,
    so if a supposedly-ignored flag actually steered config, some OTHER
    attribute would differ here — and this test would fail. Every non-deprecated
    attribute being identical proves the flag is a genuine no-op, not merely
    "accepted then quietly acted upon". (Confirmed structurally too: no code
    reads ``args.<deprecated_attr>`` anywhere in ``cli.py``.)
    """
    base = _capture_serve_args(["rapid-mlx", "serve", "qwen3.5-4b-4bit"])[0]
    with_flag = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", *extra_argv]
    )[0]

    base_d = vars(base)
    flag_d = vars(with_flag)

    # Same attribute set (the flag doesn't invent unrelated config keys).
    assert set(base_d) == set(flag_d)

    # Every attribute that is NOT one of the deprecated no-op flags must be
    # identical — i.e. the flag steered no real configuration.
    diverged = {
        k: (base_d.get(k), flag_d.get(k))
        for k in base_d
        if k not in _DEPRECATED_ATTRS and base_d.get(k) != flag_d.get(k)
    }
    assert not diverged, (
        f"{flag} changed non-deprecated config {diverged} — it is NOT an inert "
        f"no-op. A deprecated flag must never steer server/engine configuration."
    )


@pytest.mark.parametrize(
    "flag",
    [f[0] for f in _DEPRECATED_NOOP_FLAGS],
)
def test_deprecated_flag_hidden_from_help(flag):
    """Deprecated no-op flags use ``argparse.SUPPRESS`` — they must not appear
    in ``serve --help`` so we don't advertise dead knobs to new users."""
    buf = io.StringIO()
    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", "--help"]),
        pytest.raises(SystemExit) as exc,
        redirect_stdout(buf),
    ):
        cli.main()
    assert exc.value.code == 0
    assert flag not in buf.getvalue(), f"{flag} leaked into --help output"
