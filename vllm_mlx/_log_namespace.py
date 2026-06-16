# SPDX-License-Identifier: Apache-2.0
"""
Runtime log namespace rebrand: ``vllm_mlx.*`` -> ``rapid_mlx.*``.

The Python package directory is still ``vllm_mlx/`` (renaming would touch
every import in the repo and every external integration), but the product,
PyPI distribution, and CLI are all called ``rapid-mlx``. Without this shim,
``logging.getLogger(__name__)`` returns ``vllm_mlx.<submodule>`` and every
log line a user copies into a bug report or LLM reads as a stale, internal
name.

We use ``logging.setLogRecordFactory`` rather than a logger- or
handler-attached filter for three reasons:

1. **Logger filters don't propagate to children** -- a filter on the
   ``vllm_mlx`` logger would NOT see records emitted by ``vllm_mlx.scheduler``
   etc. (Python docs: "Filters added to a logger are not used by descendant
   loggers.")
2. **Handler filters are fragile** -- uvicorn, pytest's caplog, and any
   library that installs its own handler bypass a filter pinned to the root
   logger's basicConfig StreamHandler.
3. **Factory runs once per record at creation time** -- ``record.name`` is
   already ``vllm_mlx.<sub>`` when the factory runs, so we can rewrite it
   before any handler (stream, file, caplog) sees it. Third-party namespaces
   are left untouched because the prefix check filters them out.

This is install-once, idempotent, and removable in one line (revert the
``setLogRecordFactory`` call). No churn across the 70+ ``getLogger(__name__)``
sites in the codebase.
"""

from __future__ import annotations

import logging

_OLD_PREFIX = "vllm_mlx"
_NEW_PREFIX = "rapid_mlx"

# Sentinel so install_log_namespace_rebrand() is idempotent. Repeated calls
# (test fixtures, in-process restarts, CLI re-entry from a wrapper) MUST NOT
# stack factories on top of each other -- that would still produce correct
# output but waste a function call per record forever.
_INSTALLED_SENTINEL = "_rapid_mlx_log_namespace_rebrand_installed"


def _rewrite_name(name: str) -> str:
    """Map ``vllm_mlx[.*]`` -> ``rapid_mlx[.*]``; leave everything else alone.

    - Exact ``vllm_mlx`` becomes ``rapid_mlx``.
    - ``vllm_mlx.server`` becomes ``rapid_mlx.server``.
    - ``vllm_mlxfoo`` (no separator) is NOT rewritten -- defensive against any
      future logger naming collision with a different package.
    - Anything not starting with the prefix is returned as-is.
    """
    if name == _OLD_PREFIX:
        return _NEW_PREFIX
    if name.startswith(_OLD_PREFIX + "."):
        return _NEW_PREFIX + name[len(_OLD_PREFIX) :]
    return name


def install_log_namespace_rebrand() -> None:
    """Install a LogRecord factory that rebrands ``vllm_mlx.*`` -> ``rapid_mlx.*``.

    Safe to call multiple times -- subsequent calls are no-ops. Wraps any
    existing factory so other components (e.g. structlog, third-party
    libraries) that have set their own factory continue to work; their work
    runs first, then we rebrand on top.
    """
    if getattr(logging, _INSTALLED_SENTINEL, False):
        return

    previous_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        # Only rewrite when needed -- the prefix check is a fast string op and
        # avoids touching record.name for every uvicorn/asyncio/httpx record.
        if record.name.startswith(_OLD_PREFIX):
            record.name = _rewrite_name(record.name)
        return record

    logging.setLogRecordFactory(_factory)
    setattr(logging, _INSTALLED_SENTINEL, True)
