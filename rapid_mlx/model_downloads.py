# SPDX-License-Identifier: Apache-2.0
"""``--disable-model-downloads``: never fetch a model that is not on disk.

Every path that would download a model calls :func:`check` once it knows the
model is missing locally, or :func:`require_local` before it hands a model id
to a loader that downloads on a cache miss. ``rapid-mlx pull`` is the explicit
way to provision models and is not affected.
"""

from __future__ import annotations

import os

from ._env import env_truthy

ENV_VAR = "RAPID_MLX_DISABLE_MODEL_DOWNLOADS"
FLAG = "--disable-model-downloads"

_cli_disabled = False


class ModelDownloadsDisabledError(RuntimeError):
    """A model is not available locally and model downloads are disabled."""

    def __init__(self, model: str, source: str):
        self.model = model
        self.source = source
        super().__init__(
            f"{model} is not available locally and model downloads are "
            f"disabled by {source}"
        )


# Record the ``--disable-model-downloads`` flag for this process.
def configure(cli_flag: bool) -> None:
    global _cli_disabled
    _cli_disabled = bool(cli_flag)


# The setting that disables model downloads, or None when they are allowed.
def source() -> str | None:
    if _cli_disabled:
        return FLAG
    if env_truthy(ENV_VAR):
        return ENV_VAR
    return None


def disabled() -> bool:
    return source() is not None


# Call right before downloading ``model``.
def check(model: str) -> None:
    disabled_by = source()
    if disabled_by is not None:
        raise ModelDownloadsDisabledError(model, disabled_by)


# Call before handing ``model`` to a loader that downloads on a cache miss.
# Only a local path or a snapshot the cache inventory reports as runnable
# passes; an inconclusive cache probe counts as missing.
def require_local(model: str) -> None:
    if not disabled():
        return
    if os.path.exists(os.path.expanduser(model)):
        return
    from . import cli

    if cli._cache_runnability(model) is True:
        return
    check(model)
