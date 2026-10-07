# SPDX-License-Identifier: Apache-2.0
"""Optional-runtime extras whose requirements now ship in the base install.

``pip install rapid-mlx`` carries the vision (mlx-vlm), image (mflux) and
video (mlx-video) runtimes, so the historical ``[vision]`` / ``[image]`` /
``[video]`` extras — and the
mlx-vlm-only ``[dflash]`` / ``[mtp]`` extras they subsume — are kept as
accepted no-op aliases. A missing module from one of these runtimes therefore
means a damaged environment (``--no-deps``, a manual uninstall, a partial
copy), not an opt-in the user skipped: guidance should repair the base
install instead of advertising an extra.

This module is deliberately import-light so CLI startup paths can consult it.
"""

from __future__ import annotations

BASE_RUNTIME_EXTRAS: frozenset[str] = frozenset(
    {"vision", "image", "video", "dflash", "mtp"}
)


def is_base_runtime_extra(extra: str) -> bool:
    """Return whether *extra*'s requirements are part of the base install."""
    return extra in BASE_RUNTIME_EXTRAS


def runtime_install_spec(extra: str, version: str) -> str:
    """Return the pinned requirement that restores *extra*'s runtime."""
    if is_base_runtime_extra(extra):
        return f"rapid-mlx=={version}"
    return f"rapid-mlx[{extra}]=={version}"


def python_upgrade_hint() -> str:
    """Guidance for a runtime that the current Python cannot install.

    The image and video runtimes require Python 3.11+, so on the 3.10 floor
    reinstalling into the same interpreter cannot help.
    """
    from rapid_mlx import __version__

    return (
        "Reinstall rapid-mlx on Python 3.11 or newer, for example:\n"
        f"    uv tool install --force --python 3.12 'rapid-mlx=={__version__}'"
    )
