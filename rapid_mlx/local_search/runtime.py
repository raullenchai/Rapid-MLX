# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the opt-in controller."""

from ..local_search_runtime import configure, default_directory, shutdown, startup

__all__ = ["configure", "default_directory", "startup", "shutdown"]
