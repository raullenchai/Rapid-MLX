# SPDX-License-Identifier: Apache-2.0
"""Import smoke for engine modules outside the default test fan-in.

These modules are only imported on specific runtime paths (expert-cache
streaming forwards, offset-based expert loading, the disk-stream patch,
the HY3 MTP injector, the DDTree FastAPI surface); no other test pulls
them in. An import smoke catches ImportError regressions in exactly
those modules and gives the changed-lines coverage gate a lane that
executes their module-level imports. Requires mlx: most transitively
import mlx.core.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.requires_mlx

_MODULES = [
    "rapid_mlx.qwen2_moe_forward",
    "rapid_mlx.qwen3_next_forward",
    "rapid_mlx.offset_reader",
    "rapid_mlx.disk_stream_patch",
    "rapid_mlx.spec_decode.mtp.hy3_inject",
    "rapid_mlx.speculative.ddtree.server",
]


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_imports_cleanly(module_name):
    assert importlib.import_module(module_name) is not None
