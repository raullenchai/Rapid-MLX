# SPDX-License-Identifier: Apache-2.0
"""Unit tests for deferred-import entrypoints outside the MLX lanes.

These functions import their dependencies lazily and live on paths the
coverage lanes otherwise never execute (community-bench pure helpers,
the DDTree alias sweep, the agent tag-suppression structural probe).
All are stdlib-light by design, so they run in the no-MLX CI lane.
"""

from __future__ import annotations

import pytest

from rapid_mlx.agents.testing import TestStatus, _test_tag_suppression
from rapid_mlx.community_bench import hardware, runner
from rapid_mlx.speculative.ddtree.eligibility import eligible_aliases


def test_rapid_mlx_version_probe_prefers_package_metadata():
    import rapid_mlx

    assert hardware._rapid_mlx_version() == str(rapid_mlx.__version__)


def test_make_sampling_params_factory_modes():
    for mode in ("greedy", "sampled"):
        factory = runner._make_sampling_params_factory(mode)
        params = factory(128)
        assert params.max_tokens == 128
        assert params.ignore_eos is True  # tg128/tg512 contract, issue #567

    greedy = runner._make_sampling_params_factory("greedy")(64)
    assert greedy.temperature == 0.0

    with pytest.raises(ValueError, match="unknown sampling mode"):
        runner._make_sampling_params_factory("bogus")


def test_eligible_aliases_returns_sorted_profile_names():
    names = eligible_aliases()
    assert names == sorted(names)


def test_tag_suppression_probe_passes_for_valid_tags():
    result = _test_tag_suppression(
        base_url="http://127.0.0.1:1",  # never contacted — structural probe
        model_id="unused",
        extra_tags=[("open", "close")],
    )
    assert result.status == TestStatus.PASS


def test_tag_suppression_probe_fails_when_tag_leaks(monkeypatch):
    """A filter that fails to suppress a tag must surface as TestStatus.FAIL."""

    class LeakyFilter:
        def __init__(self, extra_tags):
            pass

        def process(self, text):
            return text

        def flush(self):
            return ""

    import sys
    import types

    stub = types.ModuleType("rapid_mlx.api.utils")
    stub.StreamingToolCallFilter = LeakyFilter
    # The function imports StreamingToolCallFilter lazily, so patch the
    # module in sys.modules rather than the (unset) testing-module attr.
    monkeypatch.setitem(sys.modules, "rapid_mlx.api.utils", stub)
    result = _test_tag_suppression(
        base_url="http://127.0.0.1:1",
        model_id="unused",
        extra_tags=[("open", "close")],
    )
    assert result.status == TestStatus.FAIL
