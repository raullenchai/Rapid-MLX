# SPDX-License-Identifier: Apache-2.0
"""Regression tests for PFlash's former endpoints-only data loss.

When ``sink_tokens + tail_tokens`` meet or exceed the keep budget, PFlash keeps
only the leading sink and trailing tail can fit. With the verified ``always``
defaults this affects ordinary 2.3k-11.5k token agent conversations. PFlash must
now preserve those prompts rather than return a normal success after deleting
their entire middle.
"""

from vllm_mlx.pflash import (
    PFlashConfig,
    compress_request_tokens,
    compress_tokens,
)


class TestPFlashEndpointsOnly:
    def test_default_always_config_preserves_short_prompt(self):
        # 3k tokens under the default always-mode config: keep_budget == 2048 but
        # sink(256)+tail(2048) == 2304 > budget. The old implementation dropped
        # the whole middle and even kept more than its nominal budget.
        tokens = list(range(3000))
        result = compress_tokens(tokens, PFlashConfig(mode="always"))
        assert result.tokens == tokens
        assert result.compressed is False
        assert result.reason == "insufficient_middle_budget"
        assert result.middle_tokens_kept == 0
        assert result.endpoints_only is False
        assert result.kept_tokens == 3000

    def test_generous_budget_keeps_middle_and_is_not_flagged(self):
        # 20k tokens: 0.2 * 20k == 4000 budget leaves room for middle blocks.
        result = compress_tokens(
            list(range(20000)), PFlashConfig(mode="always", keep_ratio=0.2)
        )
        assert result.compressed is True
        assert result.middle_tokens_kept > 0
        assert result.endpoints_only is False

    def test_metadata_surfaces_safe_refusal(self):
        tokens = list(range(3000))
        kept, metadata = compress_request_tokens(tokens, PFlashConfig(mode="always"))
        assert kept == tokens
        assert metadata["compressed"] is False
        assert metadata["reason"] == "insufficient_middle_budget"
        assert metadata["dropped_tokens"] == 0
        assert metadata["endpoints_only"] is False
        assert metadata["middle_tokens_kept"] == 0

    def test_metadata_middle_tokens_kept_positive_on_normal_compression(self):
        _, metadata = compress_request_tokens(
            list(range(20000)), PFlashConfig(mode="always", keep_ratio=0.2)
        )
        assert metadata["endpoints_only"] is False
        assert metadata["middle_tokens_kept"] > 0

    def test_uncompressed_result_is_not_endpoints_only(self):
        result = compress_tokens(list(range(20000)), PFlashConfig(mode="off"))
        assert result.compressed is False
        assert result.endpoints_only is False

    def test_small_sink_tail_leaves_middle_on_short_prompt(self):
        cfg = PFlashConfig(
            mode="always",
            keep_ratio=0.5,
            min_keep_tokens=64,
            sink_tokens=8,
            tail_tokens=8,
            block_size=8,
        )
        result = compress_tokens(list(range(2000)), cfg)
        assert result.compressed is True
        assert result.middle_tokens_kept > 0
        assert result.endpoints_only is False

    def test_boundary_requires_positive_middle_budget(self):
        cfg = PFlashConfig(
            mode="always",
            keep_ratio=0.5,
            min_keep_tokens=1,
            sink_tokens=2,
            tail_tokens=2,
            block_size=1,
        )
        # Four-token budget is exactly consumed by the endpoints: preserve.
        exact = compress_tokens(list(range(8)), cfg)
        assert exact.compressed is False
        assert exact.reason == "insufficient_middle_budget"
        # One extra token funds real middle selection: compression may engage.
        above = compress_tokens(list(range(10)), cfg)
        assert above.compressed is True
        assert above.middle_tokens_kept == 1
