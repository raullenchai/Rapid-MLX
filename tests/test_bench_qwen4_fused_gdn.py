# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.bench_qwen4_fused_gdn_end_to_end import counter_delta, expected_path_counts


def test_expected_path_counts_prove_fused_execution_and_prefill_fallback():
    assert expected_path_counts("stock", 36, 256) == (0, 0)
    assert expected_path_counts("fused", 36, 256) == (9252, 36)


def test_expected_path_counts_reject_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        expected_path_counts("other", 36, 256)


def test_counter_delta_preserves_complete_decline_histogram():
    assert counter_delta(
        {"uninitialized cache": 38, "speculative rollback": 7},
        {"uninitialized cache": 2, "speculative rollback": 7},
    ) == {"uninitialized cache": 36}
