"""Keep streamed terminal receipts from corrupting exact-token comparisons."""

from types import SimpleNamespace as Row

import pytest

from scripts.benchmark_m5_native_mtp import emitted_token_ids


def test_terminal_receipt_does_not_duplicate_the_last_streamed_token():
    rows = [
        Row(token=17, generation_tokens=1),
        Row(token=23, generation_tokens=2),
        Row(token=23, generation_tokens=2, token_ids=[17, 23]),
    ]
    assert emitted_token_ids(rows) == [17, 23]


def test_fallback_includes_an_eos_token_only_emitted_by_the_terminal_frame():
    rows = [Row(token=17, generation_tokens=1), Row(token=2, generation_tokens=2)]
    assert emitted_token_ids(rows) == [17, 2]


def test_fallback_deduplicates_terminal_frame_by_count_not_token_value():
    rows = [
        Row(token=17, generation_tokens=1),
        Row(token=17, generation_tokens=2),
        Row(token=17, generation_tokens=2),
    ]
    assert emitted_token_ids(rows) == [17, 17]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [Row(token=7, generation_tokens=2)],
        [Row(token=7, generation_tokens=2, token_ids=[7])],
    ],
)
def test_missing_or_inconsistent_receipts_fail_closed(rows):
    with pytest.raises(RuntimeError):
        emitted_token_ids(rows)
