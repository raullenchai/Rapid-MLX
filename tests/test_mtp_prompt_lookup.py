# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_mlx.spec_decode.mtp.prompt_lookup import PromptLookupIndex


def test_prompt_lookup_returns_prompt_continuation() -> None:
    index = PromptLookupIndex([1, 2, 3, 4, 5, 6, 7, 8], min_ngram=3)

    match = index.propose([9, 1, 2, 3], max_tokens=3)

    assert match is not None
    assert match.start == 3
    assert match.matched_suffix == 3
    assert match.tokens == (4, 5, 6)


def test_prompt_lookup_prefers_longest_suffix_match() -> None:
    prompt = [8, 1, 2, 3, 4, 9, 1, 2, 3, 5]
    index = PromptLookupIndex(prompt, min_ngram=3, max_ngram=4)

    match = index.propose([0, 8, 1, 2, 3], max_tokens=2)

    assert match is not None
    assert match.start == 4
    assert match.matched_suffix == 4
    assert match.tokens == (4, 9)


def test_prompt_lookup_never_copies_generated_only_text() -> None:
    index = PromptLookupIndex([1, 2, 3, 4], min_ngram=2)

    assert index.propose([7, 8, 7, 8], max_tokens=4) is None


def test_prompt_lookup_excludes_prompt_edge_without_continuation() -> None:
    index = PromptLookupIndex([1, 2, 3, 4], min_ngram=2)

    assert index.propose([3, 4], max_tokens=4) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_ngram": 1}, "min_ngram"),
        ({"min_ngram": 4, "max_ngram": 3}, "max_ngram"),
        ({"max_candidates": 0}, "max_candidates"),
    ],
)
def test_prompt_lookup_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PromptLookupIndex([1, 2, 3], **kwargs)
