# SPDX-License-Identifier: Apache-2.0
"""Synthetic tokenizer for OutputRouter-level regression tests.

OutputRouter regressions (issues #447 / #455 / #468) sit downstream of
the tokenizer: the router reads ``get_vocab()`` to discover special
token IDs and calls ``decode()`` to render non-control tokens. Loading
a real tokenizer for each regression is slow and brittle (HF cache,
network, version drift). A frozen synthetic vocab gives us:

  * Stable token IDs across runs — the test asserts on integers we
    chose, so any vocab drift on a real model can't silently change
    the regression's blast radius.
  * Deterministic decode — ``decode([tid])`` returns the literal
    surface form we registered, so the router's text-emission path is
    exercised faithfully.
  * Zero filesystem / network — the harness instantiates in <1ms.

Pattern lifted from vLLM ``tests/tool_parsers/conftest.py``'s
``DummyTokenizer`` (used for the same purpose) and SGLang's
``test/srt/test_special_token_routing.py`` fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeTokenizer:
    """Minimal tokenizer surface OutputRouter needs.

    Only implements ``get_vocab`` and ``decode``. Both
    ``OutputRouter.from_tokenizer`` (discovery) and ``feed`` /
    ``finalize`` (rendering) use exclusively these two methods.
    """

    vocab: dict[str, int]
    _inverse: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        # If duplicate IDs exist, last-write-wins on the inverse map
        # (test author bug — assert loudly rather than silently
        # corrupt decode output).
        seen: dict[int, str] = {}
        for tok, tid in self.vocab.items():
            assert tid not in seen, (
                f"FakeTokenizer vocab has duplicate token ID {tid}: "
                f"{seen[tid]!r} vs {tok!r}. Pick unique IDs per token."
            )
            seen[tid] = tok
        self._inverse = seen

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def decode(self, token_ids: list[int]) -> str:
        # Unknown IDs render as ``<unk:NNN>`` rather than empty string
        # so a test that accidentally feeds an unregistered ID gets a
        # loud diagnostic in the assertion message.
        return "".join(self._inverse.get(tid, f"<unk:{tid}>") for tid in token_ids)


# Token-ID conventions for synthetic Gemma 4 vocab. Numbers are
# arbitrary but chosen to match the ``MODEL_TOKEN_MAPS`` comments in
# ``vllm_mlx/output_router.py`` (e.g. ``<|channel> = 100``) so that a
# reader can pattern-match the synthetic IDs against the source.
GEMMA4_VOCAB: dict[str, int] = {
    # Channel control
    "<|channel>": 100,
    "<channel|>": 101,
    "thought": 45518,
    "content": 3955,
    "final": 10218,
    # Turn control
    "<|turn>": 105,
    "<turn|>": 106,
    # Tool control
    "<|tool_call>": 48,
    "<tool_call|>": 49,
    '<|"|>': 52,
    "<|tool>": 46,
    "<tool|>": 47,
    "<|tool_response>": 50,
    "<tool_response|>": 51,
    # Standard
    "<bos>": 2,
    "<eos>": 3,
    "<pad>": 4,
    # Content surface tokens — arbitrary but distinguishable so a
    # regression message reading ``"thoughtanalysisfinalmessage"`` is
    # obviously interpretable as "all the literal channel words leaked
    # plus the body". Use single-string tokens (no per-char split) to
    # keep the test setup ergonomic; the router doesn't care.
    "\n": 200,
    "analysis_body": 1001,
    "message_body": 1002,
    "Hello": 1003,
    " world": 1004,
}


def gemma4_fake_tokenizer() -> FakeTokenizer:
    """Build a synthetic Gemma 4 tokenizer for OutputRouter tests."""
    return FakeTokenizer(vocab=dict(GEMMA4_VOCAB))
