"""Warm agent turns must not re-tokenize the whole prompt per boundary probe.

Every chat request to a non-trimmable (hybrid) model probes the next turn's
cache boundary by rendering prompt variants and comparing their tokens with
the real prompt's.  On a 23k-token agent session each full encode costs about
20 ms on an M4 Pro, and the request used to pay five of them (context guard,
three probe variants, scheduler admission) before prefill could start.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from rapid_mlx.engine import batched
from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.prompt_host_cache import PromptHostCache
from rapid_mlx.prompt_token_reuse import (
    added_token_markers,
    common_prefix_length,
    encode_sharing_head,
)
from rapid_mlx.service import helpers

_MARKERS = ("<|im_start|>", "<|im_end|>", "<think>")


class _SectionTokenizer:
    """Toy tokenizer with the property the shortcut relies on.

    Added tokens split the input first; within every other section each
    token encodes a character together with the one before it (``^`` at the
    section start), so a tail that does not begin at a section edge
    tokenizes differently from the same text in context -- exactly what the
    shortcut must detect.
    """

    def __init__(
        self, *, bos: bool = False, split_added: bool = True, markers=_MARKERS
    ):
        self.bos = bos
        self.split_added = split_added
        self.markers = tuple(markers)
        self.encoded_chars = 0
        self.encode_calls = 0
        self._vocab: dict[str, int] = {
            marker: i + 10 for i, marker in enumerate(self.markers)
        }

    def get_added_vocab(self):
        return {marker: self._vocab[marker] for marker in self.markers}

    def _id(self, piece: str) -> int:
        return self._vocab.setdefault(piece, len(self._vocab) + 100)

    def encode(self, text, add_special_tokens=True):
        self.encode_calls += 1
        self.encoded_chars += len(text)
        # Longest added token first, like a real added-token matcher.
        ordered = sorted(self.markers, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(m) for m in ordered) + ")"
        sections = re.split(pattern, text) if self.split_added else [text]
        tokens = [1] if (self.bos and add_special_tokens) else []
        for section in sections:
            if section in self.markers and self.split_added:
                tokens.append(self._vocab[section])
                continue
            tokens.extend(
                self._id((section[i - 1] if i else "^") + section[i])
                for i in range(len(section))
            )
        return tokens


def _render(messages, tools=None, *, add_generation_prompt=True, **_kwargs):
    body = "".join(
        f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n" for m in messages
    )
    return body + ("<|im_start|>assistant\n<think>\n" if add_generation_prompt else "")


def _conversation(turns: int = 4) -> list[dict]:
    messages = [{"role": "system", "content": "tools and rules " * 400}]
    for turn in range(turns):
        messages.append({"role": "user", "content": f"question {turn} " * 7})
        messages.append({"role": "assistant", "content": f"answer {turn} " * 5})
    messages.append({"role": "user", "content": "latest odd-length request"})
    return messages


class _FakeScheduler:
    """Host-cached encoder with the scheduler's sharing contract."""

    def __init__(self, tokenizer, host_cache):
        self.tokenizer = tokenizer
        self.host_cache = host_cache

    def _encode_prompt_string(self, prompt):
        fingerprint = self.host_cache.fingerprint({"prompt": prompt})
        cached = self.host_cache.get_tokens(fingerprint)
        if cached is not None:
            return cached
        tokens = list(self.tokenizer.encode(prompt))
        self.host_cache.put_tokens(fingerprint, tokens)
        return tokens


def _engine(tokenizer, *, scheduler: bool = True):
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._processor = None
    engine._tokenizer = tokenizer
    engine._model_name = "test-model"
    host_cache = PromptHostCache(max_entries=8, max_bytes=1 << 24, enabled=True)
    engine._prompt_host_cache = host_cache
    engine._engine = (
        SimpleNamespace(
            engine=SimpleNamespace(scheduler=_FakeScheduler(tokenizer, host_cache))
        )
        if scheduler
        else None
    )
    engine._apply_chat_template = _render
    return engine


def test_common_prefix_length():
    assert common_prefix_length("", "abc") == 0
    assert common_prefix_length("abc", "abc") == 3
    assert common_prefix_length("abc", "abcdef") == 3
    assert common_prefix_length("abXd", "abYd") == 2
    assert common_prefix_length("x" * 1000 + "a", "x" * 1000 + "b") == 1000
    assert common_prefix_length("a", "b") == 0


def test_added_token_markers_tolerates_missing_or_broken_surfaces():
    assert added_token_markers(object()) == ()
    assert added_token_markers(SimpleNamespace(get_added_vocab=lambda: ["x"])) == ()

    def broken():
        raise RuntimeError("no vocab")

    assert added_token_markers(SimpleNamespace(get_added_vocab=broken)) == ()
    vocab = {"<a>": 1, "": 2, 3: 4}
    assert added_token_markers(SimpleNamespace(get_added_vocab=lambda: vocab)) == (
        "<a>",
    )


@pytest.mark.parametrize("bos", [False, True])
def test_shared_head_encoding_matches_full_encode(bos):
    tokenizer = _SectionTokenizer(bos=bos)
    messages = _conversation()
    real = _render(messages)
    real_tokens = tokenizer.encode(real)
    variants = [
        _render(messages, add_generation_prompt=False),
        _render(
            [*messages, {"role": "assistant", "content": "__probe__"}],
            add_generation_prompt=False,
        ),
        _render([*messages[:-1], {"role": "user", "content": "XXXXXXXXXX"}]),
    ]
    for variant in variants:
        reused = encode_sharing_head(
            real,
            real_tokens,
            variant,
            encode_tail=lambda tail: tokenizer.encode(tail, add_special_tokens=False),
            markers=added_token_markers(tokenizer),
        )
        assert reused == tokenizer.encode(variant)


def test_shared_head_encoding_refuses_unproven_shortcuts():
    tokenizer = _SectionTokenizer()
    real = _render(_conversation())
    variant = _render(_conversation(), add_generation_prompt=False)
    real_tokens = tokenizer.encode(real)

    def attempt(
        *,
        variant=variant,
        real_tokens=real_tokens,
        encode_tail=lambda tail: tokenizer.encode(tail, add_special_tokens=False),
        markers=_MARKERS,
    ):
        return encode_sharing_head(
            real, real_tokens, variant, encode_tail=encode_tail, markers=markers
        )

    # No added tokens to cut at, or nothing shared.
    assert attempt(markers=()) is None
    assert attempt(real_tokens=[]) is None
    assert attempt(variant="Z" + variant) is None
    # Shared text holds no marker.
    assert (
        encode_sharing_head(
            "plain shared text A",
            tokenizer.encode("plain shared text A"),
            "plain shared text B",
            encode_tail=lambda tail: tokenizer.encode(tail, add_special_tokens=False),
            markers=_MARKERS,
        )
        is None
    )
    # A tokenizer that does not split on added tokens fails the real-side proof.
    merged = _SectionTokenizer(split_added=False)
    assert (
        encode_sharing_head(
            real,
            merged.encode(real),
            variant,
            encode_tail=lambda tail: merged.encode(tail, add_special_tokens=False),
            markers=_MARKERS,
        )
        is None
    )
    # A tail encoder that injects BOS cannot be spliced.
    assert attempt(encode_tail=lambda tail: [1, *tokenizer.encode(tail)]) is None
    # A tail longer than the whole prompt is impossible.
    assert attempt(encode_tail=lambda tail: list(range(len(real_tokens) + 5))) is None

    def explode(_tail):
        raise ValueError("tokenizer failure")

    assert attempt(encode_tail=explode) is None


def test_shared_head_encoding_refuses_cut_inside_a_longer_added_token():
    """A longer added token may swallow the cut on the variant side only."""
    short, long = "<A>", "<A>" + "x" * 30 + "Y"
    tokenizer = _SectionTokenizer(markers=(short, long))
    real = "<A>" + "x" * 30 + "Z"
    variant = long
    reused = encode_sharing_head(
        real,
        tokenizer.encode(real),
        variant,
        encode_tail=lambda tail: tokenizer.encode(tail, add_special_tokens=False),
        markers=added_token_markers(tokenizer),
    )
    assert reused is None
    # The same cut with no overlapping token is still taken.
    plain = _SectionTokenizer(markers=(short,))
    assert encode_sharing_head(
        real,
        plain.encode(real),
        "<A>" + "x" * 30 + "Y",
        encode_tail=lambda tail: plain.encode(tail, add_special_tokens=False),
        markers=(short,),
    ) == plain.encode("<A>" + "x" * 30 + "Y")


def test_warm_turn_tokenizes_the_full_prompt_once():
    """Context guard + boundary probe + admission share one full encode.

    Before the fix this request encoded ~5x the prompt's characters: the
    guard's count, the probe's real/stable/next-turn encodes and admission.
    """
    tokenizer = _SectionTokenizer()
    engine = _engine(tokenizer)
    messages = _conversation()
    real = _render(messages)

    expected = tokenizer.encode(real)
    tokenizer.encoded_chars = 0

    # The request's own flow on a cold host cache: guard, probe, admission.
    assert helpers.count_prompt_tokens(engine, real) == len(expected)
    boundary = engine._compute_prefix_boundary(messages, generation_prompt=real)
    admitted = engine._engine.engine.scheduler._encode_prompt_string(real)

    # One full encode (the guard's); the probe variants only encode tails.
    assert len(real) <= tokenizer.encoded_chars < len(real) + 400
    assert admitted == expected

    # Same boundary as the full-encode path the probe used before.
    assert boundary == _reference_boundary(messages, real)
    stable_tokens = _SectionTokenizer().encode(
        _render(messages, add_generation_prompt=False)
    )
    assert boundary == len(stable_tokens) - batched._PREFIX_BOUNDARY_REPLAY_TOKENS


def _reference_boundary(messages, real, **kwargs):
    reference = _engine(_SectionTokenizer(), scheduler=False)
    reference._tokenizer.get_added_vocab = lambda: {}
    return reference._compute_prefix_boundary(
        messages, generation_prompt=real, **kwargs
    )


def test_transient_probe_uses_the_shared_head():
    tokenizer = _SectionTokenizer()
    engine = _engine(tokenizer)
    messages = [
        *_conversation(),
        {"role": "developer", "content": "temporary progress checkpoint"},
    ]
    real = _render(messages)
    engine.encode_prompt_text(real)
    tokenizer.encoded_chars = 0
    kwargs = {"transient_message_start": len(messages) - 1}
    boundary = engine._compute_prefix_boundary(
        messages, generation_prompt=real, **kwargs
    )
    assert tokenizer.encoded_chars < 400
    assert boundary > 0
    assert boundary == _reference_boundary(messages, real, **kwargs)


def test_dummy_user_fallback_uses_the_shared_head(monkeypatch):
    """Templates whose no-generation form is not a prefix take the fallback."""

    def render(messages, tools=None, *, add_generation_prompt=True, **_kwargs):
        if not add_generation_prompt:
            return _render(messages, add_generation_prompt=False) + "trailer"
        return _render(messages)

    tokenizer = _SectionTokenizer()
    engine = _engine(tokenizer)
    engine._apply_chat_template = render
    messages = _conversation()
    real = render(messages)
    engine.encode_prompt_text(real)
    tokenizer.encoded_chars = 0
    boundary = engine._compute_prefix_boundary(messages, generation_prompt=real)
    assert tokenizer.encoded_chars < 400

    reference = _engine(_SectionTokenizer(), scheduler=False)
    reference._tokenizer.get_added_vocab = lambda: {}
    reference._apply_chat_template = render
    assert boundary == reference._compute_prefix_boundary(
        messages, generation_prompt=real
    )
    assert boundary > 0


def test_engines_without_a_text_scheduler_keep_their_tokenizer_calls():
    class _RecordingTokenizer(_SectionTokenizer):
        def __init__(self):
            super().__init__()
            self.kwargs = []

        def encode(self, text, **kwargs):
            self.kwargs.append(kwargs)
            return super().encode(text, **kwargs)

    tokenizer = _RecordingTokenizer()
    engine = _engine(tokenizer, scheduler=False)
    assert engine.encode_prompt_text("<|im_start|>hi") is None
    assert helpers.count_prompt_tokens(engine, "<|im_start|>hi") == 3
    assert tokenizer.kwargs == [{"add_special_tokens": True}]


def test_count_prompt_tokens_keeps_direct_path_for_bos_prefixed_prompts():
    class _BosTokenizer(_SectionTokenizer):
        bos_token = "<s>"

    tokenizer = _BosTokenizer(bos=True)
    engine = _engine(tokenizer)
    prompt = "<s>hello"
    assert helpers.count_prompt_tokens(engine, prompt) == len(
        tokenizer.encode(prompt, add_special_tokens=False)
    )
    assert engine._prompt_host_cache.stats()["stores"] == 0


def test_token_list_generation_prompt_keeps_full_variant_encodes():
    """Harmony-style token-id prompts have no text to share a head with."""
    tokenizer = _SectionTokenizer()
    engine = _engine(tokenizer)
    messages = _conversation()
    real = _render(messages)
    boundary = engine._compute_prefix_boundary(
        messages, generation_prompt=tokenizer.encode(real)
    )
    assert boundary == _reference_boundary(messages, real)
    assert engine._prompt_host_cache.stats()["stores"] == 0
