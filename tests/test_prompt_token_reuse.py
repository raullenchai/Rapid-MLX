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

    def decode(self, token_ids, skip_special_tokens=False):
        inverse = {token_id: piece for piece, token_id in self._vocab.items()}
        return "".join(
            inverse[token_id]
            if inverse[token_id] in self.markers
            else inverse[token_id][-1]
            for token_id in token_ids
        )


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


def test_system_tools_only_completion_saves_exact_cross_session_boundary():
    tokenizer = _SectionTokenizer()
    engine = _engine(tokenizer)
    messages = [{"role": "system", "content": "unchanged guidance " * 400}]
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]

    def requires_user(items, template_tools=None, **kwargs):
        if not any(item["role"] == "user" for item in items):
            raise ValueError("No user query found in messages")
        return _render(items, template_tools, **kwargs)

    engine._apply_chat_template = requires_user
    prompt, boundary = engine._shared_system_tools_prompt(messages, tools, False, None)
    real = tokenizer.encode(
        requires_user([*messages, {"role": "user", "content": "hi"}], tools)
    )
    assert boundary > 0
    assert prompt[:boundary] == real[:boundary]
    assert prompt[boundary] != real[boundary]
    assert prompt == tokenizer.encode(
        requires_user([*messages, {"role": "user", "content": "A"}], tools)
    )
    assert tokenizer.decode(prompt).endswith("<|im_start|>assistant\n<think>\n")


def test_system_only_without_tools_saves_shared_boundary():
    tokenizer = _SectionTokenizer()
    engine = _engine(tokenizer)
    messages = [{"role": "system", "content": "unchanged guidance " * 400}]
    warm = _render(messages)
    boundary = engine._compute_prefix_boundary(messages, generation_prompt=warm)
    next_turn = tokenizer.encode(
        _render([*messages, {"role": "user", "content": "hi"}])
    )
    warm_tokens = tokenizer.encode(warm)
    assert boundary > 0
    assert warm_tokens[:boundary] == next_turn[:boundary]
    assert warm_tokens[boundary] != next_turn[boundary]

    def requires_user(items, template_tools=None, **kwargs):
        if not any(item["role"] == "user" for item in items):
            raise ValueError("No user query found in messages")
        return _render(items, template_tools, **kwargs)

    engine._apply_chat_template = requires_user
    prompt, fallback_boundary = engine._shared_system_tools_prompt(
        messages, None, False, None
    )
    assert fallback_boundary > 0
    assert prompt[:fallback_boundary] == next_turn[:fallback_boundary]


def test_system_tools_only_keeps_a_template_that_accepts_it():
    engine = _engine(_SectionTokenizer())
    engine._loaded = True
    engine._prepare_harmony_no_thinking_prompt = lambda prompt, **_kwargs: (
        prompt,
        None,
    )
    messages = [{"role": "system", "content": "guidance"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    assert engine.build_prompt(messages, tools) == _render(messages)


def test_mllm_shared_warm_prefix_uses_actual_no_bos_tokenization():
    tokenizer = _SectionTokenizer(bos=True)
    engine = _engine(tokenizer)
    engine._is_mllm = True
    messages = [{"role": "system", "content": "shared guidance"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    def requires_user(items, template_tools=None, **kwargs):
        if not any(item["role"] == "user" for item in items):
            raise ValueError("No user query found in messages")
        return _render(items, template_tools, **kwargs)

    engine._apply_chat_template = requires_user
    warm, boundary = engine._shared_system_tools_prompt(messages, tools, False, None)
    real = tokenizer.encode(
        requires_user([*messages, {"role": "user", "content": "hi"}], tools),
        add_special_tokens=False,
    )
    assert warm[0] == real[0] != 1
    assert warm[:boundary] == real[:boundary]
    assert boundary > 0


def test_build_prompt_system_only_fallback_and_unrelated_template_error():
    engine = _engine(_SectionTokenizer())
    engine._loaded = True
    messages = [{"role": "system", "content": "guidance"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    def requires_user(items, template_tools=None, **kwargs):
        if not any(item["role"] == "user" for item in items):
            raise ValueError("No user query found in messages")
        return _render(items, template_tools, **kwargs)

    engine._apply_chat_template = requires_user
    assert engine.build_prompt(messages, tools).startswith("<|im_start|>system")

    engine._apply_chat_template = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ValueError("other template failure")
    )
    with pytest.raises(ValueError, match="other template failure"):
        engine.build_prompt(messages, tools)


def test_system_tools_prompt_rejects_templates_without_common_tokens():
    engine = _engine(_SectionTokenizer())
    engine._apply_chat_template = lambda items, *_args, **_kwargs: items[-1]["content"]
    with pytest.raises(ValueError, match="no distinct shared"):
        engine._shared_system_tools_prompt(
            [{"role": "system", "content": "guidance"}], [], False, None
        )


def test_system_only_boundary_handles_native_template_and_render_failures():
    engine = _engine(_SectionTokenizer())
    messages = [{"role": "system", "content": "guidance"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    assert engine._compute_prefix_boundary(messages) > 0
    assert engine._compute_prefix_boundary(messages, tools) > 0
    assert engine._compute_prefix_boundary([]) == 0
    assert (
        engine._compute_prefix_boundary([{"role": "assistant", "content": "hello"}])
        == 0
    )
    assert engine._compute_prefix_boundary([*messages, *messages]) == 0

    engine._apply_chat_template = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ValueError("cannot render")
    )
    assert engine._compute_prefix_boundary(messages, tools) == 0


@pytest.mark.asyncio
async def test_system_tools_warm_prompt_contains_forced_assistant_prefix():
    tokenizer = _SectionTokenizer()
    engine = _engine(tokenizer)
    engine._loaded = True
    engine._needs_prefix_boundary_snapshot = lambda: True
    messages = [{"role": "system", "content": "guidance"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    def requires_user(items, template_tools=None, **kwargs):
        if not any(item["role"] == "user" for item in items):
            raise ValueError("No user query found in messages")
        return _render(items, template_tools, **kwargs)

    engine._apply_chat_template = requires_user
    seen = {}

    async def generate(**kwargs):
        seen.update(kwargs)
        return None

    engine.generate = generate
    await engine.chat(messages, tools=tools, forced_assistant_prefix="<tool_call>")
    boundary = seen["prefix_boundary"]
    assert boundary > 0
    assert seen["exact_prefix_boundary"] is True
    assert seen["prompt"][
        -len(tokenizer.encode("<tool_call>", add_special_tokens=False)) :
    ] == tokenizer.encode("<tool_call>", add_special_tokens=False)
    assert tokenizer.decode(
        seen["prompt"][
            : -len(tokenizer.encode("<tool_call>", add_special_tokens=False))
        ]
    ).endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.asyncio
async def test_native_system_only_chat_boundary_and_forced_prefix():
    engine = _engine(_SectionTokenizer())
    engine._loaded = True
    engine._needs_prefix_boundary_snapshot = lambda: True
    engine._prepare_harmony_no_thinking_prompt = lambda prompt, **_kwargs: (
        prompt,
        None,
    )
    engine._compute_prefix_boundary = lambda *_args, **_kwargs: 5
    seen = {}

    async def generate(**kwargs):
        seen.update(kwargs)
        return None

    engine.generate = generate
    messages = [{"role": "system", "content": "guidance"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    await engine.chat(messages, tools=tools, forced_assistant_prefix="<tool_call>")
    assert seen["prompt"].endswith("<tool_call>")
    assert seen["prefix_boundary"] == 5
    assert seen["exact_prefix_boundary"] is True

    engine._apply_chat_template = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ValueError("other template failure")
    )
    with pytest.raises(ValueError, match="other template failure"):
        await engine.chat(messages, tools=tools)


@pytest.mark.asyncio
async def test_stream_system_tools_warm_boundary_and_template_errors():
    tokenizer = _SectionTokenizer()
    engine = _engine(tokenizer)
    engine._loaded = True
    engine._needs_prefix_boundary_snapshot = lambda: True
    engine._create_output_router = lambda: None
    messages = [{"role": "system", "content": "guidance"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    def requires_user(items, template_tools=None, **kwargs):
        if not any(item["role"] == "user" for item in items):
            raise ValueError("No user query found in messages")
        return _render(items, template_tools, **kwargs)

    engine._apply_chat_template = requires_user
    seen = {}

    class CapturedError(Exception):
        pass

    async def stream_generate(**kwargs):
        seen.update(kwargs)
        raise CapturedError
        yield  # pragma: no cover

    engine.stream_generate = stream_generate
    with pytest.raises(CapturedError):
        async for _ in engine.stream_chat(
            messages, tools=tools, forced_assistant_prefix="<tool_call>"
        ):
            pass
    assert seen["exact_prefix_boundary"] is True
    forced_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    assert seen["prompt"][-len(forced_ids) :] == forced_ids
    assert tokenizer.decode(seen["prompt"][: -len(forced_ids)]).endswith(
        "<|im_start|>assistant\n<think>\n"
    )

    engine._apply_chat_template = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ValueError("other template failure")
    )
    with pytest.raises(ValueError, match="other template failure"):
        await anext(engine.stream_chat(messages, tools=tools))

    engine._apply_chat_template = _render
    engine._prepare_harmony_no_thinking_prompt = lambda prompt, **_kwargs: (
        prompt,
        None,
    )
    engine._compute_prefix_boundary = lambda *_args, **_kwargs: 5
    with pytest.raises(CapturedError):
        await anext(engine.stream_chat(messages, tools=tools))
    assert seen["prefix_boundary"] == 5
    assert seen["exact_prefix_boundary"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("exact", [False, True])
async def test_text_stream_passes_exact_prefix_flag_without_breaking_admission(exact):
    engine = _engine(_SectionTokenizer())
    engine._loaded = True
    engine._select_mllm_request_path = lambda *_args: False
    engine._ensure_generation_admission = lambda: ("admission", False)
    seen = {}

    class AdmittedError(Exception):
        pass

    async def add_request(**kwargs):
        seen.update(kwargs)
        raise AdmittedError

    engine._engine = SimpleNamespace(add_request=add_request)
    with pytest.raises(AdmittedError):
        await anext(
            engine.stream_generate(
                "prompt", prefix_boundary=100, exact_prefix_boundary=exact
            )
        )
    assert seen["prefix_boundary"] == 100
    assert seen["exact_prefix_boundary"] is exact


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


def test_scheduler_still_exposes_the_shared_encoder():
    """``encode_prompt_text`` reaches into the scheduler by name.

    The scheduler module needs MLX, so check its source: a rename would
    otherwise silently turn the shared encode back into three.
    """
    import ast
    from pathlib import Path

    import rapid_mlx

    source = Path(rapid_mlx.__file__).with_name("scheduler.py").read_text()
    scheduler = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    methods = {
        node.name: [arg.arg for arg in node.args.args]
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef)
    }
    assert methods.get("_encode_prompt_string") == ["self", "prompt"]
