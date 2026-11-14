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
from rapid_mlx.engine.base import GenerationOutput
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
        return GenerationOutput(text="")

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
        return GenerationOutput(text="")

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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mllm", [False, True])
async def test_custom_tokenizer_request_ids_preserve_legacy_contract(stream, mllm):
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    tokenizer = _SectionTokenizer()
    tokenizer.bos_token = None
    tokenizer.apply_chat_template = lambda messages, tokenize=True, **kw: (
        tokenizer.encode(_render(messages, **kw), add_special_tokens=False)
        if tokenize
        else _render(messages, **kw)
    )
    engine = _engine(tokenizer, scheduler=False)
    engine._is_mllm = mllm
    engine._loaded = True
    engine._needs_prefix_boundary_snapshot = lambda: True
    messages = _conversation()
    rendered = _render(messages)
    expected = tokenizer.encode(rendered, add_special_tokens=not mllm)
    tokenizer.encoded_chars = 0
    tokens = RequestPromptTokens()
    assert helpers.enforce_context_length_for_messages(
        engine,
        messages,
        request_prompt_tokens=tokens,
    ) == len(expected)
    seen = {}

    async def generate(**kwargs):
        seen.update(kwargs)
        return GenerationOutput(text="ok", new_text="ok", finished=True)

    async def stream_generate(**kwargs):
        yield await generate(**kwargs)

    engine.generate = generate
    engine.stream_generate = stream_generate
    if stream:
        _ = [out async for out in engine.stream_chat(messages, _prompt_tokens=tokens)]
    else:
        await engine.chat(messages, _prompt_tokens=tokens)
    # A custom MLLM tokenizer keeps the processor's own ``tokenizer(...)`` call,
    # which need not match ``encode`` (review P1); text lanes reuse exact IDs.
    assert seen["prompt"] == (rendered if mllm else expected)
    full_encodes = (
        3 if mllm else 1
    )  # Unknown special-token policy keeps its exact fallback.
    assert len(rendered) <= tokenizer.encoded_chars < full_encodes * len(rendered) + 400
    assert seen["prefix_boundary"] == _reference_boundary(messages, rendered)


def _spy_backend_encodes(monkeypatch, tokenizer, spy):
    """Count full encodes at the Rust backend, so the Python tokenizer hooks
    stay the library's own and remain eligible for ID reuse.

    ``spy(text, method, *args)`` must call the returned function with the same
    arguments to perform the real backend encode.
    """
    real = tokenizer._tokenizer

    def text_of(inputs):
        return inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else inputs

    class Backend:
        def __getattr__(self, name):
            return getattr(real, name)

        def __setattr__(self, name, value):
            setattr(real, name, value)

        def encode(self, sequence, *args, **kwargs):
            return spy(sequence, "encode", sequence, *args, **kwargs)

        def encode_batch(self, inputs, *args, **kwargs):
            return spy(text_of(inputs), "encode_batch", inputs, *args, **kwargs)

        def encode_batch_fast(self, inputs, *args, **kwargs):
            return spy(text_of(inputs), "encode_batch_fast", inputs, *args, **kwargs)

    monkeypatch.setattr(tokenizer, "_tokenizer", Backend())
    return lambda _text, method, *args, **kwargs: getattr(real, method)(*args, **kwargs)


def _fast_tokenizer(style="chatml", *, bos=False):
    from tokenizers import Tokenizer, models, processors
    from transformers import PreTrainedTokenizerFast

    markers = {
        "chatml": ("<|im_start|>", "<|im_end|>"),
        "gemma": ("<start_of_turn>", "<end_of_turn>"),
        "llama": ("<|start_header_id|>", "<|eot_id|>"),
        "deepseek": ("<｜User｜>", "<｜end▁of▁sentence｜>"),
    }[style]
    special = ["<unk>", "<bos>", *markers, "<think>"]
    chars = sorted(
        set(
            "tools and rules systemuserassistantlatest questionanswer0123456789_<>|/\\n界"
        )
    )
    vocab = {piece: i for i, piece in enumerate([*special, *chars])}
    backend = Tokenizer(models.BPE(vocab, [], unk_token="<unk>"))
    if bos:
        backend.post_processor = processors.TemplateProcessing(
            single="<bos> $A",
            special_tokens=[("<bos>", vocab["<bos>"])],
        )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        additional_special_tokens=list(markers) + ["<think>"],
    )
    start, end = markers
    tokenizer.chat_template = (
        "{{ prompt_label | default('') }}{% if tools %}{{ tools | tojson }}{% endif %}"
        "{% for m in messages %}"
        + start
        + "{{m['role']}}\n{{m['content']}}"
        + end
        + "\n"
        "{% endfor %}{% if add_generation_prompt %}"
        + start
        + "assistant\n{% if enable_thinking is not defined or enable_thinking %}<think>\n{% endif %}"
        "{% endif %}"
    )
    return tokenizer


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mllm", [False, True])
@pytest.mark.parametrize("style", ["chatml", "gemma", "llama", "deepseek"])
@pytest.mark.parametrize("bos", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
async def test_fast_chat_tokenizes_once_and_preserves_ids_and_boundary(
    monkeypatch, stream, mllm, style, bos, thinking
):
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    tokenizer = _fast_tokenizer(style, bos=bos)
    engine = _engine(tokenizer, scheduler=False)
    engine._loaded = True
    engine._is_mllm = mllm
    engine._needs_prefix_boundary_snapshot = lambda: True
    del (
        engine._apply_chat_template
    )  # Exercise production rendering and kwargs handling.
    engine._prompt_host_cache.enabled = False
    messages = _conversation()
    tools = (
        [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ]
        if style in ("chatml", "gemma")
        else None
    )
    template_kwargs = {
        "enable_thinking": thinking,
        "chat_template_kwargs": {"prompt_label": "rules\n"},
    }
    from rapid_mlx.api.tool_calling import convert_tools_for_template

    rendered = engine._apply_chat_template(
        messages,
        convert_tools_for_template(tools) if tools else None,
        **template_kwargs,
    )
    expected = tokenizer.encode(rendered, add_special_tokens=not mllm)
    # Force the historical full-encode reference, with exactly the same BOS policy.
    monkeypatch.setattr(tokenizer, "get_added_vocab", lambda: {})
    boundary = engine._compute_prefix_boundary(
        messages, tools, generation_prompt=rendered, **template_kwargs
    )
    monkeypatch.undo()
    calls = []

    def spy(text, *args, **kwargs):
        calls.append(text)
        return encode_plus(text, *args, **kwargs)

    encode_plus = _spy_backend_encodes(monkeypatch, tokenizer, spy)
    state = RequestPromptTokens()
    helpers.enforce_context_length_for_messages(
        engine,
        messages,
        tools=tools,
        request_prompt_tokens=state,
        **template_kwargs,
    )
    seen = {}

    async def generate(**kwargs):
        seen.update(kwargs)
        if isinstance(kwargs["prompt"], str):
            # Emulate the legacy scheduler admission when the engine submits text.
            seen["prompt"] = tokenizer.encode(
                kwargs["prompt"], add_special_tokens=not mllm
            )
        return GenerationOutput(text="ok", new_text="ok", finished=True)

    async def stream_generate(**kwargs):
        yield await generate(**kwargs)

    engine.generate = generate
    engine.stream_generate = stream_generate
    if stream:
        _ = [
            out
            async for out in engine.stream_chat(
                messages, tools=tools, _prompt_tokens=state, **template_kwargs
            )
        ]
    else:
        await engine.chat(
            messages, tools=tools, _prompt_tokens=state, **template_kwargs
        )
    assert seen["prompt"] == expected
    assert seen["prefix_boundary"] == boundary
    assert calls.count(rendered) == 1
    assert all(len(text) < len(rendered) // 2 for text in calls if text != rendered), [
        (len(t), t[-80:]) for t in calls
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mllm", [False, True])
@pytest.mark.parametrize("style", ["chatml", "gemma", "llama", "deepseek"])
@pytest.mark.parametrize("bos", [False, True])
async def test_engine_tokenizes_once_when_gate_text_differs(
    monkeypatch, stream, mllm, style, bos
):
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    tokenizer = _fast_tokenizer(style, bos=bos)
    engine = _engine(tokenizer, scheduler=False)
    engine._loaded = True
    engine._is_mllm = mllm
    engine._needs_prefix_boundary_snapshot = lambda: True
    del (
        engine._apply_chat_template
    )  # Exercise production rendering and kwargs handling.
    engine._prompt_host_cache.enabled = False
    messages = _conversation()
    tools = (
        [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ]
        if style in ("chatml", "gemma")
        else None
    )
    template_kwargs = {
        "enable_thinking": False,
        "chat_template_kwargs": {"prompt_label": "rules\n"},
    }
    from rapid_mlx.api.tool_calling import convert_tools_for_template

    rendered = engine._apply_chat_template(
        messages,
        convert_tools_for_template(tools) if tools else None,
        **template_kwargs,
    )
    expected = tokenizer.encode(rendered, add_special_tokens=not mllm)
    # Force the historical full-encode reference, with exactly the same BOS policy.
    monkeypatch.setattr(tokenizer, "get_added_vocab", lambda: {})
    boundary = engine._compute_prefix_boundary(
        messages, tools, generation_prompt=rendered, **template_kwargs
    )
    monkeypatch.undo()
    calls = []

    def spy(text, *args, **kwargs):
        calls.append(text)
        return encode_plus(text, *args, **kwargs)

    encode_plus = _spy_backend_encodes(monkeypatch, tokenizer, spy)
    # The MLLM gate renders with the tokenizer template, while the engine may
    # render with the processor template; model the gate's IDs as unrelated
    # text, so the engine's own boundary + admission must share one encode.
    state = RequestPromptTokens()
    state.encode(tokenizer, "gate text", add_special_tokens=False)
    calls.clear()
    seen = {}

    async def generate(**kwargs):
        seen.update(kwargs)
        if isinstance(kwargs["prompt"], str):
            # Emulate the legacy scheduler admission when the engine submits text.
            seen["prompt"] = tokenizer.encode(
                kwargs["prompt"], add_special_tokens=not mllm
            )
        return GenerationOutput(text="ok", new_text="ok", finished=True)

    async def stream_generate(**kwargs):
        yield await generate(**kwargs)

    engine.generate = generate
    engine.stream_generate = stream_generate
    if stream:
        _ = [
            out
            async for out in engine.stream_chat(
                messages, tools=tools, _prompt_tokens=state, **template_kwargs
            )
        ]
    else:
        await engine.chat(
            messages, tools=tools, _prompt_tokens=state, **template_kwargs
        )
    assert seen["prompt"] == expected
    assert seen["prefix_boundary"] == boundary
    assert calls.count(rendered) == 1
    assert all(len(text) < len(rendered) // 2 for text in calls if text != rendered), [
        (len(t), t[-80:]) for t in calls
    ]


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mllm", [False, True])
def test_http_chat_reuses_request_ids(monkeypatch, api, stream, mllm):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.routes.anthropic import router as anthropic_router
    from rapid_mlx.routes.chat import router as chat_router

    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._is_mllm = mllm
    engine._tokenizer = _fast_tokenizer()
    engine.check_admission = lambda: None  # No scheduler in this CPU-only route test.
    engine._needs_prefix_boundary_snapshot = lambda: True
    # The request-local IDs must work even when the process LRU retains nothing.
    engine._prompt_host_cache.enabled = False
    tokenizer = engine.tokenizer
    calls = []
    threads = []

    def spy(text, *args, **kwargs):
        import threading

        calls.append(text)
        threads.append(threading.get_ident())
        return encode_plus(text, *args, **kwargs)

    encode_plus = _spy_backend_encodes(monkeypatch, tokenizer, spy)
    seen = {}

    async def generate(**kwargs):
        import threading

        seen.update(kwargs)
        seen["loop_thread"] = threading.get_ident()
        return GenerationOutput(
            text="ok",
            raw_text="ok",
            new_text="ok",
            prompt_tokens=len(kwargs["prompt"]),
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    async def stream_generate(**kwargs):
        yield await generate(**kwargs)

    engine.generate = generate
    engine.stream_generate = stream_generate
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.tool_call_parser = None
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(anthropic_router)
    messages = _conversation()
    payload = {
        "model": "test-model",
        "messages": messages,
        "max_tokens": 8,
        "stream": stream,
    }
    path = "/v1/chat/completions"
    if api == "anthropic":
        path = "/v1/messages"
        payload["system"] = messages[0]["content"]
        payload["messages"] = messages[1:]
    try:
        with TestClient(app) as client:
            response = client.post(path, json=payload)
        assert response.status_code == 200, response.text
        assert isinstance(seen["prompt"], list)
        full_calls = [
            (text, thread) for text, thread in zip(calls, threads) if len(text) > 1000
        ]
        assert len(full_calls) == 1, [(len(text), text[-80:]) for text, _ in full_calls]
        assert full_calls[0][1] != seen["loop_thread"]
        assert seen["prefix_boundary"] > 0
    finally:
        reset_config()


def test_request_reuses_both_bos_policies_without_reencoding(monkeypatch):
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    tokenizer = _fast_tokenizer(bos=True)
    prompt = "<bos><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    expected_count_ids = tokenizer.encode(prompt, add_special_tokens=False)
    expected_model_ids = tokenizer.encode(prompt)
    assert expected_count_ids != expected_model_ids
    calls = []

    def spy(text, *args, **kwargs):
        calls.append(text)
        return encode_plus(text, *args, **kwargs)

    encode_plus = _spy_backend_encodes(monkeypatch, tokenizer, spy)
    engine = _engine(tokenizer, scheduler=False)
    state = RequestPromptTokens()
    assert helpers.count_prompt_tokens(engine, prompt, prompt_tokens=state) == len(
        expected_count_ids
    )
    assert engine._request_prompt_ids(prompt, state) == expected_model_ids
    assert (
        engine._request_prompt_ids(prompt, state, mllm_step=True) == expected_count_ids
    )
    assert calls == [prompt]


@pytest.mark.asyncio
async def test_harmony_request_reuses_initial_encode(monkeypatch):
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens
    from tests.test_engine_router_non_stream import _HARMONY_VOCAB, _FakeTokenizer

    tokenizer = _FakeTokenizer(_HARMONY_VOCAB)
    engine = _engine(tokenizer, scheduler=False)
    engine._loaded = True
    engine._needs_prefix_boundary_snapshot = lambda: False
    engine._apply_chat_template = lambda *a, **kw: "<|start|>assistant"
    messages = [{"role": "user", "content": "hello"}]
    old_prompt = engine.build_prompt(messages, enable_thinking=False)
    expected = tokenizer.encode(old_prompt)
    calls = []
    encode = tokenizer.encode

    def spy(text, **kwargs):
        calls.append(text)
        return encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", spy)
    state = RequestPromptTokens()
    assert helpers.enforce_context_length_for_messages(
        engine,
        messages,
        enable_thinking=False,
        request_prompt_tokens=state,
    ) == len(expected)
    seen = {}

    async def generate(**kwargs):
        seen.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = generate
    await engine.chat(messages, enable_thinking=False, _prompt_tokens=state)
    assert seen["prompt"] == expected
    assert calls == ["<|start|>assistant"]


def test_mllm_preparation_consumes_ids_without_tokenization():
    import numpy as np

    from tests.test_mllm_prompt_admission_cpu import _load_function

    prepare = _load_function(
        "rapid_mlx/mllm_batch_generator.py",
        "_preprocess_request",
        {"mx": np},
        "MLLMBatchGenerator",
    )
    ids = [1, 4, 9]
    request = SimpleNamespace(prompt=ids, images=None, videos=None)
    # The pre-tokenized branch must not access a tokenizer or processor.
    prepare(object(), request)
    assert request.input_ids.tolist() == [ids]
    assert request.attention_mask is None
    assert request.extra_kwargs == {}


@pytest.mark.asyncio
async def test_concurrent_frontend_tokenization_is_serial_and_request_local(
    monkeypatch,
):
    import asyncio
    import time

    from rapid_mlx.prompt_token_reuse import RequestPromptTokens, run_prompt_work

    tokenizer = _fast_tokenizer()
    engine = _engine(tokenizer, scheduler=False)
    active = 0
    maximum = 0

    def spy(text, *args, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            time.sleep(0.01)
            return encode_plus(text, *args, **kwargs)
        finally:
            active -= 1

    expected = [tokenizer.encode("a" * 1000), tokenizer.encode("b" * 1000)]
    encode_plus = _spy_backend_encodes(monkeypatch, tokenizer, spy)
    states = [RequestPromptTokens(), RequestPromptTokens()]
    results = await asyncio.gather(
        *[
            asyncio.to_thread(
                run_prompt_work, engine, engine._request_prompt_ids, prompt, state
            )
            for prompt, state in zip(["a" * 1000, "b" * 1000], states)
        ]
    )
    assert results == expected
    assert maximum == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mllm_step", [False, True])
async def test_deferred_lane_uses_selected_scheduler_bos_policy(stream, mllm_step):
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens
    from rapid_mlx.request import RequestOutput

    tokenizer = _fast_tokenizer(bos=True)
    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._is_mllm = True
    engine._tokenizer = tokenizer
    engine._ensure_generation_admission = lambda: ("test", False)
    engine._select_mllm_request_path = lambda *args: mllm_step
    engine._transfer_admission_to_scheduler = lambda *a, **kw: None
    seen = {}
    output = RequestOutput(
        request_id="test", output_text="ok", new_text="ok", finished=True
    )

    async def generate(**kwargs):
        seen.update(kwargs)
        return output

    async def add_request(**kwargs):
        seen.update(kwargs)
        return "test"

    async def outputs(request_id):
        yield output

    scheduler = SimpleNamespace(
        generate=generate,
        add_request=add_request,
        add_request_async=add_request,
        stream_outputs=outputs,
    )
    engine._mllm_scheduler = scheduler
    engine._engine = scheduler
    state = RequestPromptTokens()
    prompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    expected = tokenizer.encode(prompt, add_special_tokens=not mllm_step)
    state.encode(tokenizer, prompt, add_special_tokens=False)
    if stream:
        _ = [out async for out in engine.stream_generate(prompt, _prompt_tokens=state)]
    else:
        await engine.generate(prompt, _prompt_tokens=state)
    assert seen["prompt"] == expected
    assert "_prompt_tokens" not in seen


def test_guided_prefill_consumes_exact_request_ids(monkeypatch):
    import sys
    from types import ModuleType

    from tests.test_mllm_prompt_admission_cpu import _load_function

    class PrefillReachedError(Exception):
        pass

    expected = [7, 4, 8]

    def array(ids):
        assert ids == expected
        raise PrefillReachedError

    cache_module = ModuleType("mlx_lm.models.cache")
    cache_module.make_prompt_cache = lambda model: []
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)
    decode = _load_function(
        "rapid_mlx/api/guided.py",
        "_decode_constrained",
        {
            "LLMatcher": lambda *a: SimpleNamespace(get_error=lambda: ""),
            "allocate_token_bitmask": lambda *a: None,
            "mx": SimpleNamespace(array=array),
            "_PREFILL_STEP_SIZE": 8,
        },
        "GuidedGenerator",
    )
    generator = SimpleNamespace(
        _model=object(),
        _tokenizer=object(),
        _get_lltokenizer=lambda: SimpleNamespace(vocab_size=10),
    )
    with pytest.raises(PrefillReachedError):
        decode(generator, "grammar", expected, max_tokens=1, temperature=0)


def test_custom_template_ids_cannot_replace_model_tokenization():
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    tokenizer = _SectionTokenizer()
    tokenizer.apply_chat_template = lambda *a, tokenize=True, **kw: (
        [999] if tokenize else "different text"
    )
    state = RequestPromptTokens()
    assert state.apply_template(tokenizer, _conversation()) == [999]
    engine = _engine(tokenizer, scheduler=False)
    engine._is_mllm = True
    prompt = _render(_conversation())
    # Custom MLLM tokenizers keep the processor's own tokenization of the text.
    assert engine._request_prompt_ids(prompt, state) == prompt


def test_custom_mllm_tokenizer_keeps_processor_call_contract():
    """Review P1: a tokenizer whose ``__call__`` differs from ``encode`` must
    reach the MLLM processor as text, so the processor's call decides the IDs."""
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    class CallDiffers(_SectionTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [[7]]}

        def encode(self, text, add_special_tokens=True):
            return [99, 7]

    engine = _engine(CallDiffers(), scheduler=False)
    engine._is_mllm = True
    prompt = "hello"
    assert (
        engine._request_prompt_ids(prompt, RequestPromptTokens(), mllm_step=True)
        == prompt
    )


def test_fast_mllm_ids_match_processor_call():
    """For the standard fast tokenizer the reused IDs equal what the MLLM
    processor's ``tokenizer(prompt, add_special_tokens=False, padding=True)``
    call produces, with and without a BOS post-processor."""
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    for style in ("chatml", "gemma", "llama"):
        for bos in (False, True):
            tokenizer = _fast_tokenizer(style, bos=bos)
            tokenizer.pad_token = "<unk>"  # Real MLLM processors define one.
            engine = _engine(tokenizer, scheduler=False)
            engine._is_mllm = True
            prompt = _render(_conversation())
            processor_ids = tokenizer(
                prompt, add_special_tokens=False, padding=True, padding_side="left"
            )["input_ids"]
            assert (
                engine._request_prompt_ids(
                    prompt, RequestPromptTokens(), mllm_step=True
                )
                == processor_ids
            )


def test_fast_tokenizer_with_custom_conversion_hook_is_not_reused():
    """Review round 2 P1: a fast-tokenizer subclass that rewrites the encoding
    must keep its own tokenization instead of reused backend IDs."""
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    base = _fast_tokenizer("chatml")

    class InjectsId(type(base)):
        def _convert_encoding(self, *args, **kwargs):
            encoding_dict, encodings = super()._convert_encoding(*args, **kwargs)
            encoding_dict["input_ids"] = [
                [2, *ids] for ids in encoding_dict["input_ids"]
            ]
            return encoding_dict, encodings

    tokenizer = _fast_tokenizer("chatml")
    tokenizer.__class__ = InjectsId
    assert RequestPromptTokens._is_fast_tokenizer(base)
    assert not RequestPromptTokens._is_fast_tokenizer(tokenizer)
    engine = _engine(tokenizer, scheduler=False)
    engine._is_mllm = True
    prompt = _render(_conversation())
    assert (
        engine._request_prompt_ids(prompt, RequestPromptTokens(), mllm_step=True)
        == prompt
    )


@pytest.mark.parametrize("bos", [False, True])
def test_mlx_lm_wrapper_is_unwrapped_for_reuse(bos):
    """The served text tokenizer is mlx-lm's ``TokenizerWrapper``; reuse must
    see through it (otherwise the text path keeps re-encoding) and produce
    exactly ``wrapper.encode`` for both special-token policies."""
    pytest.importorskip("mlx_lm")
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    inner = _fast_tokenizer("llama", bos=bos)
    wrapper = TokenizerWrapper(inner)
    assert RequestPromptTokens._is_fast_tokenizer(wrapper)
    prompt = _render(_conversation())
    state = RequestPromptTokens()
    assert state.encode(wrapper, prompt, add_special_tokens=False) == wrapper.encode(
        prompt, add_special_tokens=False
    )
    calls = []
    real = inner._tokenizer

    class Counting:
        def __getattr__(self, name):
            return getattr(real, name)

        def encode_batch(self, *args, **kwargs):
            calls.append(1)
            return real.encode_batch(*args, **kwargs)

        def encode_batch_fast(self, *args, **kwargs):
            calls.append(1)
            return real.encode_batch_fast(*args, **kwargs)

    inner._tokenizer = Counting()
    # The other policy is served from the same Rust encoding.
    assert state.encode(wrapper, prompt) == wrapper.encode(prompt)
    inner._tokenizer = real
    assert len(calls) == 1  # only the reference ``wrapper.encode`` call


def test_instance_level_hook_override_is_not_reused():
    """Review round 3 P1: a hook installed on the instance must also disable
    reuse, directly and through the mlx-lm wrapper."""
    from rapid_mlx.prompt_token_reuse import RequestPromptTokens

    tokenizer = _fast_tokenizer("chatml")
    original = tokenizer._convert_encoding

    def injects(*args, **kwargs):
        encoding_dict, encodings = original(*args, **kwargs)
        encoding_dict["input_ids"] = [[2, *ids] for ids in encoding_dict["input_ids"]]
        return encoding_dict, encodings

    tokenizer._convert_encoding = injects
    assert not RequestPromptTokens._is_fast_tokenizer(tokenizer)
    engine = _engine(tokenizer, scheduler=False)
    engine._is_mllm = True
    prompt = _render(_conversation())
    assert (
        engine._request_prompt_ids(prompt, RequestPromptTokens(), mllm_step=True)
        == prompt
    )
    mlx_lm = pytest.importorskip("mlx_lm.tokenizer_utils")
    assert not RequestPromptTokens._is_fast_tokenizer(
        mlx_lm.TokenizerWrapper(tokenizer)
    )
