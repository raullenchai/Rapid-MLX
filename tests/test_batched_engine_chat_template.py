# SPDX-License-Identifier: Apache-2.0
"""Regression tests for BatchedEngine chat-template applicator selection.

Some MLLM processors (notably ``Gemma3nProcessor`` and ``Gemma3Processor`` as
loaded by ``mlx_vlm.load``) expose ``apply_chat_template`` as a method but
ship ``chat_template=None`` at the processor layer — only the inner tokenizer
carries the Jinja template. ``_apply_chat_template`` must detect this and
fall back to the tokenizer; otherwise every request to those models returns
zero tokens.
"""

from vllm_mlx.engine.batched import BatchedEngine


class _RecordingApplicator:
    """Tokenizer/processor stub that records which applicator handled the call."""

    def __init__(self, label: str, chat_template: object):
        self._label = label
        self.chat_template = chat_template
        self.last_messages = None
        self.last_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return f"<{self._label}>"


class _ProcessorStub:
    """MLLM processor stub.

    Mirrors the real ``Gemma3Processor``/``Gemma3nProcessor`` shape: exposes
    ``apply_chat_template`` and ``chat_template`` at the processor layer, plus
    an inner ``tokenizer`` attribute that the engine's ``tokenizer`` property
    unwraps to.
    """

    def __init__(self, label: str, chat_template: object, inner_tokenizer):
        self._applicator = _RecordingApplicator(label, chat_template)
        self.chat_template = chat_template
        self.tokenizer = inner_tokenizer

    def apply_chat_template(self, messages, **kwargs):
        return self._applicator.apply_chat_template(messages, **kwargs)

    @property
    def last_messages(self):
        return self._applicator.last_messages


def _make_engine(
    *,
    is_mllm: bool,
    processor: _ProcessorStub | None,
    tokenizer: _RecordingApplicator,
) -> BatchedEngine:
    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._is_mllm = is_mllm
    engine._processor = processor
    engine._tokenizer = tokenizer
    return engine


def test_falls_back_to_tokenizer_when_mllm_processor_chat_template_is_none():
    """Gemma-family processors expose apply_chat_template but chat_template=None."""
    tokenizer = _RecordingApplicator(label="TOK", chat_template="{{ messages }}")
    processor = _ProcessorStub(
        label="PROC", chat_template=None, inner_tokenizer=tokenizer
    )
    engine = _make_engine(is_mllm=True, processor=processor, tokenizer=tokenizer)

    result = engine._apply_chat_template([{"role": "user", "content": "hi"}])

    assert result == "<TOK>"
    assert processor.last_messages is None  # processor was never invoked


def test_falls_back_to_tokenizer_when_mllm_processor_chat_template_is_empty_string():
    """Empty string chat_template is just as broken as None — same fallback path."""
    tokenizer = _RecordingApplicator(label="TOK", chat_template="{{ messages }}")
    processor = _ProcessorStub(
        label="PROC", chat_template="", inner_tokenizer=tokenizer
    )
    engine = _make_engine(is_mllm=True, processor=processor, tokenizer=tokenizer)

    result = engine._apply_chat_template([{"role": "user", "content": "hi"}])

    assert result == "<TOK>"
    assert processor.last_messages is None


def test_uses_processor_when_chat_template_is_present():
    """When the processor has a real template, MLLM path keeps using it."""
    tokenizer = _RecordingApplicator(label="TOK", chat_template="{{ messages }}")
    processor = _ProcessorStub(
        label="PROC", chat_template="{{ messages }}", inner_tokenizer=tokenizer
    )
    engine = _make_engine(is_mllm=True, processor=processor, tokenizer=tokenizer)

    result = engine._apply_chat_template([{"role": "user", "content": "hi"}])

    assert result == "<PROC>"
    assert tokenizer.last_messages is None  # tokenizer was never invoked


def test_text_only_engine_always_uses_tokenizer():
    """Non-MLLM engines bypass the processor branch regardless of processor state."""
    tokenizer = _RecordingApplicator(label="TOK", chat_template="{{ messages }}")
    processor = _ProcessorStub(
        label="PROC", chat_template="{{ messages }}", inner_tokenizer=tokenizer
    )
    engine = _make_engine(is_mllm=False, processor=processor, tokenizer=tokenizer)

    result = engine._apply_chat_template([{"role": "user", "content": "hi"}])

    assert result == "<TOK>"
    assert processor.last_messages is None


def test_mllm_without_processor_uses_tokenizer():
    """MLLM engine that hasn't populated its processor still works via tokenizer."""
    tokenizer = _RecordingApplicator(label="TOK", chat_template="{{ messages }}")
    engine = _make_engine(is_mllm=True, processor=None, tokenizer=tokenizer)

    result = engine._apply_chat_template([{"role": "user", "content": "hi"}])

    assert result == "<TOK>"
