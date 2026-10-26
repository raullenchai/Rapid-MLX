"""Token-proven controls are omitted before decoding, not by spelling cleanup."""

from types import SimpleNamespace

import pytest

from rapid_mlx.parsing.token_controls import control_token_kwargs
from rapid_mlx.utils.decode import IncrementalDecoder, visible_token_ids


class Tokenizer:
    all_special_ids = [256, 257, 258]
    specials = {256: b"<|im_start|>", 257: b"<think>", 258: b"<tool_call>"}

    def decode(self, ids, skip_special_tokens=False):
        return b"".join(
            self.specials.get(i, bytes([i]) if i < 256 else b"") for i in ids
        ).decode("utf-8", errors="replace")


def config(reasoning="qwen3"):
    return SimpleNamespace(
        reasoning_parser_name=reasoning,
        reasoning_parser=None,
        tool_call_parser="hermes",
    )


def test_selected_policy_retains_grammar_ids_and_native_families_are_unchanged():
    engine = SimpleNamespace(tokenizer=Tokenizer())
    request = SimpleNamespace(tools=[])
    assert control_token_kwargs(config(), engine, request) == {
        "output_control_token_ids": (256,)
    }
    assert control_token_kwargs(config("harmony"), engine, request) == {}
    assert (
        control_token_kwargs(config(), SimpleNamespace(tokenizer=None), request) == {}
    )
    only_grammar = Tokenizer()
    only_grammar.all_special_ids = [257, 258]
    assert (
        control_token_kwargs(config(), SimpleNamespace(tokenizer=only_grammar), request)
        == {}
    )


def test_concurrent_decoder_policies_preserve_unicode_and_literal_spellings():
    tokenizer = Tokenizer()
    selected = IncrementalDecoder(tokenizer, control_ids=(256,))
    native = IncrementalDecoder(tokenizer)
    ids = list("Hi 😀".encode())
    # Insert a structural control between incomplete UTF-8 bytes.
    ids.insert(5, 256)
    ids += list(b"<|im_start|>") + [257, 258]
    selected_deltas, native_deltas = [], []
    for token_id in ids:
        selected_deltas.append(selected.add_token(token_id))
        native_deltas.append(native.add_token(token_id))
    expected = "Hi 😀<|im_start|><think><tool_call>"
    assert "".join(selected_deltas) == selected.get_full_text() == expected
    assert "\ufffd" not in "".join(selected_deltas)
    assert native.token_ids == ids
    assert selected.token_ids == [i for i in ids if i != 256]
    params = SimpleNamespace(output_control_token_ids=(256,))
    assert tokenizer.decode(visible_token_ids(ids, params)) == expected
    assert visible_token_ids(ids, SimpleNamespace()) is ids
    selected.reset()
    assert selected.add_token(256) == "" and selected.get_full_text() == ""
    assert selected.add_token(ord("x")) == "x"


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_real_routes_forward_selected_control_policy(monkeypatch, surface, stream):
    from rapid_mlx.engine.base import GenerationOutput
    from tests.parser_migration.harness import Engine, route

    original_init = Engine.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.tokenizer = Tokenizer()

    monkeypatch.setattr(Engine, "__init__", init)
    ids = [256] + list(b"Answer <|im_start|>.")

    async def buffered(self, **kwargs):
        assert kwargs["output_control_token_ids"] == (256,)
        text = self.tokenizer.decode(visible_token_ids(ids, SimpleNamespace(**kwargs)))
        return GenerationOutput(
            text=text, raw_text=text, prompt_parse_context=self.parse_context
        )

    async def streaming(self, **kwargs):
        decoder = IncrementalDecoder(
            self.tokenizer, control_ids=kwargs["output_control_token_ids"]
        )
        for token_id in ids:
            text = decoder.add_token(token_id)
            yield GenerationOutput(
                text=decoder.get_full_text(),
                new_text=text,
                tokens=[token_id],
                finished=False,
                prompt_parse_context=self.parse_context,
            )
        yield GenerationOutput(
            text=decoder.get_full_text(),
            finished=True,
            prompt_parse_context=self.parse_context,
        )

    monkeypatch.setattr(Engine, "chat", buffered)
    monkeypatch.setattr(Engine, "stream_chat", streaming)
    result = route("qwen3", "hermes", "unused", surface, stream, thinking=False)
    assert result["content"] == "Answer <|im_start|>."
    assert not result["reasoning"] and not result["calls"]


@pytest.mark.parametrize("with_decoder", [False, True])
def test_text_scheduler_control_policy_keeps_generation_accounting(with_decoder):
    from unittest.mock import MagicMock

    from rapid_mlx.request import Request, RequestStatus, SamplingParams
    from tests.test_stop_string_enforcement import _make_scheduler

    scheduler = _make_scheduler()
    tokenizer = Tokenizer()
    scheduler._decode_tokens = tokenizer.decode
    params = SamplingParams(max_tokens=100, output_control_token_ids=(256,))
    request = Request(request_id="controls", prompt="ignored", sampling_params=params)
    request.status = RequestStatus.RUNNING
    if with_decoder:
        request._decoder = IncrementalDecoder(tokenizer, control_ids=(256,))
    scheduler.running[request.request_id] = request
    scheduler.uid_to_request_id[0] = request.request_id
    ids = [256] + list(b"ok <|im_start|>")
    chunks = []
    for index, token_id in enumerate(ids):
        response = MagicMock(uid=0, token=token_id, logprobs=None)
        response.finish_reason = "length" if index == len(ids) - 1 else None
        outputs, _ = scheduler._process_batch_responses([response])
        chunks.append(outputs[0].new_text)
    assert "".join(chunks) == outputs[0].output_text == "ok <|im_start|>"
    assert request.output_token_ids == ids
    assert request.num_output_tokens == len(ids)


def test_media_scheduler_control_policy_keeps_generation_accounting():
    from rapid_mlx.mllm_batch_generator import MLLMBatchResponse
    from tests.test_mllm_usage_tracking import _make_mllm_request, _make_scheduler

    scheduler = _make_scheduler()
    request = _make_mllm_request(scheduler, "controls-media")
    request.sampling_params.output_control_token_ids = (256,)
    scheduler.uid_to_request_id[0] = request.request_id

    class Detokenizer:
        def __init__(self):
            self.decoder = IncrementalDecoder(Tokenizer())
            self.last_segment = self.text = ""

        def add_token(self, token_id):
            self.last_segment = self.decoder.add_token(token_id)
            self.text = self.decoder.get_full_text()

        def finalize(self):
            self.text = self.decoder.get_full_text()

    scheduler._detokenizer_pool[request.request_id] = Detokenizer()
    ids = [256] + list(b"ok <|im_start|>")
    chunks = []
    for index, token_id in enumerate(ids):
        response = MLLMBatchResponse(
            uid=0,
            request_id=request.request_id,
            token=token_id,
            logprobs=None,
            finish_reason="length" if index == len(ids) - 1 else None,
        )
        outputs, _ = scheduler._process_batch_responses([response])
        chunks.append(outputs[0].new_text)
    assert "".join(chunks) == outputs[0].output_text == "ok <|im_start|>"
    assert request.output_tokens == ids
    assert request.num_output_tokens == len(ids)


def test_terminal_media_control_flushes_held_text_into_stream():
    from rapid_mlx.mllm_batch_generator import MLLMBatchResponse
    from tests.test_mllm_usage_tracking import _make_mllm_request, _make_scheduler

    scheduler = _make_scheduler()
    request = _make_mllm_request(scheduler, "held-control")
    request.sampling_params.output_control_token_ids = (256,)
    scheduler.uid_to_request_id[0] = request.request_id

    class Held:
        text = ""

        def finalize(self):
            self.text = "held"

    scheduler._detokenizer_pool[request.request_id] = Held()
    response = MLLMBatchResponse(
        uid=0,
        request_id=request.request_id,
        token=256,
        logprobs=None,
        finish_reason="length",
    )
    outputs, _ = scheduler._process_batch_responses([response])
    assert outputs[0].new_text == outputs[0].output_text == "held"
    assert request.output_tokens == [256]
