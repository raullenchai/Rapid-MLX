from types import SimpleNamespace
from unittest.mock import MagicMock

from rapid_mlx.request import Request, SamplingParams
from rapid_mlx.scheduler import Scheduler


def test_terminal_output_surfaces_prompt_compression_without_mlx_runtime():
    """The hosted headless lane covers the terminal metadata assignment."""
    tokenizer = SimpleNamespace(
        eos_token_id=2,
        encode=lambda _text: [1, 2],
        decode=lambda _tokens: "",
    )
    scheduler = Scheduler(SimpleNamespace(), tokenizer)
    request = Request(
        request_id="compressed",
        prompt="",
        prompt_token_ids=list(range(128)),
        sampling_params=SamplingParams(max_tokens=1),
    )
    request.pflash_metadata = {
        "compressed": True,
        "original_tokens": 128,
        "kept_tokens": 32,
    }

    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.remove.return_value = {}
    scheduler.running[request.request_id] = request
    scheduler.uid_to_request_id[0] = request.request_id
    scheduler._decode_tokens = lambda _tokens: ""  # type: ignore[method-assign]
    response = MagicMock()
    response.uid = 0
    response.token = 42
    response.finish_reason = "stop"
    response.logprobs = None
    del response.prompt_cache

    outputs, finished = scheduler._process_batch_responses([response])

    assert finished == {request.request_id}
    assert outputs[0].prompt_compression == {
        "original_tokens": 128,
        "kept_tokens": 32,
    }
