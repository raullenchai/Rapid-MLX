# SPDX-License-Identifier: Apache-2.0
"""
Tests for request management classes.

Tests RequestStatus, SamplingParams, Request, and RequestOutput
from vllm_mlx/request.py. No MLX dependency.
"""

import time

from vllm_mlx.request import Request, RequestOutput, RequestStatus, SamplingParams


class TestRequestStatus:
    """Tests for RequestStatus enum."""

    def test_status_values_exist(self):
        assert RequestStatus.WAITING is not None
        assert RequestStatus.RUNNING is not None
        assert RequestStatus.PREEMPTED is not None
        assert RequestStatus.FINISHED_STOPPED is not None
        assert RequestStatus.FINISHED_LENGTH_CAPPED is not None
        assert RequestStatus.FINISHED_ABORTED is not None

    def test_is_finished_waiting(self):
        assert RequestStatus.is_finished(RequestStatus.WAITING) is False

    def test_is_finished_running(self):
        assert RequestStatus.is_finished(RequestStatus.RUNNING) is False

    def test_is_finished_preempted(self):
        assert RequestStatus.is_finished(RequestStatus.PREEMPTED) is False

    def test_is_finished_stopped(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED) is True

    def test_is_finished_length_capped(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED) is True

    def test_is_finished_aborted(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED) is True

    def test_get_finish_reason_stopped(self):
        assert RequestStatus.get_finish_reason(RequestStatus.FINISHED_STOPPED) == "stop"

    def test_get_finish_reason_length_capped(self):
        assert (
            RequestStatus.get_finish_reason(RequestStatus.FINISHED_LENGTH_CAPPED)
            == "length"
        )

    def test_get_finish_reason_aborted(self):
        assert (
            RequestStatus.get_finish_reason(RequestStatus.FINISHED_ABORTED) == "abort"
        )

    def test_get_finish_reason_cancelled(self):
        assert (
            RequestStatus.get_finish_reason(RequestStatus.FINISHED_CANCELLED)
            == "cancelled"
        )

    def test_get_finish_reason_waiting(self):
        assert RequestStatus.get_finish_reason(RequestStatus.WAITING) is None

    def test_get_finish_reason_running(self):
        assert RequestStatus.get_finish_reason(RequestStatus.RUNNING) is None

    def test_get_finish_reason_preempted(self):
        assert RequestStatus.get_finish_reason(RequestStatus.PREEMPTED) is None

    def test_ordering(self):
        assert RequestStatus.WAITING < RequestStatus.RUNNING
        assert RequestStatus.PREEMPTED < RequestStatus.FINISHED_STOPPED


class TestSamplingParams:
    """Tests for SamplingParams dataclass."""

    def test_defaults(self):
        params = SamplingParams()
        assert params.max_tokens == 256
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 0
        assert params.min_p == 0.0
        assert params.repetition_penalty == 1.0
        assert params.stop == []
        assert params.stop_token_ids == []

    def test_custom_values(self):
        params = SamplingParams(
            max_tokens=100,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            min_p=0.05,
            repetition_penalty=1.2,
            stop=["END", "STOP"],
            stop_token_ids=[1, 2],
        )
        assert params.max_tokens == 100
        assert params.temperature == 0.5
        assert params.stop == ["END", "STOP"]
        assert params.stop_token_ids == [1, 2]

    def test_none_stop_becomes_empty_list(self):
        params = SamplingParams(stop=None, stop_token_ids=None)
        assert params.stop == []
        assert params.stop_token_ids == []


class TestRequest:
    """Tests for Request dataclass."""

    def test_basic_creation(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req.request_id == "test-1"
        assert req.prompt == "Hello"
        assert req.status == RequestStatus.WAITING
        assert req.num_computed_tokens == 0
        assert req.output_token_ids == []
        assert req.output_text == ""
        assert req._mtp_safe_logits_processors == ()

    def test_arrival_time_set(self):
        before = time.time()
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        after = time.time()
        assert before <= req.arrival_time <= after

    def test_num_output_tokens(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            output_token_ids=[1, 2, 3],
        )
        assert req.num_output_tokens == 3

    def test_num_output_tokens_empty(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req.num_output_tokens == 0

    def test_num_tokens(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            num_prompt_tokens=10,
            output_token_ids=[1, 2, 3],
        )
        assert req.num_tokens == 13

    def test_max_tokens(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=50),
        )
        assert req.max_tokens == 50

    def test_is_finished_false(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req.is_finished() is False

    def test_is_finished_true(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            status=RequestStatus.FINISHED_STOPPED,
        )
        assert req.is_finished() is True

    def test_get_finish_reason_from_status(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            status=RequestStatus.FINISHED_STOPPED,
        )
        assert req.get_finish_reason() == "stop"

    def test_get_finish_reason_custom(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            finish_reason="tool_calls",
        )
        assert req.get_finish_reason() == "tool_calls"

    def test_get_finish_reason_none(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req.get_finish_reason() is None

    def test_append_output_token(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req.append_output_token(42)
        assert req.output_token_ids == [42]
        assert req.num_computed_tokens == 1
        assert req.num_output_tokens == 1

    def test_append_multiple_tokens(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        for tok in [10, 20, 30]:
            req.append_output_token(tok)
        assert req.output_token_ids == [10, 20, 30]
        assert req.num_computed_tokens == 3

    def test_set_finished(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req.set_finished(RequestStatus.FINISHED_STOPPED)
        assert req.status == RequestStatus.FINISHED_STOPPED
        assert req.finish_reason == "stop"
        assert req.is_finished() is True

    def test_set_finished_with_custom_reason(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req.set_finished(RequestStatus.FINISHED_STOPPED, reason="tool_calls")
        assert req.finish_reason == "tool_calls"

    def test_set_finished_length_capped(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
        assert req.finish_reason == "length"

    def test_priority_ordering(self):
        req_low = Request(
            request_id="low",
            prompt="Hello",
            sampling_params=SamplingParams(),
            priority=10,
        )
        req_high = Request(
            request_id="high",
            prompt="Hello",
            sampling_params=SamplingParams(),
            priority=1,
        )
        assert req_high < req_low

    def test_arrival_time_ordering(self):
        req_first = Request(
            request_id="first",
            prompt="Hello",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=1000.0,
        )
        req_second = Request(
            request_id="second",
            prompt="Hello",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=2000.0,
        )
        assert req_first < req_second

    def test_hash(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert hash(req) == hash("test-1")

    def test_equality(self):
        req1 = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req2 = Request(
            request_id="test-1",
            prompt="Different prompt",
            sampling_params=SamplingParams(max_tokens=999),
        )
        assert req1 == req2

    def test_inequality(self):
        req1 = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req2 = Request(
            request_id="test-2",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req1 != req2

    def test_inequality_with_non_request(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req != "not a request"

    def test_token_list_prompt(self):
        req = Request(
            request_id="test-1",
            prompt=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(),
        )
        assert req.prompt == [1, 2, 3, 4, 5]

    def test_multimodal_fields(self):
        req = Request(
            request_id="test-1",
            prompt="Describe this image",
            sampling_params=SamplingParams(),
            images=["img1.png", "img2.png"],
            videos=["vid.mp4"],
            is_multimodal=True,
        )
        assert req.is_multimodal is True
        assert len(req.images) == 2
        assert len(req.videos) == 1

    def test_prefix_cache_fields(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            cached_tokens=50,
            remaining_tokens=[51, 52, 53],
        )
        assert req.cached_tokens == 50
        assert req.remaining_tokens == [51, 52, 53]

    def test_default_optional_fields(self):
        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        assert req.prompt_token_ids is None
        assert req.prompt_cache is None
        assert req.block_table is None
        assert req.images is None
        assert req.videos is None
        assert req.pixel_values is None
        assert req.is_multimodal is False
        assert req.batch_uid is None

    def test_in_set(self):
        req1 = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        req2 = Request(
            request_id="test-2",
            prompt="World",
            sampling_params=SamplingParams(),
        )
        req_set = {req1, req2}
        assert len(req_set) == 2
        assert req1 in req_set

    def test_sortable(self):
        reqs = [
            Request(
                request_id="low",
                prompt="a",
                sampling_params=SamplingParams(),
                priority=5,
                arrival_time=1.0,
            ),
            Request(
                request_id="high",
                prompt="b",
                sampling_params=SamplingParams(),
                priority=1,
                arrival_time=2.0,
            ),
            Request(
                request_id="med",
                prompt="c",
                sampling_params=SamplingParams(),
                priority=3,
                arrival_time=0.5,
            ),
        ]
        sorted_reqs = sorted(reqs)
        assert sorted_reqs[0].request_id == "high"
        assert sorted_reqs[1].request_id == "med"
        assert sorted_reqs[2].request_id == "low"


class TestRequestOutput:
    """Tests for RequestOutput dataclass."""

    def test_basic_creation(self):
        output = RequestOutput(request_id="test-1")
        assert output.request_id == "test-1"
        assert output.new_token_ids == []
        assert output.new_text == ""
        assert output.finished is False
        assert output.finish_reason is None

    def test_with_tokens(self):
        output = RequestOutput(
            request_id="test-1",
            new_token_ids=[42, 43],
            new_text="Hello",
            output_token_ids=[42, 43],
            output_text="Hello",
        )
        assert output.new_token_ids == [42, 43]
        assert output.new_text == "Hello"

    def test_finished_output(self):
        output = RequestOutput(
            request_id="test-1",
            finished=True,
            finish_reason="stop",
        )
        assert output.finished is True
        assert output.finish_reason == "stop"

    def test_usage_property(self):
        output = RequestOutput(
            request_id="test-1",
            prompt_tokens=10,
            completion_tokens=20,
        )
        usage = output.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_usage_defaults(self):
        output = RequestOutput(request_id="test-1")
        usage = output.usage
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
