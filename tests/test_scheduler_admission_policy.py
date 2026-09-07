# SPDX-License-Identifier: Apache-2.0
"""Scope-locked tests for contention-aware prompt-slot admission."""

from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from vllm_mlx.cli import build_parser
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _request(request_id: str, tail: int, *, stop_ids=()) -> Request:
    request = Request(
        request_id=request_id,
        prompt=request_id,
        sampling_params=SamplingParams(stop_token_ids=list(stop_ids)),
    )
    request.prompt_token_ids = list(range(max(tail, 1)))
    request.remaining_tokens = list(range(tail))
    return request


def _selector(*requests: Request, max_deferrals: int = 8) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(
        scheduling_policy="shortest_validated_tail",
        scheduling_max_deferrals=max_deferrals,
    )
    scheduler.waiting = deque(requests)
    scheduler.running = {}
    scheduler.batch_generator = None
    scheduler._current_sampler_params = None
    scheduler._admission_prefill_uids = set()
    scheduler.num_admission_deferrals = 0
    scheduler.num_admission_forced_grants = 0
    return scheduler


def test_default_policy_preserves_fcfs_and_validates_opt_in():
    assert SchedulerConfig().scheduling_policy == "fcfs"
    assert SchedulerConfig().scheduling_max_deferrals == 8
    with pytest.raises(ValueError, match="scheduling_policy"):
        SchedulerConfig(scheduling_policy="shortest_prompt")
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="scheduling_max_deferrals"):
            SchedulerConfig(scheduling_max_deferrals=invalid)


def test_shortest_validated_tail_wins_and_ties_remain_fifo():
    long = _request("long", 4096)
    short_first = _request("short-first", 8)
    short_second = _request("short-second", 8)
    scheduler = _selector(long, short_first, short_second)

    assert scheduler._select_waiting_request() is short_first
    assert list(scheduler.waiting) == [long, short_second]
    assert long._admission_deferrals == 1
    assert short_second._admission_deferrals == 1


def test_incompatible_head_does_not_block_or_accrue_deferrals():
    incompatible = _request("incompatible", 1, stop_ids=(9,))
    compatible = _request("compatible", 20, stop_ids=(7,))
    scheduler = _selector(incompatible, compatible)
    scheduler.batch_generator = object()
    scheduler.running = {"live": object()}
    scheduler._current_sampler_params = (frozenset({7}), False)

    assert scheduler._select_waiting_request() is compatible
    assert list(scheduler.waiting) == [incompatible]
    assert not hasattr(incompatible, "_admission_deferrals")


def test_max_deferrals_forces_oldest_compatible_request():
    old_long = _request("old-long", 4096)
    old_long._admission_deferrals = 2
    new_short = _request("new-short", 1)
    scheduler = _selector(old_long, new_short, max_deferrals=2)

    assert scheduler._select_waiting_request() is old_long
    assert scheduler.num_admission_forced_grants == 1
    assert list(scheduler.waiting) == [new_short]


def test_cost_probe_does_not_mutate_invalid_cache_fallback():
    request = _request("invalid-cache", 0)
    request.prompt_token_ids = list(range(12))
    request.prompt_cache = []
    scheduler = _selector(request)

    assert scheduler._validated_prompt_tail_cost(request) == 12
    assert request.prompt_cache == []
    assert request.remaining_tokens == []


def test_policy_grants_only_real_prompt_and_completion_headroom():
    scheduler = _selector()
    scheduler.config.prefill_batch_size = 2
    scheduler.config.completion_batch_size = 4
    scheduler.config.max_num_seqs = 9
    scheduler.running = {"a": object(), "b": object(), "c": object()}
    scheduler._admission_prefill_uids = {11}

    assert scheduler._shortest_tail_admission_capacity() == 1
    scheduler._admission_prefill_uids.add(12)
    assert scheduler._shortest_tail_admission_capacity() == 0


def test_prompt_promotion_reopens_exactly_one_slot():
    scheduler = _selector()
    scheduler._admission_prefill_uids = {11, 12}
    scheduler._record_prompt_promotions(
        [
            SimpleNamespace(uid=11, end_of_prompt=True),
            SimpleNamespace(uid=12, end_of_prompt=False),
        ]
    )
    assert scheduler._admission_prefill_uids == {12}


def test_opt_in_keeps_excess_prompts_out_of_generator_but_fcfs_is_unchanged():
    def run(policy: str):
        scheduler = Scheduler(
            model=object(),
            tokenizer=SimpleNamespace(
                encode=lambda value: value,
                decode=lambda value: str(value),
                eos_token_id=0,
                eos_token_ids={0},
            ),
            config=SchedulerConfig(
                max_num_seqs=3,
                prefill_batch_size=1,
                completion_batch_size=3,
                enable_prefix_cache=False,
                scheduling_policy=policy,
            ),
        )
        scheduler.batch_generator = SimpleNamespace(
            insert=lambda *args, **kwargs: [100 + len(scheduler.running)]
        )
        scheduler._current_sampler_params = (frozenset(), False)
        for request in (
            _request("long", 100),
            _request("short", 2),
            _request("medium", 20),
        ):
            scheduler.requests[request.request_id] = request
            scheduler.waiting.append(request)
        return scheduler, scheduler._schedule_waiting()

    shortest, scheduled = run("shortest_validated_tail")
    assert [request.request_id for request in scheduled] == ["short"]
    assert [request.request_id for request in shortest.waiting] == ["long", "medium"]

    fcfs, scheduled = run("fcfs")
    assert [request.request_id for request in scheduled] == ["long", "short", "medium"]
    assert not fcfs.waiting


def test_serve_cli_exposes_policy_and_starvation_bound():
    args = build_parser().parse_args(
        [
            "serve",
            "model",
            "--scheduling-policy",
            "shortest_validated_tail",
            "--scheduling-max-deferrals",
            "3",
        ]
    )
    assert args.scheduling_policy == "shortest_validated_tail"
    assert args.scheduling_max_deferrals == 3
