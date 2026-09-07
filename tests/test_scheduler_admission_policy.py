# SPDX-License-Identifier: Apache-2.0
"""Scope-locked tests for contention-aware prompt-slot admission."""

import sys
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from vllm_mlx import server
from vllm_mlx.cli import build_parser
from vllm_mlx.request import Request, RequestStatus, SamplingParams
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

    selected, forced = scheduler._select_waiting_request()
    assert selected is short_first
    assert not forced
    assert list(scheduler.waiting) == [long, short_first, short_second]
    scheduler._commit_waiting_selection(selected, forced)
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

    selected, forced = scheduler._select_waiting_request()
    assert selected is compatible
    scheduler._commit_waiting_selection(selected, forced)
    assert list(scheduler.waiting) == [incompatible]
    assert incompatible._admission_deferrals == 0


def test_max_deferrals_forces_oldest_compatible_request():
    old_long = _request("old-long", 4096)
    old_long._admission_deferrals = 2
    new_short = _request("new-short", 1)
    scheduler = _selector(old_long, new_short, max_deferrals=2)

    selected, forced = scheduler._select_waiting_request()
    assert selected is old_long
    assert forced
    scheduler._commit_waiting_selection(selected, forced)
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


def test_cost_probe_rejects_stale_tail_and_accepts_current_prompt_suffix():
    request = _request("cache-hit", 100)
    request.prompt_cache = object()
    request.cached_tokens = 98
    request.remaining_tokens = [98, 99]
    scheduler = _selector(request)

    assert scheduler._validated_prompt_tail_cost(request) == 2
    request.remaining_tokens = [7, 8]
    assert scheduler._validated_prompt_tail_cost(request) == 100
    request.remaining_tokens = [98, 99]
    request.prompt_cache = SimpleNamespace(offset=97)
    assert scheduler._validated_prompt_tail_cost(request) == 100
    request.prompt_cache = SimpleNamespace(offset=-1)
    assert scheduler._validated_prompt_tail_cost(request) == 100
    request.prompt_cache = SimpleNamespace(offset="corrupt")
    assert scheduler._validated_prompt_tail_cost(request) == 100
    request.prompt_cache = SimpleNamespace(offset=True)
    assert scheduler._validated_prompt_tail_cost(request) == 100
    request.prompt_cache = SimpleNamespace(offset=98.9)
    assert scheduler._validated_prompt_tail_cost(request) == 100


def test_cost_probe_observes_nested_cache_offsets_without_mutation():
    cache = SimpleNamespace(
        caches=[SimpleNamespace(offset=2), [SimpleNamespace(offset=2)]]
    )

    assert Scheduler._observable_prompt_cache_offsets(cache) == ((2, 2), True)


def test_exact_cache_match_uses_trim_probe_and_safely_falls_back(monkeypatch):
    request = _request("exact-cache", 0)
    request.prompt_token_ids = [0, 1, 2]
    request.cached_tokens = 3
    request.prompt_cache = SimpleNamespace(offset=3)
    scheduler = _selector(request)
    scheduler._validate_cache = lambda _cache: True

    import mlx_lm.models.cache as cache_module

    monkeypatch.setattr(cache_module, "can_trim_prompt_cache", lambda _cache: True)
    assert scheduler._validated_prompt_tail_cost(request) == 1

    monkeypatch.setattr(cache_module, "can_trim_prompt_cache", lambda _cache: False)
    assert scheduler._validated_prompt_tail_cost(request) == 3

    def fail_trim(_cache):
        raise RuntimeError("unsupported cache")

    monkeypatch.setattr(cache_module, "can_trim_prompt_cache", fail_trim)
    assert scheduler._validated_prompt_tail_cost(request) == 3


def test_selection_is_read_only_until_admission_commit():
    old = _request("old", 100)
    old._admission_deferrals = 3
    short = _request("short", 1)
    scheduler = _selector(old, short)

    selected, forced = scheduler._select_waiting_request()

    assert selected is short and not forced
    assert list(scheduler.waiting) == [old, short]
    assert old._admission_deferrals == 3


def test_selection_returns_none_when_live_generator_is_incompatible():
    request = _request("incompatible", 1, stop_ids=(9,))
    scheduler = _selector(request)
    scheduler.batch_generator = object()
    scheduler.running = {"live": object()}
    scheduler._current_sampler_params = (frozenset({7}), False)

    assert scheduler._select_waiting_request() is None
    assert scheduler._schedule_waiting() == []
    assert list(scheduler.waiting) == [request]


@pytest.mark.parametrize("generator_ready", [False, True])
@pytest.mark.parametrize("policy", ["shortest_validated_tail", "fcfs"])
def test_admission_failure_preserves_queue_and_deferrals(generator_ready, policy):
    request = _request("retry", 1)
    request._admission_deferrals = 2
    scheduler = _selector(request)
    scheduler.config.scheduling_policy = policy
    scheduler._ensure_batch_generator = lambda _params: generator_ready
    if generator_ready:
        scheduler.batch_generator = None

    assert scheduler._schedule_waiting() == []
    assert list(scheduler.waiting) == [request]
    assert request._admission_deferrals == 2


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

    scheduler.running = {"a": object()}
    scheduler._admission_prefill_uids.clear()
    scheduler.config.prefill_batch_size = 4
    scheduler.config.completion_batch_size = 2
    assert scheduler._shortest_tail_admission_capacity() == 1


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


def test_empty_uid_insert_preserves_opt_in_request_without_spinning():
    scheduler = Scheduler(
        model=object(),
        tokenizer=SimpleNamespace(
            encode=lambda value: value,
            decode=lambda value: str(value),
            eos_token_id=0,
            eos_token_ids={0},
        ),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            scheduling_policy="shortest_validated_tail",
        ),
    )
    scheduler.batch_generator = SimpleNamespace(insert=lambda *args, **kwargs: [])
    scheduler._current_sampler_params = (frozenset(), False)
    request = _request("retry-empty-uid", 1)
    scheduler.requests[request.request_id] = request
    scheduler.waiting.append(request)

    assert scheduler._schedule_waiting() == []
    assert list(scheduler.waiting) == [request]
    assert request._admission_deferrals == 0


def test_legacy_flat_response_runtime_falls_back_to_fcfs_without_sticking_slots():
    scheduler = Scheduler(
        MagicMock(),
        SimpleNamespace(
            encode=lambda value: value,
            decode=lambda value: str(value),
            eos_token_id=0,
            eos_token_ids={0},
        ),
        SchedulerConfig(
            enable_prefix_cache=False,
            scheduling_policy="shortest_validated_tail",
        ),
    )
    request = _request("live", 1)
    request.status = RequestStatus.RUNNING
    scheduler.requests = {request.request_id: request}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = SimpleNamespace(
        next=lambda: [],
        _generation_batch=None,
        _prompt_batch=None,
        _currently_processing=(),
        _unprocessed_sequences=(),
    )
    scheduler._admission_prefill_uids = {11}

    scheduler.step()

    assert scheduler._shortest_tail_runtime_supported is False
    assert not scheduler._admission_prefill_uids
    stats = scheduler.get_stats()
    assert stats["configured_scheduling_policy"] == "shortest_validated_tail"
    assert stats["scheduling_policy"] == "fcfs"


def test_tuple_response_records_prompt_promotion_and_runtime_support():
    scheduler = Scheduler(
        MagicMock(),
        SimpleNamespace(
            encode=lambda value: value,
            decode=lambda value: str(value),
            eos_token_id=0,
            eos_token_ids={0},
        ),
        SchedulerConfig(
            enable_prefix_cache=False,
            scheduling_policy="shortest_validated_tail",
        ),
    )
    request = _request("live", 1)
    request.status = RequestStatus.RUNNING
    scheduler.requests = {request.request_id: request}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = SimpleNamespace(
        next=lambda: ([SimpleNamespace(uid=11, end_of_prompt=True)], []),
        _generation_batch=None,
        _prompt_batch=None,
        _currently_processing=(),
        _unprocessed_sequences=(),
    )
    scheduler._admission_prefill_uids = {11}

    scheduler.step()

    assert scheduler._shortest_tail_runtime_supported is True
    assert not scheduler._admission_prefill_uids


def test_admission_mirror_is_cleared_on_generator_close_and_uid_forget():
    scheduler = Scheduler(
        MagicMock(),
        SimpleNamespace(eos_token_id=0, eos_token_ids={0}),
        SchedulerConfig(enable_prefix_cache=False),
    )
    scheduler._admission_prefill_uids = {7, 8}
    scheduler._close_batch_generator()
    assert not scheduler._admission_prefill_uids

    scheduler._admission_prefill_uids = {7}
    scheduler._forget_uid_grammar(7)
    assert not scheduler._admission_prefill_uids


def test_runtime_fallback_is_idempotent_and_ignored_for_fcfs():
    scheduler = _selector()
    scheduler._shortest_tail_runtime_supported = False
    scheduler._fallback_from_unobservable_prompt_runtime()
    assert scheduler._shortest_tail_runtime_supported is False

    scheduler.config.scheduling_policy = "fcfs"
    scheduler._shortest_tail_runtime_supported = None
    scheduler._fallback_from_unobservable_prompt_runtime()
    assert scheduler._shortest_tail_runtime_supported is None


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


def test_standalone_server_parser_registers_admission_flags(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm_mlx.server", "--scheduling-policy", "invalid"],
    )

    with pytest.raises(SystemExit) as exc:
        server.main()

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
