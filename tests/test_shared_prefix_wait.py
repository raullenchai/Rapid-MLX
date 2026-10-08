"""A request waits for a running prefill of the same prompt prefix."""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from collections import deque
from types import SimpleNamespace

from rapid_mlx.request import Request, SamplingParams
from rapid_mlx.scheduler import Scheduler, SchedulerConfig, _common_prefix_len

DOC = list(range(1000, 9000))


def _request(name, tokens, *, emitted=0):
    request = Request(
        request_id=name, prompt="", sampling_params=SamplingParams(max_tokens=8)
    )
    request.prompt_token_ids = list(tokens)
    request.remaining_tokens = list(tokens)
    request.output_token_ids = [0] * emitted
    return request


def _scheduler(*running, waiting=(), wait_tokens=1024):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(shared_prefix_wait_tokens=wait_tokens)
    scheduler.running = {r.request_id: r for r in running}
    scheduler.waiting = deque(waiting)
    scheduler._shared_prefix_waits = 0
    scheduler._reclaim_prefix_cache_for_prefill = lambda request: 0
    scheduler.stored = {}

    def fetch(tokens):
        for key, cache in scheduler.stored.items():
            if list(tokens[: len(key)]) == list(key):
                scheduler.memory_aware_cache._last_match_type = "prefix"
                return cache, list(tokens[len(key) :])
        scheduler.memory_aware_cache._last_match_type = "miss"
        return None, list(tokens)

    scheduler.memory_aware_cache = SimpleNamespace(
        fetch=fetch, _last_match_type="miss", _entries={}
    )
    return scheduler


def test_common_prefix_len():
    assert _common_prefix_len([], [1]) == 0
    assert _common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert _common_prefix_len(DOC, DOC + [7]) == len(DOC)
    assert _common_prefix_len([5] + DOC, DOC) == 0


def test_request_waits_for_a_prefill_of_the_same_document():
    leader = _request("leader", DOC + [1, 2])
    follower = _request("follower", DOC + [3, 4, 5])
    other = _request("other", list(range(50)))
    scheduler = _scheduler(leader, waiting=[follower, other])

    # The held request does not block an unrelated one behind it.
    assert scheduler._pop_waiting_for_admission() is other
    assert scheduler._pop_waiting_for_admission() is None
    assert list(scheduler.waiting) == [follower]
    assert scheduler._shared_prefix_waits == 1

    # The leader stored its prompt state and produced its first token.
    scheduler.stored[tuple(DOC)] = ["state"]
    leader.output_token_ids = [0]
    assert scheduler._pop_waiting_for_admission() is follower
    assert follower.prompt_cache == ["state"]
    assert follower.cached_tokens == len(DOC)
    assert follower.remaining_tokens == [3, 4, 5]
    assert scheduler._shared_prefix_waits == 1


def test_request_is_released_when_the_leader_leaves_without_storing():
    leader = _request("leader", DOC + [1])
    follower = _request("follower", DOC + [2])
    scheduler = _scheduler(leader, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is None

    del scheduler.running["leader"]  # aborted mid-prefill
    assert scheduler._pop_waiting_for_admission() is follower
    assert follower.prompt_cache is None
    assert follower.cached_tokens == 0
    assert follower.remaining_tokens == DOC + [2]


def test_request_follows_only_the_request_it_was_held_for():
    first = _request("first", DOC + [1])
    second = _request("second", DOC + [3])
    follower = _request("follower", DOC + [2])
    scheduler = _scheduler(first, second, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is None
    assert follower.prefix_wait_leader == "first"

    # Its leader is done; another prefill of the document is still running.
    first.output_token_ids = [0]
    assert scheduler._pop_waiting_for_admission() is follower
    assert follower.prefix_wait_leader is None


def test_request_waits_only_once():
    first = _request("first", DOC + [1])
    follower = _request("follower", DOC + [2])
    scheduler = _scheduler(first, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is None
    del scheduler.running["first"]
    assert scheduler._pop_waiting_for_admission() is follower

    # Requeued (generator not ready) beside another prefill of the document.
    scheduler.running["second"] = _request("second", DOC + [3])
    scheduler.waiting.appendleft(follower)
    assert scheduler._pop_waiting_for_admission() is follower


@pytest.mark.parametrize(
    ("leader_tokens", "follower_tokens"),
    [
        # Too little shared to be worth a wait.
        (DOC[:1000] + [1] * 3000, DOC[:1000] + [2] * 3000),
        # The leader still has far more prompt to process than is shared.
        (DOC[:2000] + [1] * 30_000, DOC[:2000] + [2] * 100),
    ],
)
def test_request_is_not_held_when_waiting_would_not_pay(leader_tokens, follower_tokens):
    leader = _request("leader", leader_tokens)
    follower = _request("follower", follower_tokens)
    scheduler = _scheduler(leader, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is follower
    assert scheduler._shared_prefix_waits == 0


def test_request_is_not_held_behind_a_decoding_request_or_its_own_cached_prefix():
    decoding = _request("decoding", DOC + [1], emitted=5)
    follower = _request("follower", DOC + [2])
    scheduler = _scheduler(decoding, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is follower

    leader = _request("leader", DOC + [1])
    cached = _request("cached", DOC + [2])
    cached.remaining_tokens = [2]  # the shared part is already cached
    scheduler = _scheduler(leader, waiting=[cached])
    assert scheduler._pop_waiting_for_admission() is cached


def test_zero_disables_the_wait_and_negative_is_rejected():
    leader = _request("leader", DOC + [1])
    follower = _request("follower", DOC + [2])
    scheduler = _scheduler(leader, waiting=[follower], wait_tokens=0)
    assert scheduler._pop_waiting_for_admission() is follower
    with pytest.raises(ValueError, match="shared_prefix_wait_tokens"):
        SchedulerConfig(shared_prefix_wait_tokens=-1)
    with pytest.raises(ValueError, match="shared_prefix_wait_tokens"):
        SchedulerConfig(shared_prefix_wait_tokens=True)


def test_admission_leaves_a_held_request_in_the_queue():
    leader = _request("leader", DOC + [1])
    follower = _request("follower", DOC + [2])
    scheduler = _scheduler(leader, waiting=[follower])
    scheduler._max_running_sequences = lambda: 8
    assert scheduler._schedule_waiting() == []
    assert list(scheduler.waiting) == [follower]
