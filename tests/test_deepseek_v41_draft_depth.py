"""CPU-only contracts for draft-depth configuration and profitability."""

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Load this stdlib-only module without the model package's optional MLX imports.
_SPEC = importlib.util.spec_from_file_location(
    "_dsv41_draft_depth_cpu",
    Path(__file__).resolve().parents[1]
    / "rapid_mlx/models/deepseek_v41_native/draft_depth.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
DEPTH_ENV = _MODULE.DEPTH_ENV
DraftDepthPolicy = _MODULE.DraftDepthPolicy
draft_depth = _MODULE.draft_depth


@pytest.mark.parametrize(
    "value,expected", [(str(k), k) for k in range(5)] + [(" AUTO ", "auto")]
)
def test_depth_environment(monkeypatch, value, expected):
    monkeypatch.setenv(DEPTH_ENV, value)
    assert draft_depth() == expected


def test_depth_default(monkeypatch):
    monkeypatch.delenv(DEPTH_ENV, raising=False)
    assert draft_depth() == 4


@pytest.mark.parametrize("value", ["", "-1", "5", "1.0", "yes"])
def test_invalid_depth_warns(monkeypatch, caplog, value):
    monkeypatch.setenv(DEPTH_ENV, value)
    assert draft_depth() == 4
    assert DEPTH_ENV in caplog.text and "using 4" in caplog.text


def test_policy_profitability_hysteresis_and_probe():
    policy = DraftDepthPolicy("auto", acceptance=0.0, probe_interval=3)
    assert [policy.choose() for _ in range(3)] == [0, 0, 4]
    policy.observe(4, 4)
    assert policy.acceptance == 0.25
    policy.acceptance = 1.0
    assert policy.choose() == 4
    # A 10% switching margin keeps a nearly optimal current depth stable.
    policy.acceptance = 0.9
    policy.current = 3
    assert policy.choose() == 3
    policy.acceptance = 0.0
    assert policy.choose() == 0
    assert policy.choose(0) == 0
    assert policy.choose() == 4
    policy.observe(0, 0)
    assert policy.acceptance == 0.0
    policy.observe(1, 4)
    assert policy.acceptance == 0.125  # One success and one tested rejection.


@pytest.mark.parametrize("depth", range(5))
def test_fixed_depth_and_remaining_limit(depth):
    policy = DraftDepthPolicy(depth)
    assert policy.choose() == depth
    assert policy.choose(1) == min(depth, 1)


def test_policy_selects_best_rate_with_custom_calibration():
    policy = DraftDepthPolicy(
        "auto", acceptance=1, verify_ms=(10, 11, 12, 100, 200), draft_ms=1
    )
    assert policy.choose() == 2
    assert policy.rate(2) == 3 / 13


def test_hysteresis_cannot_retain_an_unprofitable_depth():
    policy = DraftDepthPolicy(
        "auto",
        acceptance=1,
        current=4,
        verify_ms=(10, 18.8, 40, 60, 51),
        draft_ms=0,
    )
    # Best K=1 is 6.4% above baseline, within the switching margin of
    # unprofitable K=4. Discard K=4 rather than letting hysteresis retain it.
    assert policy.choose() == 0


def test_v2_priors_and_learned_phase_costs():
    policy = DraftDepthPolicy("auto", acceptance=1)
    assert policy.verify_ms == (28.7, 46.2, 58.0, 69.5, 81.7)
    assert policy.draft_ms == 21
    assert policy.rate(4) == 5 / (81.7 + 21)
    policy.observe_cost(4, draft_ms=3)
    assert policy.rate(4) == 5 / (81.7 + 3)
    policy.observe_cost(4, verify_ms=50)
    assert policy.rate(4) == 5 / 53
    policy.observe_cost(4, draft_ms=7, verify_ms=70)
    assert policy.learned_draft_ms == {4: 4}
    assert policy.learned_verify_ms == {4: 55}
    assert policy.rate(4) == 5 / 59
    # Depths and requests do not share learned costs.
    assert policy.rate(3) == 4 / (69.5 + 21)
    assert DraftDepthPolicy("auto").learned_draft_ms == {}


def test_cheaper_draft_moves_auto_to_deeper_depth():
    policy = DraftDepthPolicy("auto", acceptance=0.8)
    assert policy.choose() == 0
    for depth in range(1, 5):
        policy.observe_cost(depth, draft_ms=1)
    assert policy.choose() == 3


def test_whole_round_cost_replaces_both_priors_and_learns_per_depth():
    policy = DraftDepthPolicy("auto", acceptance=1)
    policy.observe_cost(0, round_ms=30)
    policy.observe_cost(4, round_ms=60)
    assert policy.rate(0) == 1 / 30
    assert policy.rate(4) == 5 / 60
    policy.observe_cost(4, round_ms=100)
    assert policy.learned_round_ms == {0: 30, 4: 70}
    assert policy.rate(4) == 5 / 70
    assert policy.rate(3) == 4 / (69.5 + 21)
    assert DraftDepthPolicy("auto").learned_round_ms == {}


def test_learned_unprofitable_costs_disable_draft_but_probe_recovers():
    policy = DraftDepthPolicy("auto", acceptance=1, probe_interval=3)
    assert policy.choose() == 4
    for depth in range(1, 5):
        policy.observe_cost(depth, round_ms=1000)
    assert [policy.choose() for _ in range(3)] == [0, 0, 4]
    # A cheap probe is learned using the same EMA, without dropping history.
    for _ in range(20):
        policy.observe_cost(4, round_ms=50)
    assert policy.choose() == 4


@pytest.mark.parametrize("depth", [0, 1, 4])
def test_serving_learns_completed_round_at_existing_sync(depth):
    """Execute production host orchestration with a deferred array test double.

    Loading just the host functions avoids importing Metal on CPU-only hosts.
    The clock advances only when the existing eval completes queued work.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "rapid_mlx/models/deepseek_v41_native/serving.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            or isinstance(node, ast.Import)
            and node.names[0].name == "time"
            or isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in ("stream_generate", "_match", "GenerationChunk")
        )
    ]
    clock = 0.0
    pending = 0.0
    evals = []
    costs = []
    policy = DraftDepthPolicy(depth)
    original_observe = policy.observe_cost

    def observe_cost(used_depth, **kwargs):
        costs.append((used_depth, kwargs))
        original_observe(used_depth, **kwargs)

    policy.observe_cost = observe_cost

    def evaluate(*_args):
        nonlocal clock, pending
        evals.append(pending)
        clock += pending
        pending = 0.0

    class Drafter:
        def observe(self, *_args):
            pass

        def propose(self, seed, *, depth, device):
            nonlocal pending
            assert device
            pending += depth * 0.01
            return np.ones(depth + 1, dtype=int), np.array(True)

    def model(ids, _cache, **kwargs):
        nonlocal pending
        if kwargs.get("enable_rollback"):
            pending += 0.02
        logits = np.zeros((*ids.shape, 2))
        logits[..., 1] = 1
        return logits, np.zeros((*ids.shape, 1))

    model.make_cache = lambda **_kw: SimpleNamespace(offset=2, layers=[])
    namespace = {
        "dataclass": dataclass,
        "DraftDepthPolicy": lambda _mode: policy,
        "draft_depth": lambda: depth,
        "DSpark": Drafter,
        "_observe_verified": lambda *_args: None,
        "NaiveStreamingDetokenizer": lambda _p: SimpleNamespace(
            text="", add_token=lambda _t: None, finalize=lambda: None
        ),
        # No synchronize method: adding a timing sync must fail this test.
        "mx": SimpleNamespace(
            array=np.array, argmax=np.argmax, int32=np.int32, eval=evaluate
        ),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["time"] = SimpleNamespace(perf_counter=lambda: clock)
    runtime = SimpleNamespace(drafter=Drafter(), reset=lambda: None)
    chunks = list(
        namespace["stream_generate"](
            model,
            SimpleNamespace(eos_token_id=9),
            [1, 2],
            runtime=runtime,
            max_tokens=6,
            optimize_prefill=False,
        )
    )
    assert [chunk.token for chunk in chunks] == [1] * 6
    assert len(evals) == len(costs) + 1  # One prefill eval, one per round.
    assert [used for used, _ in costs] == {
        0: [0, 0, 0, 0, 0, 0],
        1: [1, 1, 1],
        4: [4, 0],
    }[depth]
    for used, measurement in costs:
        assert measurement == {"round_ms": pytest.approx(20 + used * 10)}
        assert policy.learned_round_ms[used] == pytest.approx(20 + used * 10)
    assert policy.learned_draft_ms == policy.learned_verify_ms == {}
