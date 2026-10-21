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


@pytest.mark.parametrize("depth", range(5))
def test_fixed_depth_and_remaining_limit(depth):
    policy = DraftDepthPolicy(depth)
    assert policy.choose() == depth
    assert policy.choose(1) == min(depth, 1)


def calibrated_policy(histogram):
    policy = DraftDepthPolicy("auto", warmup_rounds=0)
    policy.accepted_histogram = list(histogram)
    for depth, cost in enumerate((28.7, 46.2, 58.0, 69.5, 81.7)):
        policy.observe_cost(depth, verify_ms=cost)
    policy.observe_cost(4, draft_ms=19.6)
    policy.observe_cost(0, round_ms=31.7)
    return policy


def test_empirical_bimodal_yield_and_workload_choices():
    # QA-like full-depth counts: both zero and four acceptance have real mass.
    qa = calibrated_policy([26, 24, 13, 8, 22])
    assert qa.choose() == 0
    assert [qa.expected_tokens(k) for k in range(5)] == pytest.approx(
        [1, 1 + 67 / 93, 1 + 110 / 93, 1 + 140 / 93, 1 + 162 / 93]
    )
    code = calibrated_policy([10, 5, 5, 5, 75])
    assert code.choose() == 4
    assert code.rate(4) == pytest.approx(4.3 / 104.3)
    assert DraftDepthPolicy("auto").expected_tokens(4) == 1


def test_histogram_adapts_both_directions_and_excludes_censored_rounds():
    policy = calibrated_policy([0, 0, 0, 0, 1])
    assert policy.choose() == 4
    original = policy.accepted_histogram.copy()
    policy.observe(0, 1)
    policy.observe(2, 2)
    assert policy.accepted_histogram == original
    for _ in range(20):
        policy.observe(0, 4)
    assert policy.choose() == 0
    assert policy.full_rounds == 20
    for _ in range(30):
        policy.observe(4, 4)
    assert policy.choose() == 4
    assert DraftDepthPolicy("auto").accepted_histogram == [0] * 5


def test_warmup_probes_and_output_limits():
    policy = DraftDepthPolicy("auto", probe_interval=3)
    for _ in range(8):
        assert policy.choose() == 4
        policy.observe(0, 4)
    assert policy.full_rounds == 8
    for _ in range(3):
        assert policy.choose() == 0
        policy.observe(0, 0)
    assert policy.choose() == 4
    policy.observe(4, 4)
    # The second probe is mandatory even if the first changed the best rate.
    assert policy.choose() == 4
    policy.observe(0, 4)
    assert policy.choose() == 0
    assert policy.choose(0) == 0
    assert policy.idle_rounds == 0
    # A short tail must not count toward full-depth warmup or acceptance.
    tail = DraftDepthPolicy("auto")
    assert tail.choose(2) == 2
    tail.observe(2, 2)
    assert tail.full_rounds == 0
    assert tail.expected_tokens(4) == 1
    # Cache-ring ceiling sets the maximum observable acceptance depth.
    ring = DraftDepthPolicy("auto", max_depth=2, warmup_rounds=1)
    assert ring.choose() == 2
    ring.observe(2, 2)
    assert ring.full_rounds == 1
    assert ring.expected_tokens(2) == 3


def test_probe_pair_finishes_before_resuming_rate_selection():
    policy = calibrated_policy([0.01, 0, 0, 0, 0])
    policy.probe_interval = 1
    assert policy.choose() == 0
    policy.observe(0, 0)
    assert policy.choose() == 4
    policy.observe(4, 4)
    assert max(range(5), key=policy.rate) == 4
    # current stays parked during both probes, despite the new winning rate.
    assert policy.choose() == 4
    assert policy.current == 0
    policy.observe(4, 4)
    assert policy.choose() == 4
    assert policy.current == 4
    assert policy.idle_rounds == 0


def test_default_probe_interval_is_32_completed_plain_steps():
    policy = DraftDepthPolicy("auto", warmup_rounds=0)
    for _ in range(32):
        assert policy.choose() == 0
        policy.observe(0, 0)
    assert policy.choose() == policy.choose() == 4
    assert policy.choose() == 0


def test_hysteresis_and_custom_curve():
    policy = calibrated_policy([0, 0, 0, 0.2, 0.8])
    policy.current = 3
    assert policy.choose() == 3  # K4 improves rate by less than 10%.
    policy.observe_cost(3, verify_ms=200)
    assert policy.choose() == 4  # Cannot retain an unprofitable depth.
    policy.verify_ms = (10, 11, 12, 100, 200)
    policy.learned_verify_ms.clear()
    policy.learned_plain_ms = 13
    policy.learned_draft_ms = 1
    policy.current = 0
    assert policy.choose() == 2


def test_shared_draft_ema_and_verify_prior_scaling():
    policy = DraftDepthPolicy("auto")
    assert policy.verify_ms == (28.7, 46.2, 58.0, 69.5, 81.7)
    assert policy.draft_ms == 19.6
    assert policy.verify_cost(2) == 58
    policy.observe_cost(4, draft_ms=20, verify_ms=163.4)
    assert policy.verify_cost(2) == 116  # Unobserved costs scale by 2.
    policy.observe_cost(1, draft_ms=12, verify_ms=46.2)
    assert policy.learned_draft_ms == 18  # One EMA across depths.
    assert policy.verify_cost(2) == 87  # Mean observed ratio: (2 + 1) / 2.
    assert policy.verify_cost(1) == 46.2  # Observed costs remain exact.
    policy.observe_cost(1, verify_ms=86.2)
    assert policy.learned_verify_ms[1] == 56.2
    assert policy.rate(3) == pytest.approx(1 / (policy.verify_cost(3) + 18 + 3))
    fresh = DraftDepthPolicy("auto")
    assert fresh.learned_draft_ms is None
    assert fresh.learned_verify_ms == {}
    assert fresh.rate(4) == pytest.approx(1 / (81.7 + 19.6 + 3))


def test_plain_and_host_cost_learning():
    policy = DraftDepthPolicy("auto")
    policy.observe_cost(0, verify_ms=20, round_ms=30)
    policy.observe_cost(0, round_ms=50)
    assert policy.learned_plain_ms == 35
    assert policy.rate(0) == 1 / 35
    assert policy.verify_cost(4) == pytest.approx(81.7 * 20 / 28.7)
    policy.observe_cost(4, draft_ms=20, verify_ms=80, round_ms=107)
    assert policy.host_ms == 4  # EMA of host residual 7 and prior 3.
    assert policy.rate(4) == 1 / 104
    policy.observe_cost(4, draft_ms=20, verify_ms=80, round_ms=99)
    assert policy.host_ms == 3  # Negative timing noise is clamped to zero.
    policy.observe_cost(4, round_ms=1000)  # Cannot decompose an async total.
    assert policy.host_ms == 3


@pytest.mark.parametrize("depth", [0, 1, 4, "auto"])
@pytest.mark.parametrize("eos_id", [9, 1])
@pytest.mark.parametrize("reject", [False, True])
def test_serving_learns_completed_phases(depth, eos_id, reject):
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
            and node.name
            in ("stream_generate", "_match", "_observe_verified", "GenerationChunk")
        )
    ]
    clock = 0.0
    pending = 0.0
    evals = []
    syncs = []
    costs = []
    policy = DraftDepthPolicy(depth)
    original_observe = policy.observe_cost

    def observe_cost(used_depth, **kwargs):
        if depth == "auto":
            assert pending == 0  # Observation must finish before round_ms is read.
        costs.append((used_depth, kwargs))
        original_observe(used_depth, **kwargs)

    policy.observe_cost = observe_cost

    def evaluate(*args):
        nonlocal clock, pending
        if len(args) == 1 and args[0] is runtime.drafter.windows:
            syncs.append(pending)
        else:
            evals.append(pending)
        clock += pending
        pending = 0.0

    class Drafter:
        def __init__(self):
            self.windows = {}

        def observe(self, *_args):
            pass

        def observe_many(self, hidden, position, *, defer):
            nonlocal pending
            assert defer
            self.windows = {0: [hidden]}
            pending += 0.007

        def propose(self, seed, *, depth, device):
            nonlocal pending
            assert device
            pending += 0.0196
            candidate = np.ones(depth + 1, dtype=int)
            if reject:
                candidate[1] = 0
            return candidate, np.array(True)

    def model(ids, _cache, **kwargs):
        nonlocal pending
        if kwargs.get("enable_rollback"):
            pending += 0.02
        logits = np.zeros((*ids.shape, 2))
        logits[..., 1] = 1
        return logits, np.zeros((*ids.shape, 1))

    rollbacks = []
    model.make_cache = lambda **_kw: SimpleNamespace(
        offset=2, layers=[], rollback=rollbacks.append
    )
    namespace = {
        "dataclass": dataclass,
        "DraftDepthPolicy": lambda _mode: policy,
        "draft_depth": lambda: depth,
        "DSpark": Drafter,
        "NaiveStreamingDetokenizer": lambda _p: SimpleNamespace(
            text="", add_token=lambda _t: None, finalize=lambda: None
        ),
        # No synchronize method: adding a timing sync must fail this test.
        "mx": SimpleNamespace(
            array=np.array, argmax=np.argmax, int32=np.int32, eval=evaluate
        ),
    }
    # Exercise the production synchronization method without importing MLX.
    dspark_path = path.with_name("dspark.py")
    dspark_class = next(
        node
        for node in ast.parse(dspark_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "DSpark"
    )
    sync_method = next(
        node
        for node in dspark_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "sync_pending"
    )
    exec(
        compile(
            ast.Module(body=[sync_method], type_ignores=[]), str(dspark_path), "exec"
        ),
        namespace,
    )
    Drafter.sync_pending = namespace["sync_pending"]
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["time"] = SimpleNamespace(perf_counter=lambda: clock)
    runtime = SimpleNamespace(drafter=Drafter(), reset=lambda: None)
    chunks = list(
        namespace["stream_generate"](
            model,
            SimpleNamespace(eos_token_id=eos_id),
            [1, 2],
            runtime=runtime,
            max_tokens=6,
            optimize_prefill=False,
        )
    )
    assert [chunk.token for chunk in chunks] == [1] * (1 if eos_id == 1 else 6)
    draft_syncs = sum(bool(used) for used, _ in costs) if depth == "auto" else 0
    assert len(evals) == len(costs) + 1 + draft_syncs
    expected_depths = {
        0: [0, 0, 0, 0, 0, 0],
        1: [1] * 5 if reject else [1, 1, 1],
        4: [4, 4, 3, 2, 1] if reject else [4, 0],
        "auto": [4, 4, 3, 2, 1] if reject else [4, 0],
    }[depth]
    assert [used for used, _ in costs] == (
        expected_depths[:1] if eos_id == 1 else expected_depths
    )
    assert len(rollbacks) == (len(costs) if reject and depth and eos_id != 1 else 0)
    assert syncs == (pytest.approx([0.007] * len(costs)) if depth == "auto" else [])
    for index, (used, measurement) in enumerate(costs):
        observation_ms = 7 if depth == "auto" or (depth and index) else 0
        assert measurement["round_ms"] == pytest.approx(
            20 + (19.6 if used else 0) + observation_ms
        )
        if depth == "auto":
            assert measurement["verify_ms"] == pytest.approx(20)
            assert measurement["draft_ms"] == (pytest.approx(19.6) if used else None)
        else:
            assert measurement["verify_ms"] is None
            assert measurement["draft_ms"] is None
    if depth == "auto":
        assert policy.learned_draft_ms == pytest.approx(19.6)
        assert policy.learned_verify_ms[4] == pytest.approx(20)
        assert policy.host_ms == pytest.approx(7 - 4 * 0.75**draft_syncs)
        full_rounds = (2 if reject else 1) * int(eos_id != 1)
        assert policy.full_rounds == full_rounds
        mass = sum(0.9**i for i in range(full_rounds))
        assert policy.accepted_histogram == pytest.approx(
            [mass, 0, 0, 0, 0] if reject else [0, 0, 0, 0, mass]
        )
    else:
        assert policy.learned_draft_ms is None
        assert policy.learned_verify_ms == {}
