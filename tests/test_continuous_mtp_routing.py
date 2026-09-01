"""Model-free and AST tests for PR 9 continuous-MTP routing."""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from vllm_mlx.spec_decode.config import (
    SpeculativeConfigError,
    parse_speculative_config,
)
from vllm_mlx.spec_decode.mtp import continuous_routing as routing_module
from vllm_mlx.spec_decode.mtp.batched import (
    AdmissionDecision,
    BatchedMTPRoute,
    SamplingContract,
)
from vllm_mlx.spec_decode.mtp.continuous_routing import (
    ContinuousMTPAPCHit,
    ContinuousMTPIntegrationRoute,
    ContinuousMTPRequestMetadata,
    plan_router_install,
)
from vllm_mlx.spec_decode.mtp.prepared_state import (
    PreparedStateIdentity,
    prepare_mtp_state,
)

ROOT = Path(__file__).resolve().parents[1]


def _descriptor(family="qwen3_5"):
    return MappingProxyType(
        {
            "protocol_version": 1,
            "model_family": family,
            "batch_forward": "mtp_batch_forward",
            "recursive_draft_depth": 2,
            "fixed_membership": True,
            "dynamic_join": True,
            "quantized_cache": False,
            "windowed_cache": False,
            "xtc": False,
        }
    )


class _Model:
    batched_mtp_capability = _descriptor()
    mtp = object()

    def __call__(self, *args, **kwargs):
        return args, kwargs

    def mtp_batch_forward(self, *args, **kwargs):
        return args, kwargs

    def mtp_forward(self, *args, **kwargs):
        return args, kwargs

    def make_mtp_cache(self):
        return object()


class _SchedulerResponse:
    """Strict mlx-lm response twin used to exercise compatibility filling."""

    def __init__(
        self,
        uid,
        token,
        logprobs,
        finish_reason,
        required_bookkeeping,
    ):
        self.uid = uid
        self.token = token
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.required_bookkeeping = required_bookkeeping


class _SchedulerBatchGenerator:
    class _GenerationBatch:
        Response = _SchedulerResponse

    def __init__(self, *, raw_next=([], ()), with_response=True):
        if not with_response:
            self._generation_batch = SimpleNamespace()
        else:
            self._generation_batch = self._GenerationBatch()
        self._unprocessed_sequences = deque()
        self.raw_next = raw_next
        self.closed = False
        self.removed = []

    def next(self):
        return self.raw_next

    def close(self):
        self.closed = True
        return "closed"

    def remove(self, uids, *_args, **_kwargs):
        self.removed.append(tuple(uids))
        return {uid: (f"plain-cache-{uid}", [uid]) for uid in uids}


def _scheduler_sequence(uid, *, caches=None, segments=None, processors=None):
    return (
        uid,
        [[10 + uid, 20 + uid]] if segments is None else segments,
        16,
        [SimpleNamespace(offset=0, nbytes=0)] if caches is None else caches,
        [],
        object(),
        [] if processors is None else processors,
        None,
        None,
    )


def _scheduler_request(*, temperature=0.0, stop=None, has_tools=False):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=temperature, stop=stop),
        has_tools=has_tools,
    )


def _request(lane, uid, **changes):
    values = {
        "lane_id": lane,
        "uid": uid,
        "prompt_tokens": (10 + uid, 20 + uid),
        "max_tokens": 16,
    }
    values.update(changes)
    return ContinuousMTPRequestMetadata(**values)


def _router(**changes):
    values = {"enabled": True, "hard_reserve_bytes": 0}
    values.update(changes)
    decision = plan_router_install(_Model(), **values)
    assert decision.admitted is True
    assert decision.router is not None
    return decision.router


def _identity():
    return PreparedStateIdentity.from_config(
        model_id="Qwen/Qwen3.8-Flash-Next",
        model_revision="revision-a",
        speculative_config={"method": "mtp", "continuous_batching": True},
        target_cache_layout="qwen4-batch-kv:bf16",
        mtp_cache_layout="qwen4-mtp-batch-kv:bf16",
        seed_hidden_layout="bf16[1,1,2048]",
    )


def _apc_hit(prefix, *, identity=None):
    expected = _identity()
    state = prepare_mtp_state(
        identity=identity or expected,
        prefix_tokens=prefix,
        target_cache="target-cache",
        target_cache_tokens=len(prefix),
        mtp_cache="mtp-cache",
        mtp_cache_pairs=len(prefix) - 1,
        seed_hidden="seed-hidden",
        captured_at=10.0,
    )
    return ContinuousMTPAPCHit(
        state=state,
        expected_identity=expected,
        target_cache_tokens=len(prefix),
        mtp_cache_pairs=len(prefix) - 1,
        now=11.0,
        max_age_seconds=60.0,
    )


def test_speculative_config_continuous_batching_is_explicit_and_mtp_only():
    default = parse_speculative_config('{"method":"mtp"}')
    enabled = parse_speculative_config('{"method":"mtp","continuous_batching":true}')
    assert default is not None and default.continuous_batching is False
    assert default.allow_dynamic_membership is False
    assert enabled is not None and enabled.continuous_batching is True

    dynamic = parse_speculative_config(
        '{"method":"mtp","continuous_batching":true,"allow_dynamic_membership":true}'
    )
    assert dynamic is not None and dynamic.allow_dynamic_membership is True

    with pytest.raises(SpeculativeConfigError, match="must be a boolean"):
        parse_speculative_config('{"method":"mtp","continuous_batching":1}')
    with pytest.raises(SpeculativeConfigError, match="must be a boolean"):
        parse_speculative_config('{"method":"mtp","allow_dynamic_membership":1}')
    with pytest.raises(SpeculativeConfigError, match="requires continuous_batching"):
        parse_speculative_config('{"method":"mtp","allow_dynamic_membership":true}')
    with pytest.raises(SpeculativeConfigError, match="unsupported speculative-config"):
        parse_speculative_config('{"method":"suffix","continuous_batching":true}')


def test_install_plan_is_default_off_and_fails_closed_without_mutation():
    model = _Model()
    before = dict(vars(model))

    disabled = plan_router_install(model, enabled=False, hard_reserve_bytes=0)
    quantized = plan_router_install(
        model,
        enabled=True,
        cache_quantized=True,
        hard_reserve_bytes=0,
    )

    assert disabled.admitted is False
    assert disabled.fallback is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert "disabled" in " ".join(disabled.reasons)
    assert quantized.admitted is False
    assert "quantized cache" in " ".join(quantized.reasons)
    assert vars(model) == before


def test_install_plan_threads_attested_dynamic_membership():
    admitted = plan_router_install(
        _Model(),
        enabled=True,
        allow_dynamic_membership=True,
        hard_reserve_bytes=0,
    )
    assert admitted.admitted is True
    assert admitted.router is not None
    assert admitted.router.config.allow_dynamic_membership is True
    assert admitted.router.runtime.capabilities.dynamic_membership is True


def test_supported_requests_build_an_immutable_fixed_cohort_plan():
    router = _router()

    decision = router.plan([_request("a", 1), _request("b", 2)], free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert decision.live_token_delivery is False
    assert [lane.lane_id for lane in decision.cohort] == ["a", "b"]
    assert [lane.spec.uid for lane in decision.cohort] == [1, 2]
    assert [lane.spec.num_draft for lane in decision.cohort] == [2, 2]
    assert all(lane.prepared_state is None for lane in decision.cohort)


def test_exact_apc_sidecar_is_validated_and_carried_as_a_resume_plan():
    router = _router()
    prefix = tuple(range(64))
    hit = _apc_hit(prefix)
    requests = [
        _request(
            "a",
            1,
            prompt_tokens=prefix + (999,),
            apc_hit=hit,
        ),
        _request("b", 2),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    restored = decision.cohort[0]
    assert restored.resume_at == 64
    assert restored.spec.prompt == (999,)
    assert restored.spec.prompt_cache == "target-cache"
    assert restored.spec.mtp_cache == "mtp-cache"
    assert restored.prepared_state is hit.state
    assert decision.live_token_delivery is False


def test_bad_apc_sidecar_routes_plain_while_other_lanes_form_cohort():
    router = _router()
    prefix = tuple(range(64))
    foreign = PreparedStateIdentity.from_config(
        model_id="other/model",
        model_revision="revision-a",
        speculative_config={"method": "mtp", "continuous_batching": True},
        target_cache_layout="qwen4-batch-kv:bf16",
        mtp_cache_layout="qwen4-mtp-batch-kv:bf16",
        seed_hidden_layout="bf16[1,1,2048]",
    )
    requests = [
        _request(
            "bad-apc",
            1,
            prompt_tokens=prefix + (999,),
            apc_hit=_apc_hit(prefix, identity=foreign),
        ),
        _request("a", 2),
        _request("b", 3),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert [lane.lane_id for lane in decision.cohort] == ["a", "b"]
    assert decision.plain_lane_ids == ("bad-apc",)
    assert "model_mismatch" in " ".join(decision.reasons)


def test_unsupported_sampling_falls_back_to_legacy_without_a_cohort():
    router = _router()
    transformed = SamplingContract(greedy=False)
    requests = [
        _request("a", 1, sampling=transformed, temperature=0.8),
        _request("b", 2, sampling=transformed, temperature=0.8),
    ]

    decision = router.plan(requests, free_bytes=1)

    assert decision.route is ContinuousMTPIntegrationRoute.LEGACY_MTP
    assert decision.cohort == ()
    assert decision.legacy_lane_ids == ("a", "b")
    assert "transformed_distribution_verify" in " ".join(decision.reasons)


def test_scheduler_wiring_diverts_next_and_refusal_precedes_mutation():
    tree = ast.parse((ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8"))
    installer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_install_continuous_mtp_router"
    )
    assignments = [
        node
        for node in ast.walk(installer)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_step"
            for target in node.targets
        )
        for node in assignments
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "next"
            and isinstance(target.value, ast.Name)
            and target.value.id == "batch_gen"
            for target in node.targets
        )
        for node in assignments
    )
    source = ast.get_source_segment(
        (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8"),
        installer,
    )
    assert source is not None
    assert "if not decision.admitted" in source
    assert source.index("if not decision.admitted") < source.index(
        "batch_gen._continuous_mtp_router"
    )
    assert "ContinuousMTPDriver.create" in source
    assert "continuous_responses + list(fallback_responses)" in source
    assert "scheduler_owned_termination" in source
    assert "bool(params.stop) or bool(request.has_tools)" in source
    assert source.index("_remove_queued(selected)") < source.index(
        "raw = original_next()"
    )
    assert "finishing_package.target_cache" in (
        ROOT / "vllm_mlx" / "spec_decode" / "mtp" / "continuous_driver.py"
    ).read_text(encoding="utf-8")

    scheduler_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    create_generator = next(
        node
        for node in scheduler_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_create_batch_generator"
    )
    create_source = ast.get_source_segment(
        (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8"),
        create_generator,
    )
    assert create_source is not None
    router_call = create_source.index("_install_continuous_mtp_router(")
    vendored_call = create_source.index("_install_mtp_vendored(", router_call)
    assert router_call < vendored_call


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mtp_continuous_batching": 1}, "must be a boolean"),
        ({"mtp_allow_dynamic_membership": 1}, "must be a boolean"),
        (
            {"mtp_continuous_batching": True, "spec_decode": "none"},
            "requires spec_decode='mtp'",
        ),
        (
            {"mtp_allow_dynamic_membership": True},
            "requires mtp_continuous_batching",
        ),
    ],
)
def test_scheduler_config_rejects_ambiguous_continuous_mtp_policy(changes, message):
    from vllm_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match=message):
        SchedulerConfig(**changes)


@pytest.mark.requires_mlx
def test_live_installer_fails_closed_before_mutating_unsupported_generators(
    monkeypatch,
):
    from vllm_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from vllm_mlx.spec_decode.mtp import continuous_runtime

    enabled = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert not _install_continuous_mtp_router(
        SimpleNamespace(next=lambda: ([], [])), _Model(), enabled
    )

    no_capacity = _SchedulerBatchGenerator()
    assert not _install_continuous_mtp_router(
        no_capacity,
        _Model(),
        SimpleNamespace(
            spec_decode="mtp",
            mtp_continuous_batching=True,
            max_num_seqs=0,
            completion_batch_size=4,
        ),
    )
    assert not hasattr(no_capacity, "_continuous_mtp_router")

    disabled = _SchedulerBatchGenerator()
    assert not _install_continuous_mtp_router(
        disabled, _Model(), SchedulerConfig(spec_decode="mtp")
    )
    assert not hasattr(disabled, "_continuous_mtp_router")

    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("refused")),
    )
    assembly_refused = _SchedulerBatchGenerator()
    assert not _install_continuous_mtp_router(assembly_refused, _Model(), enabled)
    assert not hasattr(assembly_refused, "_continuous_mtp_router")

    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: object(),
    )
    missing_response = _SchedulerBatchGenerator(with_response=False)
    assert not _install_continuous_mtp_router(missing_response, _Model(), enabled)
    assert not hasattr(missing_response, "_continuous_mtp_router")


@pytest.mark.requires_mlx
def test_live_installer_drives_join_detach_remove_and_compat_response(monkeypatch):
    from vllm_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from vllm_mlx.spec_decode.mtp import continuous_runtime
    from vllm_mlx.spec_decode.mtp.continuous_driver import ContinuousMTPDriver
    from vllm_mlx.spec_decode.mtp.continuous_engine import (
        ContinuousSelfMTPCapabilities,
    )

    capabilities = ContinuousSelfMTPCapabilities(
        target_return_hidden=True,
        mtp_return_hidden=True,
        confirmed_target_forward=True,
        ragged_rollback=True,
        atomic_cache_commit=True,
        dynamic_membership=True,
    )
    runtime = SimpleNamespace(capabilities=capabilities)
    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: runtime,
    )

    class _Driver:
        dynamic_membership = True
        lane_uids = (1, 2)
        pending_join_uids = ()
        has_work = True
        closed = False
        has_pending_responses = False

        def __init__(self):
            self.calls = []
            self.fail_discard = False

        def next(self):
            self.calls.append("next")
            return [SimpleNamespace(uid=1, token=31)]

        def take_terminal_detaches(self):
            self.calls.append("take-terminal")
            return ()

        def resume_turnover(self):
            self.calls.append("resume")
            return ()

        def queue_lanes(self, specs, *, stop_tokens):
            self.calls.append(("join", tuple(spec.uid for spec in specs), stop_tokens))
            return tuple(spec.uid for spec in specs)

        def discard_all(self):
            self.calls.append("discard")
            if self.fail_discard:
                raise RuntimeError("already detached")

        def remove_uids(self, uids):
            self.calls.append(("remove", tuple(uids)))
            return (
                SimpleNamespace(
                    uid=1,
                    target_cache="target-cache",
                    token_ids=(10, 11),
                    draft_cache="draft-cache",
                    lane=SimpleNamespace(seed_hidden="seed-hidden"),
                ),
                SimpleNamespace(
                    uid=99,
                    target_cache="foreign",
                    token_ids=(),
                    draft_cache="foreign",
                    lane=SimpleNamespace(seed_hidden=None),
                ),
            )

    driver = _Driver()
    captured = {}

    def _create(_cls, specs, installed_runtime, **kwargs):
        captured["uids"] = tuple(spec.uid for spec in specs)
        captured["runtime"] = installed_runtime
        captured["response_factory"] = kwargs["response_factory"]
        return driver

    monkeypatch.setattr(ContinuousMTPDriver, "create", classmethod(_create))

    batch_gen = _SchedulerBatchGenerator(raw_next=[SimpleNamespace(uid=50)])
    requests = {}
    uid_to_request_id = {}
    for uid in (1, 2, 3, 4, 5, 6, 95):
        request_id = f"req-{uid}"
        requests[request_id] = _scheduler_request()
        uid_to_request_id[uid] = request_id
    uid_to_request_id[97] = "req-97"
    batch_gen._unprocessed_sequences.extend(
        [
            (0,),
            _scheduler_sequence(98),
            _scheduler_sequence(97, segments=[]),
            _scheduler_sequence(95, segments=[]),
            _scheduler_sequence(
                96,
                caches=[SimpleNamespace(caches=[SimpleNamespace(offset="bad")])],
                processors=[object()],
            ),
            _scheduler_sequence(1),
            _scheduler_sequence(2),
        ]
    )
    requests["req-96"] = _scheduler_request()
    uid_to_request_id[96] = "req-96"

    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        mtp_allow_dynamic_membership=True,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests=requests,
        uid_to_request_id=uid_to_request_id,
        free_bytes_getter=lambda: 16 * 1024**3,
        stop_tokens={99},
    )

    prompt, fallback = batch_gen.next()
    assert prompt == []
    assert [item.uid for item in fallback] == [50]
    assert captured["uids"] == (1, 2)
    assert captured["runtime"] is runtime

    compatible = captured["response_factory"](
        uid=1,
        token=2,
        logprobs=None,
        finish_reason=None,
        from_draft=True,
    )
    assert compatible.required_bookkeeping is None
    assert compatible.from_draft is True

    batch_gen.raw_next = ([], [])
    batch_gen._unprocessed_sequences.append(
        _scheduler_sequence(3, processors=[object()])
    )
    _, responses = batch_gen.next()
    assert [item.uid for item in responses] == [1]
    assert driver.calls[:2] == ["next", "take-terminal"]

    batch_gen._unprocessed_sequences.extend(
        [_scheduler_sequence(uid) for uid in (4, 5, 6)]
    )
    batch_gen.next()
    join = next(
        call for call in driver.calls if isinstance(call, tuple) and call[0] == "join"
    )
    assert join[1] == (4, 5)

    driver.lane_uids = (1, 2, 4, 5)
    batch_gen.next()
    driver.dynamic_membership = False
    batch_gen.next()
    driver.closed = True
    batch_gen.next()
    assert "resume" in driver.calls

    removed = batch_gen.remove([1], return_prompt_caches=True)
    assert removed[1] == ("target-cache", [10, 11])
    assert batch_gen._continuous_mtp_removed_states[1] == (
        "draft-cache",
        "seed-hidden",
    )

    assert batch_gen.close() == "closed"
    driver.fail_discard = True
    assert batch_gen.close() == "closed"


@pytest.mark.requires_mlx
def test_live_installer_handles_empty_pressure_and_optional_base_hooks(monkeypatch):
    from vllm_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from vllm_mlx.spec_decode.mtp import continuous_runtime

    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    batch_gen = _SchedulerBatchGenerator()
    batch_gen.close = None
    batch_gen.remove = None
    batch_gen._unprocessed_sequences.append(_scheduler_sequence(1))
    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests={"req-1": _scheduler_request()},
        uid_to_request_id={1: "req-1"},
        free_bytes_getter=lambda: (_ for _ in ()).throw(RuntimeError("pressure")),
    )
    assert batch_gen.next() == ([], [])
    assert batch_gen.remove([1], return_prompt_caches=True) == {}
    assert batch_gen.close() is None

    no_getter = _SchedulerBatchGenerator()
    no_getter._unprocessed_sequences.append(_scheduler_sequence(2))
    assert _install_continuous_mtp_router(
        no_getter,
        _Model(),
        config,
        requests={"req-2": _scheduler_request()},
        uid_to_request_id={2: "req-2"},
    )
    assert no_getter.next() == ([], [])

    empty = _SchedulerBatchGenerator()
    assert _install_continuous_mtp_router(empty, _Model(), config)
    assert empty.next() == ([], [])


@pytest.mark.requires_mlx
def test_live_installer_forms_fixed_initial_cohort_without_dynamic_join(monkeypatch):
    """Exercise the real scheduler import and installer on the Apple lane.

    The scheduler intentionally imports MLX and mlx-lm at module load so its
    hardware compatibility hooks run before BatchGenerator is captured.  This
    integration contract therefore belongs on the real-MLX lane; the pure
    routing and AST contracts above remain in the hosted no-MLX matrix.
    """
    from vllm_mlx.scheduler import SchedulerConfig, _install_continuous_mtp_router
    from vllm_mlx.spec_decode.mtp import continuous_runtime
    from vllm_mlx.spec_decode.mtp.continuous_driver import ContinuousMTPDriver
    from vllm_mlx.spec_decode.mtp.continuous_engine import (
        ContinuousSelfMTPCapabilities,
    )

    class _BatchGenerator:
        class _GenerationBatch:
            Response = SimpleNamespace

        def __init__(self):
            self._generation_batch = self._GenerationBatch()
            self._unprocessed_sequences = deque()

        def next(self):
            return [], []

        def close(self):
            return None

        def remove(self, _uids, *_args, **_kwargs):
            return {}

    batch_gen = _BatchGenerator()
    requests = {}
    uid_to_request_id = {}
    for uid in (1, 2):
        request_id = f"req-{uid}"
        requests[request_id] = SimpleNamespace(
            sampling_params=SimpleNamespace(
                temperature=0.0,
                stop=None,
            ),
            has_tools=False,
        )
        uid_to_request_id[uid] = request_id
        batch_gen._unprocessed_sequences.append(
            (
                uid,
                [[10 + uid, 20 + uid]],
                16,
                [SimpleNamespace(offset=0, nbytes=0)],
                [],
                object(),
                [],
                None,
                None,
            )
        )

    capabilities = ContinuousSelfMTPCapabilities(
        target_return_hidden=True,
        mtp_return_hidden=True,
        confirmed_target_forward=True,
        ragged_rollback=True,
        atomic_cache_commit=True,
        dynamic_membership=False,
    )
    runtime = SimpleNamespace(capabilities=capabilities)
    monkeypatch.setattr(
        continuous_runtime,
        "assemble_continuous_self_mtp_runtime",
        lambda *_args, **_kwargs: runtime,
    )

    created = {}
    driver = SimpleNamespace(dynamic_membership=False, has_work=True)

    def _create(_cls, specs, installed_runtime, **_kwargs):
        created["uids"] = tuple(spec.uid for spec in specs)
        created["runtime"] = installed_runtime
        return driver

    monkeypatch.setattr(ContinuousMTPDriver, "create", classmethod(_create))

    config = SchedulerConfig(
        spec_decode="mtp",
        mtp_continuous_batching=True,
        mtp_allow_dynamic_membership=False,
        max_num_seqs=4,
        completion_batch_size=4,
    )
    assert _install_continuous_mtp_router(
        batch_gen,
        _Model(),
        config,
        requests=requests,
        uid_to_request_id=uid_to_request_id,
        free_bytes_getter=lambda: 16 * 1024**3,
    )

    prompt_responses, completion_responses = batch_gen.next()

    assert prompt_responses == []
    assert completion_responses == []
    assert created == {"uids": (1, 2), "runtime": runtime}
    assert batch_gen._continuous_mtp_driver is driver
    assert list(batch_gen._unprocessed_sequences) == []


def test_cli_and_scheduler_config_carry_the_default_off_opt_in_by_ast():
    cli_source = (ROOT / "vllm_mlx" / "cli.py").read_text(encoding="utf-8")
    scheduler_source = (ROOT / "vllm_mlx" / "scheduler.py").read_text(encoding="utf-8")
    assert "args.mtp_continuous_batching = config.continuous_batching" in cli_source
    assert (
        'mtp_continuous_batching=getattr(args, "mtp_continuous_batching", False)'
        in cli_source
    )
    assert "mtp_continuous_batching: bool = False" in scheduler_source
    assert "mtp_allow_dynamic_membership: bool = False" in scheduler_source
    assert (
        'if self.mtp_continuous_batching and self.spec_decode != "mtp"'
        in scheduler_source
    )
    assert "mtp_allow_dynamic_membership=getattr(" in cli_source


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"uid": True}, "uid must be an integer"),
        ({"prompt_tokens": ()}, "non-negative integers"),
        ({"prompt_tokens": (1, -1)}, "non-negative integers"),
        ({"max_tokens": True}, "max_tokens must be an integer"),
        ({"max_tokens": 0}, "max_tokens must be positive"),
        ({"stop_tokens": frozenset({True})}, "stop_tokens must contain integers"),
        ({"temperature": 0.5}, "greedy sampling requires"),
        (
            {"sampling": SamplingContract(greedy=False), "temperature": 0.0},
            "non-greedy sampling requires",
        ),
    ],
)
def test_request_metadata_rejects_ambiguous_scheduler_facts(changes, message):
    values = {
        "lane_id": "lane",
        "uid": 1,
        "prompt_tokens": (1, 2),
        "max_tokens": 8,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        ContinuousMTPRequestMetadata(**values)


def test_router_rejects_duplicate_lane_and_uid_identity():
    router = _router()
    with pytest.raises(ValueError, match="lane_id values must be unique"):
        router.plan([_request("same", 1), _request("same", 2)], free_bytes=1)
    with pytest.raises(ValueError, match="uid values must be unique"):
        router.plan([_request("a", 1), _request("b", 1)], free_bytes=1)


def test_static_refusals_preserve_plain_fallback_without_legacy_mtp():
    class _PlainOnly:
        batched_mtp_capability = _descriptor("qwen4_exp")

    decision = plan_router_install(
        _PlainOnly(),
        enabled=True,
        cache_windowed=True,
        allow_dynamic_membership=True,
        hard_reserve_bytes=0,
    )
    assert decision.admitted is False
    assert decision.fallback is ContinuousMTPIntegrationRoute.PLAIN_DECODE
    joined = " ".join(decision.reasons)
    assert "unsupported model family" in joined
    assert "windowed cache" in joined
    assert "dynamic membership" in joined


def test_descriptor_mismatch_and_plain_runtime_fallback_are_explicit():
    class _Mismatched(_Model):
        batched_mtp_capability = MappingProxyType(
            {**dict(_descriptor()), "protocol_version": 2}
        )

    refused = plan_router_install(_Mismatched(), enabled=True, hard_reserve_bytes=0)
    assert refused.admitted is False
    assert "protocol_version" in " ".join(refused.reasons)

    plain_router = _router()
    plain_router.runtime = type(plain_router.runtime)(
        model_family=plain_router.runtime.model_family,
        capability_descriptor=plain_router.runtime.capability_descriptor,
        capabilities=plain_router.runtime.capabilities,
        legacy_mtp_available=False,
    )
    transformed = SamplingContract(greedy=False)
    decision = plain_router.plan(
        [
            _request("a", 1, sampling=transformed, temperature=0.8),
            _request("b", 2, sampling=transformed, temperature=0.8),
        ],
        free_bytes=1,
    )
    assert decision.route is ContinuousMTPIntegrationRoute.PLAIN_DECODE
    assert decision.plain_lane_ids == ("a", "b")


def test_runtime_static_refusal_routes_legacy_or_plain_per_loaded_surface():
    for legacy in (True, False):
        router = _router()
        router._static_refusals = ("runtime drift",)
        router.runtime = type(router.runtime)(
            model_family=router.runtime.model_family,
            capability_descriptor=router.runtime.capability_descriptor,
            capabilities=router.runtime.capabilities,
            legacy_mtp_available=legacy,
        )
        decision = router.plan([_request("a", 1)], free_bytes=1)
        expected = (
            ContinuousMTPIntegrationRoute.LEGACY_MTP
            if legacy
            else ContinuousMTPIntegrationRoute.PLAIN_DECODE
        )
        assert decision.route is expected
        assert decision.reasons == ("runtime drift",)


def test_request_local_cache_refusal_does_not_block_other_cohort_lanes():
    decision = _router().plan(
        [
            _request("quantized", 1, cache_quantized=True),
            _request("a", 2),
            _request("b", 3),
        ],
        free_bytes=1,
    )
    assert decision.route is ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED
    assert decision.plain_lane_ids == ("quantized",)
    assert "quantized/windowed" in " ".join(decision.reasons)


def test_queue_only_admission_remains_queue_when_every_lane_is_plain(monkeypatch):
    monkeypatch.setattr(
        routing_module,
        "plan_admission",
        lambda *_args, **_kwargs: AdmissionDecision(BatchedMTPRoute.QUEUE),
    )
    decision = _router().plan(
        [_request("quantized", 1, cache_quantized=True)],
        free_bytes=0,
    )
    assert decision.route is ContinuousMTPIntegrationRoute.QUEUE
