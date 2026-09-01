"""Per-model Metal memory budgeting tests (#2858).

Covers the three pieces the issue's acceptance criteria name:

* ``plan_metal_limit`` — auto per-model budget selection, including the two
  required fixtures: a model that fits at the default (0.90 floor) budget
  and one that requires a higher per-model budget, plus the explicit
  operator override.
* ``Scheduler.preflight_metal_admission`` — a model whose resident
  footprint can never admit a request fails startup with the actionable
  required-vs-available message instead of reporting healthy and 503ing
  every request.
* The rewritten D-METAL-CAP admission 503 — required vs available plus a
  concrete remediation, while keeping the tokens existing regression tests
  grep for.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from vllm_mlx.memory_budget import (  # noqa: E402
    AUTO_UTILIZATION_CEILING,
    AUTO_UTILIZATION_FLOOR,
    MetalPreflightError,
    format_preflight_error,
    plan_metal_limit,
)
from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import (  # noqa: E402
    BackpressureError,
    Scheduler,
    SchedulerConfig,
)

GB = 10**9


def _make_scheduler(*, gpu_memory_utilization: float = 0.9) -> Scheduler:
    config = SchedulerConfig(
        max_num_seqs=8,
        max_concurrent_requests=64,
        enable_prefix_cache=False,
        use_memory_aware_cache=False,
        use_paged_cache=False,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s)))
    return Scheduler(model=MagicMock(), tokenizer=tokenizer, config=config)


class TestPlanMetalLimit:
    def test_small_model_fits_at_default_floor(self):
        """#2858 acceptance fixture 1: a model with plenty of headroom
        resolves to exactly the historical 0.90 default — auto mode is
        byte-identical to the old behavior for every model that already
        fit comfortably."""
        plan = plan_metal_limit(
            weights_bytes=2 * GB,
            device_budget_bytes=20 * GB,
        )
        assert plan.mode == "auto"
        assert plan.resolved_utilization == AUTO_UTILIZATION_FLOOR
        assert plan.limit_bytes == int(20 * GB * AUTO_UTILIZATION_FLOOR)

    def test_large_model_gets_higher_per_model_budget(self):
        """#2858 acceptance fixture 2: the issue's 20B-MoE-on-16GB shape —
        weights close to the device budget push the resolved utilization
        ABOVE the 0.90 floor without the operator touching anything."""
        weights = 11 * GB
        device = 12_800_000_000  # ~the working-set budget of a 16 GB Mac
        plan = plan_metal_limit(weights_bytes=weights, device_budget_bytes=device)
        assert plan.mode == "auto"
        assert plan.resolved_utilization > AUTO_UTILIZATION_FLOOR
        assert plan.resolved_utilization <= AUTO_UTILIZATION_CEILING
        # The resolved limit must actually cover the weights.
        assert plan.limit_bytes > weights

    def test_oversized_model_clamps_at_ceiling(self):
        """Auto mode never plans past the ceiling — a model that cannot
        fit resolves to the ceiling and the scheduler preflight (not the
        planner) is what refuses startup."""
        plan = plan_metal_limit(
            weights_bytes=15 * GB,
            device_budget_bytes=12 * GB,
        )
        assert plan.resolved_utilization == AUTO_UTILIZATION_CEILING

    def test_explicit_override_is_honored_verbatim(self):
        """The advanced manual override must bypass auto sizing entirely,
        including values BELOW what auto would pick."""
        plan = plan_metal_limit(
            weights_bytes=11 * GB,
            device_budget_bytes=12 * GB,
            requested_utilization=0.5,
        )
        assert plan.mode == "manual"
        assert plan.resolved_utilization == 0.5
        assert plan.limit_bytes == 6 * GB

    def test_missing_measurement_falls_back_to_floor(self):
        plan = plan_metal_limit(weights_bytes=0, device_budget_bytes=20 * GB)
        assert plan.resolved_utilization == AUTO_UTILIZATION_FLOOR

    def test_invalid_device_budget_raises(self):
        with pytest.raises(ValueError):
            plan_metal_limit(weights_bytes=GB, device_budget_bytes=0)


class TestPreflightErrorMessage:
    def test_message_reports_required_available_and_remediation(self):
        """#2858 acceptance: a rejection must report required vs available
        memory and at least one concrete remediation."""
        message = format_preflight_error(
            required_bytes=11_200_000_000,
            active_bytes=10_700_000_000,
            min_kv_bytes=500_000_000,
            cap_bytes=9_100_000_000,
            utilization=0.75,
            device_budget_bytes=12_100_000_000,
        )
        assert "11.2 GB" in message  # required
        assert "9.1 GB" in message  # current limit
        assert "--gpu-memory-utilization" in message  # remediation 1
        assert "smaller model" in message  # remediation 2
        assert "close memory-heavy apps" in message.lower()  # remediation 3

    def test_advice_survives_at_auto_ceiling(self):
        """Codex round 3 NIT: at the 0.97 auto ceiling an explicit
        override can still legally raise the knob toward 1.0, so the
        advice must stay."""
        message = format_preflight_error(
            required_bytes=15_000_000_000,
            active_bytes=14_500_000_000,
            min_kv_bytes=500_000_000,
            cap_bytes=11_700_000_000,
            utilization=0.97,
            device_budget_bytes=12_100_000_000,
        )
        assert "Increase --gpu-memory-utilization" in message

    def test_no_impossible_advice_at_full_utilization(self):
        """Codex rounds 1+3 NITs: only at a true 1.0 is 'increase
        --gpu-memory-utilization' impossible advice — the message must
        say the Mac lacks the memory instead."""
        message = format_preflight_error(
            required_bytes=15_000_000_000,
            active_bytes=14_500_000_000,
            min_kv_bytes=500_000_000,
            cap_bytes=12_100_000_000,
            utilization=1.0,
            device_budget_bytes=12_100_000_000,
        )
        assert "Increase --gpu-memory-utilization" not in message
        assert "does not have enough unified memory" in message
        assert "smaller model" in message


class TestSchedulerPreflight:
    def test_noop_when_cap_disabled(self):
        sched = _make_scheduler(gpu_memory_utilization=0.0)
        with patch.object(sched, "_current_metal_active_bytes", return_value=10**15):
            sched.preflight_metal_admission()  # must not raise

    def test_noop_when_active_unreadable(self):
        """Non-Metal CI hosts and unit-test stubs read 0 active bytes —
        the preflight must stay silent there exactly like the gate."""
        sched = _make_scheduler()
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=10 * GB),
            patch.object(sched, "_current_metal_active_bytes", return_value=0),
        ):
            sched.preflight_metal_admission()

    def test_passes_when_model_fits(self):
        sched = _make_scheduler()
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=12 * GB),
            patch.object(sched, "_current_metal_active_bytes", return_value=8 * GB),
        ):
            sched.preflight_metal_admission()

    def test_fails_startup_when_weights_exceed_cap(self):
        """The healthy-but-every-request-503s configuration must be caught
        at startup with the actionable message (#2858 acceptance)."""
        sched = _make_scheduler()
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=9_100_000_000),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=11_200_000_000
            ),
            pytest.raises(MetalPreflightError) as exc_info,
        ):
            sched.preflight_metal_admission()
        message = str(exc_info.value)
        assert "9.1 GB" in message
        assert "--gpu-memory-utilization" in message

    def test_fails_when_smallest_request_cannot_fit(self):
        """Codex round 2 BLOCKING #1: positive but insufficient headroom
        for even a one-token exchange is still a deterministic-503 config
        and must be refused."""
        sched = _make_scheduler()
        per_tok = 100_000
        sched.config.metal_cap_kv_bytes_per_token = per_tok
        cap = 10 * GB
        # Room for one token of KV, but the smallest request needs two.
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(
                sched,
                "_current_metal_active_bytes",
                return_value=cap - per_tok,
            ),
            pytest.raises(MetalPreflightError),
        ):
            sched.preflight_metal_admission()

    def test_passes_when_weights_just_under_cap(self):
        """Codex round 1 BLOCKING #1: a memory-tight config whose weights
        leave room for the smallest valid request can still serve short
        requests, so preflight must NOT refuse it even though a nominal
        1024-token request's projected KV would overflow."""
        sched = _make_scheduler()
        per_tok = 100_000  # bytes per token, via the operator override path
        sched.config.metal_cap_kv_bytes_per_token = per_tok
        cap = 10 * GB
        min_kv = per_tok * Scheduler.PREFLIGHT_NOMINAL_TOKENS
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(
                sched,
                "_current_metal_active_bytes",
                return_value=cap - min_kv // 2,
            ),
        ):
            sched.preflight_metal_admission()

    def test_recovers_when_cache_clear_frees_enough(self):
        """Codex round 1 BLOCKING #4: reclaimable allocator cache must not
        fail a load. When the post-clear re-measure drops below the cap,
        preflight passes."""
        sched = _make_scheduler()
        cap = 10 * GB
        readings = iter([cap + GB, cap - 2 * GB])  # over, then under
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(
                sched,
                "_current_metal_active_bytes",
                side_effect=lambda: next(readings),
            ),
        ):
            sched.preflight_metal_admission()

    def test_counts_sliding_window_kv_and_survives_clear_cache_failure(self):
        """Sliding-window KV is part of the smallest-request estimate, and
        a failing ``mx.clear_cache`` must not mask the real verdict."""
        import vllm_mlx.scheduler as sched_mod

        sched = _make_scheduler()
        per_tok = 100_000
        cap = 10 * GB
        with (
            patch.object(sched, "_resolve_kv_bytes_per_token", return_value=per_tok),
            patch.object(sched, "_resolve_kv_fixed_baseline_bytes", return_value=0),
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=cap - per_tok
            ),
            patch.object(
                sched_mod.mx,
                "clear_cache",
                side_effect=RuntimeError("no metal"),
                create=True,
            ),
            pytest.raises(MetalPreflightError),
        ):
            sched._kv_sliding_slot_bytes = 50_000
            sched._kv_sliding_window = 8
            sched.preflight_metal_admission()


class TestProcessUtilizationRatchet:
    """Codex round 2 BLOCKING #2: a resident scheduler's cap must follow
    the process-wide utilization ratchet upward."""

    @pytest.fixture(autouse=True)
    def _isolated_floor(self):
        import vllm_mlx.memory_budget as mb

        with mb._process_floor_lock:
            saved = (mb._process_utilization_floor, mb._process_floor_generation)
            mb._process_utilization_floor = 0.0
            mb._process_floor_generation += 1
        yield
        with mb._process_floor_lock:
            mb._process_utilization_floor = saved[0]
            mb._process_floor_generation += 1

    def _fake_device(self):
        import vllm_mlx.scheduler as sched_mod

        metal = MagicMock()
        metal.is_available.return_value = True
        device_info = MagicMock(return_value={"memory_size": 100 * GB})
        return patch.multiple(
            sched_mod.mx, metal=metal, device_info=device_info, create=True
        )

    def test_cap_follows_ratchet_upward(self):
        from vllm_mlx.memory_budget import note_resolved_utilization

        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with self._fake_device():
            assert sched._resolve_metal_cap_bytes() == 50 * GB
            note_resolved_utilization(0.9)
            assert sched._resolve_metal_cap_bytes() == 90 * GB
            # A LOWER later resolution must not lower the enforced cap.
            note_resolved_utilization(0.6)
            assert sched._resolve_metal_cap_bytes() == 90 * GB

    def test_disabled_cap_stays_disabled(self):
        from vllm_mlx.memory_budget import note_resolved_utilization

        sched = _make_scheduler(gpu_memory_utilization=0.0)
        with self._fake_device():
            note_resolved_utilization(0.97)
            assert sched._resolve_metal_cap_bytes() == 0

    def test_cap_disables_when_device_info_unreadable(self):
        import vllm_mlx.scheduler as sched_mod

        sched = _make_scheduler(gpu_memory_utilization=0.5)
        metal = MagicMock()
        metal.is_available.return_value = True
        device_info = MagicMock(side_effect=RuntimeError("no device info"))
        with patch.multiple(
            sched_mod.mx, metal=metal, device_info=device_info, create=True
        ):
            assert sched._resolve_metal_cap_bytes() == 0

    def test_ratchet_and_apply_is_atomic_and_serialized(self):
        """Codex round 4 BLOCKING #1: the setter callback must run under
        the floor lock with the post-ratchet effective value, so a stale
        lower limit can never be installed after a newer higher one."""
        import vllm_mlx.memory_budget as mb
        from vllm_mlx.memory_budget import ratchet_utilization_and_apply

        applied: list[float] = []

        def _apply(effective: float) -> None:
            # The lock is held across the callback — a concurrent ratchet
            # cannot interleave between the floor read and this apply.
            assert mb._process_floor_lock.locked()
            applied.append(effective)

        eff, gen1 = ratchet_utilization_and_apply(0.95, _apply)
        assert eff == 0.95
        # A later, LOWER resolution applies the ratcheted floor, not its
        # own value — the last setter call carries the highest floor.
        eff, gen2 = ratchet_utilization_and_apply(0.5, _apply)
        assert eff == 0.95
        assert gen2 == gen1  # no upward ratchet, no invalidation
        assert applied == [0.95, 0.95]


class TestActionableAdmission503:
    def test_backpressure_message_carries_remediation(self):
        """The runtime D-METAL-CAP 503 must state required vs available and
        a remediation, while keeping the historical grep tokens."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        req = Request(
            request_id="req-503",
            prompt="x" * 16,
            prompt_token_ids=list(range(16)),
            sampling_params=SamplingParams(max_tokens=1),
        )
        req.num_prompt_tokens = 16
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * GB),
            patch.object(sched, "_current_metal_active_bytes", return_value=100 * GB),
            pytest.raises(BackpressureError) as exc_info,
        ):
            sched._enforce_metal_cap_at_admission(req)
        message = str(exc_info.value)
        assert "D-METAL-CAP" in message
        assert "reserved KV" in message
        assert "current limit is" in message
        assert "--gpu-memory-utilization" in message

    def test_no_utilization_advice_when_cap_maxed(self):
        """Codex round 2 NIT: at an enforced utilization with no headroom
        left, the 503 must not suggest raising --gpu-memory-utilization."""
        sched = _make_scheduler(gpu_memory_utilization=1.0)
        sched._metal_cap_effective_utilization = 1.0
        req = Request(
            request_id="req-503-max",
            prompt="x" * 16,
            prompt_token_ids=list(range(16)),
            sampling_params=SamplingParams(max_tokens=1),
        )
        req.num_prompt_tokens = 16
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * GB),
            patch.object(sched, "_current_metal_active_bytes", return_value=100 * GB),
            pytest.raises(BackpressureError) as exc_info,
        ):
            sched._enforce_metal_cap_at_admission(req)
        message = str(exc_info.value)
        assert "D-METAL-CAP" in message
        assert "--gpu-memory-utilization" not in message


class TestBatchedEngineBudgetInstall:
    """The BatchedEngine-side wiring: resolve on the worker, install the
    result, and run the admission preflight (#2858)."""

    @pytest.fixture(autouse=True)
    def _isolated_floor(self):
        import vllm_mlx.memory_budget as mb

        with mb._process_floor_lock:
            saved = (mb._process_utilization_floor, mb._process_floor_generation)
            mb._process_utilization_floor = 0.0
            mb._process_floor_generation += 1
        yield
        with mb._process_floor_lock:
            mb._process_utilization_floor = saved[0]
            mb._process_floor_generation += 1

    def _bare_engine(self, requested=None):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._gpu_memory_utilization = requested
        engine._model_load_executor = None
        engine._engine = None
        return engine

    def _patched_mx(self, *, available=True, budget=100 * GB, active=50 * GB):
        device_info = {"max_recommended_working_set_size": budget}
        return patch.multiple(
            "mlx.core",
            device_info=MagicMock(return_value=device_info),
            clear_cache=MagicMock(),
            get_active_memory=MagicMock(return_value=active),
            set_memory_limit=MagicMock(),
            set_cache_limit=MagicMock(),
        ), patch("mlx.core.metal.is_available", return_value=available)

    def test_resolve_auto_success_installs_ratcheted_limit(self):
        import mlx.core as mx

        engine = self._bare_engine(requested=None)
        core_patch, metal_patch = self._patched_mx(active=50 * GB)
        with core_patch, metal_patch:
            resolved = engine._resolve_and_set_metal_limits()
            # 50GB weights + headroom on a 100GB budget needs < the floor,
            # so auto resolves to the historical 0.90.
            assert resolved == AUTO_UTILIZATION_FLOOR
            mx.set_memory_limit.assert_called_once_with(int(100 * GB * 0.90))
            mx.clear_cache.assert_called_once()
            assert mx.set_cache_limit.called

    def test_resolve_falls_back_when_metal_unavailable(self):
        engine = self._bare_engine(requested=0.42)
        core_patch, metal_patch = self._patched_mx(available=False)
        with core_patch, metal_patch:
            assert engine._resolve_and_set_metal_limits() == 0.42
        engine = self._bare_engine(requested=None)
        core_patch, metal_patch = self._patched_mx(available=False)
        with core_patch, metal_patch:
            assert engine._resolve_and_set_metal_limits() == AUTO_UTILIZATION_FLOOR

    def test_resolve_falls_back_without_device_budget(self):
        engine = self._bare_engine(requested=None)
        core_patch, metal_patch = self._patched_mx(budget=0)
        with core_patch, metal_patch:
            assert engine._resolve_and_set_metal_limits() == AUTO_UTILIZATION_FLOOR

    def test_resolve_treats_measurement_failure_as_no_measurement(self):
        import mlx.core as mx

        engine = self._bare_engine(requested=None)
        core_patch, metal_patch = self._patched_mx()
        with core_patch, metal_patch:
            mx.get_active_memory.side_effect = RuntimeError("boom")
            assert engine._resolve_and_set_metal_limits() == AUTO_UTILIZATION_FLOOR
            mx.set_memory_limit.assert_called_once_with(int(100 * GB * 0.90))

    def test_resolve_measures_weights_even_when_clear_cache_fails(self):
        """Codex round 6 #1: a failing ``clear_cache`` must not suppress a
        still-usable weight measurement — a big model must keep its
        auto-raised budget."""
        import mlx.core as mx

        engine = self._bare_engine(requested=None)
        core_patch, metal_patch = self._patched_mx(active=95 * GB)
        with core_patch, metal_patch:
            mx.clear_cache.side_effect = RuntimeError("no cache to clear")
            resolved = engine._resolve_and_set_metal_limits()
            # 95GB weights + headroom on 100GB pins auto at the ceiling,
            # not the no-measurement floor.
            assert resolved == AUTO_UTILIZATION_CEILING

    def test_resolve_survives_setter_failure(self):
        """Codex round 1 BLOCKING #2: a setter failure must not poison the
        resolved utilization the admission cap enforces."""
        import mlx.core as mx

        engine = self._bare_engine(requested=0.95)
        core_patch, metal_patch = self._patched_mx()
        with core_patch, metal_patch:
            mx.set_memory_limit.side_effect = RuntimeError("metal says no")
            assert engine._resolve_and_set_metal_limits() == 0.95

    def test_install_stores_resolved_value(self):
        import concurrent.futures

        engine = self._bare_engine(requested=None)
        engine._model_load_executor = concurrent.futures.ThreadPoolExecutor(1)
        try:
            with patch.object(
                type(engine), "_resolve_and_set_metal_limits", return_value=0.93
            ):
                engine._install_resolved_metal_budget()
            assert engine._gpu_memory_utilization == 0.93
        finally:
            engine._model_load_executor.shutdown(wait=True)

    def test_install_failure_falls_back_to_floor_only_when_unset(self):
        import concurrent.futures

        for requested, expected in ((None, AUTO_UTILIZATION_FLOOR), (0.5, 0.5)):
            engine = self._bare_engine(requested=requested)
            engine._model_load_executor = concurrent.futures.ThreadPoolExecutor(1)
            try:
                with patch.object(
                    type(engine),
                    "_resolve_and_set_metal_limits",
                    side_effect=RuntimeError("worker died"),
                ):
                    engine._install_resolved_metal_budget()
                assert engine._gpu_memory_utilization == expected
            finally:
                engine._model_load_executor.shutdown(wait=True)

    def test_preflight_runs_on_worker_and_propagates(self):
        import concurrent.futures
        from types import SimpleNamespace

        engine = self._bare_engine()
        engine._model_load_executor = concurrent.futures.ThreadPoolExecutor(1)
        preflight = MagicMock()
        engine._engine = SimpleNamespace(
            engine=SimpleNamespace(
                scheduler=SimpleNamespace(preflight_metal_admission=preflight)
            )
        )
        try:
            engine._run_metal_admission_preflight()
            preflight.assert_called_once()
            preflight.side_effect = MetalPreflightError("impossible budget")
            with pytest.raises(MetalPreflightError):
                engine._run_metal_admission_preflight()
        finally:
            engine._model_load_executor.shutdown(wait=True)
