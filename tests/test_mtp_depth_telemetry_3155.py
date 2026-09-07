"""#3155: per-depth MTP verify telemetry on the process counter and /metrics.

A single accept ratio cannot separate the three ways MTP loses (drafts wrong,
verify too expensive, rollback churn).  The counter now records every verify
call with its draft depth and accepted prefix, and /metrics renders the
per-depth histogram plus verify / correction / bonus counters.
"""

from __future__ import annotations

import pytest

from vllm_mlx.spec_decode.mtp.accept_counter import (
    MTPAcceptCounter,
    get_global_counter,
    reset_global_counter_for_tests,
)


def test_record_verify_builds_prefix_histogram():
    c = MTPAcceptCounter()
    c.record_verify(3, 3)  # all accepted -> bonus
    c.record_verify(3, 1)  # depth 2 rejected -> correction
    c.record_verify(2, 0)  # first draft rejected -> correction
    snap = c.snapshot()
    assert snap.verify_calls == 3
    assert snap.bonus_tokens == 1
    assert snap.correction_tokens == 2
    assert snap.drafted_by_depth == ((1, 3), (2, 3), (3, 2))
    assert snap.accepted_by_depth == ((1, 2), (2, 1), (3, 1))
    assert snap.mean_accepted_per_verify == pytest.approx(4 / 3)
    # verify-only bookkeeping: the legacy triplet is untouched
    assert (snap.attempts, snap.accepts, snap.tokens_saved) == (0, 0, 0)


def test_record_round_is_the_legacy_triplet_plus_verify():
    c = MTPAcceptCounter()
    c.record_round(3, 2)
    c.record_round(0, 0)  # target-only cycle: nothing was drafted
    snap = c.snapshot()
    assert (snap.attempts, snap.accepts, snap.tokens_saved) == (3, 2, 2)
    assert snap.verify_calls == 1
    assert snap.correction_tokens == 1
    assert snap.drafted_by_depth == ((1, 1), (2, 1), (3, 1))
    assert snap.accepted_by_depth == ((1, 1), (2, 1))


@pytest.mark.parametrize("depth,accepted", [(-1, 0), (2, 3), (1, -1)])
def test_invalid_outcomes_are_rejected(depth, accepted):
    c = MTPAcceptCounter()
    with pytest.raises(ValueError):
        c.record_verify(depth, accepted)
    with pytest.raises(ValueError):
        c.record_round(depth, accepted)


def test_reset_clears_depth_state():
    c = MTPAcceptCounter()
    c.record_round(2, 2)
    c.reset()
    snap = c.snapshot()
    assert snap.verify_calls == 0
    assert snap.drafted_by_depth == ()
    assert snap.accepted_by_depth == ()
    assert snap.mean_accepted_per_verify == 0.0


def test_snapshot_defaults_keep_old_constructor_sites_working():
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptSnapshot

    snap = MTPAcceptSnapshot(attempts=4, accepts=3, tokens_saved=3)
    assert snap.verify_calls == 0
    assert snap.drafted_by_depth == ()


def test_metrics_render_per_depth_families(monkeypatch):
    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters

    reset_global_counter_for_tests()
    try:
        counter = get_global_counter()
        counter.record_round(3, 3)
        counter.record_round(3, 1)

        class _Cfg:
            model_alias = "qwen3.8-27b-4bit"
            model = "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"

        body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    finally:
        reset_global_counter_for_tests()
    lbl = 'family="qwen3.8-27b-4bit",method="mtp"'
    assert f"rapid_mlx_spec_decode_verify_calls_total{{{lbl}}} 2" in body
    assert f"rapid_mlx_spec_decode_correction_tokens_total{{{lbl}}} 1" in body
    assert f"rapid_mlx_spec_decode_bonus_tokens_total{{{lbl}}} 1" in body
    for depth, drafted, accepted in ((1, 2, 2), (2, 2, 1), (3, 2, 1)):
        assert (
            f'rapid_mlx_spec_decode_drafted_by_depth_total{{{lbl},depth="{depth}"}} {drafted}'
            in body
        )
        assert (
            f'rapid_mlx_spec_decode_accepted_by_depth_total{{{lbl},depth="{depth}"}} {accepted}'
            in body
        )
    # legacy counters still fed by record_round
    assert f"rapid_mlx_spec_decode_attempts_total{{{lbl}}} 6" in body
    assert f"rapid_mlx_spec_decode_accepts_total{{{lbl}}} 4" in body


def test_metrics_render_cold_start_has_no_depth_samples():
    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters

    reset_global_counter_for_tests()

    class _Cfg:
        model_alias = "qwen3.5-9b-4bit"
        model = "mlx-community/Qwen3.5-9B-4bit"

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    # depth families appear only once sampled (parser rejects empty families)
    assert "rapid_mlx_spec_decode_drafted_by_depth_total" not in body
    assert "rapid_mlx_spec_decode_accepted_by_depth_total" not in body
    assert (
        'rapid_mlx_spec_decode_verify_calls_total{family="qwen3.5-9b-4bit",method="mtp"} 0'
        in body
    )


def test_zero_depth_verify_is_a_noop():
    """codex r1 on #3158: a verify with no draft (EOS cut every position)
    must not count as a verify call or a bonus token, matching
    ``record_round(0, 0)``."""
    counter = MTPAcceptCounter()
    counter.record_verify(0, 0)
    counter.record_round(0, 0)
    snap = counter.snapshot()
    assert snap.verify_calls == 0
    assert snap.bonus_tokens == 0
    assert snap.correction_tokens == 0
    assert snap.drafted_by_depth == ()


def test_record_round_is_atomic_under_snapshot_and_reset():
    """codex r1 on #3158: legacy counters and the verify histogram move
    under one lock, so a concurrent snapshot never sees attempts without
    the matching verify call (or vice versa) and a reset never strands
    verify counters."""
    import threading

    counter = MTPAcceptCounter()
    rounds = 2000
    stop = threading.Event()
    torn: list[tuple[int, int]] = []

    def observer():
        while not stop.is_set():
            snap = counter.snapshot()
            if snap.attempts != 2 * snap.verify_calls:
                torn.append((snap.attempts, snap.verify_calls))

    t = threading.Thread(target=observer)
    t.start()
    try:
        for _ in range(rounds):
            counter.record_round(2, 1)
    finally:
        stop.set()
        t.join()
    assert torn == []
    counter.reset()
    snap = counter.snapshot()
    assert (snap.attempts, snap.verify_calls, snap.correction_tokens) == (0, 0, 0)


def test_metrics_accepted_family_zero_filled_when_nothing_accepted():
    """codex r1 on #3158: drafts recorded but none accepted must not leave
    the accepted-depth family header-only (parser-invalid); it renders a
    zero sample for every drafted depth."""
    from prometheus_client.parser import text_string_to_metric_families

    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters

    reset_global_counter_for_tests()
    try:
        get_global_counter().record_round(2, 0)

        class _Cfg:
            model_alias = "qwen3.5-9b-4bit"
            model = "mlx-community/Qwen3.5-9B-4bit"

        body = "\n".join(_render_spec_decode_mtp_counters(_Cfg())) + "\n"
    finally:
        reset_global_counter_for_tests()
    lbl = 'family="qwen3.5-9b-4bit",method="mtp"'
    for depth in (1, 2):
        assert (
            f'rapid_mlx_spec_decode_accepted_by_depth_total{{{lbl},depth="{depth}"}} 0'
            in body
        )
    families = {f.name: f for f in text_string_to_metric_families(body)}
    assert len(families["rapid_mlx_spec_decode_accepted_by_depth"].samples) == 2
