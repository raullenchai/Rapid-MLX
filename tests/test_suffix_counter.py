# SPDX-License-Identifier: Apache-2.0
"""SuffixDecoding telemetry counter + its /metrics rendering.

``_suffix_stats`` in the scheduler was write-only: the verify/acceptance
counts and the seven-way fallthrough breakdown were all maintained and
none of it was readable. "I enabled suffix decoding and nothing got
faster" therefore had no answer short of patching in a log line.

These tests pin the counter's arithmetic and the Prometheus surface, so
the exit stays open.
"""

from types import SimpleNamespace

import pytest

from vllm_mlx.routes.metrics import _render_suffix_decode_counters
from vllm_mlx.speculative.suffix_counter import (
    SuffixAcceptCounter,
    get_global_counter,
    reset_global_counter,
)


@pytest.fixture
def counter():
    return SuffixAcceptCounter()


# ── counter arithmetic ────────────────────────────────────────────────


def test_accept_ratio_is_zero_not_nan_before_any_proposal(counter):
    """A fresh process must still render a scrapeable series. NaN would
    break Prometheus ingestion on the very first scrape."""
    snap = counter.snapshot()
    assert snap["accept_ratio"] == 0.0
    assert snap["draft_tokens_proposed"] == 0


def test_accept_ratio_is_accepted_over_proposed(counter):
    counter.record_verify(proposed=8, accepted=6)
    counter.record_verify(proposed=8, accepted=7)
    snap = counter.snapshot()
    assert snap["verify_steps"] == 2
    assert snap["draft_tokens_proposed"] == 16
    assert snap["draft_tokens_accepted"] == 13
    assert snap["accept_ratio"] == pytest.approx(13 / 16)


def test_fallthrough_reasons_are_tracked_separately(counter):
    counter.record_fallthrough("cooldown")
    counter.record_fallthrough("cooldown")
    counter.record_fallthrough("non_greedy")
    snap = counter.snapshot()
    assert snap["fallthrough_steps"] == 3
    assert snap["ft_cooldown"] == 2
    assert snap["ft_non_greedy"] == 1
    assert snap["ft_no_draft"] == 0


def test_unknown_fallthrough_reason_still_counts_the_step(counter):
    """A typo'd reason must not silently lose the step from the total —
    the breakdown is advisory, ``fallthrough_steps`` is the ground truth
    that must reconcile against ``verify_steps``."""
    counter.record_fallthrough("not-a-real-reason")
    assert counter.snapshot()["fallthrough_steps"] == 1


def test_state_gauges_are_last_write_wins(counter):
    counter.set_state(current_k=2, backoff_level=3)
    counter.set_state(current_k=8, backoff_level=0)
    snap = counter.snapshot()
    assert snap["current_k"] == 8
    assert snap["backoff_level"] == 0


def test_counters_are_monotonic_across_records(counter):
    for _ in range(5):
        counter.record_verify(proposed=4, accepted=1)
    counter.record_cooldown_trip(level=2)
    snap = counter.snapshot()
    assert snap["verify_steps"] == 5
    assert snap["cooldown_trips"] == 1
    assert snap["backoff_level"] == 2


# ── /metrics rendering ────────────────────────────────────────────────


def _render(alias="gemma-4-12b-4bit"):
    return "\n".join(_render_suffix_decode_counters(SimpleNamespace(model_alias=alias)))


def test_metrics_render_even_when_nothing_has_run():
    """Series must exist at cold start so a dashboard's rate() has a
    stable series set across restarts."""
    reset_global_counter()
    out = _render()
    assert "rapid_mlx_suffix_decode_verify_steps_total" in out
    assert "rapid_mlx_suffix_decode_accept_ratio" in out
    assert 'reason="cooldown"' in out


def test_metrics_reflect_recorded_activity():
    reset_global_counter()
    c = get_global_counter()
    c.record_verify(proposed=8, accepted=6)
    c.record_fallthrough("cooldown")
    c.set_state(current_k=8, backoff_level=0)
    out = _render()
    assert (
        'rapid_mlx_suffix_decode_draft_tokens_accepted_total{family="gemma-4-12b-4bit",method="suffix"} 6'
        in out
    )
    assert (
        'rapid_mlx_suffix_decode_accept_ratio{family="gemma-4-12b-4bit",method="suffix"} 0.7500'
        in out
    )
    assert 'reason="cooldown"} 1' in out
    reset_global_counter()


def test_every_fallthrough_reason_gets_a_series():
    """MUTATION-KILL: dropping a reason from the render loop would hide
    exactly the diagnosis the breakdown exists for."""
    reset_global_counter()
    out = _render()
    for reason in (
        "batch_size",
        "uids_size",
        "non_greedy",
        "logits_processors",
        "no_draft",
        "cooldown",
        "non_trimmable_cache",
        "error",
    ):
        assert f'reason="{reason}"' in out, f"missing series for {reason}"


def test_reset_is_test_only_but_works():
    c = get_global_counter()
    c.record_verify(proposed=4, accepted=4)
    assert c.snapshot()["verify_steps"] >= 1
    reset_global_counter()
    assert c.snapshot()["verify_steps"] == 0


def test_label_value_is_escaped():
    """An alias containing a quote must not break the whole scrape — one
    malformed line makes Prometheus reject the entire exposition, not
    just that series. Validated by actually parsing it."""
    text_parser = pytest.importorskip(
        "prometheus_client.parser"
    ).text_string_to_metric_families

    reset_global_counter()
    lines = _render_suffix_decode_counters(SimpleNamespace(model_alias='we"ird\\alias'))
    out = "\n".join(lines)
    assert 'family="we\\"ird\\\\alias"' in out

    families = list(text_parser(out + "\n"))
    assert families, "rendered block did not parse as exposition text"
    seen = {
        s.labels["family"] for f in families for s in f.samples if "family" in s.labels
    }
    assert seen == {'we"ird\\alias'}, f"label round-tripped as {seen!r}"


def test_error_counter_is_exported():
    """MUTATION-KILL: ``record_error`` was collected but never rendered, so
    the one fallthrough that indicates a real fault stayed invisible."""
    reset_global_counter()
    c = get_global_counter()
    c.record_error()
    c.record_error()
    out = _render()
    assert "rapid_mlx_suffix_decode_errors_total" in out
    assert 'method="suffix"} 2' in out
    reset_global_counter()


def test_error_fallback_also_counts_as_a_fallthrough_step():
    """An error path still takes a plain forward. If it is not in the
    breakdown, verify + fallthrough stops reconciling with actual decode
    steps exactly when something is going wrong — which is when an
    operator is reading these numbers."""
    reset_global_counter()
    c = get_global_counter()
    c.record_verify(proposed=8, accepted=8)
    c.record_fallthrough("cooldown")
    c.record_error()
    snap = c.snapshot()
    assert snap["errors"] == 1
    assert snap["ft_error"] == 1
    # fallthrough_steps must equal the sum of its reasons.
    reasons = sum(v for k, v in snap.items() if k.startswith("ft_"))
    assert snap["fallthrough_steps"] == reasons == 2
    reset_global_counter()


def test_state_gauges_do_not_report_a_previous_requests_values():
    """A request that never reaches a successful verify must still publish
    its own state. Otherwise /metrics keeps showing the last request's
    width and backoff level, which is exactly backwards when diagnosing a
    request that is not drafting."""
    reset_global_counter()
    c = get_global_counter()
    # Request A ran hot and finished wide.
    c.set_state(current_k=8, backoff_level=3)
    # Request B starts; publish at creation, before any verify.
    c.set_state(current_k=2, backoff_level=0)
    snap = c.snapshot()
    assert snap["current_k"] == 2
    assert snap["backoff_level"] == 0
    reset_global_counter()


def test_state_gauges_return_to_rest_when_no_request_is_drafting():
    """Once the last suffix-decoded request is reaped the gauges describe
    nothing, so they must read at-rest rather than freeze on whatever the
    final request ended on.

    The failure this pins is a diagnostic one: a request that ends deep in
    a back-off window would otherwise leave ``backoff_level`` pegged, and
    an idle server would keep reporting a suffix decoder in trouble long
    after the traffic that caused it stopped. The scheduler calls
    ``set_state(_K_MIN, 0)`` from both reap paths once ``_uid_state`` is
    empty; this pins the value those paths must publish.
    """
    reset_global_counter()
    c = get_global_counter()
    # A request backs off hard, then finishes.
    c.set_state(current_k=8, backoff_level=5)
    assert c.snapshot()["backoff_level"] == 5
    # Last request reaped -> gauges go back to the at-rest pair.
    c.set_state(current_k=2, backoff_level=0)
    snap = c.snapshot()
    assert snap["current_k"] == 2, "idle width must not keep a finished request's K"
    assert snap["backoff_level"] == 0, "idle server must not report a back-off level"
    reset_global_counter()
