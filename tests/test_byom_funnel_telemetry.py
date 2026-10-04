# SPDX-License-Identifier: Apache-2.0
"""BYOM funnel telemetry: preflight outcome, suggestion, support request,
suggestion acceptance and ``rapid-mlx import`` — all closed registry values."""

from __future__ import annotations

import argparse
import json
import os
import stat
from contextlib import nullcontext
from pathlib import Path

import pytest

import rapid_mlx
from rapid_mlx.byom import imports as im
from rapid_mlx.byom import preflight as pf
from rapid_mlx.telemetry import (
    byom_funnel,
    consent_runtime,
    model_events,
    posthog_sender,
    registry,
    state,
)
from rapid_mlx.telemetry import track as track_module
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts
from tests.test_byom_preflight import (  # noqa: F401 - fixtures
    GIB,
    MLX_CONFIG,
    _args,
    _info,
    hook,
    uncached,
)

REAL_EMIT_REJECTION = pf._emit_rejection
STAMP = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
FACTS = PlatformFacts(
    os="darwin",
    os_version="25.3",
    arch="arm64",
    chip="m3-pro",
    memory_gb=36,
    python_version="3.11",
)


@pytest.fixture(autouse=True)
def telemetry_on(monkeypatch, tmp_path):
    """An official build with uploads allowed; events captured after validation."""
    for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.5")
    monkeypatch.setattr(track_module.common_props, "read_platform_facts", lambda: FACTS)
    monkeypatch.setattr(
        state, "get_or_create_client_id", lambda: "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
    )
    monkeypatch.setattr(
        state, "session_id", lambda: "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
    )
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(posthog_sender.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(
        model_events, "_submit_model_served", lambda callback: (callback(), True)[1]
    )
    monkeypatch.setattr(track_module.store, "days_since_first_run_bucket", lambda: None)
    monkeypatch.setattr(track_module.store, "note_model_served", lambda model: 0)
    events: list[tuple[str, dict[str, object]]] = []

    def enqueue(accepted) -> bool:
        events.append((accepted.event, dict(accepted.props)))
        return True

    monkeypatch.setattr(track_module, "_enqueue_accepted", enqueue)
    track_module._reset_for_tests()
    model_events._reset_for_tests()
    byom_funnel._reset_for_tests()
    yield events
    track_module._reset_for_tests()
    model_events._reset_for_tests()
    byom_funnel._reset_for_tests()


def _ledger() -> Path:
    return byom_funnel._ledger_path()


def _payload(events, name):
    found = [props for event, props in events if event == name]
    assert len(found) == 1, events
    return found[0]


# --------------------------------------------------------------------------
# Registry shape


def test_registry_accepts_the_funnel_props_and_scopes_refusal_ones():
    served = {
        "model": "<custom>",
        "model_type": "other",
        "auto_selected": False,
        "quant": "4bit",
        "preflight": "passed",
        "via_suggestion": True,
    }
    assert registry.validate("model_served", served) == served
    failed = {
        "error_class": "unsupported_format",
        "preflight": "refused",
        "suggestion": "mlx_build",
        "support_request": "declined",
    }
    assert registry.validate("model_pull_failed", failed) == failed
    assert registry.validate("model_serve_failed", failed) == failed
    # Refusal-only context outside a refusal drops the whole event.
    assert (
        registry.validate(
            "model_serve_failed",
            {"error_class": "other", "preflight": "passed", "suggestion": "none"},
        )
        is None
    )
    # Never on the success twin, never a free value.
    assert registry.validate("model_served", {**served, "suggestion": "none"}) is None
    assert registry.validate("model_pulled", {"preflight": "o/r"}) is None
    imported = {"model": "<local>", "quant": "4bit"}
    assert registry.validate("model_imported", imported) == imported
    assert registry.validate(
        "model_import_failed", {"error_class": "smoke_failed"}
    ) == {"error_class": "smoke_failed"}


def test_funnel_constants_match_the_registry_enums():
    enums = registry.load_registry()["enums"]
    assert set(enums["byom_preflight"]["values"]) == byom_funnel.PREFLIGHT_OUTCOMES
    assert set(enums["byom_suggestion"]["values"]) == byom_funnel.SUGGESTION_KINDS
    assert (
        set(enums["support_request_outcome"]["values"])
        == byom_funnel.SUPPORT_REQUEST_OUTCOMES
    )
    from rapid_mlx.byom import support_request as sr

    assert {
        sr.NOT_ELIGIBLE,
        sr.NON_INTERACTIVE,
        sr.DECLINED,
        sr.NO_ANSWER,
        sr.SENT,
        sr.BUSY,
        sr.UNREACHABLE,
    } == byom_funnel.SUPPORT_REQUEST_OUTCOMES


# --------------------------------------------------------------------------
# Context scoping


def test_props_only_for_this_invocations_model():
    byom_funnel.begin(["my-alias", "Org/Repo"])
    byom_funnel.set_preflight("passed")
    assert byom_funnel.props_for("org/repo", failed=False) == {"preflight": "passed"}
    assert byom_funnel.props_for("my-alias", failed=True) == {"preflight": "passed"}
    # A secondary lane / later swap never inherits the context.
    assert byom_funnel.props_for("other/model", failed=False) == {}
    assert byom_funnel.props_for(None, failed=False) == {}
    assert byom_funnel.props_for(" ", failed=False) == {}
    byom_funnel.set_preflight("bogus")
    assert byom_funnel.props_for("org/repo", failed=False) == {"preflight": "passed"}


def test_refusal_context_only_on_failure_events():
    byom_funnel.begin(["o/r"])
    byom_funnel.note_refusal(suggestion="catalog", support_request="sent")
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "refused"}
    assert byom_funnel.props_for("o/r", failed=True) == {
        "preflight": "refused",
        "suggestion": "catalog",
        "support_request": "sent",
    }
    byom_funnel.begin(["o/r"])
    byom_funnel.note_refusal(suggestion="weird", support_request=None)
    assert byom_funnel.props_for("o/r", failed=True) == {"preflight": "refused"}


def test_local_paths_match_by_real_path(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    byom_funnel.begin([str(model)])
    byom_funnel.set_preflight("passed")
    assert byom_funnel.props_for(f"{tmp_path}/./model", failed=False) == {
        "preflight": "passed"
    }


def test_funnel_never_raises(monkeypatch):
    monkeypatch.setattr(byom_funnel, "_norm", lambda ref: 1 / 0)
    byom_funnel.begin(["o/r"])
    assert byom_funnel.props_for("o/r", failed=False) == {}
    byom_funnel.note_refusal(
        suggestion="none", support_request="sent", suggested_refs=["x"]
    )


def test_norm_survives_a_failing_exists(monkeypatch):
    monkeypatch.setattr(byom_funnel.os.path, "exists", lambda p: 1 / 0)
    assert byom_funnel._norm("Some/Repo") == "some/repo"


# --------------------------------------------------------------------------
# Suggestion ledger (via_suggestion)


def test_suggestion_round_trip_is_one_shot_and_hashed():
    byom_funnel.begin(["bad/gguf"])
    byom_funnel.note_refusal(
        suggestion="mlx_build",
        support_request="declined",
        suggested_refs=["mlx-community/Good-4bit", "qwen3.5-4b"],
    )
    raw = _ledger().read_text()
    assert "mlx-community" not in raw.lower() and "qwen" not in raw
    assert len(json.loads(raw)) == 2
    assert stat.S_IMODE(os.stat(_ledger()).st_mode) == 0o600

    byom_funnel.begin(["qwen3.5-4b", "mlx-community/Qwen3.5-4B-4bit"])
    props = byom_funnel.props_for("qwen3.5-4b", failed=False)
    assert props == {"via_suggestion": True}
    assert len(json.loads(_ledger().read_text())) == 2, "building props never consumes"
    byom_funnel.note_emitted(props, True)
    assert len(json.loads(_ledger().read_text())) == 1
    # Consumed: the next run of the same model is not "via suggestion".
    byom_funnel.begin(["qwen3.5-4b"])
    assert byom_funnel.props_for("qwen3.5-4b", failed=False) == {}


def test_no_ledger_when_uploads_are_not_allowed(monkeypatch):
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: False)
    byom_funnel.begin(["bad/gguf"])
    byom_funnel.note_refusal(
        suggestion="catalog", support_request="declined", suggested_refs=["a-b"]
    )
    assert not _ledger().exists()
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    byom_funnel.note_refusal(
        suggestion="catalog", support_request="declined", suggested_refs=["a-b"]
    )
    assert _ledger().exists()
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: False)
    byom_funnel.begin(["a-b"])
    assert byom_funnel.props_for("a-b", failed=False) == {}
    # Not consumed while uploads are off.
    assert len(json.loads(_ledger().read_text())) == 1


def test_upload_check_failure_is_treated_as_off(monkeypatch):
    monkeypatch.setattr(track_module, "_upload_allowed", lambda: 1 / 0)
    assert byom_funnel._upload_allowed() is False


def test_ledger_expires_and_is_capped(monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(byom_funnel, "_clock", lambda: now[0])
    byom_funnel._record_suggestions(["old/one"])
    now[0] += byom_funnel.SUGGESTION_WINDOW_SECONDS
    assert byom_funnel._consume_suggestion(["old/one"]) is False
    byom_funnel._record_suggestions([f"r/{i}" for i in range(40)])
    assert len(json.loads(_ledger().read_text())) == byom_funnel._SUGGESTION_MAX_KEYS
    byom_funnel._record_suggestions([None, ""])
    now[0] = float("nan")
    assert byom_funnel._consume_suggestion(["r/1"]) is False


def test_ledger_failures_fail_closed(monkeypatch):
    from rapid_mlx.telemetry import model_events as me
    from rapid_mlx.telemetry import server_start

    assert byom_funnel._consume_suggestion([]) is False
    byom_funnel._record_suggestions(["x/y"])
    monkeypatch.setattr(me, "_acquire_serve_failed_lock", lambda fd: False)
    assert byom_funnel._consume_suggestion(["x/y"]) is False
    monkeypatch.setattr(me, "_acquire_serve_failed_lock", lambda fd: 1 / 0)
    assert byom_funnel._consume_suggestion(["x/y"]) is False
    monkeypatch.setattr(server_start, "_prepare_state_dir", lambda path: False)
    assert byom_funnel._mutate_ledger(lambda recent, now: True) is False
    monkeypatch.setattr(server_start, "_prepare_state_dir", lambda path: 1 / 0)
    assert byom_funnel._mutate_ledger(lambda recent, now: True) is False


# --------------------------------------------------------------------------
# The preflight hook end to end (real emitters, registry-validated payloads)


@pytest.fixture
def real_emit(monkeypatch):
    monkeypatch.setattr(pf, "_emit_rejection", REAL_EMIT_REJECTION)


def test_refusal_reports_suggestion_and_support_outcome(
    hook, real_emit, monkeypatch, telemetry_on
):
    from rapid_mlx.byom import alternatives, support_request

    def suggest(*a, targets=None, **kw):
        targets.append("mlx-community/Good-4bit")
        return ["  hint"], True

    monkeypatch.setattr(alternatives, "suggest", suggest)
    monkeypatch.setattr(support_request, "offer", lambda *a: "declined")
    with pytest.raises(SystemExit):
        hook(_info({"m-Q4_K_M.gguf": 7 * GIB}))
    props = _payload(telemetry_on, "model_serve_failed")
    assert props["error_class"] == "unsupported_format"
    assert props["failure_stage"] == "preflight"
    assert props["preflight"] == "refused"
    assert props["suggestion"] == "mlx_build"
    assert props["support_request"] == "declined"
    assert "via_suggestion" not in props
    assert _ledger().exists()

    # Accepting the suggestion: a later pull of the suggested repo.
    telemetry_on.clear()
    byom_funnel.begin(["mlx-community/Good-4bit"])
    model_events.emit_model_pulled("mlx-community/Good-4bit", "hf", GIB)
    assert _payload(telemetry_on, "model_pulled")["via_suggestion"] is True


@pytest.mark.parametrize(
    ("hints", "found", "kind"),
    [([], False, "none"), (["  x"], False, "catalog")],
)
def test_suggestion_kind(
    hook, real_emit, monkeypatch, telemetry_on, hints, found, kind
):
    hook.hints["value"] = (hints, found)
    with pytest.raises(SystemExit):
        hook(
            _info({"m-Q4_K_M.gguf": 7 * GIB}),
            _args(command="pull", bits=None, format=None),
        )
    props = _payload(telemetry_on, "model_pull_failed")
    assert props["preflight"] == "refused"
    assert props["suggestion"] == kind
    # The fixture's offer() stub returns no outcome: the key is omitted.
    assert "support_request" not in props


def test_pass_no_verdict_and_unchecked_outcomes(hook, monkeypatch, tmp_path):
    hook(_info({"model.safetensors": 5 * GIB}, config=dict(MLX_CONFIG)))
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "passed"}
    monkeypatch.setattr(pf, "inspect_hub", lambda ref: None)
    hook(None)
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "no_verdict"}
    hook(None, _args(no_preflight=True))
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "skipped"}
    hook(None, _args(model=str(tmp_path), no_preflight=True))
    assert byom_funnel.props_for(str(tmp_path), failed=False) == {
        "preflight": "skipped"
    }
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    hook(None)
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "no_verdict"}
    monkeypatch.delenv("HF_HUB_OFFLINE")
    from rapid_mlx import _download_gate

    monkeypatch.setattr(_download_gate, "is_repo_cached", lambda name: True)
    hook(None)
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "cached"}
    # Catalog models carry no preflight outcome at all.
    hook(None, _args(model="qwen3.5-4b"))
    assert byom_funnel.props_for("qwen3.5-4b", failed=False) == {}


def test_pull_format_gguf_is_a_selector_pull(hook, real_emit, telemetry_on):
    # Selector pulls carry no preflight outcome (events.json byom_preflight).
    with pytest.raises(SystemExit):
        hook(None, _args(command="pull", bits=None, format="gguf"))
    props = _payload(telemetry_on, "model_pull_failed")
    assert props["error_class"] == "unsupported_format"
    assert "preflight" not in props and "suggestion" not in props


def test_a_vanished_local_path_still_matches(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    byom_funnel.begin([str(model)])
    byom_funnel.set_preflight("passed")
    model.rmdir()
    assert byom_funnel.props_for(str(model), failed=True) == {"preflight": "passed"}


def test_a_failed_consume_is_retried_by_the_next_event(monkeypatch):
    _seed_suggestion("q-4bit")
    real = byom_funnel._consume_suggestion
    monkeypatch.setattr(byom_funnel, "_consume_suggestion", lambda refs: False)
    props = byom_funnel.props_for("q-4bit", failed=False)
    byom_funnel.note_emitted(props, True)
    assert len(json.loads(_ledger().read_text())) == 1
    monkeypatch.setattr(byom_funnel, "_consume_suggestion", real)
    byom_funnel.note_emitted(props, True)
    assert json.loads(_ledger().read_text()) == {}
    byom_funnel.note_emitted(props, True)  # already consumed: a no-op


def test_served_and_failed_serve_carry_the_context(telemetry_on):
    byom_funnel.begin(["o/r"])
    byom_funnel.set_preflight("passed")
    model_events.emit_model_served(None, "o/r", False)
    assert _payload(telemetry_on, "model_served")["preflight"] == "passed"
    model_events.emit_model_serve_failed(RuntimeError("x"), alias_or_path="o/r")
    assert _payload(telemetry_on, "model_serve_failed")["preflight"] == "passed"
    model_events.emit_model_pull_failed(TimeoutError(), model_ref="o/r")
    assert _payload(telemetry_on, "model_pull_failed")["preflight"] == "passed"
    model_events.emit_model_pull_failed(TimeoutError())
    assert [p for e, p in telemetry_on if e == "model_pull_failed"][-1] == {
        "error_class": "network"
    }


# --------------------------------------------------------------------------
# rapid-mlx import


def test_import_emitters(telemetry_on, tmp_path):
    model_events.emit_model_imported(str(tmp_path), 4)
    assert _payload(telemetry_on, "model_imported") == {
        "model": "<local>",
        "quant": "4bit",
    }
    model_events.emit_model_import_failed("convert_failed", "o/r", 3)
    assert _payload(telemetry_on, "model_import_failed") == {
        "model": "<custom>",
        "quant": "3bit",
        "error_class": "convert_failed",
    }
    telemetry_on.clear()
    model_events.emit_model_import_failed("free text", None, True)
    assert _payload(telemetry_on, "model_import_failed") == {"error_class": "other"}
    # No identity, no event: model_imported requires both props.
    model_events.emit_model_imported(None, 4)
    model_events.emit_model_imported("o/r", None)
    assert not [e for e, _ in telemetry_on if e == "model_imported"]


@pytest.fixture
def import_events(monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(
        model_events,
        "emit_model_import_failed",
        lambda cls, source, bits: seen.append(("failed", cls, source, bits)),
    )
    monkeypatch.setattr(
        model_events,
        "emit_model_imported",
        lambda source, bits: seen.append(("ok", source, bits)),
    )
    return seen


@pytest.fixture
def run_import(monkeypatch, import_events):
    from tests.test_byom_imports import SUPPORTED, _insp

    monkeypatch.setattr(im, "convertible_model_types", lambda: SUPPORTED)
    monkeypatch.setattr(pf, "physical_ram_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(im, "revision_cached", lambda repo, insp: False)
    monkeypatch.setattr(im, "_free_bytes", lambda path: 1 << 50)
    monkeypatch.setattr(im, "_same_filesystem", lambda a, b: True)
    state_ = {"insp": _insp(), "execute": None}
    monkeypatch.setattr(pf, "inspect_hub", lambda ref: state_["insp"])

    def _execute(plan, *, force, spinner_factory):
        behaviour = state_["execute"]
        if callable(behaviour):
            return behaviour(plan)
        return Path("/imports") / plan.name, behaviour == "reused"

    monkeypatch.setattr(im, "execute", _execute)

    def run(source="o/My-FT-bf16", bits=4):
        args = argparse.Namespace(source=source, quantize=bits, name=None, force=False)
        im.import_command(args, spinner_factory=lambda label: nullcontext())

    run.state = state_
    return run


def test_import_command_success_and_reuse(run_import, import_events):
    run_import()
    assert import_events == [("ok", "o/My-FT-bf16", 4)]
    run_import.state["execute"] = "reused"
    run_import()
    assert import_events == [("ok", "o/My-FT-bf16", 4)]


def test_import_command_failure_classes(run_import, import_events, monkeypatch):
    with pytest.raises(SystemExit):
        run_import("not-a-repo")
    run_import.state["insp"] = None
    with pytest.raises(SystemExit):
        run_import()
    from tests.test_byom_imports import _insp

    run_import.state["insp"] = _insp(
        config={"model_type": "qwen3", "quantization": {"bits": 4}}
    )
    with pytest.raises(SystemExit):
        run_import()
    run_import.state["insp"] = _insp()

    def interrupted(plan):
        raise KeyboardInterrupt

    run_import.state["execute"] = interrupted
    with pytest.raises(SystemExit):
        run_import()

    def smoke(plan):
        raise im.ImportRefusedError("smoke step failed", error_class="smoke_failed")

    run_import.state["execute"] = smoke
    with pytest.raises(SystemExit):
        run_import()

    def crash(plan):
        im._steps.current = "download_failed"
        raise ConnectionError("hub down")

    run_import.state["execute"] = crash
    with pytest.raises(ConnectionError):
        run_import()
    assert [event[1] for event in import_events] == [
        "invalid_ref",
        "metadata_unavailable",
        "already_quantized",
        "interrupted",
        "smoke_failed",
        "download_failed",
    ]


def test_import_refusal_classes_are_closed():
    allowed = set(registry.load_registry()["enums"]["import_error_class"]["values"])
    assert im.ImportRefusedError("x").error_class == "other"
    source = Path(im.__file__).read_text(encoding="utf-8")
    import re

    used = set(re.findall(r'error_class="([a-z_]+)"', source))
    used |= {f"{step}_failed" for step in ("convert", "smoke")}
    used |= {"download_failed", "invalid_ref", "interrupted"}
    assert used <= allowed, used - allowed


def test_execute_tags_a_download_failure(tmp_path, monkeypatch):
    from tests.test_byom_imports import _plan

    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    im._steps.current = "other"

    def download(plan):
        raise ConnectionError("hub down")

    with pytest.raises(ConnectionError):
        im.execute(
            _plan(),
            force=False,
            spinner_factory=lambda label: nullcontext(),
            download=download,
        )
    assert im._steps.current == "download_failed"

    def worker(step, *rest):
        raise im.ImportRefusedError(f"{step} failed", error_class=f"{step}_failed")

    with pytest.raises(im.ImportRefusedError) as exc:
        im.execute(
            _plan(),
            force=False,
            spinner_factory=lambda label: nullcontext(),
            download=lambda plan: str(tmp_path),
            run_worker=worker,
        )
    assert exc.value.error_class == "convert_failed"
    assert im._steps.current == "other"


def test_run_worker_failures_are_typed(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "boom"),
    )
    with pytest.raises(im.ImportRefusedError) as exc:
        im._run_worker("smoke", "/out")
    assert exc.value.error_class == "smoke_failed"

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(im.ImportRefusedError) as exc:
        im._run_worker("smoke", "/out")
    assert exc.value.error_class == "smoke_failed"


def test_a_run_that_emits_nothing_keeps_the_suggestion():
    byom_funnel.begin(["bad/gguf"])
    byom_funnel.note_refusal(
        suggestion="catalog", support_request="declined", suggested_refs=["q-4bit"]
    )
    byom_funnel.begin(["q-4bit"])  # e.g. a cached pull: no lifecycle event
    assert len(json.loads(_ledger().read_text())) == 1
    byom_funnel.begin(["q-4bit"])
    props = byom_funnel.props_for("q-4bit", failed=True)
    assert props == {"via_suggestion": True}
    # A rejected / unsent event never spends it.
    byom_funnel.note_emitted(props, False)
    byom_funnel.note_emitted(None, True)
    byom_funnel.note_emitted({"preflight": "passed"}, True)
    assert len(json.loads(_ledger().read_text())) == 1
    byom_funnel.note_emitted(props, True)
    assert json.loads(_ledger().read_text()) == {}
    # Consumed once; later events of the same run still report it.
    assert byom_funnel.props_for("q-4bit", failed=False) == {"via_suggestion": True}
    byom_funnel.note_emitted(props, True)


def test_note_emitted_never_raises(monkeypatch):
    monkeypatch.setattr(byom_funnel, "_consume_suggestion", lambda refs: 1 / 0)
    byom_funnel.begin(["x"])
    byom_funnel._context["via_suggestion"] = True
    byom_funnel.note_emitted({"via_suggestion": True}, True)


def _seed_suggestion(ref):
    byom_funnel.begin(["bad/gguf"])
    byom_funnel.note_refusal(
        suggestion="catalog", support_request="declined", suggested_refs=[ref]
    )
    byom_funnel.begin([ref])


def test_every_lifecycle_emitter_consumes_only_when_accepted(monkeypatch, telemetry_on):
    # Opted out between preflight and emit: the event is refused, nothing spent.
    _seed_suggestion("q-4bit")
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: False)
    model_events.emit_model_pulled("q-4bit", "hf", GIB)
    assert telemetry_on == []
    assert len(json.loads(_ledger().read_text())) == 1
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    for emit in (
        lambda: model_events.emit_model_pulled("q-4bit", "hf", GIB),
        lambda: model_events.emit_model_pull_failed(TimeoutError(), model_ref="q-4bit"),
        lambda: model_events.emit_model_served(None, "q-4bit", False),
        lambda: model_events.emit_model_serve_failed(
            RuntimeError("x"), alias_or_path="q-4bit"
        ),
    ):
        _seed_suggestion("q-4bit")
        model_events._reset_for_tests()
        telemetry_on.clear()
        emit()
        assert telemetry_on and telemetry_on[0][1]["via_suggestion"] is True
        assert json.loads(_ledger().read_text()) == {}


def test_full_config_confirmation_counts_as_passed(hook):
    config = {
        "model_type": "mystery",
        "architectures": ["MysteryForCausalLM"],
        "tokenizer_config": {"chat_template": "{{ x }}"},
    }
    info = _info({"model.safetensors": GIB}, config=config)
    hook(info, full_config=None)
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "no_verdict"}
    hook(info, full_config={**config, "model_type": "qwen3"})
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "passed"}


def test_offline_never_probes_the_cache(hook, monkeypatch):
    from rapid_mlx import _download_gate

    probed = []
    monkeypatch.setattr(
        _download_gate, "is_repo_cached", lambda name: probed.append(name) or True
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    hook(None)
    assert byom_funnel.props_for("o/r", failed=False) == {"preflight": "no_verdict"}
    assert probed == []


def test_unreadable_import_source_classes(run_import, import_events, tmp_path):
    run_import.state["insp"] = None
    for source in ("foo/bar/baz", "o/My-FT-bf16"):
        with pytest.raises(SystemExit):
            run_import(source)
    assert [event[1] for event in import_events] == [
        "invalid_ref",
        "metadata_unavailable",
    ]
