# SPDX-License-Identifier: Apache-2.0
"""Regression for #225 — startup ordering.

`_detect_native_tool_support()` reads `cfg.enable_auto_tool_choice` and
`cfg.tool_call_parser` via `get_config()`. If `_sync_config()` runs
*after* the detection call (the pre-fix layout), those fields are still
at their dataclass defaults (False, None), the guard short-circuits to
False, and `_engine.preserve_native_tool_format` is silently set to
False even though the configured parser supports native format.

Downstream symptom (per the bug report on Qwen3.5-9B-4bit and
Qwen3.6-35B-A3B-4bit-DWQ): assistant tool history gets serialised by
`api/utils.py::process_messages` as
`[Calling tool: name({json})]` text. The model sees prose-format
examples in context and mimics that pattern on subsequent turns —
streaming chunks emit the literal string instead of structured
`tool_calls`. Looks like a model failure but is a startup ordering
bug.
"""

from __future__ import annotations

import pytest


class _StubEngine:
    """Minimal stand-in for `BatchedEngine` — only the surface `load_model`
    actually accesses between construction and the model-registry add.

    This is intentionally explicit (not `MagicMock`) so that any future
    `load_model` change touching a new attribute fails LOUDLY with
    `AttributeError`, not silently with a fabricated MagicMock value.
    """

    is_mllm = False
    preserve_native_tool_format = False
    _tokenizer = None
    _tool_logits_processor_factory = None

    def __init__(self, *args, **kwargs):
        # Accept positional too in case `BatchedEngine.__init__` ever takes any.
        self.args = args
        self.kwargs = kwargs

    async def start(self):
        pass

    def generate_warmup(self):
        pass


@pytest.fixture(autouse=True)
def _reset_cfg_around_each_test():
    """Reset the ServerConfig singleton before AND after every test.

    `monkeypatch.setattr` on module globals is restored automatically, but
    the cfg singleton is a separate process-level object that must be
    explicitly reset on both sides — otherwise a mid-test failure leaks
    cfg state into the next test.
    """
    from vllm_mlx.config import reset_config

    reset_config()
    yield
    reset_config()


def test_load_model_enables_native_tool_format_when_parser_supports_it(monkeypatch):
    """After load_model() returns, the engine MUST reflect the parser's
    native-format support. Pre-fix this asserted False because cfg was
    unsynced when detection ran.
    """
    from vllm_mlx import server

    monkeypatch.setattr(server, "BatchedEngine", _StubEngine)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", True, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", "hermes", raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_model_alias", None, raising=False)

    server.load_model("mlx-community/Qwen3.5-9B-4bit")

    assert server._engine is not None
    # hermes parser sets SUPPORTS_NATIVE_TOOL_FORMAT = True; with the
    # ordering fix, detection sees the synced cfg and propagates that
    # to the engine.
    assert server._engine.preserve_native_tool_format is True


@pytest.mark.parametrize("served,expected", [("studio-assistant", True), (None, False)])
def test_load_model_tracks_explicit_served_model_name(monkeypatch, served, expected):
    """Issue #2353: ``load_model(..., served_model_name=...)`` must set the
    flag the readiness banner consumes, and leave it clear when no override
    is supplied — otherwise the banner would silently fall back to the
    catalog alias."""
    from vllm_mlx import server

    monkeypatch.setattr(server, "BatchedEngine", _StubEngine)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_model_alias", None, raising=False)
    monkeypatch.setattr(server, "_served_model_name_set", not expected, raising=False)

    server.load_model("mlx-community/Qwen3.5-9B-4bit", served_model_name=served)

    assert server._served_model_name_set is expected


def _stub_routing_globals(monkeypatch, server):
    """Neutralize the load_model globals that the routing tests don't exercise."""
    monkeypatch.setattr(server, "BatchedEngine", _StubEngine)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_model_alias", None, raising=False)


def test_load_model_materializes_config_before_hybrid_routing_probe(
    monkeypatch, caplog
):
    """BLOCKING (#1178 codex): the hybrid→text-only fallback probe reads the
    checkpoint config from the local cache. On a first-time uncached remote
    startup that config is absent, so the probe must run AFTER the model is
    materialized — otherwise a hybrid VLM probes "not hybrid" and is routed
    into the MLLM engine that cannot serve it (#352).

    Simulate exactly that race: the hybrid backbone is only "visible" once
    ``_ensure_routing_config`` has run. Assert the engine is still built for
    the text lane (``force_text=True``), proving the materialize-then-probe
    ordering holds — and that the automatic fallback is NOT reported as an
    explicit ``--no-mllm``.
    """
    import logging

    from vllm_mlx import server
    from vllm_mlx.api import utils as api_utils

    _stub_routing_globals(monkeypatch, server)

    state = {"materialized": False}

    def _fake_ensure(model_name):
        state["materialized"] = True

    monkeypatch.setattr(server, "_ensure_routing_config", _fake_ensure)
    # A multimodal checkpoint whose hybrid backbone only becomes visible once
    # its config has been materialized (i.e. after the download).
    monkeypatch.setattr(api_utils, "is_mllm_model", lambda name: True)
    monkeypatch.setattr(
        api_utils,
        "mllm_backbone_cache_mode",
        lambda name: "arrays" if state["materialized"] else None,
    )
    monkeypatch.setattr(api_utils, "mllm_hybrid_runtime_supported", lambda: False)

    with caplog.at_level(logging.INFO, logger="vllm_mlx.server"):
        server.load_model("some/uncached-hybrid-vlm-4bit")

    assert server._engine is not None
    # Materialization ran before the probe → hybrid detected → text lane.
    assert server._engine.kwargs.get("force_text") is True, (
        "auto-fallback to the text lane must fire once config is materialized"
    )
    # force_mllm must remain False (auto mode, no explicit flag).
    assert server._engine.kwargs.get("force_mllm") is False
    joined = " ".join(rec.message for rec in caplog.records)
    # Diagnostics attribute the reason to the automatic downgrade, NOT --no-mllm.
    assert "auto-downgraded to the text-only" in joined
    assert "Force text-only mode enabled via --no-mllm flag" not in joined


def test_load_model_genuine_vlm_stays_on_mllm_lane(monkeypatch):
    """A multimodal checkpoint with a NON-hybrid backbone (gemma-4 shape) must
    keep its MLLM routing — the auto-fallback fires only for hybrid backbones,
    so a working VLM is never downgraded."""
    from vllm_mlx import server
    from vllm_mlx.api import utils as api_utils

    _stub_routing_globals(monkeypatch, server)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda name: None)
    monkeypatch.setattr(api_utils, "is_mllm_model", lambda name: True)
    monkeypatch.setattr(api_utils, "mllm_backbone_cache_mode", lambda name: None)

    server.load_model("some/genuine-vlm-4bit")

    assert server._engine is not None
    # No downgrade → force_text stays False; BatchedEngine does its own MLLM
    # auto-detection from there.
    assert server._engine.kwargs.get("force_text") is False


def test_materialized_checkpoint_keeps_catalog_vision_memory_floor(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx import model_aliases, model_metadata, server
    from vllm_mlx.api import utils as api_utils
    from vllm_mlx.model_profile import ModelProfile

    monkeypatch.setattr(model_aliases, "resolve_model", lambda _name: "publisher/model")
    monkeypatch.setattr(
        model_aliases,
        "resolve_profile",
        lambda _name: ModelProfile(vision_min_memory_gb=32.0),
    )
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _name: None)
    monkeypatch.setattr(
        model_metadata,
        "read_model_metadata",
        lambda _name: SimpleNamespace(snapshot_dir="/cache/snapshots/revision"),
    )
    monkeypatch.setattr(api_utils, "is_mllm_model", lambda _name: True)
    monkeypatch.setattr(
        api_utils, "mllm_arch_unsupported_but_text_vendored", lambda _name: False
    )
    monkeypatch.setattr(api_utils, "mllm_backbone_cache_mode", lambda _name: "arrays")
    monkeypatch.setattr(api_utils, "mllm_hybrid_runtime_supported", lambda: True)
    monkeypatch.setattr(api_utils, "physical_ram_gb", lambda: 16.0)

    resolved = server._resolve_serving_checkpoint("qwen3.5-4b-4bit")

    assert resolved.load_path == "/cache/snapshots/revision"
    assert resolved.auto_text_fallback is True
    assert resolved.lane_reason == "vision_memory_insufficient"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auto_text_fallback", "lane_reason", "expected_force_text"),
    [
        (True, "vision_hybrid_runtime_unsupported", True),
        (False, "vision_hybrid_runtime_supported", False),
    ],
)
async def test_startup_and_runtime_use_identical_checkpoint_lane_contract(
    monkeypatch,
    scheduler_config_stub,
    auto_text_fallback,
    lane_reason,
    expected_force_text,
):
    """Startup and residency must hand the same resolved path/lane to engine."""
    pytest.importorskip("mlx")  # checkpoint-lane contract drives real mlx engine
    from vllm_mlx import server
    from vllm_mlx.model_profile import ModelProfile

    _stub_routing_globals(monkeypatch, server)
    calls = []
    resolved = server._ServingCheckpoint(
        model_path="publisher/model",
        load_path="/cache/snapshots/revision",
        auto_text_fallback=auto_text_fallback,
        lane_reason=lane_reason,
    )

    def resolve_once(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return resolved

    monkeypatch.setattr(server, "_resolve_serving_checkpoint", resolve_once)
    monkeypatch.setattr("vllm_mlx.model_aliases.resolve_profile", lambda _name: None)
    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config",
        lambda _name: ModelProfile(is_hybrid=False, experimental=True),
    )

    server.load_model("publisher/model")
    startup_kwargs = dict(server._engine.kwargs)
    runtime = await server._load_dynamic_resident_model("publisher/model", None)

    startup = server._model_registry.get_entry("publisher/model")
    assert startup.experimental is True
    assert runtime.experimental is True

    assert calls == [
        (
            "publisher/model",
            {
                "force_mllm": False,
                "force_text": False,
            },
        ),
        (
            "publisher/model",
            {"force_text": False},
        ),
    ]
    assert startup_kwargs["model_name"] == runtime.engine.kwargs["model_name"]
    assert (
        startup_kwargs["force_text"]
        == runtime.engine.kwargs["force_text"]
        is expected_force_text
    )
    assert (
        startup_kwargs["serving_lane_reason"]
        == runtime.engine.kwargs["serving_lane_reason"]
        == lane_reason
    )


def test_ensure_routing_config_raises_when_prefetch_does_not_materialize(monkeypatch):
    """BLOCKING (#1178 codex r4): ``_ensure_routing_config`` must NOT swallow a
    prefetch failure and let the caller route on a guess. If, after the
    prefetch attempt, the checkpoint config is still absent, the MLLM-vs-text
    probe would fall back to "not hybrid" and misroute a hybrid VLM into the
    crashing MLLM engine (#352). Assert it fails fast with an actionable error
    instead.
    """
    from vllm_mlx import cli as cli_mod
    from vllm_mlx import model_metadata as mm
    from vllm_mlx import server

    # Uncached remote repo id (not a local path → os.path.exists False).
    model = "some/uncached-and-unmaterializable-4bit"
    # Config NEVER becomes readable, even after the prefetch runs.
    monkeypatch.setattr(mm, "read_model_metadata", lambda name: None)
    called = {"prefetch": False}

    original_err = OSError("network unreachable")

    def _failing_prefetch(name):
        called["prefetch"] = True  # ran, but errored + put no config on disk
        raise original_err

    monkeypatch.setattr(cli_mod, "_ensure_model_downloaded", _failing_prefetch)

    with pytest.raises(RuntimeError) as excinfo:
        server._ensure_routing_config(model)

    assert called["prefetch"] is True, "prefetch must be attempted before failing"
    msg = str(excinfo.value)
    assert model in msg
    # Actionable: names the routing consequence and the escape hatches.
    assert "--no-mllm" in msg
    assert "#352" in msg
    # NIT (#1178 codex r5): the real prefetch cause is preserved via chaining,
    # not discarded.
    assert excinfo.value.__cause__ is original_err


def test_ensure_routing_config_warns_when_prefetch_errors_but_config_lands(
    monkeypatch, caplog
):
    """NIT (#1178 codex r5): if the prefetch raises a concrete error (auth /
    network / partial download) but config.json is present afterward, don't
    silently discard that error — resolve the lane (config is readable) but
    surface the original cause at WARNING so a later weight-load failure is
    attributable."""
    import logging

    from vllm_mlx import cli as cli_mod
    from vllm_mlx import model_metadata as mm
    from vllm_mlx import server

    state = {"materialized": False}
    monkeypatch.setattr(
        mm,
        "read_model_metadata",
        lambda name: object() if state["materialized"] else None,
    )

    def _partial_prefetch(name):
        # config.json lands, but the download errors out (weights incomplete).
        state["materialized"] = True
        raise OSError("connection reset mid-download")

    monkeypatch.setattr(cli_mod, "_ensure_model_downloaded", _partial_prefetch)

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        # Config is readable afterward → no raise.
        server._ensure_routing_config("some/partially-downloaded-4bit")

    joined = " ".join(rec.message for rec in caplog.records)
    assert "connection reset mid-download" in joined
    assert "partially downloaded" in joined


def test_ensure_routing_config_succeeds_when_prefetch_materializes(monkeypatch):
    """Happy path for the first-time uncached startup: config is absent, the
    prefetch materializes it, and ``_ensure_routing_config`` returns cleanly."""
    from vllm_mlx import cli as cli_mod
    from vllm_mlx import model_metadata as mm
    from vllm_mlx import server

    state = {"materialized": False}
    monkeypatch.setattr(
        mm,
        "read_model_metadata",
        lambda name: object() if state["materialized"] else None,
    )

    def _fake_prefetch(name):
        state["materialized"] = True

    monkeypatch.setattr(cli_mod, "_ensure_model_downloaded", _fake_prefetch)

    # Must not raise.
    server._ensure_routing_config("some/uncached-but-fetchable-4bit")
    assert state["materialized"] is True


def test_ensure_routing_config_skips_prefetch_when_config_already_readable(monkeypatch):
    """Warm cache / local checkpoint: config already readable → no download is
    attempted (keeps warm starts and the unit suite fully offline)."""
    from vllm_mlx import cli as cli_mod
    from vllm_mlx import model_metadata as mm
    from vllm_mlx import server

    monkeypatch.setattr(mm, "read_model_metadata", lambda name: object())

    def _must_not_run(name):  # pragma: no cover - asserted never called
        raise AssertionError("prefetch must be skipped when config is readable")

    monkeypatch.setattr(cli_mod, "_ensure_model_downloaded", _must_not_run)

    server._ensure_routing_config("mlx-community/Qwen3.5-9B-4bit")


def test_ensure_routing_config_propagates_disk_gate_systemexit(monkeypatch):
    """The intentional hard disk-space gate (``SystemExit``) from the prefetch
    must propagate unchanged — it is a fail-fast, not a swallowable hiccup."""
    from vllm_mlx import cli as cli_mod
    from vllm_mlx import model_metadata as mm
    from vllm_mlx import server

    monkeypatch.setattr(mm, "read_model_metadata", lambda name: None)

    def _disk_gate(name):
        raise SystemExit(1)

    monkeypatch.setattr(cli_mod, "_ensure_model_downloaded", _disk_gate)

    with pytest.raises(SystemExit):
        server._ensure_routing_config("some/uncached-4bit")


def test_load_model_infers_programmatic_max_tokens_explicit(monkeypatch):
    from vllm_mlx import server
    from vllm_mlx.config import get_config, reset_config

    monkeypatch.setattr(server, "BatchedEngine", _StubEngine)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_model_alias", None, raising=False)

    server.load_model("mlx-community/Qwen3.5-9B-4bit")
    cfg = get_config()
    assert cfg.default_max_tokens == 32768
    assert cfg.default_max_tokens_is_explicit is False

    reset_config()
    monkeypatch.setattr(server, "_engine", None, raising=False)

    server.load_model("mlx-community/Qwen3.5-9B-4bit", max_tokens=32)
    cfg = get_config()
    assert cfg.default_max_tokens == 32
    assert cfg.default_max_tokens_is_explicit is True

    reset_config()
    monkeypatch.setattr(server, "_engine", None, raising=False)

    server.load_model(
        "mlx-community/Qwen3.5-9B-4bit",
        max_tokens=4096,
        max_tokens_is_explicit=False,
    )
    cfg = get_config()
    assert cfg.default_max_tokens == 4096
    assert cfg.default_max_tokens_is_explicit is False


def test_load_model_mtp_kwarg_translates_to_scheduler_config(
    monkeypatch, scheduler_config_stub
):
    pytest.importorskip("mlx")  # mtp spec-decode path imports mlx.core (no-MLX job)
    from vllm_mlx import server

    monkeypatch.setattr(server, "BatchedEngine", _StubEngine)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_model_alias", None, raising=False)

    with pytest.warns(DeprecationWarning, match="load_model\\(mtp=True\\)"):
        server.load_model("mlx-community/Qwen3.5-9B-4bit", mtp=True)

    assert server._engine is not None
    cfg = server._engine.kwargs["scheduler_config"]
    assert cfg.spec_decode == "mtp"
    assert cfg.enable_mtp is True


def test_load_model_mtp_kwarg_rejects_conflicting_spec_decode(scheduler_config_stub):
    pytest.importorskip(
        "mlx"
    )  # scheduler/spec-decode path requires mlx (no-MLX coverage job)
    from vllm_mlx import server

    cfg = scheduler_config_stub()
    cfg.spec_decode = "suffix"

    with pytest.raises(ValueError, match="mtp=True.*spec_decode='suffix'"):
        server.load_model(
            "mlx-community/Qwen3.5-9B-4bit",
            scheduler_config=cfg,
            mtp=True,
        )


def test_load_model_mtp_kwarg_rejects_conflicting_suffix_config(
    scheduler_config_stub,
):
    pytest.importorskip(
        "mlx"
    )  # scheduler/spec-decode path requires mlx (no-MLX coverage job)
    from vllm_mlx import server

    with pytest.raises(ValueError, match="enable_suffix_decoding=True"):
        server.load_model(
            "mlx-community/Qwen3.5-9B-4bit",
            scheduler_config=scheduler_config_stub(enable_suffix_decoding=True),
            mtp=True,
        )


def test_load_model_mtp_kwarg_rejects_conflicting_dflash_config(
    scheduler_config_stub,
):
    pytest.importorskip(
        "mlx"
    )  # scheduler/spec-decode path requires mlx (no-MLX coverage job)
    from vllm_mlx import server

    with pytest.raises(ValueError, match="dflash_drafter_path"):
        server.load_model(
            "mlx-community/Qwen3.5-9B-4bit",
            scheduler_config=scheduler_config_stub(dflash_drafter_path="local/draft"),
            mtp=True,
        )


def test_load_model_response_cache_reconfigure_failure_forces_disabled(monkeypatch):
    """If ``configure_response_cache`` raises during ``load_model``, the
    fail-safe must NOT leave the PREVIOUS cache live under the NEW model
    (that would serve stale cross-model output). It rebinds the singleton to
    a FRESH disabled instance — an independent fail-closed path that does not
    reuse the possibly-wedged instance/method that just failed.

    Mutation-kill: remove the ``force_disable_response_cache()`` call from
    the except path → the pre-seeded, enabled cache object survives with its
    entries, so this fails.
    """
    from vllm_mlx import response_cache as rc
    from vllm_mlx import server

    # Pre-seed a live, populated cache — simulating the PREVIOUS model's
    # cache still holding entries when the reload begins.
    rc.reset_response_cache_for_tests()
    old_cache = rc.get_response_cache()
    old_cache.reconfigure(16)  # enabled
    ep = old_cache.current_epoch()
    old_cache.put("prev-model-key", "prev-model-output", ep)
    assert old_cache.enabled is True
    assert old_cache.snapshot()["entries"] == 1

    # Make the load-path reconfigure blow up (e.g. a parse error on the
    # resolved capacity, or any internal failure).
    def _boom(_capacity):
        raise RuntimeError("simulated reconfigure failure")

    monkeypatch.setattr(rc, "configure_response_cache", _boom)

    monkeypatch.setattr(server, "BatchedEngine", _StubEngine)
    monkeypatch.setattr(server, "_engine", None, raising=False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_model_alias", None, raising=False)

    # load_model must NOT raise (best-effort), but must force the cache safe.
    server.load_model("mlx-community/Qwen3.5-9B-4bit")

    new_cache = rc.get_response_cache()
    # The fail-safe rebinds to a BRAND-NEW instance — not the wedged old one.
    assert new_cache is not old_cache, (
        "reconfigure failure did not rebind the singleton — the old "
        "(possibly wedged) instance is still live"
    )
    assert new_cache.enabled is False, (
        "reconfigure failure left the cache ENABLED — it could serve stale "
        "cross-model completions"
    )
    assert new_cache.snapshot()["entries"] == 0, (
        "reconfigure failure left the PREVIOUS model's entries live"
    )
    # The old object's entries are irrelevant now that it is unreferenced by
    # the singleton, but confirm the live singleton exposes none.
    assert new_cache.capacity == 0

    rc.reset_response_cache_for_tests()


def test_load_model_mtp_kwarg_rejects_legacy_optimistic_config(
    scheduler_config_stub,
):
    """PR #1050 hard-reject: server.load_model(mtp=True) with a
    scheduler_config carrying ``mtp_optimistic=True`` must fail because
    the direct mutation of ``spec_decode='mtp'`` below would bypass
    ``__post_init__`` and silently drop the flag under the vendored path."""
    pytest.importorskip(
        "mlx"
    )  # scheduler/spec-decode path requires mlx (no-MLX coverage job)
    from vllm_mlx import server

    # SchedulerConfig(mtp_optimistic=True) alone (spec_decode="none") is
    # legal — the reject is triggered only once mtp=True elevates the
    # config into the unified spec-decode interface path.
    cfg = scheduler_config_stub(mtp_optimistic=True)

    with pytest.raises(
        ValueError, match="mtp_optimistic.*not supported under the unified"
    ):
        server.load_model(
            "mlx-community/Qwen3.5-9B-4bit",
            scheduler_config=cfg,
            mtp=True,
        )


def test_detect_native_tool_support_requires_synced_config(monkeypatch):
    """Contract test for the ordering invariant: detection short-circuits
    to False when cfg has not been synced yet, so callers MUST run
    `_sync_config()` first.
    """
    from vllm_mlx import server
    from vllm_mlx.config import get_config

    monkeypatch.setattr(server, "_enable_auto_tool_choice", True, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", "hermes", raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_engine", None, raising=False)

    cfg = get_config()
    assert cfg.enable_auto_tool_choice is False
    assert cfg.tool_call_parser is None
    assert server._detect_native_tool_support() is False

    server._sync_config()

    cfg = get_config()
    assert cfg.enable_auto_tool_choice is True
    assert cfg.tool_call_parser == "hermes"
    assert server._detect_native_tool_support() is True


def test_sync_config_is_idempotent(monkeypatch):
    """`_sync_config()` is called twice in `load_model` (early before native
    tool detection, late after the model registry add). Both calls must
    leave cfg in the same state — if the function ever grows non-idempotent
    side effects (counter increments, callback fires, cache invalidations),
    the late re-sync becomes a latent bug.
    """
    from vllm_mlx import server
    from vllm_mlx.config import get_config

    monkeypatch.setattr(server, "_enable_auto_tool_choice", True, raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", "hermes", raising=False)
    monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
    monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
    monkeypatch.setattr(server, "_engine", None, raising=False)

    server._sync_config()
    cfg = get_config()
    snapshot = {
        "engine": cfg.engine,
        "model_name": cfg.model_name,
        "model_alias": cfg.model_alias,
        "model_path": cfg.model_path,
        "enable_auto_tool_choice": cfg.enable_auto_tool_choice,
        "tool_call_parser": cfg.tool_call_parser,
        "tool_parser_instance": cfg.tool_parser_instance,
        "enable_tool_logits_bias": cfg.enable_tool_logits_bias,
        "reasoning_parser": cfg.reasoning_parser,
        "reasoning_parser_name": cfg.reasoning_parser_name,
        "mcp_manager": cfg.mcp_manager,
        "model_registry": cfg.model_registry,
    }

    server._sync_config()
    cfg2 = get_config()

    for k, v in snapshot.items():
        assert getattr(cfg2, k) == v, f"_sync_config() not idempotent on cfg.{k}"


def test_sync_config_propagates_mcp_manager(monkeypatch):
    """After init_mcp() sets the global _mcp_manager, _sync_config() must
    copy it into cfg so MCP routes read a live manager instead of None.

    Regression for #986: load_model() stamped cfg.mcp_manager = None before
    lifespan init_mcp() ran, and no later _sync_config() updated it.
    """
    from unittest.mock import MagicMock

    from vllm_mlx import server
    from vllm_mlx.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg, "mcp_manager", None, raising=False)
    monkeypatch.setattr(cfg, "mcp_executor", None, raising=False)

    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_mcp_executor", None, raising=False)

    server._sync_config()
    assert get_config().mcp_manager is None

    mock_manager = MagicMock()
    mock_executor = MagicMock()
    monkeypatch.setattr(server, "_mcp_manager", mock_manager, raising=False)
    monkeypatch.setattr(server, "_mcp_executor", mock_executor, raising=False)

    server._sync_config()
    cfg = get_config()
    assert cfg.mcp_manager is mock_manager
    assert cfg.mcp_executor is mock_executor


def test_sync_config_preserves_unrelated_config_on_mcp_update(monkeypatch):
    """Updating MCP globals and re-syncing must not clobber unrelated cfg.

    Regression for the fix to #986: the post-init_mcp _sync_config() runs
    late in startup; if it overwrote fields that were intentionally set
    earlier, it would introduce ordering bugs.
    """
    from unittest.mock import MagicMock

    from vllm_mlx import server
    from vllm_mlx.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg, "mcp_manager", None, raising=False)
    monkeypatch.setattr(cfg, "mcp_executor", None, raising=False)

    monkeypatch.setattr(server, "_model_name", "model-a", raising=False)
    monkeypatch.setattr(server, "_tool_call_parser", "hermes", raising=False)
    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_mcp_executor", None, raising=False)

    server._sync_config()
    cfg_before = get_config()
    assert cfg_before.model_name == "model-a"
    assert cfg_before.tool_call_parser == "hermes"
    assert cfg_before.mcp_manager is None

    mock_manager = MagicMock()
    monkeypatch.setattr(server, "_mcp_manager", mock_manager, raising=False)

    server._sync_config()
    cfg_after = get_config()
    assert cfg_after.mcp_manager is mock_manager
    assert cfg_after.model_name == "model-a"
    assert cfg_after.tool_call_parser == "hermes"


async def test_init_mcp_syncs_config_into_cfg(monkeypatch):
    """init_mcp() must publish the initialized manager/executor to cfg.

    Regression for #986: this guards against deleting the `_sync_config()`
    call inside init_mcp() and re-introducing the stale cfg bug.
    """
    from unittest.mock import AsyncMock, MagicMock

    import vllm_mlx.mcp as mcp_module
    from vllm_mlx import server
    from vllm_mlx.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg, "mcp_manager", None, raising=False)
    monkeypatch.setattr(cfg, "mcp_executor", None, raising=False)

    mock_manager = MagicMock()
    mock_manager.start = AsyncMock()
    mock_manager.get_all_tools.return_value = []
    mock_executor = MagicMock()

    mock_config = MagicMock()
    mock_config.allowed_high_risk_tools = []
    # ``_start_mcp`` now loads tolerantly and reads ``config.rejected``; the
    # mock has to accept the ``tolerant`` kwarg and expose an iterable.
    mock_config.rejected = []

    monkeypatch.setattr(
        mcp_module, "load_mcp_config", lambda _path, tolerant=False: mock_config
    )
    monkeypatch.setattr(mcp_module, "MCPClientManager", lambda _cfg: mock_manager)
    monkeypatch.setattr(mcp_module, "ToolExecutor", lambda _mgr: mock_executor)
    monkeypatch.setattr(mcp_module, "set_sandbox", MagicMock())

    monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
    monkeypatch.setattr(server, "_mcp_executor", None, raising=False)

    await server.init_mcp("/tmp/fake-mcp.json")

    cfg = get_config()
    assert cfg.mcp_manager is mock_manager
    assert cfg.mcp_executor is mock_executor
