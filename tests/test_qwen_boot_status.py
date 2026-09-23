from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.qwen_runtime_plan import SpeculativeIntent, TargetLane
from rapid_mlx.routes import health


class _TextEngine:
    def __init__(
        self,
        *,
        supports_spec_decode: bool = True,
        runtime_attempted: bool = False,
        runtime_method: str | None = None,
    ):
        self.engine = SimpleNamespace(
            scheduler=SimpleNamespace(
                spec_decode_runtime_attempted=runtime_attempted,
                spec_decode_runtime_method=runtime_method,
                model_config=SimpleNamespace(
                    supports_spec_decode=supports_spec_decode,
                ),
            )
        )

    def get_stats(self) -> dict:
        return {}


def _checkpoint(tmp_path: Path, *, qwen: bool = True, moe: bool = True) -> Path:
    checkpoint = tmp_path / "neutral-checkpoint"
    checkpoint.mkdir(parents=True)
    if qwen:
        config = {
            "model_type": "qwen3_5_moe" if moe else "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text"
            },
        }
    else:
        config = {"model_type": "llama", "text_config": {"model_type": "llama"}}
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return checkpoint


def _engine(
    checkpoint: Path,
    *,
    intent: SpeculativeIntent,
    is_mllm: bool,
    companion: bool,
    requested_spec_method: str | None = "mtp",
    mtp_dispatch_result: str | None = "attached",
    mtp_model_type: str | None = "qwen3_5_moe",
    supports_spec_decode: bool = True,
    operator_lane: TargetLane | None = None,
    model_name: str = "neutral-model",
) -> BatchedEngine:
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model_name = model_name
    engine._profile_name = "neutral-profile"
    engine._qwen_artifact_repo_id = "not-a-hub-repo"
    engine._qwen_artifact_snapshot_source = str(checkpoint)
    engine._qwen_runtime_plan = None
    engine._qwen_artifact_truth = None
    engine._qwen_mtp_dispatch_result = mtp_dispatch_result
    engine._qwen_speculative_intent = intent
    engine._qwen_operator_target_lane = operator_lane
    engine._is_mllm = is_mllm
    engine._mllm_native_text_engine = companion
    engine._mllm_instance = (
        SimpleNamespace(
            config=json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        )
        if is_mllm
        else None
    )
    engine._serving_lane_reason = "text_lane_forced"
    engine._scheduler_config = SimpleNamespace(
        spec_decode=requested_spec_method,
        mtp_model_type=mtp_model_type,
    )
    engine._engine = (
        None if is_mllm else _TextEngine(supports_spec_decode=supports_spec_decode)
    )
    engine._mllm_scheduler = None
    engine._prompt_host_cache = None
    engine._loaded = True
    engine._stream_interval = 1
    engine._start_time = None
    return engine


@pytest.mark.parametrize(
    ("model", "moe"),
    [("qwen3.6-35b-4bit", True), ("qwen3.8-27b-4bit", False)],
)
def test_alias_default_plan_preserves_current_text_mtp_boot(
    tmp_path: Path, model: str, moe: bool, caplog
) -> None:
    caplog.set_level(logging.INFO)
    engine = _engine(
        _checkpoint(tmp_path, moe=moe),
        intent=SpeculativeIntent.ALIAS_DEFAULT,
        is_mllm=False,
        companion=False,
        model_name=model,
    )
    before = (engine._is_mllm, engine._engine, engine._scheduler_config.spec_decode)

    engine._finalize_qwen_runtime_observability()

    assert (
        engine._is_mllm,
        engine._engine,
        engine._scheduler_config.spec_decode,
    ) == before
    assert engine._qwen_runtime_plan.to_status_dict() == {
        "target_lane": "text",
        "text_mode": "mtp",
        "selection_source": "alias_default",
        "qualification_id": None,
        "receipt_id": None,
        "target_verification_id": None,
        "target_verification_authority": None,
        "reason": "legacy_alias_default",
        "media_enabled": False,
        "fallback_chain": [],
        "recovery_action": "none",
        "pending_fallback_target": None,
    }
    boot_log = next(
        record.message
        for record in caplog.records
        if "Qwen runtime boot:" in record.message
    )
    assert '"activation":"pending_first_request"' in boot_log


@pytest.mark.parametrize(
    ("model", "moe", "companion", "expected_text_mode"),
    [
        ("qwen3.6-35b-4bit", True, True, "native_ar"),
        ("qwen3.8-27b-4bit", False, False, "none"),
    ],
)
def test_explicit_mllm_preserves_final_vision_shape(
    tmp_path: Path,
    model: str,
    moe: bool,
    companion: bool,
    expected_text_mode: str,
) -> None:
    engine = _engine(
        _checkpoint(tmp_path, moe=moe),
        intent=SpeculativeIntent.NONE,
        is_mllm=True,
        companion=companion,
        operator_lane=TargetLane.VISION,
        model_name=model,
    )

    engine._finalize_qwen_runtime_observability()

    status = engine._qwen_runtime_plan.to_status_dict()
    assert status["target_lane"] == "vision"
    assert status["text_mode"] == expected_text_mode
    assert status["media_enabled"] is True
    assert status["selection_source"] == "operator"
    assert status["reason"] == "legacy_operator"
    assert engine._qwen_runtime_activation().value == "not_applicable"


@pytest.mark.parametrize(
    ("model", "moe"),
    [("qwen3.6-35b-4bit", True), ("qwen3.8-27b-4bit", False)],
)
def test_explicit_no_spec_preserves_final_text_native_ar_shape(
    tmp_path: Path,
    model: str,
    moe: bool,
) -> None:
    engine = _engine(
        _checkpoint(tmp_path, moe=moe),
        intent=SpeculativeIntent.EXPLICIT_DISABLED,
        is_mllm=False,
        companion=False,
        requested_spec_method="none",
        mtp_dispatch_result=None,
        model_name=model,
    )

    engine._finalize_qwen_runtime_observability()

    status = engine._qwen_runtime_plan.to_status_dict()
    assert status["target_lane"] == "text"
    assert status["text_mode"] == "native_ar"
    assert status["media_enabled"] is False
    assert status["selection_source"] == "operator"
    assert status["reason"] == "legacy_operator"
    assert engine._qwen_runtime_activation().value == "not_applicable"


def test_mllm_loaded_config_builds_plan_without_local_snapshot(tmp_path: Path) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.EXPLICIT_DISABLED,
        is_mllm=True,
        companion=True,
    )
    engine._qwen_artifact_snapshot_source = "org/mutable-looking-repo-id"

    engine._finalize_qwen_runtime_observability()

    assert engine._qwen_runtime_plan.target_lane.value == "vision"
    assert engine._qwen_runtime_plan.text_mode.value == "native_ar"
    assert engine._qwen_artifact_truth is None


def test_mllm_live_config_wins_over_qwen_looking_disk_metadata(tmp_path: Path) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.NONE,
        is_mllm=True,
        companion=False,
    )
    engine._mllm_instance.config = {
        "model_type": "llama",
        "text_config": {"model_type": "llama"},
    }

    engine._finalize_qwen_runtime_observability()

    assert engine._qwen_runtime_plan is None


def test_invalid_outer_text_model_type_cross_pair_fails_closed(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"model_type": "qwen3_5_moe_text"},
            }
        ),
        encoding="utf-8",
    )
    engine = _engine(
        checkpoint,
        intent=SpeculativeIntent.NONE,
        is_mllm=False,
        companion=False,
    )

    engine._finalize_qwen_runtime_observability()

    assert engine._qwen_runtime_plan is None


@pytest.mark.parametrize("dispatch_result", [None, "unresolved", "no_inject"])
def test_requested_mtp_soft_skip_reports_native_ar(
    tmp_path: Path, dispatch_result: str | None
) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.EXPLICIT_ENABLED,
        is_mllm=False,
        companion=False,
        mtp_dispatch_result=dispatch_result,
    )

    engine._finalize_qwen_runtime_observability()

    assert engine._qwen_runtime_plan.text_mode.value == "native_ar"


def test_config_vetted_gate_can_disable_attached_mtp(tmp_path: Path) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.EXPLICIT_ENABLED,
        is_mllm=False,
        companion=False,
        mtp_dispatch_result="attached",
        mtp_model_type=None,
        supports_spec_decode=False,
    )

    engine._finalize_qwen_runtime_observability()

    assert engine._qwen_runtime_plan.text_mode.value == "native_ar"


@pytest.mark.parametrize("method", ["dflash", "dspark", "suffix"])
def test_non_mtp_decoder_omits_unrepresentable_plan(
    tmp_path: Path, method: str
) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.EXPLICIT_ENABLED,
        is_mllm=False,
        companion=False,
        requested_spec_method=method,
        mtp_dispatch_result=None,
    )

    engine._finalize_qwen_runtime_observability()

    assert engine._qwen_runtime_plan is None


def test_exact_config_not_qwen_like_name_controls_publication(tmp_path: Path) -> None:
    misleading = _engine(
        _checkpoint(tmp_path, qwen=False),
        intent=SpeculativeIntent.NONE,
        is_mllm=False,
        companion=False,
        model_name="qwen3.8-lookalike",
    )
    misleading._finalize_qwen_runtime_observability()
    assert misleading._qwen_runtime_plan is None

    neutral_root = tmp_path / "neutral"
    neutral_root.mkdir()
    neutral = _engine(
        _checkpoint(neutral_root),
        intent=SpeculativeIntent.NONE,
        is_mllm=False,
        companion=False,
        model_name="custom/local-model",
    )
    neutral._finalize_qwen_runtime_observability()
    assert neutral._qwen_runtime_plan is not None


def test_alias_forced_text_reason_is_not_operator_provenance(tmp_path: Path) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.NONE,
        is_mllm=False,
        companion=False,
        operator_lane=None,
    )

    engine._finalize_qwen_runtime_observability()

    status = engine._qwen_runtime_plan.to_status_dict()
    assert engine._serving_lane_reason == "text_lane_forced"
    assert status["selection_source"] == "fallback"
    assert status["reason"] == "legacy_default"


def test_plan_stored_once_reset_on_stop_and_recomputed(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.NONE,
        is_mllm=False,
        companion=False,
    )
    engine._abort_all_guided_requests = lambda: None
    engine._engine = None
    engine._model_load_executor = None
    engine._model = None
    engine._tokenizer = None
    engine._processor = None
    engine._mllm_instance = None
    engine._engine_started = False

    engine._qwen_mtp_dispatch_result = None
    engine._engine = _TextEngine()
    engine._finalize_qwen_runtime_observability()
    first = engine._qwen_runtime_plan
    engine._finalize_qwen_runtime_observability()
    assert engine._qwen_runtime_plan is first
    assert sum("Qwen runtime boot:" in record.message for record in caplog.records) == 1

    engine._engine = None
    asyncio.run(engine.stop())
    assert engine._qwen_runtime_plan is None
    assert engine._qwen_artifact_truth is None
    assert engine._qwen_mtp_dispatch_result is None

    engine._qwen_artifact_snapshot_source = str(_checkpoint(tmp_path / "reload"))
    engine._qwen_mtp_dispatch_result = "attached"
    engine._engine = _TextEngine()
    engine._finalize_qwen_runtime_observability()
    assert engine._qwen_runtime_plan.text_mode.value == "mtp"
    assert engine._qwen_runtime_plan is not first


def test_stop_failure_still_clears_qwen_boot_state(tmp_path: Path) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.ALIAS_DEFAULT,
        is_mllm=False,
        companion=False,
    )
    engine._finalize_qwen_runtime_observability()
    engine._abort_all_guided_requests = lambda: None

    class _FailingEngine:
        async def stop(self) -> None:
            raise RuntimeError("synthetic stop failure")

    engine._engine = _FailingEngine()

    with pytest.raises(RuntimeError, match="synthetic stop failure"):
        asyncio.run(engine.stop())

    assert engine._qwen_runtime_plan is None
    assert engine._qwen_artifact_truth is None
    assert engine._qwen_artifact_snapshot_source is None
    assert engine._qwen_mtp_dispatch_result is None


def test_selected_mtp_activation_tracks_lazy_installer_state(tmp_path: Path) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.ALIAS_DEFAULT,
        is_mllm=False,
        companion=False,
    )
    engine._finalize_qwen_runtime_observability()
    scheduler = engine._engine.engine.scheduler

    assert engine.get_stats()["qwen_runtime_activation"] == "pending_first_request"

    scheduler.spec_decode_runtime_attempted = True
    scheduler.spec_decode_runtime_method = "mtp"
    assert engine.get_stats()["qwen_runtime_activation"] == "active"

    scheduler.spec_decode_runtime_method = None
    assert engine.get_stats()["qwen_runtime_activation"] == "fallback_native_ar"


def test_get_stats_and_status_forward_only_redacted_qwen_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(
        _checkpoint(tmp_path),
        intent=SpeculativeIntent.NONE,
        is_mllm=False,
        companion=False,
    )
    engine._finalize_qwen_runtime_observability()
    stats = engine.get_stats()
    assert stats["qwen_auto_enabled"] is False
    assert stats["qwen_runtime_plan"]["reason"] == "legacy_default"
    assert stats["qwen_runtime_activation"] == "pending_first_request"
    assert "qwen_artifact_truth" not in stats

    monkeypatch.setattr(
        health,
        "get_config",
        lambda: SimpleNamespace(engine=engine, model_name="public-model"),
    )
    payload = asyncio.run(health.status())
    assert payload["qwen_auto_enabled"] is False
    assert payload["qwen_runtime_plan"] == stats["qwen_runtime_plan"]
    assert payload["qwen_runtime_activation"] == "pending_first_request"
    assert "model_name" not in payload["qwen_runtime_plan"]
    assert str(tmp_path) not in json.dumps(payload)
