# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the DFlash qualification harness."""

from unittest.mock import MagicMock

import pytest

from scripts import bench_dflash
from scripts.bench_dflash import WORKLOADS, _qualify


def _passing_speedups() -> dict[str, float]:
    return {name: 1.4 for name in WORKLOADS}


def test_rejected_chat_cannot_qualify_from_code_results_alone() -> None:
    speedups = _passing_speedups()
    speedups.pop("chat")

    result = _qualify(speedups, gate=1.3, non_code_floor=1.0)

    assert result.ship is False
    assert "missing valid workloads: chat" in result.decision


def test_partially_rejected_code_cannot_qualify_from_one_code_result() -> None:
    speedups = {"fibonacci": 1.8, "chat": 1.1}

    result = _qualify(speedups, gate=1.3, non_code_floor=1.0)

    assert result.ship is False
    assert "quicksort" in result.decision
    assert "hashtable" in result.decision
    assert "sortedlist" in result.decision


def test_complete_mixed_workload_can_qualify() -> None:
    result = _qualify(_passing_speedups(), gate=1.3, non_code_floor=1.0)

    assert result.ship is True
    assert result.decision == "SHIP (supports_dflash=true)"


def test_complete_workload_without_immutable_pair_receipt_cannot_qualify() -> None:
    result = _qualify(
        _passing_speedups(),
        gate=1.3,
        non_code_floor=1.0,
        immutable_receipt=False,
    )

    assert result.ship is False
    assert "immutable" in result.decision


def test_start_server_requires_health_algorithm_receipt(monkeypatch) -> None:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    popen = MagicMock(return_value=proc)
    monkeypatch.setattr(bench_dflash.subprocess, "Popen", popen)

    def _get(url: str, timeout: float):
        del timeout
        response = MagicMock(status_code=200)
        response.json.return_value = (
            {"algorithm": "dflash2"} if url.endswith("/healthz") else {}
        )
        return response

    monkeypatch.setattr(bench_dflash.httpx, "get", _get)

    handle = bench_dflash.start_server(
        "target",
        8765,
        True,
        draft_model="draft",
        expected_algorithm="dflash2",
    )
    try:
        assert handle.algorithm == "dflash2"
        cmd = popen.call_args.args[0]
        assert cmd[cmd.index("serve") + 1] == "target"
        assert cmd[cmd.index("--dflash-drafter-path") + 1] == "draft"
    finally:
        handle.stop()


def test_start_server_requires_expected_algorithm_before_spawn(monkeypatch) -> None:
    popen = MagicMock()
    monkeypatch.setattr(bench_dflash.subprocess, "Popen", popen)

    with pytest.raises(ValueError, match="requires expected_algorithm"):
        bench_dflash.start_server("target", 8765, True, draft_model="draft")

    popen.assert_not_called()


def test_baseline_server_command_uses_pinned_target_and_explicit_mtp(
    monkeypatch,
) -> None:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    popen = MagicMock(return_value=proc)
    monkeypatch.setattr(bench_dflash.subprocess, "Popen", popen)
    monkeypatch.setattr(
        bench_dflash.httpx,
        "get",
        lambda *_args, **_kwargs: MagicMock(status_code=200),
    )
    config = '{"method":"mtp","model":"/cache/target","num_speculative_tokens":3}'

    handle = bench_dflash.start_server(
        "/cache/target",
        8765,
        False,
        speculative_config=config,
    )
    try:
        cmd = popen.call_args.args[0]
        assert cmd[cmd.index("serve") + 1] == "/cache/target"
        assert cmd[cmd.index("--speculative-config") + 1] == config
    finally:
        handle.stop()


def test_expected_algorithm_is_inferred_for_known_alias_pair() -> None:
    assert (
        bench_dflash._resolve_expected_algorithm(
            "qwen3.8-27b-4bit", "z-lab/Qwen3.8-27B-DFlash2", None
        )
        == "dflash2"
    )


def test_alias_pair_receipt_records_effective_repositories_and_revisions() -> None:
    receipt = bench_dflash._resolve_pair_receipt(
        "qwen3.8-27b-4bit", "z-lab/Qwen3.8-27B-DFlash2", None
    )

    assert receipt.target_model == "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"
    assert receipt.target_revision == "aa985c29ff5b334cbfdcbbc787d47e66e9d9e456"
    assert receipt.draft_model == "z-lab/Qwen3.8-27B-DFlash2"
    assert receipt.draft_revision == "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
    assert receipt.algorithm == "dflash2"
    assert receipt.immutable is True


def test_local_drafter_receipt_keeps_target_pin_but_cannot_qualify() -> None:
    receipt = bench_dflash._resolve_pair_receipt(
        "qwen3.8-27b-4bit", "/tmp/local-drafter", "dflash2"
    )

    assert receipt.target_revision == "aa985c29ff5b334cbfdcbbc787d47e66e9d9e456"
    assert receipt.draft_revision is None
    assert receipt.immutable is False


def test_main_uses_same_pinned_target_for_baseline_and_dflash(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(
        bench_dflash, "_materialize_target", lambda _pair: "/cache/target-snapshot"
    )
    monkeypatch.setattr(
        bench_dflash, "_materialize_drafter", lambda _pair: "/cache/draft-snapshot"
    )

    def _bench(model, port, dflash, runs, max_tokens, **kwargs):
        calls.append({"model": model, "dflash": dflash, **kwargs})
        return bench_dflash.ModeResult(
            median_tps={name: 10.0 for name in WORKLOADS},
            raw_runs={name: [] for name in WORKLOADS},
            algorithm="dflash2" if dflash else None,
        )

    monkeypatch.setattr(bench_dflash, "bench_one_mode", _bench)
    monkeypatch.setattr(bench_dflash, "write_bench_json", lambda *_args: None)

    assert (
        bench_dflash.main(
            [
                "--model",
                "qwen3.8-27b-4bit",
                "--draft-model",
                "z-lab/Qwen3.8-27B-DFlash2",
                "--runs",
                "1",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
        == 1
    )
    assert [call["model"] for call in calls] == [
        "/cache/target-snapshot",
        "/cache/target-snapshot",
    ]
    baseline_config = calls[0]["speculative_config"]
    assert '"method":"mtp"' in baseline_config
    assert '"model":"/cache/target-snapshot"' in baseline_config
    assert calls[1]["draft_model"] == "/cache/draft-snapshot"


def test_expected_algorithm_requires_receipt_for_unknown_override() -> None:
    with pytest.raises(ValueError, match="cannot infer"):
        bench_dflash._resolve_expected_algorithm(
            "qwen3.8-27b-4bit", "/tmp/local-drafter", None
        )


def test_start_server_stops_process_on_algorithm_mismatch(monkeypatch) -> None:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    monkeypatch.setattr(bench_dflash.subprocess, "Popen", lambda *args, **kwargs: proc)

    def _get(url: str, timeout: float):
        del timeout
        response = MagicMock(status_code=200)
        response.json.return_value = (
            {"algorithm": "dflash"} if url.endswith("/healthz") else {}
        )
        return response

    monkeypatch.setattr(bench_dflash.httpx, "get", _get)

    try:
        bench_dflash.start_server(
            "target",
            8765,
            True,
            draft_model="draft",
            expected_algorithm="dflash2",
        )
    except RuntimeError as exc:
        assert "algorithm mismatch" in str(exc)
    else:
        raise AssertionError("mismatched runtime algorithm must fail")
    proc.send_signal.assert_called()
    proc.wait.assert_called()
