"""Flash-Next defaults to its qualified embedded-head engine path."""

import json

import pytest

from rapid_mlx import cli
from rapid_mlx.cli_parser import build_parser


@pytest.mark.parametrize(
    "target",
    ["qwen3.8-flash-next-4bit", "rapid-mlx/Qwen3.8-Flash-Next-4bit"],
)
@pytest.mark.parametrize("opt_out", [False, True])
def test_flash_next_default_uses_embedded_head_and_honors_opt_out(
    monkeypatch, target, opt_out
):
    def unexpected_runtime_probe(_):
        pytest.fail("embedded MTP must not require the external native runtime")

    monkeypatch.setattr(cli, "_native_mtp_runtime_ready", unexpected_runtime_probe)
    flags = ["--no-spec-decode"] if opt_out else []
    args = build_parser().parse_args(["serve", target, *flags])
    cli._normalize_speculative_config_or_exit(args)

    assert cli.serve_runtime_lane(args) == "engine"
    if opt_out:
        assert args._speculative_config is None
        assert args.spec_decode == "none"
    else:
        assert args._speculative_config.method == "mtp"
        assert args._speculative_config.backend is None
        assert args.mtp_backend is None
        assert args.mtp_sidecar is None
        assert args.mtp_max_k == 1
        assert args.spec_decode == "mtp"


def test_flash_next_explicit_k2_keeps_engine_path():
    payload = json.dumps({"method": "mtp", "num_speculative_tokens": 2})
    args = build_parser().parse_args(
        ["serve", "qwen3.8-flash-next-4bit", "--speculative-config", payload]
    )
    cli._normalize_speculative_config_or_exit(args)
    assert args.mtp_max_k == 2
    assert cli.serve_runtime_lane(args) == "engine"


def test_flash_next_decode_snapshot_tracks_mtp_without_borrowing_ar_cohorts():
    from rapid_mlx.decode_expectations import snapshot

    model = snapshot()["models"]["qwen3.8-flash-next-4bit"]
    assert model["default_method"] == "mtp"
    # Existing ordinary-decode submissions do not qualify the new default.
    assert model["default_config_key"] is None
    assert all(cohort["method"] == "none" for cohort in model["cohorts"])
