# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import tomllib

from rapid_mlx.cli import _resolve_system_one_backend, build_parser


def test_system_one_cli_defaults_to_laya_service():
    args = build_parser().parse_args(["system-one"])
    assert args.model == "convaiinnovations/laya"
    assert args.backend == "auto"
    assert args.host == "127.0.0.1"
    assert args.port == 8700
    assert args.max_concurrent_requests == 8


def test_system_one_cli_accepts_clm_runtime_inputs():
    args = build_parser().parse_args(
        [
            "system-one",
            "clm-latest",
            "--backend",
            "clm",
            "--head",
            "/tmp/head",
            "--encoder",
            "Qwen/Qwen3-8B",
        ]
    )
    assert args.head == "/tmp/head"
    assert args.encoder == "Qwen/Qwen3-8B"


def test_system_one_auto_backend_does_not_substring_match_clm():
    assert _resolve_system_one_backend("org/aclmish-laya", "auto") == "laya"
    assert _resolve_system_one_backend("Contrastive-LM/CLM-v0.1-8B", "auto") == "clm"


def test_system_one_extra_is_optional_and_included_in_all():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert all(not item.startswith("laya-mlx") for item in project["dependencies"])
    extras = project["optional-dependencies"]
    assert any(item.startswith("laya-mlx") for item in extras["system-one"])
    assert any(item.startswith("laya-mlx") for item in extras["all"])
