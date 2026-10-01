# SPDX-License-Identifier: Apache-2.0
"""Focused CLI and catalog contracts for the qualified Qwen 27B lane."""

from types import SimpleNamespace

import pytest


def test_tensorfold_alias_injects_qualified_speculative_config() -> None:
    from rapid_mlx import cli

    args = SimpleNamespace(
        model="qwen3.8-27b-tensorfold",
        speculative_config=None,
        no_spec_decode=False,
        mllm=False,
    )

    cli._normalize_speculative_config_or_exit(args)

    assert args.speculative_config == (
        '{"method":"dflash","backend":"tensorfold","model":"z-lab/Qwen3.8-27B-DFlash2"}'
    )
    assert args._speculative_config.method == "dflash"
    assert args._speculative_config.backend == "tensorfold"
    assert args._speculative_config.model == "z-lab/Qwen3.8-27B-DFlash2"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"dflash_backend": "other"}, "must be 'tensorfold'"),
        ({"dflash_backend": "tensorfold"}, "requires a pinned DFlash pair"),
    ],
)
def test_alias_schema_rejects_invalid_tensorfold_backend(extra, message) -> None:
    from rapid_mlx.model_aliases import _coerce

    with pytest.raises(ValueError, match=message):
        _coerce("bad-tensorfold", {"hf_path": "org/model", **extra})
