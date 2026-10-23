# SPDX-License-Identifier: Apache-2.0
"""The public multimodal loader must reject checkpoint Python before dispatch."""

import json
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.requires_mlx


@pytest.mark.parametrize(
    "environment,requested", [(None, None), ("0", None), ("0", True)]
)
def test_mllm_rejects_model_file_before_upstream_loader(
    tmp_path, monkeypatch, environment, requested
):
    import mlx_vlm

    from rapid_mlx.models import mllm
    from rapid_mlx.patches import (
        glm5_next_forget_gate_quant,
        glm5_next_processor,
        glm5_next_runtime,
    )

    # The test reads a real local config through the installed dependency's
    # load_config/get_model_path. It does not execute a model or download assets.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llava", "model_file": "custom.py"})
    )
    marker = tmp_path / "executed"
    (tmp_path / "custom.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    if environment is None:
        monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", environment)
    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    monkeypatch.setattr(
        glm5_next_runtime, "install_glm5_next_runtime_fix", lambda: None
    )
    monkeypatch.setattr(
        glm5_next_processor, "install_glm5_next_processor_patch", lambda: None
    )
    monkeypatch.setattr(
        glm5_next_forget_gate_quant,
        "install_glm5_next_forget_gate_quant_fix",
        lambda: None,
    )
    loader = Mock(side_effect=AssertionError("upstream model loader must not run"))
    monkeypatch.setattr(mlx_vlm, "load", loader)

    instance = mllm.MLXMultimodalLM(
        str(tmp_path), trust_remote_code=requested, enable_cache=False
    )
    with pytest.raises(ValueError, match="explicitly opt in"):
        instance.load()
    loader.assert_not_called()
    assert not marker.exists()
    assert instance._loaded is False


@pytest.mark.parametrize("environment,requested", [(None, True), ("1", None)])
def test_mllm_explicit_consent_preserves_local_checkpoint(
    tmp_path, monkeypatch, environment, requested
):
    from types import SimpleNamespace

    import mlx_vlm
    import mlx_vlm.utils

    from rapid_mlx.models import mllm

    (tmp_path / "config.json").write_text(
        '{"model_type":"llava","model_file":"custom.py"}'
    )
    (tmp_path / "custom.py").write_text("# reviewed implementation")
    if environment is None:
        monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", environment)
    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    calls = []
    original_load_config = mlx_vlm.utils.load_config

    def load_config(path, *, trust_remote_code):
        assert trust_remote_code is True
        calls.append(("config", path))
        return original_load_config(path, trust_remote_code=trust_remote_code)

    monkeypatch.setattr(mlx_vlm.utils, "load_config", load_config)

    def load(path, *, trust_remote_code):
        assert trust_remote_code is True
        calls.append(("model", path))
        return (
            SimpleNamespace(config=SimpleNamespace()),
            SimpleNamespace(tokenizer=SimpleNamespace()),
        )

    monkeypatch.setattr(
        mlx_vlm,
        "load",
        load,
    )
    instance = mllm.MLXMultimodalLM(
        str(tmp_path), trust_remote_code=requested, enable_cache=False
    )
    instance.load()
    assert instance._loaded
    assert calls == [("config", str(tmp_path)), ("model", str(tmp_path))]
