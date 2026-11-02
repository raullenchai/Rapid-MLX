# SPDX-License-Identifier: Apache-2.0
"""Vision dependencies must be checked against the final automatic MTP lane."""

import sys
from types import SimpleNamespace

import huggingface_hub
import pytest

from rapid_mlx import cli, model_metadata
from rapid_mlx.api import utils as api_utils
from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.models import media_backend, mllm
from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing
from tests.test_continuous_mtp_qualification import _parsed_serve_args, _StopServeError


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("eligible", [True, False], ids=["mtp", "vision"])
def test_deferred_mtp_vision_guard_uses_final_lane(monkeypatch, warm, eligible):
    state = SimpleNamespace(cached=warm, downloads=0, guard_calls=[])
    profile = resolve_profile("qwen3.6-35b-4bit")
    config = {
        "model_type": "qwen3_5",
        "mtp_num_hidden_layers": 1 if eligible else 0,
    }

    def no_fetch(*args, **kwargs):
        pytest.fail("serve preflight attempted a network download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_fetch)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_fetch)
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda _: (config if state.cached else None, None),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_cached_hf_config",
        lambda *_: (
            None,
            f"local cache for {profile.hf_path} has no config.json at refs/main",
        ),
    )
    monkeypatch.setattr(cli, "_cached_mtp_sidecar", lambda _: "/cached/mtp")
    monkeypatch.setattr(model_metadata, "read_model_metadata", lambda _: None)
    monkeypatch.setattr(api_utils, "read_model_metadata", lambda _: None)
    monkeypatch.setattr(api_utils, "_prefetch_config_for_degrade_probe", lambda _: None)
    monkeypatch.setattr(media_backend, "native_media_model_type", lambda _: None)
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.INCOMPATIBLE, "0.7.0"),
    )
    monkeypatch.setattr(mllm, "_managed_desktop_runtime", lambda: False)

    args = _parsed_serve_args("qwen3.6-35b-4bit")

    def download(*_args, **_kwargs):
        state.downloads += 1
        state.cached = True

    def reject_runtime(exc, **kwargs):
        state.guard_calls.append((args.spec_decode, state.downloads))
        raise exc

    def stop(_level):
        raise _StopServeError

    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(_resolve_api_key=lambda key: key),
    )
    monkeypatch.setattr(cli, "_ensure_model_downloaded", download)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_warn_vision_text_only_degrade", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_handle_optional_runtime_missing", reject_runtime)
    monkeypatch.setattr(cli, "_configure_bootstrap_logging", stop)

    if eligible:
        with pytest.raises(_StopServeError):
            cli.serve_command(args)
        assert args.spec_decode == "mtp"
        assert args._speculative_config.method == "mtp"
        assert args.mtp_sidecar == "/cached/mtp"
        assert state.downloads == 1
        assert state.guard_calls == []
    else:
        with pytest.raises(OptionalRuntimeMissing, match="incompatible"):
            cli.serve_command(args)
        assert args.spec_decode == "none"
        # Settled vision lanes still fail before prefetch; deferred lanes are
        # checked after the fake target download resolves their eligibility.
        assert state.guard_calls == [("none", 0 if warm else 1)]
    assert not getattr(args, "_auto_mtp_deferred", False)
    assert not getattr(args, "_auto_mtp_retry", False)
