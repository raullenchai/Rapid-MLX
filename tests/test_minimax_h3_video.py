import fnmatch
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.model_aliases import resolve_profile, subfolder_allow_patterns
from rapid_mlx.video.minimax_h3 import (
    FILES,
    MiniMaxH3BackendError,
    MiniMaxH3RequestError,
    MiniMaxH3VideoEngine,
)


def test_alias_and_download_selection():
    profile = resolve_profile("minimax-h3")
    assert profile.hf_path == "MiniMaxAI/MiniMax-H3"
    assert profile.revision == "b3c7290e66afdf293bef3b9077b7a266ef421f34"
    assert subfolder_allow_patterns(profile.hf_path) == list(FILES)
    for bad in (
        "FL2VA/model.safetensors",
        "Ref2VA/model.safetensors",
        "transformer/model.py",
    ):
        assert not any(fnmatch.fnmatchcase(bad, pattern) for pattern in FILES)


def test_request_validation():
    validate = MiniMaxH3VideoEngine.validate_request
    validate(width=768, height=512, num_frames=124, image=None)
    with pytest.raises(MiniMaxH3RequestError):
        validate(width=768, height=511, num_frames=124, image=None)
    with pytest.raises(MiniMaxH3RequestError):
        validate(width=768, height=512, num_frames=125, image=None)


def test_generation_loads_once_and_maps_workflows(monkeypatch, tmp_path):
    module = ModuleType("mlx_vlm.generate")
    calls = []

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def load(path, workflow):
        calls.append(("load", path, workflow))
        return SimpleNamespace(workflow=workflow)

    def generate(model, request, *, output_path):
        calls.append((model.workflow, request))
        output_path.write_bytes(b"mp4")

    module.VideoGenerationRequest = Request
    module.generate_video = generate
    module.load_video_generation_model = load
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    args = dict(
        prompt="test",
        output_path=tmp_path / "out.mp4",
        width=768,
        height=512,
        num_frames=124,
        fps=24,
        seed=7,
        image=None,
    )
    engine.generate(**args)
    args["image"] = tmp_path / "first.png"
    engine.generate(**args, last_image=tmp_path / "last.png")
    assert [call[0] for call in calls] == ["load", "t2va", "fl2va"]
    assert calls[2][1].last_image == tmp_path / "last.png"
    engine.stop()
    with pytest.raises(MiniMaxH3BackendError, match="cancelled"):
        engine.generate(**args)


def test_runtime_failure_is_mapped(monkeypatch, tmp_path):
    module = ModuleType("mlx_vlm.generate")
    module.VideoGenerationRequest = lambda **kwargs: kwargs
    module.load_video_generation_model = lambda *args, **kwargs: SimpleNamespace(
        workflow="fl2va"
    )

    def fail(*args, **kwargs):
        raise OSError("private details")

    module.generate_video = fail
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)
    engine = MiniMaxH3VideoEngine(str(tmp_path))
    with pytest.raises(MiniMaxH3BackendError, match="OSError") as exc:
        engine.generate(
            prompt="test",
            output_path=tmp_path / "out.mp4",
            width=768,
            height=512,
            num_frames=124,
            fps=24,
            seed=7,
            image=None,
        )
    assert "private details" not in str(exc.value)


def test_runtime_missing_is_reported(monkeypatch):
    from rapid_mlx.runtime import video_lane

    monkeypatch.setattr(video_lane.importlib.util, "find_spec", lambda name: None)
    issue = video_lane.video_runtime_issue("minimax-h3", provision=False)
    assert issue is not None
    assert "mlx-vlm" in issue.detail
    assert issue.marker_reason == "runtime_dependency_missing"


def test_pinned_component_cache_completeness(monkeypatch, tmp_path):
    import huggingface_hub.constants

    from rapid_mlx._download_gate import is_repo_cached
    from rapid_mlx.video.minimax_h3 import REPO, REVISION

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / REVISION
    for name in FILES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert is_repo_cached(REPO)
    (snapshot / FILES[-1]).unlink()
    assert not is_repo_cached(REPO)
