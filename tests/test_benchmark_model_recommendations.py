from pathlib import Path

from scripts.benchmark_model_recommendations import huggingface_cache_dir


def test_huggingface_cache_dir_prefers_explicit_override(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/ignored/direct")
    monkeypatch.setenv("HF_HOME", "/ignored/home")
    assert huggingface_cache_dir("/Volumes/model-cache") == Path(
        "/Volumes/model-cache"
    )


def test_huggingface_cache_dir_understands_direct_hub_cache(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/Volumes/jetson/hf-cache")
    monkeypatch.setenv("HF_HOME", "/ignored/home")
    assert huggingface_cache_dir() == Path("/Volumes/jetson/hf-cache")


def test_huggingface_cache_dir_appends_hub_to_hf_home(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", "/Volumes/jetson/hf-home")
    assert huggingface_cache_dir() == Path("/Volumes/jetson/hf-home/hub")
