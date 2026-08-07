from pathlib import Path
from types import SimpleNamespace

from scripts import benchmark_model_recommendations as benchmark
from scripts.benchmark_model_recommendations import huggingface_cache_dir


def test_huggingface_cache_dir_prefers_explicit_override(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/ignored/direct")
    monkeypatch.setenv("HF_HOME", "/ignored/home")
    assert huggingface_cache_dir("/Volumes/model-cache") == Path("/Volumes/model-cache")


def test_huggingface_cache_dir_understands_direct_hub_cache(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/Volumes/jetson/hf-cache")
    monkeypatch.setenv("HF_HOME", "/ignored/home")
    assert huggingface_cache_dir() == Path("/Volumes/jetson/hf-cache")


def test_huggingface_cache_dir_appends_hub_to_hf_home(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", "/Volumes/jetson/hf-home")
    assert huggingface_cache_dir() == Path("/Volumes/jetson/hf-home/hub")


def test_measure_disables_prefix_cache_and_passes_selected_hf_cache(
    monkeypatch, tmp_path
):
    launched = {}

    class FakeProcess:
        pid = 123

        def poll(self):
            return 0

    def fake_popen(argv, **kwargs):
        launched["argv"] = argv
        launched["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(benchmark, "cached_repo_for", lambda *_: True)
    monkeypatch.setattr(benchmark.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(benchmark, "wait_ready", lambda *_: None)
    monkeypatch.setattr(benchmark, "footprint", lambda *_: (1.0, 1.0))
    monkeypatch.setattr(benchmark, "swap_used_mb", lambda: 0.0)
    monkeypatch.setattr(
        benchmark,
        "run_prompt",
        lambda *_: {
            "server_prompt_tps": 100.0,
            "server_generation_tps": 20.0,
        },
    )

    args = SimpleNamespace(
        allow_download=False,
        hf_cache="/Volumes/jetson/hf-cache",
        log_dir=str(tmp_path),
        port=8010,
        load_timeout=10,
        abort_ram_fraction=0.8,
        abort_swap_mb=256,
        output_tokens=4,
        serve_arg=[],
        cooldown=0,
    )
    result = benchmark.measure("qwen3-1.7b-4bit", args, {"physical_ram_gb": 32})

    assert result["status"] == "ok"
    assert "--disable-prefix-cache" in launched["argv"]
    assert launched["env"]["HF_HUB_CACHE"] == "/Volumes/jetson/hf-cache"
