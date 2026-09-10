from types import SimpleNamespace

import pytest

from scripts.bench_deepseek_v41_runtime import (
    _checkpoint_total_size,
    _counter_delta,
    _percentile,
    _resolve_runtime,
    _run_tokens,
    run,
)


def test_percentile_uses_nearest_rank():
    assert _percentile([], 0.95) is None
    assert _percentile([4, 1, 3, 2], 0.50) == 2
    assert _percentile([4, 1, 3, 2], 0.95) == 4


def test_checkpoint_total_size(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":238796133496},"weight_map":{}}',
        encoding="utf-8",
    )
    assert _checkpoint_total_size(tmp_path) == 238_796_133_496


def test_runtime_resolution_prefers_nested_runtime(tmp_path):
    nested = tmp_path / "runtime" / "runtime.py"
    nested.parent.mkdir()
    nested.touch()
    (tmp_path / "runtime.py").touch()
    assert _resolve_runtime(tmp_path) == nested


def test_runtime_resolution_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="no fallback download"):
        _resolve_runtime(tmp_path)


def test_counter_delta():
    before = {"reads": 2, "bytes": 10}
    after = {"reads": 5, "bytes": 42}
    assert _counter_delta(after, before) == {"reads": 3, "bytes": 32}


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeMetal:
    get_active_memory = staticmethod(lambda: 1)
    get_cache_memory = staticmethod(lambda: 2)
    get_peak_memory = staticmethod(lambda: 3)


class _FakeMx:
    metal = _FakeMetal()
    argmax = staticmethod(lambda logits: logits)


class _FakeBarrier:
    def reset(self):
        pass

    def snapshot(self):
        return {"calls": 0}


class _FakeWeights:
    disk_bytes_read = 0
    disk_read_calls = 0
    engram_cache_hits = 0
    engram_cache_misses = 0
    resident_bytes = 0


class _FakeRuntime:
    def __init__(self):
        self.w = _FakeWeights()
        self.tokens = []

    def step(self, token):
        self.tokens.append(token)
        return _Scalar(token + 1), None


def test_run_tokens_continues_generated_chain():
    runtime = _FakeRuntime()
    result = _run_tokens(
        runtime,
        3,
        _FakeBarrier(),
        _FakeMx(),
        initial_token=10,
    )

    assert runtime.tokens == [10, 11, 12]
    assert result["last_token_id"] == 13


def test_run_tokens_can_teacher_force_context():
    runtime = _FakeRuntime()
    result = _run_tokens(
        runtime,
        2,
        _FakeBarrier(),
        _FakeMx(),
        initial_token=99,
        teacher_tokens=[5, 6],
    )

    assert runtime.tokens == [5, 6]
    assert result["last_token_id"] == 7


def test_real_weight_harness_with_tiny_local_runtime(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":4},"weight_map":{}}', encoding="utf-8"
    )
    (runtime_dir / "runtime.py").write_text(
        """\
import mlx.core as mx

class _Encoding:
    ids = [1, 2]

class _Tokenizer:
    def encode(self, prompt):
        return _Encoding()

class _Weights:
    disk_bytes_read = 0
    disk_read_calls = 0
    engram_cache_hits = 0
    engram_cache_misses = 0
    resident_bytes = 4

class TextRuntime:
    def __init__(self, path, max_tokens, resident_backbone, execution_mode):
        self.tokenizer = _Tokenizer()
        self.w = _Weights()

    def step(self, token):
        logits = mx.array([token, token + 1])
        mx.eval(logits)
        return logits, None
""",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        model=tmp_path,
        trust_checkpoint_runtime=True,
        resident_backbone=False,
        execution_mode="compiled",
        prompt="test",
        context_tokens=2,
        warmup_tokens=1,
        measure_tokens=2,
        trace=None,
        trace_tokens=4,
    )

    result = run(args)

    assert result["checkpoint"]["indexed_total_size_bytes"] == 4
    assert result["context_build"]["tokens"] == 2
    assert result["warmup"]["tokens"] == 1
    assert result["measurement"]["tokens"] == 2
    assert result["measurement"]["eval_barriers"]["calls"] == 2
    assert result["trace_probe"] is None
