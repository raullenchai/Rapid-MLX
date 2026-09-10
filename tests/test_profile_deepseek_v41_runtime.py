from types import SimpleNamespace

import pytest

from scripts.bench_deepseek_v41_runtime import (
    _checkpoint_total_size,
    _counter_delta,
    _install_optimized_moe,
    _make_v41_hc_mix_kernel,
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


class _FakeMx:
    get_active_memory = staticmethod(lambda: 1)
    get_cache_memory = staticmethod(lambda: 2)
    get_peak_memory = staticmethod(lambda: 3)
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


@pytest.mark.requires_mlx
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
        engram_cache_rows=16,
        execution_mode="compiled",
        prompt="test",
        context_tokens=2,
        warmup_tokens=1,
        measure_tokens=2,
        diagnostic_tokens=1,
    )

    result = run(args)

    assert result["checkpoint"]["indexed_total_size_bytes"] == 4
    assert result["context_build"]["tokens"] == 2
    assert result["warmup"]["tokens"] == 1
    assert result["measurement"]["tokens"] == 2
    assert result["measurement"]["eval_barriers"] is None
    assert result["barrier_diagnostic"]["eval_barriers"]["calls"] == 1


@pytest.mark.requires_mlx
def test_expert_local_gate_up_fusion_matches_separate_quantized_matmuls():
    mx = pytest.importorskip("mlx.core")

    class Weights:
        q = {"bits": 2, "group_size": 64, "mode": "affine"}

        def __init__(self):
            self.resident = {}
            self.resident_bytes = 0

        def read(self, key):
            return self.resident[key]

        def linear(self, base, x):
            return mx.quantized_matmul(
                x,
                self.resident[base + ".weight"],
                self.resident[base + ".scales"],
                self.resident[base + ".biases"],
                transpose=True,
                group_size=64,
                bits=2,
                mode="affine",
            )

    class Runtime:
        execution_mode = "compiled"
        c = {
            "n_routed_experts": 2,
            "num_hidden_layers": 1,
            "num_experts_per_tok": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 0.0,
        }

        def __init__(self):
            self.w = Weights()

        def expert(self, base, x, routing=None):
            if base.endswith("shared_experts"):
                return mx.zeros_like(x)
            gate = self.w.linear(base + ".w1", x).astype(mx.float32)
            up = self.w.linear(base + ".w3", x).astype(mx.float32)
            hidden = gate * mx.sigmoid(gate) * up
            if routing is not None:
                hidden = hidden * routing
            return self.w.linear(base + ".w2", hidden.astype(x.dtype))

    runtime = Runtime()
    mx.random.seed(19)
    base = "layers.0.ffn"
    for expert in range(2):
        for projection in ("w1", "w2", "w3"):
            dense = mx.random.normal((64, 64)).astype(mx.float32)
            weight, scales, biases = mx.quantize(
                dense, group_size=64, bits=2, mode="affine"
            )
            prefix = f"{base}.experts.{expert}.{projection}"
            runtime.w.resident[prefix + ".weight"] = weight
            runtime.w.resident[prefix + ".scales"] = scales
            runtime.w.resident[prefix + ".biases"] = biases
    runtime.w.resident[base + ".gate.weight"] = mx.zeros((2, 64))
    runtime.w.resident[base + ".gate.bias"] = mx.array([1.0, 0.0])

    x = mx.random.normal((1, 64)).astype(mx.bfloat16)
    routing = mx.sqrt(mx.logaddexp(mx.array(0.0), mx.array(0.0)))
    expected = runtime.expert(base + ".experts.0", x, routing).astype(x.dtype)
    metadata = _install_optimized_moe(runtime, mx)
    actual = runtime.moe(base, x)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert metadata["name"] == "expert_local_fused_gate_up_quantized_matmul"
    assert base + ".experts.0.w1.weight" not in runtime.w.resident
    assert base + ".experts.0.w3.weight" not in runtime.w.resident
    assert base + ".experts.0.w2.weight" in runtime.w.resident


@pytest.mark.requires_mlx
def test_fused_hc_kernel_stays_close_to_reference_sinkhorn():
    mx = pytest.importorskip("mlx.core")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        pytest.skip("Metal GPU required")

    mx.random.seed(23)
    hc_mult = 4
    sinkhorn_iters = 20
    epsilon = 1e-6
    mixes = mx.random.normal((2, (2 + hc_mult) * hc_mult)).astype(mx.float32)
    scale = mx.random.normal((3,)).astype(mx.float32)
    base = mx.random.normal(((2 + hc_mult) * hc_mult,)).astype(mx.float32)

    pre = mx.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + epsilon
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    comb = (mixes[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]).reshape(
        2, hc_mult, hc_mult
    )
    comb = mx.softmax(comb, axis=-1) + epsilon
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + epsilon)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + epsilon)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + epsilon)

    kernel = _make_v41_hc_mix_kernel(mx)
    actual = kernel(
        inputs=[mixes, scale, base],
        template=[
            ("HC", hc_mult),
            ("ITERS", sinkhorn_iters),
            ("EPS_INT", round(epsilon / 1e-9)),
        ],
        grid=(2 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(2, 4), (2, 4), (2, 4, 4)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    mx.eval(pre, post, comb, *actual)

    for expected, observed in zip((pre, post, comb), actual):
        assert mx.allclose(expected, observed, rtol=1e-6, atol=1e-6).item()
