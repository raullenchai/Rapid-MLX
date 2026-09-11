from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_deepseek_v41_dspark.py"
    spec = importlib.util.spec_from_file_location("benchmark_deepseek_v41_dspark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_moe_script():
    path = (
        Path(__file__).parents[1] / "scripts" / "benchmark_deepseek_v41_moe_kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_deepseek_v41_moe_kernel", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_requires_explicit_checkpoint_runtime_trust(tmp_path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_deepseek_v41_dspark.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--target",
            str(tmp_path / "target"),
            "--overlay",
            str(tmp_path / "overlay"),
            "--checkpoint-runtime",
            str(tmp_path / "runtime"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--trust-checkpoint-runtime" in result.stderr


def test_greedy_prefix_stops_at_accepted_eos() -> None:
    module = _load_script()
    eos_id = 1
    candidate = [9, eos_id, 7]
    logits = mx.zeros((1, 3, 10))
    logits[0, 0, eos_id] = 1
    logits[0, 1, 7] = 1

    committed, mismatch, hit_eos, accepted = module._match_greedy_prefix(
        candidate, logits, eos_id, False
    )

    assert committed == [9, eos_id]
    assert mismatch is None
    assert hit_eos is True
    assert accepted == 1


def test_zero_depth_deferred_seed_advances_without_duplicate_output() -> None:
    module = _load_script()
    candidate = [9]
    logits = mx.zeros((1, 1, 10))

    committed, mismatch, hit_eos, accepted = module._match_greedy_prefix(
        candidate, logits, eos_id=1, seed_already_emitted=True
    )

    assert committed == []
    assert mismatch is None
    assert hit_eos is False
    assert accepted == 0


def test_checkpoint_runtime_loads_exact_files_and_restores_ambient_module(
    tmp_path,
) -> None:
    module = _load_script()
    (tmp_path / "runtime.py").write_text("MARKER = 'requested'\n")
    (tmp_path / "dspark.py").write_text(
        "from runtime import MARKER\nclass DSpark: pass\n"
    )
    ambient = ModuleType("runtime")
    ambient.MARKER = "ambient"
    previous = sys.modules.get("runtime")
    sys.modules["runtime"] = ambient
    try:
        with module._checkpoint_runtime(tmp_path) as (runtime, dspark):
            assert runtime.MARKER == "requested"
            assert dspark.MARKER == "requested"
            assert sys.modules["runtime"] is runtime
        assert sys.modules["runtime"] is ambient
    finally:
        if previous is None:
            sys.modules.pop("runtime", None)
        else:
            sys.modules["runtime"] = previous


def test_packed_mtp_uses_dspark_specific_topk() -> None:
    module = _load_script()
    config = {"num_experts_per_tok": 6, "dspark_num_experts_per_tok": 3}

    assert module._dspark_topk(config) == 3


def test_tokens_must_be_positive() -> None:
    module = _load_script()

    assert module._positive_int("1") == 1
    with pytest.raises(module.argparse.ArgumentTypeError, match="at least 1"):
        module._positive_int("0")


@pytest.mark.parametrize("conflict", ["native_moe_only", "packed_mtp_only"])
def test_target_only_rejects_modes_that_require_other_runtimes(conflict) -> None:
    module = _load_script()
    args = SimpleNamespace(
        target_only=True,
        native_moe_only=False,
        packed_mtp_only=False,
    )
    setattr(args, conflict, True)

    with pytest.raises(SystemExit, match="cannot be combined"):
        module._validate_mode(args)


def test_packed_mtp_method_binding_does_not_retain_adapter() -> None:
    module = _load_script()

    class Adapter:
        pass

    adapter = Adapter()

    def method(self):
        return self

    adapter.method = module.MethodType(method, module.weakref.proxy(adapter))
    reference = module.weakref.ref(adapter)
    del adapter

    assert reference() is None


def test_direct_down_install_is_atomic_and_idempotent() -> None:
    module = _load_script()
    first = SwitchGLU(64, 64, 2, bias=False)
    second = SwitchGLU(64, 64, 2, bias=False)
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(ffn=SimpleNamespace(experts=first)),
            SimpleNamespace(ffn=SimpleNamespace(experts=second)),
        ]
    )

    assert module._install_direct_down_qmv(model) == 2
    assert type(first) is module.ExactDirectDownSwitchGLU
    assert type(second) is module.ExactDirectDownSwitchGLU
    assert module._install_direct_down_qmv(model) == 0

    valid = SwitchGLU(64, 64, 2, bias=False)
    invalid = object()
    mixed = SimpleNamespace(
        layers=[
            SimpleNamespace(ffn=SimpleNamespace(experts=valid)),
            SimpleNamespace(ffn=SimpleNamespace(experts=invalid)),
        ]
    )
    with pytest.raises(TypeError, match="unsupported expert module"):
        module._install_direct_down_qmv(mixed)
    assert type(valid) is SwitchGLU


def test_direct_down_falls_back_for_prefill_sized_route_batch() -> None:
    module = _load_script()
    experts = SwitchGLU(64, 64, 8, bias=False)
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(experts=experts))]
    )
    inputs = mx.random.normal((7, 64))
    indices = mx.broadcast_to(mx.arange(6, dtype=mx.uint32), (7, 6))
    expected = experts(inputs, indices)
    mx.eval(expected)

    assert module._install_direct_down_qmv(model) == 1
    actual = experts(inputs, indices)
    mx.eval(actual)

    assert mx.array_equal(actual, expected).item()


def test_moe_layer_loader_merges_indexed_shards(tmp_path, monkeypatch) -> None:
    module = _load_moe_script()
    prefix = "layers.20.ffn."
    index = {
        prefix + "gate.weight": "model-1.safetensors",
        prefix + "experts.weight": "model-2.safetensors",
    }
    contents = {
        "model-1.safetensors": {prefix + "gate.weight": "gate"},
        "model-2.safetensors": {prefix + "experts.weight": "experts"},
    }
    for shard in contents:
        (tmp_path / shard).touch()
    monkeypatch.setattr(
        module.mx,
        "load",
        lambda path: contents[Path(path).name],
    )

    assert module._load_prefix_items(tmp_path, index, prefix) == [
        ("experts.weight", "experts"),
        ("gate.weight", "gate"),
    ]


def test_moe_layer_loader_rejects_traversing_shard(tmp_path) -> None:
    module = _load_moe_script()
    prefix = "layers.20.ffn."

    with pytest.raises(ValueError, match="must be a basename"):
        module._load_prefix_items(
            tmp_path, {prefix + "gate.weight": "../outside.safetensors"}, prefix
        )


def test_moe_layer_loader_rejects_symlinked_shard(tmp_path) -> None:
    module = _load_moe_script()
    prefix = "layers.20.ffn."
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model-1.safetensors").symlink_to(outside)

    with pytest.raises(ValueError, match="must not be a symlink"):
        module._load_prefix_items(
            model_path, {prefix + "gate.weight": "model-1.safetensors"}, prefix
        )
