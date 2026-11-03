# SPDX-License-Identifier: Apache-2.0
"""Right-padded Nemotron-H prefill must match independent recurrent caches."""

import importlib
import os

import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx import _mlx_compat

_mlx_compat.install()

from mlx_lm.models import nemotron_h, ssm  # noqa: E402


@pytest.fixture(autouse=True)
def installed_patch():
    from rapid_mlx.utils.tokenizer import install_nemotron_h_ssm_fix

    install_nemotron_h_ssm_fix()


def tiny_model():
    mx.random.seed(17)
    model = nemotron_h.Model(
        nemotron_h.ModelArgs(
            model_type="nemotron_h",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            max_position_embeddings=1024,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_bias=False,
            mamba_num_heads=2,
            mamba_head_dim=8,
            mamba_proj_bias=False,
            ssm_state_size=32,
            conv_kernel=4,
            n_groups=1,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            hybrid_override_pattern=["M", "*"],
        )
    )
    model.layers[0].mixer.dt_bias = mx.full((2,), -5.0)
    model.layers[0].mixer.A_log = mx.zeros(2)
    mx.eval(model.parameters())
    return model


def assert_close(actual, expected):
    mx.eval(actual, expected)
    error = float(mx.max(mx.abs(actual - expected)))
    if mx.default_device() == mx.gpu and os.environ.get("MLX_ENABLE_TF32") != "0":
        # GPU GEMM and GEMV can use different TF32 precision, notably on M5.
        # Bound both RMS and peak error; exact exhausted-state checks below
        # do not use this tolerance. The strict run disables TF32 explicitly.
        tolerance = 8 * 2**-10
        assert bool(
            mx.mean(mx.square(actual - expected))
            <= tolerance**2 * mx.mean(mx.square(expected))
        ), error
        assert error <= tolerance * float(mx.max(mx.abs(expected))), error
    else:
        assert bool(mx.allclose(actual, expected, atol=2e-5, rtol=2e-4)), error


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("chunk", [333, 256, 1])
def test_model_padded_prefill_matches_independent(prefix, chunk):
    model = tiny_model()
    lengths = [0, 255, 256, 257, 300, 333]
    tokens = (mx.arange(333)[None, :] + mx.arange(len(lengths))[:, None]) % 32
    singles = [model.make_cache() for _ in lengths]
    if prefix:
        for row, cache in enumerate(singles):
            mx.eval(model(mx.array([[row, 3, 7, 2, 9, 1, 5]]), cache=cache))
    batch = importlib.import_module("mlx_lm.generate")._merge_caches(singles)
    for entry in batch:
        entry.prepare(lengths=lengths, right_padding=[333 - n for n in lengths])
    output = []
    for start in range(0, 333, chunk):
        output.append(model(tokens[:, start : start + chunk], cache=batch))
    output = mx.concatenate(output, axis=1)
    mx.eval(output, batch[0].state)
    for entry in batch:
        entry.finalize()
    for row, (length, cache) in enumerate(zip(lengths, singles)):
        if length:
            expected = model(tokens[row : row + 1, :length], cache=cache)
            assert_close(output[row : row + 1, :length], expected)
        for got, want in zip(batch[0].state, cache[0].state):
            if want is None:
                assert bool(mx.all(got[row] == 0))
            else:
                assert_close(got[row : row + 1], want)
        # Decode exposes corrupted final state as user-visible logits.
        extracted = [entry.extract(row) for entry in batch]
        assert_close(
            model(mx.array([[4]]), cache=extracted), model(mx.array([[4]]), cache=cache)
        )


@pytest.mark.parametrize("prefix", [False, True])
def test_real_prompt_batch_right_padding(prefix):
    generate = importlib.import_module("mlx_lm.generate")
    model = tiny_model()
    tokens = [[1] * n for n in (0, 256, 300, 333)]
    caches = [model.make_cache() for _ in tokens]
    if prefix:
        for cache in caches:
            mx.eval(model(mx.array([[2, 3, 4]]), cache=cache))
    from copy import deepcopy

    independent = deepcopy(caches)
    batch = generate.PromptProcessingBatch(
        model=model,
        uids=list(range(4)),
        caches=caches,
        prefill_step_size=256,
    )
    batch.prompt(tokens)
    for row, (prompt, cache) in enumerate(zip(tokens, independent)):
        if prompt:
            mx.eval(model(mx.array([prompt]), cache=cache))
        # State errors must also be visible in the next generated-token logits.
        from copy import deepcopy

        assert_close(
            model(mx.array([[6]]), cache=batch.extract_cache(row)),
            model(mx.array([[6]]), cache=deepcopy(cache)),
        )
        for got, want in zip(batch.prompt_cache[0].state, cache[0].state):
            if want is None:
                assert bool(mx.all(got[row] == 0))
            else:
                assert_close(got[row : row + 1], want)


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("lengths", [[0, 256, 300], [-1, 255, 257], [0, 333, 500]])
def test_shared_ssm_matches_token_recurrence(with_mask, with_state, lengths):
    mx.random.seed(23)
    batch, size, heads, width, groups, state_size = 3, 333, 4, 2, 2, 32
    x = mx.random.normal((batch, size, heads, width)) * 0.1
    b = mx.random.normal((batch, size, groups, state_size)) * 0.1
    c = mx.random.normal(b.shape) * 0.1
    dt = mx.random.normal((batch, size, heads)) * 0.1 - 5
    a, d, bias = mx.zeros(heads), mx.ones(heads), mx.zeros(heads)
    initial = (
        mx.random.normal((batch, heads, width, state_size)) if with_state else None
    )
    mask = mx.arange(size)[None, :] < mx.array(lengths)[:, None]
    actual, state = ssm.ssm_attn(
        x,
        a,
        b,
        c,
        d,
        dt,
        bias,
        state=initial,
        mask=mask if with_mask else None,
        lengths=mx.array(lengths),
    )
    # Independent one-token recurrence, without segsum or gathered decay rows.
    expected_state = (
        mx.zeros((batch, heads, width, state_size)) if initial is None else initial
    )
    timestep = ssm.compute_dt(dt, bias, (0.001, 100.0))
    bs, cs = (
        mx.repeat(b, heads // groups, axis=2),
        mx.repeat(c, heads // groups, axis=2),
    )
    outputs = []
    for pos in range(size):
        delta = timestep[:, pos, :, None, None]
        next_state = (
            mx.exp(-delta) * expected_state
            + delta * x[:, pos, :, :, None] * bs[:, pos, :, None, :]
        )
        expected_state = mx.where(
            mask[:, pos, None, None, None], next_state, expected_state
        )
        outputs.append(
            mx.sum(expected_state * cs[:, pos, :, None, :], axis=-1) + x[:, pos]
        )
    expected = mx.stack(outputs, axis=1)
    assert_close(
        mx.where(mask[..., None, None], actual, 0),
        mx.where(mask[..., None, None], expected, 0),
    )
    assert_close(state, expected_state)
    if initial is not None:
        assert bool(mx.array_equal(state[0], initial[0]))


def test_left_padding_and_unpadded_chunking():
    mx.random.seed(29)
    x = mx.random.normal((2, 7, 2, 4))
    b = mx.random.normal((2, 7, 1, 32))
    c = mx.random.normal(b.shape)
    dt = mx.full((2, 7, 2), -5.0)
    args = (x, mx.zeros(2), b, c, mx.ones(2), dt, mx.zeros(2))
    initial = mx.ones((2, 2, 4, 32))
    full, state = ssm.ssm_attn(*args, state=initial)
    chunked, chunk_state = ssm.ssm_attn(*args, state=initial, step=3)
    assert_close(full, chunked)
    assert_close(state, chunk_state)
    mask = mx.array([[False, False, True, True, True, True, True], [True] * 7])
    padded, padded_state = ssm.ssm_attn(*args, mask=mask, state=initial)
    independent, independent_state = ssm.ssm_attn(
        x[:1, 2:],
        mx.zeros(2),
        b[:1, 2:],
        c[:1, 2:],
        mx.ones(2),
        dt[:1, 2:],
        mx.zeros(2),
        state=initial[:1],
    )
    assert_close(padded[:1, 2:], independent)
    assert_close(padded_state[:1], independent_state)


def test_loader_installs_idempotently_and_preserves_decode(monkeypatch):
    from rapid_mlx.patches import nemotron_h_ssm as patch
    from rapid_mlx.utils import tokenizer

    assert tokenizer.install_nemotron_h_ssm_fix is patch.install_nemotron_h_ssm_fix
    assert ssm.ssm_attn is patch.ssm_attn
    original = ssm.ssm_update
    patch.install_nemotron_h_ssm_fix()
    assert ssm.ssm_update is original
    assert nemotron_h.ssm_update is original  # Pre-imported aliases remain valid.
    model = tiny_model()
    assert_close(
        model(mx.array([[1, 2, 3]])),
        model(mx.array([[1, 2, 3]]), cache=model.make_cache()),
    )
    calls = []

    def kernel(*args):
        calls.append(args)
        return mx.zeros_like(args[0]), args[7]

    monkeypatch.setattr(ssm, "ssm_update_kernel", kernel)
    # Exercise the dispatch contract even in CPU-only numerical verification.
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2]]), cache=cache))
    mx.eval(model(mx.array([[3]]), cache=cache))
    assert len(calls) == 1
    # A masked one-token tail must use the recurrence and preserve both caches.
    before = list(cache[0].state)
    cache[0].prepare(lengths=[0])
    mx.eval(model(mx.array([[0]]), cache=cache))
    assert len(calls) == 1
    for got, want in zip(cache[0].state, before):
        assert bool(mx.array_equal(got, want))


def test_install_without_mlx_lm(monkeypatch):
    import builtins

    from rapid_mlx.patches import nemotron_h_ssm as patch

    original = builtins.__import__

    def no_models(name, *args, **kwargs):
        if name == "mlx_lm.models":
            raise ImportError("no optional runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_models)
    patch.install_nemotron_h_ssm_fix()


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "attention",
        "update",
        "mixer",
        "opaque",
        "metadata",
        "removed_attention",
        "removed_update",
        "removed_method",
        "removed_class",
    ],
)
def test_install_skips_unknown_runtime(monkeypatch, caplog, change):
    # Recover the real upstream definitions without mutating live modules.
    import types
    from pathlib import Path

    from rapid_mlx.patches import nemotron_h_ssm as patch

    def upstream(module):
        clean = types.ModuleType(module.__name__)
        clean.__package__ = module.__package__
        exec(
            compile(Path(module.__file__).read_text(), module.__file__, "exec"),
            clean.__dict__,
        )
        return clean

    clean_ssm, clean_model = upstream(ssm), upstream(nemotron_h)
    monkeypatch.setattr(ssm, "ssm_attn", clean_ssm.ssm_attn)
    monkeypatch.setattr(
        nemotron_h.NemotronHMamba2Mixer, "_ssm", clean_model.NemotronHMamba2Mixer._ssm
    )
    if change == "version":
        monkeypatch.setattr(patch, "version", lambda _: "0.32.0")
    elif change.startswith("removed_"):
        target, attr = {
            "removed_attention": (ssm, "ssm_attn"),
            "removed_update": (ssm, "ssm_update"),
            "removed_method": (nemotron_h.NemotronHMamba2Mixer, "_ssm"),
            "removed_class": (nemotron_h, "NemotronHMamba2Mixer"),
        }[change]
        monkeypatch.delattr(target, attr)
    elif change == "metadata":

        def missing(_):
            raise patch.PackageNotFoundError("mlx-lm")

        monkeypatch.setattr(patch, "version", missing)
    else:

        def incompatible(unknown):
            raise AssertionError("must not execute")

        if change == "opaque":
            incompatible.__signature__ = "not inspectable"
        target, attr = {
            "attention": (ssm, "ssm_attn"),
            "opaque": (ssm, "ssm_attn"),
            "update": (ssm, "ssm_update"),
            "mixer": (nemotron_h.NemotronHMamba2Mixer, "_ssm"),
        }[change]
        monkeypatch.setattr(target, attr, incompatible)

    def snapshot():
        mixer = getattr(nemotron_h, "NemotronHMamba2Mixer", None)
        return (
            getattr(ssm, "ssm_attn", None),
            getattr(ssm, "ssm_update", None),
            mixer,
            getattr(mixer, "_ssm", None),
        )

    before = snapshot()
    patch.install_nemotron_h_ssm_fix()
    assert before == snapshot()
    assert "Skipping Nemotron-H padding shim" in caplog.text


@pytest.mark.parametrize("adapter", [False, True])
def test_checked_loader_padding_path(tmp_path, monkeypatch, adapter):
    import json

    from mlx_lm.utils import load_model, save_model

    from rapid_mlx.model_load_errors import load_model_checked
    from rapid_mlx.models import nemotron_h_puzzle as puzzle
    from rapid_mlx.models import nemotron_h_super as super_h
    from rapid_mlx.patches import nemotron_h_ssm as patch

    args = dict(vars(tiny_model().args))
    args.update(layers_block_type=["mamba", "attention"], rapid_mlx_nemotron_h=adapter)
    model = (
        super_h.Model(super_h.ModelArgs.from_dict(args)) if adapter else tiny_model()
    )
    save_model(tmp_path, model)
    (tmp_path / "config.json").write_text(json.dumps(args))
    model, _ = load_model_checked(load_model, tmp_path)
    mixer = model.layers[0].mixer
    # The opt-in adapter already owns a corrected recurrence. Never replace it.
    assert type(mixer)._ssm is (
        puzzle.NemotronHMamba2Mixer._ssm if adapter else patch._mixer_ssm
    )
    calls = []
    original = puzzle.ssm_attn if adapter else patch.ssm_attn

    def recorded(*args, **kwargs):
        calls.append(kwargs.get("lengths"))
        return original(*args, **kwargs)

    monkeypatch.setattr(puzzle if adapter else patch, "ssm_attn", recorded)
    generate = importlib.import_module("mlx_lm.generate")
    batch = generate.PromptProcessingBatch(
        model=model,
        uids=[0, 1],
        caches=[model.make_cache(), model.make_cache()],
        prefill_step_size=256,
    )
    prompts = [[1, 2, 3], [4] * 300]
    batch.prompt(prompts)
    assert any(lengths is not None for lengths in calls)
    for row, prompt in enumerate(prompts):
        cache = model.make_cache()
        mx.eval(model(mx.array([prompt]), cache=cache))
        assert_close(
            model(mx.array([[6]]), cache=batch.extract_cache(row)),
            model(mx.array([[6]]), cache=cache),
        )
