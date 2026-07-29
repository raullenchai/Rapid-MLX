"""Unit tests for models/defs/lora_merge.py — the LoRA merge-at-load math.

Pure-numpy/MLX, no model weights or torch needed. Run either way:

    python scripts/test_lora_merge.py          # standalone (no pytest needed)
    pytest scripts/test_lora_merge.py

Correctness is established by:
  * the zero-init -> identity invariant for every adapter type (a strong guard on
    axis / reshape / transpose bugs: with B/M_xs zeroed and magnitudes at their
    init values, every forward must return W0);
  * an exact independent reconstruction for standard LoRA and PEFT;
  * the Conv1d (out,k,in) <-> PyTorch (out,in,k) layout round-trip;
  * the to_local_embed name remap;
  * --lora-strength scaling and the bit-exact strength-0 bypass;
  * the trust boundary (pickle refusal) and base-mismatch / 0-merge handling.
"""
import json
import os
import sys
import tempfile

import numpy as np
import mlx.core as mx

# Import lora_merge the same way the CLI does (REPO = optimized/mlx on sys.path).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.defs import lora_merge as lm  # noqa: E402

rng = np.random.default_rng(0)


def _save_native(path, layer, params, cfg):
    d = {f"{layer}.parametrizations.weight.0.{k}": mx.array(v.astype(np.float32))
         for k, v in params.items()}
    mx.save_safetensors(path, d, metadata={"lora_config": json.dumps(cfg)})


def _init_params(adapter_type, W0, rank, nonzero):
    """Init so a zeroed core -> identity; if nonzero, give a real core."""
    fo, fi = W0.shape
    p = {}
    if not adapter_type.endswith("-xs"):
        p["lora_A"] = rng.standard_normal((rank, fi)).astype(np.float32)
        p["lora_B"] = (rng.standard_normal((fo, rank)) if nonzero
                       else np.zeros((fo, rank))).astype(np.float32)
    else:
        p["M_xs"] = (rng.standard_normal((rank, rank)) if nonzero
                     else np.zeros((rank, rank))).astype(np.float32)
    if "magnitude" in lm._PARAMS_FOR[adapter_type]:
        nd = 1 if "rows" in adapter_type else 0
        p["magnitude"] = np.linalg.norm(W0, axis=nd).astype(np.float32)
    if "magnitude_r" in lm._PARAMS_FOR[adapter_type]:
        p["magnitude_r"] = np.linalg.norm(W0, axis=1).astype(np.float32)
        p["magnitude_c"] = np.linalg.norm(W0, axis=0).astype(np.float32)
    return p


def _np(arr):
    return np.array(arr.astype(mx.float32))


def test_trust_boundary_refuses_pickle():
    for bad in ("evil.ckpt", "evil.pt", "evil.bin", "weights.npz"):
        try:
            lm._load_safetensors(bad)
            assert False, f"should have refused {bad}"
        except lm.LoraError:
            pass


def test_all_types_zero_init_identity_and_nonzero_delta():
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        for atype in lm._PARAMS_FOR:
            path = os.path.join(tmp, f"{atype}.safetensors")
            # zero-init -> identity
            _save_native(path, "foo", _init_params(atype, W0, 4, nonzero=False),
                         {"adapter_type": atype, "rank": 4, "alpha": 4})
            w = {"foo.weight": mx.array(W0)}
            lm.merge_loras_into_weights(w, [path])
            assert np.allclose(_np(w["foo.weight"]), W0, atol=1e-4), f"{atype} identity"
            # nonzero core -> finite, changed
            _save_native(path, "foo", _init_params(atype, W0, 4, nonzero=True),
                         {"adapter_type": atype, "rank": 4, "alpha": 4})
            w = {"foo.weight": mx.array(W0)}
            lm.merge_loras_into_weights(w, [path])
            got = _np(w["foo.weight"])
            assert np.isfinite(got).all() and not np.allclose(got, W0), f"{atype} delta"


def test_standard_lora_exact_and_strength():
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((6, 4)).astype(np.float32)
    expect = W0 + (8.0 / 4.0) * (B @ A)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "std.safetensors")
        _save_native(path, "foo", {"lora_A": A, "lora_B": B},
                     {"adapter_type": "lora", "rank": 4, "alpha": 8})
        w = {"foo.weight": mx.array(W0)}
        lm.merge_loras_into_weights(w, [path])
        assert np.allclose(_np(w["foo.weight"]), expect, atol=1e-4)
        # strength 0.5 halves the delta; strength 0 is the original
        w = {"foo.weight": mx.array(W0)}
        lm.merge_loras_into_weights(w, [path], strength=0.5)
        assert np.allclose(_np(w["foo.weight"]), W0 + 0.5 * (8.0 / 4.0) * (B @ A), atol=1e-4)
        w = {"foo.weight": mx.array(W0)}
        lm.merge_loras_into_weights(w, [path], strength=0.0)
        assert np.allclose(_np(w["foo.weight"]), W0, atol=1e-6)


def test_conv1d_layout_round_trip():
    out_c, k_c, in_c = 4, 1, 3
    Wc = rng.standard_normal((out_c, k_c, in_c)).astype(np.float32)   # MLX layout
    Ac = rng.standard_normal((2, in_c * k_c)).astype(np.float32)
    Bc = rng.standard_normal((out_c, 2)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "conv.safetensors")
        _save_native(path, "conv", {"lora_A": Ac, "lora_B": Bc},
                     {"adapter_type": "lora", "rank": 2, "alpha": 2})
        w = {"conv.weight": mx.array(Wc)}
        lm.merge_loras_into_weights(w, [path])
        got = _np(w["conv.weight"])
        expect = Wc + (Bc @ Ac).reshape(out_c, in_c, k_c).transpose(0, 2, 1)
        assert got.shape == Wc.shape and np.allclose(got, expect, atol=1e-4)


def test_to_local_embed_remap():
    Wt = rng.standard_normal((5, 5)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tle.safetensors")
        _save_native(path, "transformer.layers.0.to_local_embed.0",
                     {"lora_A": rng.standard_normal((2, 5)).astype(np.float32),
                      "lora_B": rng.standard_normal((5, 2)).astype(np.float32)},
                     {"adapter_type": "lora", "rank": 2, "alpha": 2})
        w = {"transformer.layers.0.to_local_embed.seq.0.weight": mx.array(Wt)}
        stats = lm.merge_loras_into_weights(w, [path])
        assert stats["merged"] == 1 and not stats["skipped"]


def test_peft_format_exact():
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((6, 4)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "adapter_config.json"), "w") as fh:
            json.dump({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "use_dora": False}, fh)
        mx.save_safetensors(
            os.path.join(tmp, "adapter_model.safetensors"),
            {"base_model.model.foo.lora_A.weight": mx.array(A),
             "base_model.model.foo.lora_B.weight": mx.array(B)},
            metadata={"format": "pt"})
        w = {"foo.weight": mx.array(W0)}
        stats = lm.merge_loras_into_weights(w, [tmp])   # directory input
        assert stats["merged"] == 1
        assert np.allclose(_np(w["foo.weight"]), W0 + (8.0 / 4.0) * (B @ A), atol=1e-4)


def test_unknown_layers_skipped_base_untouched():
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "std.safetensors")
        _save_native(path, "foo",
                     {"lora_A": rng.standard_normal((4, 8)).astype(np.float32),
                      "lora_B": rng.standard_normal((6, 4)).astype(np.float32)},
                     {"adapter_type": "lora", "rank": 4, "alpha": 4})
        w = {"other.weight": mx.array(W0)}
        msgs = []
        stats = lm.merge_loras_into_weights(w, [path], log=msgs.append)
        assert stats["merged"] == 0 and stats["skipped"] == ["foo"]
        assert np.allclose(_np(w["other.weight"]), W0)
        # a 0-merge run must warn, not look like a successful no-op
        assert any("WARNING" in m and "0 layers" in m for m in msgs)


def test_base_mismatch_raises_clear_error():
    # base is (6, 8) but the adapter was trained for an (10, 8) layer
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mismatch.safetensors")
        _save_native(path, "foo",
                     {"lora_A": rng.standard_normal((4, 8)).astype(np.float32),
                      "lora_B": rng.standard_normal((10, 4)).astype(np.float32)},
                     {"adapter_type": "lora", "rank": 4, "alpha": 4})
        w = {"foo.weight": mx.array(W0)}
        try:
            lm.merge_loras_into_weights(w, [path])
            assert False, "should have raised on base mismatch"
        except lm.LoraError as e:
            assert "foo" in str(e) and "base" in str(e).lower()


def test_svd_bases_orthonormal():
    Wsvd = rng.standard_normal((6, 8)).astype(np.float32)
    U, V = lm._svd_bases(Wsvd, rank=6)
    assert np.allclose(U.T @ U, np.eye(6), atol=1e-4)
    assert np.allclose(V.T @ V, np.eye(6), atol=1e-4)


def test_underfit_wrapper_prefixes():
    """Underfit (github.com/dada-bots/underfit) saves full-wrapper layer names:
    DiT layers as ``model.transformer...`` and the seconds conditioner as
    ``conditioners.seconds_total.embedder.embedding.1``. Both must land on the
    bare npz keys; bare names keep working."""
    W_dit = rng.standard_normal((6, 8)).astype(np.float32)
    W_cond = rng.standard_normal((5, 7)).astype(np.float32)
    A_d = rng.standard_normal((4, 8)).astype(np.float32)
    B_d = rng.standard_normal((6, 4)).astype(np.float32)
    A_c = rng.standard_normal((4, 7)).astype(np.float32)
    B_c = rng.standard_normal((5, 4)).astype(np.float32)
    cfg = {"adapter_type": "lora", "rank": 4, "alpha": 4}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "underfit.safetensors")
        d = {}
        for layer, (A, B) in {
            "model.transformer.layers.0.self_attn.to_qkv": (A_d, B_d),
            "conditioners.seconds_total.embedder.embedding.1": (A_c, B_c),
        }.items():
            d[f"{layer}.parametrizations.weight.0.lora_A"] = mx.array(A)
            d[f"{layer}.parametrizations.weight.0.lora_B"] = mx.array(B)
        mx.save_safetensors(path, d, metadata={"lora_config": json.dumps(cfg)})
        w = {"transformer.layers.0.self_attn.to_qkv.weight": mx.array(W_dit),
             "cond.seconds_total_weight": mx.array(W_cond)}
        stats = lm.merge_loras_into_weights(w, [path])
        assert stats["merged"] == 2 and not stats["skipped"], stats
        assert np.allclose(_np(w["transformer.layers.0.self_attn.to_qkv.weight"]),
                           W_dit + B_d @ A_d, atol=1e-4)
        assert np.allclose(_np(w["cond.seconds_total_weight"]),
                           W_cond + B_c @ A_c, atol=1e-4)
    # bare names still pass through, and model.-prefixed to_local_embed remaps
    assert lm._layer_to_npz_key("transformer.layers.3.ff.ff.2") == \
        "transformer.layers.3.ff.ff.2.weight"
    assert lm._layer_to_npz_key("model.transformer.layers.3.to_local_embed.0") == \
        "transformer.layers.3.to_local_embed.seq.0.weight"


def test_lora_spec_parser():
    s = lm.parse_lora_spec(["a.safetensors"], default_strength=0.5)
    assert s == {"path": "a.safetensors", "strength": 0.5, "steps": None}
    s = lm.parse_lora_spec(["a.safetensors", "strength=0.8", "steps=2-8"])
    assert s["strength"] == 0.8 and s["steps"] == (2, 8)
    assert lm.parse_lora_spec(["a", "steps=2-"])["steps"] == (2, None)
    assert lm.parse_lora_spec(["a", "steps=-4"])["steps"] == (None, 4)
    assert lm.parse_lora_spec(["a", "steps=3"])["steps"] == (3, 3)
    for bad in (["a.st", "b.st"],            # two paths in one flag
                ["a", "steps=8-2"],          # min > max
                ["a", "steps=0-3"],          # 1-based
                ["a", "steps=-"],            # empty range
                ["a", "strength=loud"],      # not a float
                []):
        try:
            lm.parse_lora_spec(bad)
            assert False, f"should have rejected {bad}"
        except lm.LoraError:
            pass
    # resolve against an 8-step schedule (0-based inclusive out)
    assert lm.resolve_steps(None, 8) == (0, 7)
    assert lm.resolve_steps((2, 8), 8) == (1, 7)
    assert lm.resolve_steps((2, None), 8) == (1, 7)
    assert lm.resolve_steps((None, 4), 8) == (0, 3)
    assert lm.resolve_steps((2, 99), 8) == (1, 7)      # clamp
    assert lm.resolve_steps((9, None), 8) is None      # misses the schedule


class _StubModule:
    """Bare object with .weight — stands in for nn.Linear in plan tests."""
    def __init__(self, w):
        self.weight = w


def _stub_tree(weights):
    """Build a walkable module tree for keys like 'foo.weight' and
    'layers.0.lin.weight' (digits = list containers, as in the DiT)."""
    class _NS:  # namespace node
        pass
    root = _NS()
    for key, w in weights.items():
        parts = key.split(".")[:-1]          # drop trailing 'weight'
        obj = root
        for j, p in enumerate(parts[:-1]):
            nxt = parts[j + 1]
            if p.isdigit():
                obj = obj[int(p)]
            else:
                if not hasattr(obj, p):
                    setattr(obj, p, [] if nxt.isdigit() else _NS())
                obj = getattr(obj, p)
            if isinstance(obj, list) and nxt.isdigit():
                while len(obj) <= int(nxt):
                    obj.append(_NS())
        last = parts[-1]
        mod = _StubModule(w)
        if last.isdigit():
            obj[int(last)] = mod
        else:
            setattr(obj, last, mod)
    return root


def test_step_plan_matches_reference():
    """For every adapter type: gate two overlapping adapters across an 8-step
    schedule and check the in-place weights at EVERY step against the from-
    scratch reference W0 + Σ_active strength·(merged − W0)."""
    num_steps = 8
    K1, K2 = "foo.weight", "layers.0.lin.weight"
    W = {K1: rng.standard_normal((6, 8)).astype(np.float32),
         K2: rng.standard_normal((5, 7)).astype(np.float32)}
    with tempfile.TemporaryDirectory() as tmp:
        for atype in lm._PARAMS_FOR:
            p1 = os.path.join(tmp, f"{atype}-1.safetensors")
            p2 = os.path.join(tmp, f"{atype}-2.safetensors")
            prm = {k: _init_params(atype, w, 3, nonzero=True) for k, w in W.items()}
            d = {}
            for k, w in W.items():
                layer = k[: -len(".weight")]
                for pk, pv in prm[k].items():
                    d[f"{layer}.parametrizations.weight.0.{pk}"] = mx.array(pv)
            mx.save_safetensors(p1, d, metadata={"lora_config": json.dumps(
                {"adapter_type": atype, "rank": 3, "alpha": 3})})
            prm2 = {K1: _init_params("lora", W[K1], 3, nonzero=True)}
            d2 = {f"foo.parametrizations.weight.0.{pk}": mx.array(pv)
                  for pk, pv in prm2[K1].items()}
            mx.save_safetensors(p2, d2, metadata={"lora_config": json.dumps(
                {"adapter_type": "lora", "rank": 3, "alpha": 3})})

            specs = [{"path": p1, "strength": 0.7, "steps": (2, 5)},
                     {"path": p2, "strength": 1.3, "steps": (4, None)}]
            weights = {k: mx.array(w) for k, w in W.items()}
            plan = lm.prepare_loras(weights, specs, num_steps=num_steps)
            assert plan is not None and len(plan.layers) == 2

            model = _stub_tree(weights)
            plan.attach(model)
            mods = {K1: model.foo, K2: model.layers[0].lin}

            def ref(i):
                out = {k: W[k].copy() for k in W}
                if 1 <= i <= 4:  # adapter 1 (steps 2-5, 0-based 1..4)
                    for k in W:
                        m = lm._merged_weight(W[k], prm[k], atype, 1.0)
                        out[k] += 0.7 * (m - W[k])
                if i >= 3:       # adapter 2 (steps 4-, 0-based 3..7)
                    m = lm._merged_weight(W[K1], prm2[K1], "lora", 1.0)
                    out[K1] += 1.3 * (m - W[K1])
                return out

            for i in range(num_steps):
                plan.sync(i)
                expect = ref(i)
                for k in W:
                    got = _np(mods[k].weight)
                    err = np.abs(got - expect[k]).max()
                    assert err < 1e-3, f"{atype} step {i} {k}: max|d|={err:.2e}"


def test_step_plan_round_trip_drift():
    """50 on/off gating cycles on fp16 weights must not accumulate drift
    (exact-inverse pairing — measured frozen-after-cycle-1 in the design
    benchmarks; this guards the implementation)."""
    K = "foo.weight"
    W0 = rng.standard_normal((32, 48)).astype(np.float16)
    prm = _init_params("dora-rows", W0.astype(np.float32), 4, nonzero=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "d.safetensors")
        _save_native(path, "foo", prm, {"adapter_type": "dora-rows", "rank": 4, "alpha": 4})
        weights = {K: mx.array(W0)}
        plan = lm.prepare_loras(weights, [{"path": path, "strength": 1.0,
                                           "steps": (2, None)}], num_steps=2)
        model = _stub_tree(weights)
        plan.attach(model)
        drift1 = None
        for cyc in range(50):
            plan.sync(1)   # on
            plan.sync(0)   # off
            d = np.abs(_np(model.foo.weight) - W0.astype(np.float32)).max()
            if cyc == 0:
                drift1 = d
        assert d <= drift1 + 1e-6, f"drift grew: cycle1 {drift1:.2e} → cycle50 {d:.2e}"
        rel = (np.linalg.norm(_np(model.foo.weight) - W0.astype(np.float32))
               / np.linalg.norm(W0.astype(np.float32)))
        assert rel < 5e-3, f"round-trip rel drift {rel:.2e}"


def test_full_interval_uses_merge_path():
    """Specs without steps= (or covering every step) → no plan; weights match
    the plain merge_loras_into_weights result exactly."""
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    prm = _init_params("lora", W0, 4, nonzero=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "l.safetensors")
        _save_native(path, "foo", prm, {"adapter_type": "lora", "rank": 4, "alpha": 4})
        wa = {"foo.weight": mx.array(W0)}
        wb = {"foo.weight": mx.array(W0)}
        plan = lm.prepare_loras(wa, [{"path": path, "strength": 0.6,
                                      "steps": (1, 8)}], num_steps=8)
        assert plan is None
        lm.merge_loras_into_weights(wb, [path], strength=0.6)
        assert np.array_equal(_np(wa["foo.weight"]), _np(wb["foo.weight"]))


def test_gate_all_and_clear_restore_base():
    """The gradio path: gate_all=True routes even full-range adapters through
    the plan so clear() can restore a cached model to base in place; an adapter
    matching zero layers raises a clear wrong-base error."""
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    prm = _init_params("dora-rows", W0, 4, nonzero=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "g.safetensors")
        _save_native(path, "foo", prm, {"adapter_type": "dora-rows", "rank": 4, "alpha": 4})
        w = {"foo.weight": mx.array(W0)}
        plan = lm.prepare_loras(w, [{"path": path, "strength": 1.0, "steps": None}],
                                num_steps=8, gate_all=True)
        assert plan is not None, "gate_all must plan-manage full-range adapters"
        model = _stub_tree(w)
        plan.attach(model)
        assert not np.allclose(_np(model.foo.weight), W0), "step-1 state not applied"
        plan.clear()
        assert np.abs(_np(model.foo.weight) - W0).max() < 1e-4, "clear() must restore base"
        # an adapter whose layers don't exist in the target → clear error
        w2 = {"bar.weight": mx.array(W0)}
        try:
            lm.prepare_loras(w2, [{"path": path, "strength": 1.0, "steps": (2, None)}],
                             num_steps=8)
            assert False, "zero-match adapter should raise"
        except lm.LoraError as e:
            assert "different base model" in str(e)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    print("\n" + ("ALL PASS" if not failed else f"FAILURES: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
