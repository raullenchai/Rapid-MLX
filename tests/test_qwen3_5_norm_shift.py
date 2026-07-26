# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Qwen3.5/3.6 norm-shift correction.

Pins the surgical fix in ``vllm_mlx.patches.qwen3_5_norm_shift`` that undoes
mlx-lm's spurious ``+1.0`` RMSNorm-weight shift on mtp-bundled VLM checkpoints
like ``mlx-community/Qwen3.6-35B-A3B-*`` (ml-explore/mlx-lm#1197).

No model download: a stub ``TextModel.sanitize`` faithfully mimics mlx-lm's
proxy-gated shift so the wrapper's undo behavior is exercised end-to-end.

Covered:
1. Decision helper — proxy fires + standard-form norms => spurious shift.
2. Decision helper — proxy fires + zero-centered norms => NOT spurious (keep).
3. Decision helper — proxy silent => never spurious.
4. Wrapper — standard-form + mtp: shift is undone (loaded gains ~1, not ~2).
5. Wrapper — zero-centered + mtp: shift is preserved (correct behavior).
6. Wrapper — no proxy: unchanged.
7. install/uninstall swap the class method and are idempotent.
"""

from __future__ import annotations

import types

import mlx.core as mx
import pytest

from vllm_mlx.patches import qwen3_5_norm_shift as nsf

# mlx-lm's real norm-key suffixes (kept identical to the module under test).
_NORM_KEYS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _mlx_lm_mimic_sanitize(self, weights):
    """Faithful copy of ``mlx_lm.models.qwen3_5.TextModel.sanitize`` (0.31.3)."""
    has_mtp_weights = any("mtp." in k for k in weights)
    has_unsanitized_conv1d = any(
        "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
    )
    should_shift = has_mtp_weights or has_unsanitized_conv1d
    weights = {k: v for k, v in weights.items() if "mtp." not in k}
    if getattr(self.args, "tie_word_embeddings", False):
        weights.pop("lm_head.weight", None)
    for k, v in list(weights.items()):
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
        if should_shift and any(k.endswith(sfx) for sfx in _NORM_KEYS):
            if v.ndim == 1:
                weights[k] = v + 1.0
    return weights


def _norm_weights(mean: float, n_layers: int = 4):
    """Build a synthetic weight dict of ``n_layers`` layers of norm gains with
    the given aggregate mean, plus one non-norm 2-D weight for realism."""
    w = {}
    for layer in range(n_layers):
        base = f"language_model.model.layers.{layer}"
        w[f"{base}.input_layernorm.weight"] = mx.full((8,), mean, dtype=mx.float32)
        w[f"{base}.post_attention_layernorm.weight"] = mx.full(
            (8,), mean, dtype=mx.float32
        )
        w[f"{base}.self_attn.q_norm.weight"] = mx.full((4,), mean, dtype=mx.float32)
        w[f"{base}.self_attn.o_proj.weight"] = mx.zeros((8, 8), dtype=mx.float32)
    w["language_model.model.norm.weight"] = mx.full((8,), mean, dtype=mx.float32)
    return w


def _add_mtp_trigger(w):
    # A non-norm mtp tensor so ``has_mtp`` fires without polluting the norm
    # aggregate (mtp weights are stripped by sanitize anyway).
    w = dict(w)
    w["language_model.mtp.layers.0.self_attn.qkv_proj.weight"] = mx.zeros(
        (8, 8), dtype=mx.float32
    )
    return w


# ----------------------------------------------------------------------
# Decision helpers
# ----------------------------------------------------------------------


def test_standard_form_plus_mtp_is_spurious():
    w = _add_mtp_trigger(_norm_weights(mean=1.05))
    assert nsf._would_spuriously_shift(w) is True


def test_zero_centered_plus_mtp_is_not_spurious():
    w = _add_mtp_trigger(_norm_weights(mean=0.02))
    assert nsf._would_spuriously_shift(w) is False


def test_no_proxy_is_never_spurious():
    # Standard-form norms but no mtp / no unsanitized conv1d => proxy silent.
    w = _norm_weights(mean=1.05)
    assert nsf._would_spuriously_shift(w) is False


def test_unsanitized_conv1d_standard_form_is_spurious():
    w = _norm_weights(mean=1.05)
    # conv1d with shape[-1] != 1 trips has_unsanitized_conv1d.
    w["language_model.model.layers.0.linear_attn.conv1d.weight"] = mx.zeros(
        (8, 4, 3), dtype=mx.float32
    )
    assert nsf._would_spuriously_shift(w) is True


def test_sanitize_applied_shift_detects_delta():
    raw = _norm_weights(mean=1.0)
    shifted = {k: (v + 1.0 if v.ndim == 1 else v) for k, v in raw.items()}
    assert nsf._sanitize_applied_shift(raw, shifted) is True
    assert nsf._sanitize_applied_shift(raw, dict(raw)) is False


# ----------------------------------------------------------------------
# Full wrapper behavior (stub original mimicking mlx-lm)
# ----------------------------------------------------------------------


@pytest.fixture
def patched_textmodel(monkeypatch):
    """Install the fix over a stub TextModel whose original sanitize mimics
    mlx-lm. Yields a callable ``run(weights)`` invoking the patched method."""
    from mlx_lm.models import qwen3_5 as q

    # Clean slate on the real module attributes the patch stashes onto.
    nsf.uninstall_qwen3_5_norm_shift_fix()
    for attr in (
        "_RAPID_MLX_ORIG_TEXTMODEL_SANITIZE",
        "_RAPID_MLX_NORM_SHIFT_INSTALLED",
    ):
        if hasattr(q, attr):
            monkeypatch.delattr(q, attr, raising=False)

    class _StubTextModel:
        sanitize = _mlx_lm_mimic_sanitize

    monkeypatch.setattr(q, "TextModel", _StubTextModel)

    nsf.install_qwen3_5_norm_shift_fix()

    fake_self = types.SimpleNamespace(
        args=types.SimpleNamespace(tie_word_embeddings=False)
    )

    def run(weights):
        return q.TextModel.sanitize(fake_self, weights)

    yield run
    nsf.uninstall_qwen3_5_norm_shift_fix()


def _mean(v):
    return float(v.astype(mx.float32).mean())


def test_wrapper_undoes_shift_on_standard_form(patched_textmodel):
    w = _add_mtp_trigger(_norm_weights(mean=1.05))
    out = patched_textmodel(w)
    key = "language_model.model.layers.0.input_layernorm.weight"
    # mlx-lm would leave this at ~2.05; the fix restores ~1.05.
    assert _mean(out[key]) == pytest.approx(1.05, abs=1e-4)
    # mtp keys still stripped by the delegated original.
    assert not any("mtp." in k for k in out)


def test_wrapper_preserves_shift_on_zero_centered(patched_textmodel):
    w = _add_mtp_trigger(_norm_weights(mean=0.02))
    out = patched_textmodel(w)
    key = "language_model.model.norm.weight"
    # Genuinely zero-centered => the shift is correct and must be kept (~1.02).
    assert _mean(out[key]) == pytest.approx(1.02, abs=1e-4)


def test_wrapper_noop_without_proxy(patched_textmodel):
    w = _norm_weights(mean=1.05)
    out = patched_textmodel(w)
    key = "language_model.model.layers.0.post_attention_layernorm.weight"
    # No mtp / no unsanitized conv1d => no shift, no undo.
    assert _mean(out[key]) == pytest.approx(1.05, abs=1e-4)


# ----------------------------------------------------------------------
# install / uninstall wiring
# ----------------------------------------------------------------------


def test_install_uninstall_roundtrip(monkeypatch):
    from mlx_lm.models import qwen3_5 as q

    nsf.uninstall_qwen3_5_norm_shift_fix()
    for attr in (
        "_RAPID_MLX_ORIG_TEXTMODEL_SANITIZE",
        "_RAPID_MLX_NORM_SHIFT_INSTALLED",
    ):
        if hasattr(q, attr):
            monkeypatch.delattr(q, attr, raising=False)

    class _StubTextModel:
        sanitize = _mlx_lm_mimic_sanitize

    monkeypatch.setattr(q, "TextModel", _StubTextModel)
    original = q.TextModel.sanitize

    nsf.install_qwen3_5_norm_shift_fix()
    assert nsf.is_installed() is True
    assert q.TextModel.sanitize is not original
    # Idempotent.
    nsf.install_qwen3_5_norm_shift_fix()
    assert nsf.is_installed() is True

    nsf.uninstall_qwen3_5_norm_shift_fix()
    assert nsf.is_installed() is False
    assert q.TextModel.sanitize is original


def test_install_fires_on_real_serve_import_path():
    """The fix must install when a SERVE-path module is imported, not only
    when the patch module itself is imported directly.

    The production ``rapid-mlx serve`` boot path is::

        cli -> server -> engine.batched -> utils.tokenizer.
        load_model_with_fallback -> mlx_lm.load -> mlx_lm.utils.load_model

    None of those import ``model_runner``, so wiring the install ONLY at
    ``model_runner.py`` (as this fix first did) never fired in production and
    Qwen3.6-35B still served garbage. The in-process tests above would pass
    anyway because they import the patch module directly, masking the gap —
    exactly the trap the deepseek_v32 indexer gate documents. This test runs
    in a SUBPROCESS (pristine interpreter, no state leak) and imports a
    serve-path module, then asserts the fix installed.

    A future refactor that removes the ``utils/tokenizer.py`` install hook
    makes this fail — the same symptom (garbage Qwen3.6 output) it prevents.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys

        # Import a SERVE-path module (NOT the patch module directly).
        import vllm_mlx.utils.tokenizer  # noqa: F401

        from vllm_mlx.patches.qwen3_5_norm_shift import is_installed

        if not is_installed():
            print("FAIL: is_installed() False after serve-path import")
            sys.exit(1)

        from mlx_lm.models import qwen3_5 as q

        if not getattr(q, "_RAPID_MLX_NORM_SHIFT_INSTALLED", False):
            print(
                "FAIL: upstream marker mlx_lm.models.qwen3_5."
                "_RAPID_MLX_NORM_SHIFT_INSTALLED is missing"
            )
            sys.exit(1)

        print("OK")
        """
    ).strip()

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"subprocess install-on-serve-path check failed (exit={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
        "Wiring regression — Qwen3.6-35B would serve garbage again."
    )
    assert "OK" in result.stdout, (
        f"subprocess did not print OK. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
