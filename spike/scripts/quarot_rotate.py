"""QuaRot R1+R2+R4(weight-only) rotation for Qwen3-family models on MLX.

Adapted from spcl/QuaRot e2e/checkpoint_utils/rotation_utils.py (PyTorch).
Target: W4A16 — weight-only offline rotation; no runtime kernel.

Pipeline:
    input dir (FP16/BF16 Qwen3 safetensors)
        -> fuse_layer_norms (gamma absorbed into adjacent linears)
        -> rotate with random Hadamard Q on hidden_size
        -> rotate V-O with per-head Hadamard H on head_dim
        -> rotate down_proj input with exact Hadamard on intermediate_size
        -> save rotated safetensors to output dir

After this, run `mlx_lm.convert -q --q-bits 4` to produce QuaRot-rotated 4-bit weights.
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Hadamard utilities
# ---------------------------------------------------------------------------

# Precomputed odd-size Hadamard matrices from spcl/QuaRot.
# Used in Kronecker construction for hidden_size / intermediate_size that
# aren't pure powers of 2.

_HAD12 = np.array([
    [+1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [+1,+1,-1,+1,-1,-1,-1,+1,+1,+1,-1,+1],
    [+1,+1,+1,-1,+1,-1,-1,-1,+1,+1,+1,-1],
    [+1,-1,+1,+1,-1,+1,-1,-1,-1,+1,+1,+1],
    [+1,+1,-1,+1,+1,-1,+1,-1,-1,-1,+1,+1],
    [+1,+1,+1,-1,+1,+1,-1,+1,-1,-1,-1,+1],
    [+1,+1,+1,+1,-1,+1,+1,-1,+1,-1,-1,-1],
    [+1,-1,+1,+1,+1,-1,+1,+1,-1,+1,-1,-1],
    [+1,-1,-1,+1,+1,+1,-1,+1,+1,-1,+1,-1],
    [+1,-1,-1,-1,+1,+1,+1,-1,+1,+1,-1,+1],
    [+1,+1,-1,-1,-1,+1,+1,+1,-1,+1,+1,-1],
    [+1,-1,+1,-1,-1,-1,+1,+1,+1,-1,+1,+1],
], dtype=np.float32)

_HAD20 = np.array([
    [+1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1],
    [+1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1],
    [+1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1],
    [+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1],
    [+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1],
    [+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1],
    [+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1],
    [+1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1],
    [+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1],
    [+1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1,-1],
    [+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1,+1],
    [+1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1,-1],
    [+1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1,-1],
    [+1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1,-1],
    [+1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1,-1],
    [+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,+1],
    [+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1],
    [+1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1,-1],
    [+1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,-1,+1,+1,+1,+1,-1,-1,+1,+1],
], dtype=np.float32)


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def sylvester_hadamard(k: int) -> np.ndarray:
    """Construct an unnormalized 2^k x 2^k Sylvester Hadamard matrix.

    H[0] = [[1]]; H[k] = [[H[k-1], H[k-1]], [H[k-1], -H[k-1]]]
    """
    H = np.array([[1]], dtype=np.float32)
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard_matrix(n: int) -> np.ndarray:
    """Build an n x n NORMALIZED Hadamard matrix (H @ H.T = I).

    Tries: pure pow-of-2 (Sylvester), then 12k Kronecker, then 20k Kronecker.
    Raises ValueError if n cannot be factored.
    """
    if is_pow2(n):
        H = sylvester_hadamard(int(math.log2(n)))
        return H / math.sqrt(n)
    if n % 12 == 0 and is_pow2(n // 12):
        H_small = _HAD12
        H_big = sylvester_hadamard(int(math.log2(n // 12)))
        H = np.kron(H_big, H_small)
        return H / math.sqrt(n)
    if n % 20 == 0 and is_pow2(n // 20):
        H_small = _HAD20
        H_big = sylvester_hadamard(int(math.log2(n // 20)))
        H = np.kron(H_big, H_small)
        return H / math.sqrt(n)
    raise ValueError(
        f"Cannot construct exact Hadamard of size {n}. "
        f"Need n = pow2, or n = pow2 * {{12, 20}}."
    )


def random_hadamard(n: int, seed: int) -> np.ndarray:
    """Random Hadamard: H @ diag(rand_signs).

    This is the QuaRot R1: a random orthogonal matrix whose product structure
    keeps it Hadamard-fast at small cost in matmul-quality.
    """
    rng = np.random.default_rng(seed)
    H = hadamard_matrix(n)
    signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    return H * signs[None, :]  # H @ diag(signs)


# ---------------------------------------------------------------------------
# Layer norm fusion (gamma absorbed into adjacent linears)
# ---------------------------------------------------------------------------


def fuse_ln_into_linears(state: dict, ln_key: str, linear_keys: list[str]) -> None:
    """Multiply each linear's input dim (last axis of weight) by RMSNorm gamma.

    Assumes weight shape (out, in). After fusion, the linear's input is
    effectively scaled by gamma, so the LN can be replaced by RMSNorm
    without gamma (gamma == 1). The script writes gamma = 1 to ``ln_key``.
    """
    gamma = state[ln_key].astype(mx.float32)
    for lk in linear_keys:
        W = state[lk].astype(mx.float32)
        # weight shape (out, in); broadcast gamma over `out`
        W = W * gamma[None, :]
        state[lk] = W.astype(state[lk].dtype)
    # Set ln gamma to 1 so the LN is now identity (rotation-passable).
    state[ln_key] = mx.ones_like(state[ln_key])


# ---------------------------------------------------------------------------
# Rotation application
# ---------------------------------------------------------------------------


def rotate_state(state: dict, config: dict, seed: int) -> dict:
    """Apply QuaRot R1 + R2 + R4(weight-only) rotations to a Qwen3 state dict.

    Mutates `state` and returns it.

    R1: random Hadamard Q on hidden_size, absorbed into embed, q/k/v/o, gate/up/down, lm_head.
    R2: exact Hadamard on head_dim, absorbed into v_proj output and o_proj input.
    R4(weight-only): exact Hadamard on intermediate_size, absorbed into down_proj input.
    """
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", hidden // n_heads)
    n_layers = config["num_hidden_layers"]
    tied = bool(config.get("tie_word_embeddings", False))

    print(f"  hidden={hidden}, intermediate={intermediate}, head_dim={head_dim}")
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, layers={n_layers}, tied={tied}")

    # ------------------------------------------------------------
    # Step 0: untie lm_head from embed_tokens (if tied).
    # We need a separate lm_head so we can fuse final-norm gamma into it
    # without touching the input embeddings.
    # ------------------------------------------------------------
    if tied:
        if "lm_head.weight" not in state:
            print("  untying: lm_head.weight = embed_tokens.weight.copy()")
            state["lm_head.weight"] = state["model.embed_tokens.weight"]

    # ------------------------------------------------------------
    # Step 1: fuse layer norms (gamma absorbed into adjacent linears).
    # ------------------------------------------------------------
    print("  fusing layer norms...")
    for layer in range(n_layers):
        # input_layernorm -> q/k/v_proj
        fuse_ln_into_linears(
            state,
            f"model.layers.{layer}.input_layernorm.weight",
            [
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"model.layers.{layer}.self_attn.k_proj.weight",
                f"model.layers.{layer}.self_attn.v_proj.weight",
            ],
        )
        # post_attention_layernorm -> gate/up_proj
        fuse_ln_into_linears(
            state,
            f"model.layers.{layer}.post_attention_layernorm.weight",
            [
                f"model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
            ],
        )
    # final model.norm -> lm_head
    fuse_ln_into_linears(state, "model.norm.weight", ["lm_head.weight"])

    # ------------------------------------------------------------
    # Step 2: build random Hadamard Q for R1.
    # ------------------------------------------------------------
    print(f"  building random Hadamard Q ({hidden}x{hidden})...")
    Q_np = random_hadamard(hidden, seed)
    Q = mx.array(Q_np, dtype=mx.float32)

    # ------------------------------------------------------------
    # Step 3: apply R1 (rotate weights along the hidden_size axis).
    # ------------------------------------------------------------
    print("  applying R1 to embed/lm_head/attention/mlp...")

    # embed_tokens: (vocab, hidden) — rotate output dim (axis=1)
    _W = state["model.embed_tokens.weight"].astype(mx.float32)
    state["model.embed_tokens.weight"] = (_W @ Q).astype(state["model.embed_tokens.weight"].dtype)
    # lm_head: (vocab, hidden) — rotate input dim (axis=1) [final layer output is Q-rotated]
    _W = state["lm_head.weight"].astype(mx.float32)
    state["lm_head.weight"] = (_W @ Q).astype(state["lm_head.weight"].dtype)

    for layer in range(n_layers):
        # q/k/v_proj: (out, hidden) — rotate input dim (axis=1)
        for which in ("q_proj", "k_proj", "v_proj"):
            key = f"model.layers.{layer}.self_attn.{which}.weight"
            _W = state[key].astype(mx.float32)
            state[key] = (_W @ Q).astype(state[key].dtype)
        # o_proj: (hidden, in) — rotate output dim (axis=0)
        key = f"model.layers.{layer}.self_attn.o_proj.weight"
        _W = state[key].astype(mx.float32)
        state[key] = (Q.T @ _W).astype(state[key].dtype)
        # gate/up_proj: (intermediate, hidden) — rotate input dim (axis=1)
        for which in ("gate_proj", "up_proj"):
            key = f"model.layers.{layer}.mlp.{which}.weight"
            _W = state[key].astype(mx.float32)
            state[key] = (_W @ Q).astype(state[key].dtype)
        # down_proj: (hidden, intermediate) — rotate output dim (axis=0)
        key = f"model.layers.{layer}.mlp.down_proj.weight"
        _W = state[key].astype(mx.float32)
        state[key] = (Q.T @ _W).astype(state[key].dtype)

    # ------------------------------------------------------------
    # Step 4: R2 — exact Hadamard on head_dim, V-O path.
    # ------------------------------------------------------------
    print(f"  applying R2 (head_dim Hadamard, head_dim={head_dim})...")
    H_head_np = hadamard_matrix(head_dim)
    H_head = mx.array(H_head_np, dtype=mx.float32)

    for layer in range(n_layers):
        # v_proj: (n_kv_heads*head_dim, hidden) — rotate per-kv-head OUTPUT.
        # Reshape: (n_kv_heads, head_dim, hidden) -> (n_kv_heads, H @ head_dim, hidden)
        key = f"model.layers.{layer}.self_attn.v_proj.weight"
        W = state[key].astype(mx.float32)
        W = W.reshape(n_kv_heads, head_dim, hidden)
        W = mx.einsum("ij,kjh->kih", H_head, W)  # H_head @ W along head_dim
        W = W.reshape(n_kv_heads * head_dim, hidden)
        state[key] = W.astype(state[key].dtype)

        # o_proj: (hidden, n_heads*head_dim) — rotate per-head INPUT.
        # Reshape: (hidden, n_heads, head_dim) -> (hidden, n_heads, head_dim @ H)
        key = f"model.layers.{layer}.self_attn.o_proj.weight"
        W = state[key].astype(mx.float32)
        W = W.reshape(hidden, n_heads, head_dim)
        W = mx.einsum("hki,ij->hkj", W, H_head)  # W @ H_head along head_dim
        W = W.reshape(hidden, n_heads * head_dim)
        state[key] = W.astype(state[key].dtype)

    # ------------------------------------------------------------
    # NOTE: R4 (down_proj input Hadamard on intermediate_size) is INTENTIONALLY
    # SKIPPED. R4 requires an online Hadamard on the activation flowing into
    # down_proj. In W4A4 (full QuaRot e2e) that runtime Hadamard suppresses
    # activation outliers before quant. In W4A16 (weight-only) there is no
    # runtime Hadamard, so absorbing R4 into down_proj.weight produces
    # mathematically wrong outputs (y = W @ H @ x != W @ x). Confirmed by the
    # garbage-output smoke test on Qwen3-0.6B-bf16 (rev 1).
    #
    # For W4A16 weight-only, only R1 + R2 are absorbable. R3/R4 require
    # runtime kernels.

    return state


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_state(input_dir: Path) -> tuple[dict, dict]:
    """Load all safetensors files in `input_dir` into a flat dict; also load config."""
    state = {}
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files in {input_dir}")
    print(f"loading {len(safetensor_files)} safetensors file(s) from {input_dir}")
    for f in safetensor_files:
        partial = mx.load(str(f))
        state.update(partial)
    config = json.loads((input_dir / "config.json").read_text())
    return state, config


def save_state(state: dict, input_dir: Path, output_dir: Path, config: dict, untied: bool) -> None:
    """Save state to output_dir; copy non-weight files verbatim.

    If `untied`, write a modified config.json with tie_word_embeddings=False so
    mlx_lm.load uses the saved lm_head.weight instead of tying embed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save as a single safetensors for simplicity (4B fits, 27B may need sharding).
    print(f"saving {len(state)} tensors to {output_dir}/model.safetensors")
    mx.save_safetensors(str(output_dir / "model.safetensors"), state)
    # Copy non-weight files, EXCEPT config.json which we rewrite if untied.
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "generation_config.json",
        "chat_template.jinja",
    ]:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    # Write updated config.
    out_config = dict(config)
    if untied:
        out_config["tie_word_embeddings"] = False
        print("  config: tie_word_embeddings -> False (untied)")
    (output_dir / "config.json").write_text(json.dumps(out_config, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="QuaRot rotation precompute for Qwen3 (MLX)")
    parser.add_argument("--input", required=True, help="Path to FP16/BF16 Qwen3 model dir")
    parser.add_argument("--output", required=True, help="Output dir for rotated weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Hadamard sign matrix")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        sys.exit(f"input dir not found: {input_dir}")

    print(f"=== QuaRot rotate: {input_dir} -> {output_dir} ===")
    state, config = load_state(input_dir)

    model_type = config.get("model_type", "?")
    if model_type != "qwen3":
        print(f"WARNING: model_type={model_type!r}, this script is designed for 'qwen3'")

    untied_flag = bool(config.get("tie_word_embeddings", False))
    rotated = rotate_state(state, config, args.seed)
    save_state(rotated, input_dir, output_dir, config, untied=untied_flag)
    print("=== done ===")


if __name__ == "__main__":
    main()
