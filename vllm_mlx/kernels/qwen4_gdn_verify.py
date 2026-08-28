# SPDX-License-Identifier: MIT
"""Gated-delta verify kernel that also returns rollback boundaries.

The ordinary MLX gated-delta kernel returns only the final recurrent state.
Native MTP verification needs the state after each potentially committed
position so a rejected draft can restore the target cache without replaying
the accepted prefix.  This batch-one Metal path performs the same recurrence
once and writes only the intermediate states that can become rollback
boundaries.  Unsupported shapes use the transparent MLX reference path.

The kernel is adapted from the MLX Qwen hybrid-model implementation.
"""

from __future__ import annotations

from functools import cache, partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def _compute_g_beta(a_log, alpha, beta, dt_bias):
    decay = mx.exp(-mx.exp(a_log.astype(mx.float32)) * nn.softplus(alpha + dt_bias))
    return decay, mx.sigmoid(beta)


@cache
def _kernel(has_mask: bool):
    if not mx.metal.is_available():  # pragma: no cover - Apple-only module
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        boundaries += ((b_idx * StateT * Hv + hv_idx) * Dv) * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        auto boundary_ = boundaries + dv_idx * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        for (int t = 0; t < T; ++t) {{
            if ({mask_source}) {{
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    kv_mem += state[i] * k_[s_idx];
                }}
                kv_mem = simd_sum(kv_mem);
                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                    out += state[i] * q_[s_idx];
                }}
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {{
                    y[dv_idx] = static_cast<InT>(out);
                }}
            }} else {{
                y[dv_idx] = static_cast<InT>(0);
            }}

            if (t < StateT) {{
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    boundary_[s_idx] = static_cast<StT>(state[i]);
                }}
                boundary_ += Hv * Dv * Dk;
            }}
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    return mx.fast.metal_kernel(
        name=f"rapid_qwen4_gdn_verify{'_mask' if has_mask else ''}",
        input_names=inputs,
        output_names=["y", "state_out", "boundaries"],
        source=source,
    )


def _reference(q, k, v, g, beta, state, mask):
    batch, steps, key_heads, _ = q.shape
    value_heads = v.shape[-2]
    repeat = value_heads // key_heads
    if repeat > 1:
        q = mx.repeat(q, repeat, axis=-2)
        k = mx.repeat(k, repeat, axis=-2)
    outputs = []
    boundaries = []
    for position in range(steps):
        old_state = state
        state = state * g[:, position, :, None, None]
        memory = (state * k[:, position, :, None, :]).sum(axis=-1)
        delta = (v[:, position] - memory) * beta[:, position, :, None]
        state = state + k[:, position, :, None, :] * delta[..., None]
        output = (state * q[:, position, :, None, :]).sum(axis=-1)
        if mask is not None:
            valid = mask[:, position]
            state = mx.where(valid[:, None, None, None], state, old_state)
            output = mx.where(valid[:, None, None], output, 0)
        outputs.append(output.astype(q.dtype))
        if position < steps - 1:
            boundaries.append(state)
    if boundaries:
        rollback_states = mx.stack(boundaries, axis=1)
    else:
        rollback_states = mx.zeros(
            (batch, 0, value_heads, v.shape[-1], k.shape[-1]),
            dtype=state.dtype,
        )
    return mx.stack(outputs, axis=1), state, rollback_states


def gated_delta_verify_with_states(
    query,
    key,
    value,
    alpha,
    beta,
    a_log,
    dt_bias,
    state=None,
    mask=None,
    *,
    use_kernel: bool = True,
):
    """Return ``(output, final_state, rollback_states)`` for a verify block."""

    decay, update = _compute_g_beta(a_log, alpha, beta, dt_bias)
    batch, steps, key_heads, key_dim = key.shape
    value_heads, value_dim = value.shape[2:]
    if state is None:
        state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32)
    kernel = _kernel(mask is not None) if use_kernel else None
    if kernel is None or mx.default_device() != mx.gpu or key_dim % 32 or steps < 2:
        return _reference(query, key, value, decay, update, state, mask)

    inputs = [query, key, value, decay, update, state, steps]
    if mask is not None:
        inputs.append(mask)
    state_steps = steps - 1
    return kernel(
        inputs=inputs,
        template=[
            ("InT", query.dtype),
            ("StT", state.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", key_heads),
            ("Hv", value_heads),
            ("StateT", state_steps),
        ],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (batch, steps, value_heads, value_dim),
            state.shape,
            (batch, state_steps, value_heads, value_dim, key_dim),
        ],
        output_dtypes=[query.dtype, state.dtype, state.dtype],
    )
