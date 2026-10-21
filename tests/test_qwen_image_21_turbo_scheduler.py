# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 Turbo sampler against the pinned mflux scheduler contract."""

from __future__ import annotations

import types

import pytest

pytestmark = pytest.mark.requires_mlx

pytest.importorskip("mlx.core")
pytest.importorskip("mflux")

import mlx.core as mx  # noqa: E402
from mflux.models.common.schedulers import try_import_external_scheduler  # noqa: E402
from mflux.models.common.schedulers.base_scheduler import BaseScheduler  # noqa: E402

from rapid_mlx.image.qwen21_turbo import SCHEDULER_PATH, TURBO_SIGMAS  # noqa: E402
from rapid_mlx.image.qwen21_turbo_scheduler import Qwen21TurboScheduler  # noqa: E402


def _config(steps):
    return types.SimpleNamespace(num_inference_steps=steps)


def test_mflux_resolves_the_dotted_scheduler_path():
    cls = try_import_external_scheduler(SCHEDULER_PATH)
    assert cls is Qwen21TurboScheduler
    assert issubclass(cls, BaseScheduler)


def test_sigmas_timesteps_and_euler_step():
    scheduler = Qwen21TurboScheduler(_config(8))
    assert scheduler.sigmas.tolist() == pytest.approx([*TURBO_SIGMAS, 0.0])
    assert scheduler.timesteps.tolist() == list(range(8))
    latents = mx.ones((1, 4), dtype=mx.bfloat16)
    noise = mx.full((1, 4), 2.0, dtype=mx.bfloat16)
    stepped = scheduler.step(noise=noise, timestep=7, latents=latents)
    assert stepped.dtype == mx.bfloat16
    # Final node 0.414568 -> 0: x + v * (0 - 0.414568).
    assert stepped.tolist()[0] == pytest.approx([1.0 - 2 * 0.414568] * 4, abs=1e-2)


def test_rejects_other_step_counts():
    with pytest.raises(ValueError, match="8-step schedule only"):
        Qwen21TurboScheduler(_config(4))
