# SPDX-License-Identifier: Apache-2.0
"""Model-validation benchmarks.

This package owns the model-side checks (smoke/speed/harness/stress/agents)
that used to live under ``vllm_mlx.doctor.checks``.  ``doctor`` is now
strictly environment-health; anything that boots a server or measures a
model belongs here.
"""
