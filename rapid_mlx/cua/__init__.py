"""Productized computer-use agent: native AX execution layer + configurable brains.

Fast thinking (outcome routing, fixation detection) is always on-device.
Slow thinking (planning) is whatever the user configures: cloud GLM, a local
Rapid-MLX server, or any loopback OpenAI-compatible endpoint.
"""

from rapid_mlx.cua.config import CUAConfig, PlannerConfig, resolve_planner

__all__ = ["CUAConfig", "PlannerConfig", "resolve_planner"]
