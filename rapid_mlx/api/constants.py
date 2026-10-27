# SPDX-License-Identifier: Apache-2.0
"""
Wire-level constants shared across the api / service layers.

Lives in ``rapid_mlx.api`` to preserve the layering rule that
``rapid_mlx.api`` is the lower layer (no engine / MLX imports) and
``rapid_mlx.service`` may depend on it. Constants placed here must
remain pure-literal so that importing this module never pulls in
MLX, Metal, or any engine code — adapter unit tests in headless
environments depend on this.
"""
