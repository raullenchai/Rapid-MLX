"""HDR utilities: LogC3 compression for HDR IC-LoRA training and inference.

MLX-native port of upstream ``ltx_core.hdr``. Provides compress /
decompress + postprocess helpers for HDR video generation. Used by
the HDR IC-LoRA pipeline at decode time.

LogC3 is the ARRI camera log curve (EI 800). It maps linear scene
intensities ``[0, ∞)`` into a perceptually-uniform ``[0, 1]`` window so
the VAE — trained on ``[-1, 1]`` SDR pixel ranges — can carry HDR
information without clipping highlights. The decode-time inverse
expands the compressed range back to linear HDR.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx


class LogC3:
    """ARRI LogC3 (EI 800) HDR compression.

    Compresses linear ``[0, ∞)`` to LogC3 ``[0, 1]`` via a piecewise
    log/linear curve. The log branch allocates more precision to
    shadows / midtones and compresses highlights smoothly. Below the
    cut, a linear branch keeps the curve continuous and differentiable.

    Callers are responsible for mapping the LogC3 ``[0, 1]`` output to
    the VAE's ``[-1, 1]`` input range (and the inverse on decode).

    Constants match upstream ``ltx_core.hdr.LogC3`` verbatim.
    """

    name = "LogC3"
    A = 5.555556
    B = 0.052272
    C = 0.247190
    D = 0.385537
    E = 5.367655
    F = 0.092809
    CUT = 0.010591

    def compress(self, hdr: mx.array) -> mx.array:
        """Compress linear HDR ``[0, ∞)`` → LogC3 ``[0, 1]``."""
        x = mx.maximum(hdr, 0.0)
        log_part = self.C * mx.log10(self.A * x + self.B) + self.D
        lin_part = self.E * x + self.F
        logc = mx.where(x >= self.CUT, log_part, lin_part)
        return mx.clip(logc, 0.0, 1.0)

    def compress_ldr(self, ldr: mx.array) -> mx.array:
        """Compress LDR ``[0, 1]`` → ``[0, 1]`` (no log curve, just clamp)."""
        return mx.clip(ldr, 0.0, 1.0)

    def decompress(self, logc: mx.array) -> mx.array:
        """Decompress LogC3 ``[0, 1]`` → linear HDR ``[0, ∞)``."""
        logc = mx.clip(logc, 0.0, 1.0)
        cut_log = self.E * self.CUT + self.F
        lin_from_log = (mx.power(mx.array(10.0), (logc - self.D) / self.C) - self.B) / self.A
        lin_from_lin = (logc - self.F) / self.E
        return mx.where(logc >= cut_log, lin_from_log, lin_from_lin)

    def decompress_ldr(self, logc: mx.array) -> mx.array:
        """Decompress ``[0, 1]`` → LDR ``[0, 1]`` (identity clamp)."""
        return mx.clip(logc, 0.0, 1.0)


def apply_hdr_decode_postprocess(
    decoded_video: mx.array,
    transform: Literal["logc3"] = "logc3",
) -> mx.array:
    """Apply HDR decompress to VAE decode output for HDR recovery.

    Args:
        decoded_video: VAE decode output in ``[0, 1]``, shape
            ``(B, C, F, H, W)``. Must be ``float32`` for sufficient
            color resolution (HDR linear values can span many orders
            of magnitude).
        transform: HDR transform to apply. Currently only ``"logc3"``.

    Returns:
        Linear HDR video tensor, ``float32``.
    """
    decoded_video = decoded_video.astype(mx.float32)
    if transform == "logc3":
        return LogC3().decompress(decoded_video)
    raise ValueError(f"Unsupported HDR transform: {transform}")
