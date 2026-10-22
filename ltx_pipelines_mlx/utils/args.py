"""CLI argument helpers — mirrors upstream ``ltx_pipelines.utils.args``.

Currently exposes the multi-image I2V conditioning input used by the
text-to-video pipelines: ``ImageConditioningInput`` + ``ImageAction``.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

# Re-export for legacy import paths. Single source of truth lives in
# ``ltx_pipelines_mlx.utils.media_io`` (mirrors upstream's
# ``ltx_pipelines.utils.media_io.DEFAULT_IMAGE_CRF``).
from ltx_pipelines_mlx.utils.media_io import DEFAULT_IMAGE_CRF


class ImageConditioningInput(NamedTuple):
    """One image conditioning entry for multi-anchor I2V.

    Mirrors upstream ``ltx_pipelines.utils.args.ImageConditioningInput``
    verbatim.

    Args:
        path: Path to the image file.
        frame_idx: Target latent frame index. ``0`` replaces the first
            latent frame (``VideoConditionByLatentIndex``); any other
            value appends a keyframe at that pixel-frame position
            (``VideoConditionByKeyframeIndex``).
        strength: Conditioning strength in ``[0, 1]``. ``1.0`` = fully
            preserved.
        crf: Optional H.264 CRF for input degradation (default 33).
            Currently accepted for API parity but not applied.
    """

    path: str
    frame_idx: int
    strength: float
    crf: int = DEFAULT_IMAGE_CRF


class ImageAction(argparse.Action):
    """Variadic argparse action accepting ``PATH [FRAME_IDX [STRENGTH [CRF]]]``.

    Repeatable. Each invocation appends one
    :class:`ImageConditioningInput` to the namespace list.

    Backward-compat: ``--image PATH`` alone is accepted and defaults to
    ``frame_idx=0, strength=1.0, crf=33`` — matches the prior single-arg
    API. The strict upstream form is ``--image PATH FRAME_IDX STRENGTH
    [CRF]`` (3 or 4 args).
    """

    def __call__(  # type: ignore[override]
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        if not isinstance(values, list):
            values = [values]
        if len(values) not in (1, 3, 4):
            parser.error(
                f"{option_string}: expected 1, 3, or 4 values "
                "(PATH [FRAME_IDX STRENGTH [CRF]]), got "
                f"{len(values)}: {values}"
            )

        path = values[0]
        if len(values) == 1:
            frame_idx, strength, crf = 0, 1.0, DEFAULT_IMAGE_CRF
        else:
            try:
                frame_idx = int(values[1])
                strength = float(values[2])
                crf = int(values[3]) if len(values) == 4 else DEFAULT_IMAGE_CRF
            except (ValueError, TypeError) as e:
                parser.error(f"{option_string}: could not parse FRAME_IDX/STRENGTH/CRF from {values[1:]}: {e}")

        item = ImageConditioningInput(
            path=path,
            frame_idx=frame_idx,
            strength=strength,
            crf=crf,
        )

        existing = getattr(namespace, self.dest, None) or []
        existing.append(item)
        setattr(namespace, self.dest, existing)


__all__ = ["DEFAULT_IMAGE_CRF", "ImageAction", "ImageConditioningInput"]
