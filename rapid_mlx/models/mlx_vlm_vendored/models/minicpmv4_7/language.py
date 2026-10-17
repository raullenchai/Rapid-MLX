# SPDX-License-Identifier: MIT
# Copyright © 2025 Prince Canuma
# Vendored from mlx-vlm @ b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0; see LICENSE and provenance.json.
# VENDOR-DEVIATION(import-redirects): use pinned mlx_vlm modules and the
# local 0.7.2 compatibility slices; all other code is upstream verbatim.
import mlx.core as mx

from mlx_vlm.models.minicpmv4_6.language import LanguageModel as MiniCPMVLanguageModel
from mlx_vlm.models.qwen3_5_moe.language import LanguageModel as Qwen35MoELanguageModel


class LanguageModel(MiniCPMVLanguageModel):
    def get_rope_index(self, input_ids, *args, **kwargs):
        # APC cannot rebuild a canvas from token IDs without the crop grids.
        visual = (input_ids == self.config.image_token_id) | (
            input_ids == self.config.video_token_id
        )
        if mx.any(visual).item():
            raise ValueError(
                "MiniCPM-V 4.7 visual positions require crop grids from get_input_embeddings."
            )
        return super().get_rope_index(input_ids, *args, **kwargs)


class MoELanguageModel(Qwen35MoELanguageModel, LanguageModel):
    """Qwen3.5 experts with the MiniCPM canvas-position contract."""
