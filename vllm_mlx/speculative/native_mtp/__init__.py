# SPDX-License-Identifier: Apache-2.0
"""Native MTP speculative decoding helpers."""

from .sampling import (
    MTPDistributionParams,
    acceptance_mask,
    distribution_logprobs,
    residual_logprobs,
    sample_from_logprobs,
)

__all__ = [
    "MTPDistributionParams",
    "acceptance_mask",
    "distribution_logprobs",
    "residual_logprobs",
    "sample_from_logprobs",
]
