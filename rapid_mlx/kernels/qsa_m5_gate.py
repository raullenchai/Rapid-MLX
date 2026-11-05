# SPDX-License-Identifier: Apache-2.0
"""Device-independent admission rules for experimental M5 QSA prefill."""

from __future__ import annotations

import re


def architecture_supported(architecture: str, macos_version: str) -> bool:
    """Mirror the runtime architecture/OS gate; never use a compile probe."""
    match = re.fullmatch(r"applegpu_g(\d+)([a-z])", architecture)
    try:
        release = tuple(int(p) for p in macos_version.split(".")[:2])
    except ValueError:
        return False
    return bool(
        match
        and len(release) == 2
        and release >= (26, 2)
        and int(match[1]) >= (18 if match[2] == "p" else 17)
    )


def envelope_supported(query_length: int, *, batch_size: int, training: bool) -> bool:
    return batch_size == 1 and query_length >= 64 and not training


def admit(query_length, *, batch_size, training, enabled, hardware, on_gpu) -> bool:
    """Short-circuit before touching the runtime on every declined envelope."""
    return bool(
        enabled
        and envelope_supported(query_length, batch_size=batch_size, training=training)
        and hardware()
        and on_gpu()
    )
