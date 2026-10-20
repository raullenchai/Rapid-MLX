# SPDX-License-Identifier: Apache-2.0
"""The one TensorFold runtime every qualified Rapid profile is measured on.

One environment holds one ``tensorfold`` distribution, so every profile pins
the same release and source revision; a profile qualified on another revision
could not be installed beside the others.
"""

from __future__ import annotations

import shlex
import sys

SUPPORTED_VERSION = "0.6.6"
SUPPORTED_REVISION = "cb2ebf0540f42604e2759b2ddef497861e928248"
SUPPORTED_RUNTIME_URL = "https://github.com/ashhart/TensorFold.git"
SUPPORTED_MLX_VERSION = "0.32.3"
# Keep aligned with the release-tested Desktop sidecar constraints.
SUPPORTED_MLX_AUDIO_VERSION = "0.5.3"
INSTALL_HINT = (
    "Install the qualified TensorFold runtime from its vetted revision with:\n"
    f'    {shlex.quote(sys.executable)} -m pip install "mlx=={SUPPORTED_MLX_VERSION}" '
    f'"mlx-audio=={SUPPORTED_MLX_AUDIO_VERSION}" "tensorfold @ '
    f'git+{SUPPORTED_RUNTIME_URL}@{SUPPORTED_REVISION}"'
)
