# SPDX-License-Identifier: Apache-2.0
"""DDTree speculative-decoding integration (issue #879).

DDTree builds a draft tree from DFlash draft-model logits and verifies
multiple candidate continuations per target-model pass. The MVP keeps it
behind an explicit ``--enable-ddtree`` flag and a dedicated single-user
server, using the external ``dtree_mlx`` package as the runtime boundary.
"""

from .eligibility import DDTreeUnavailable, check
from .runtime import load_runtime

__all__ = ["DDTreeUnavailable", "check", "load_runtime"]
