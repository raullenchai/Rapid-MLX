# SPDX-License-Identifier: Apache-2.0
"""Allow ``python -m scripts.pr_validate <PR#>`` as the README claims.

Without this file, the package can only be invoked as
``python -m scripts.pr_validate.pr_validate <PR#>`` — surprising to
anyone who follows the documented invocation.
"""

from __future__ import annotations

import sys

from .pr_validate import main

if __name__ == "__main__":
    sys.exit(main())
