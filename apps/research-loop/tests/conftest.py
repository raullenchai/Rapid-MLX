"""Make the standalone workbench modules importable in tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
