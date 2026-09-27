"""Model-agnostic computer-use tool layer for macOS (Orca-style).

Agents observe and act through the `rapid-mlx computer` CLI; this package
never calls a model. See cli.py for the command surface and errors.py for
the typed error taxonomy with recovery hints.
"""

from .errors import ComputerUseError

__all__ = ["ComputerUseError"]
