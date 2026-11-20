"""``python -m rapid_mlx`` — same behaviour as the ``rapid-mlx`` console script.

Importing :mod:`rapid_mlx.cli` (instead of executing it as ``__main__``)
keeps a single CLI module, so startup instrumentation installed on it is
not bypassed. The supervised serve runner launches through this module.
"""

from rapid_mlx.cli import cli_entrypoint

if __name__ == "__main__":
    cli_entrypoint()
