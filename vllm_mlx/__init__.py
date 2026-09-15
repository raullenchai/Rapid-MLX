# SPDX-License-Identifier: Apache-2.0
"""Deprecated import shim: the package moved to ``rapid_mlx``.

Rapid-MLX 0.14.2 and earlier shipped the importable package as
``vllm_mlx`` while the distribution, CLI, and product were already named
``rapid-mlx``. The Python package has now been renamed to ``rapid_mlx``
to match; this shim keeps ``import vllm_mlx`` working for scripts written
against pre-rename releases.

What still works through this shim:

- ``import vllm_mlx`` / ``from vllm_mlx import SamplingParams``
- ``import vllm_mlx.<anything>`` — submodules resolve to the
  corresponding ``rapid_mlx`` module (same objects, no duplicate import)
- ``python -m vllm_mlx.server`` (see ``vllm_mlx/server.py``)

Every entry emits a ``DeprecationWarning`` pointing at the ``rapid_mlx``
name. The shim will be retired after a deprecation window (at least one
minor series); move your imports now::

    from rapid_mlx import SamplingParams   # instead of vllm_mlx
    python -m rapid_mlx.server             # instead of vllm_mlx.server
"""

# NOTE: deliberately NOT ``from rapid_mlx import *`` — rapid_mlx's
# ``__init__`` is lazily loaded so that a bare ``import rapid_mlx`` never
# pulls in mlx.core (which can SIGABRT on non-Apple-Silicon hosts). A
# star-import here would eagerly trigger every lazy attribute. Attribute
# forwarding through ``__getattr__`` below preserves the laziness.

import importlib
import importlib.util
import sys
import warnings

import rapid_mlx as _rapid_mlx

__version__ = _rapid_mlx.__version__

_DEPRECATION_MESSAGE = (
    "The 'vllm_mlx' package has been renamed to 'rapid_mlx'; this import "
    "shim is deprecated and will be removed in a future release. "
    "Update your imports (e.g. 'from rapid_mlx import SamplingParams', "
    "'python -m rapid_mlx.server')."
)

_PREFIX = "vllm_mlx."
_TARGET_PREFIX = "rapid_mlx."


class _RapidMlxModuleAliasLoader:
    """Loader that reuses an already-imported ``rapid_mlx`` module object.

    ``create_module`` returns the target module itself, so the import
    system registers the *same* module instance under the legacy
    ``vllm_mlx.*`` name — no duplicate module objects, no double
    execution, and ``isinstance`` checks keep working across both names.
    """

    def __init__(self, target_module):
        self._target = target_module

    def create_module(self, spec):
        return self._target

    def exec_module(self, module):
        pass  # already executed as rapid_mlx.<sub>

    def is_package(self, fullname):
        return hasattr(self._target, "__path__")

    def get_filename(self, fullname):
        return getattr(self._target, "__file__", None)

    def get_code(self, fullname):
        # Lets ``python -m vllm_mlx.cli`` (and any other legacy -m
        # invocation) work: runpy pulls the code object through the
        # aliased module's real loader.
        target_loader = getattr(self._target, "__spec__", None)
        target_loader = target_loader.loader if target_loader else None
        if target_loader is None:
            return None
        return target_loader.get_code(self._target.__name__)

    def get_source(self, fullname):
        target_loader = getattr(self._target, "__spec__", None)
        target_loader = target_loader.loader if target_loader else None
        if target_loader is None:
            return None
        return target_loader.get_source(self._target.__name__)


class _RapidMlxModuleAliasFinder:
    """Resolve ``import vllm_mlx.<sub>`` to ``rapid_mlx.<sub>``.

    Registered at the END of ``sys.meta_path`` so a physical file inside
    this shim package (e.g. ``vllm_mlx/server.py``) always wins, and the
    finder only fills in the names this shim does not ship itself.
    """

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        target_name = _TARGET_PREFIX + fullname[len(_PREFIX) :]
        try:
            target_module = importlib.import_module(target_name)
        except ImportError:
            return None  # propagate the real ImportError for unknown names
        loader = _RapidMlxModuleAliasLoader(target_module)
        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=getattr(target_module, "__file__", None),
            is_package=hasattr(target_module, "__path__"),
        )


def _install_alias_finder() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _RapidMlxModuleAliasFinder):
            return  # idempotent (e.g. interpreter reload scenarios)
    sys.meta_path.append(_RapidMlxModuleAliasFinder())


_install_alias_finder()


def __getattr__(name):
    """Forward attribute access to ``rapid_mlx`` (preserves its laziness)."""
    return getattr(_rapid_mlx, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_rapid_mlx)))


warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
