# SPDX-License-Identifier: Apache-2.0
"""Deprecated import shim: the package moved to ``rapid_mlx``.

Rapid-MLX 0.14.2 and earlier shipped the importable package as
``vllm_mlx`` while the distribution, CLI, and product were already named
``rapid-mlx``. The Python package has now been renamed to ``rapid_mlx``
to match; this shim keeps ``import vllm_mlx`` working for scripts written
against pre-rename releases.

What still works through this shim:

- ``import vllm_mlx`` / ``from vllm_mlx import SamplingParams``
- ``import vllm_mlx.<anything>`` — submodules (at any depth) resolve to
  the corresponding ``rapid_mlx`` module: the SAME module object, no
  duplicate import, so ``isinstance`` identity survives across both names
- ``python -m vllm_mlx.server`` / ``python -m vllm_mlx.cli``

Note on warning visibility: the ``DeprecationWarning`` fires on
``import vllm_mlx`` and is shown by default for scripts (anything run as
``__main__``), which is the primary migration audience. Under the
default warning filter it is hidden for imports that happen inside
installed libraries — pass ``-W default::DeprecationWarning`` to see it
everywhere.

The shim will remain available across multiple minor releases and can only be
retired in a separately announced breaking release; move your imports now::

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


class _RapidMlxLegacyResourceLoader:
    """Keep package discovery/resources working through the legacy name."""

    def __init__(self, legacy_loader, target_loader):
        self._legacy_loader = legacy_loader
        self._target_loader = target_loader

    def __getattr__(self, name):
        return getattr(self._legacy_loader, name)

    def get_resource_reader(self, fullname):
        get_reader = getattr(self._target_loader, "get_resource_reader", None)
        return None if get_reader is None else get_reader(_rapid_mlx.__name__)


# The shim directory intentionally contains only this module. Point package
# discovery at the renamed tree so integrations that enumerate
# ``vllm_mlx.__path__`` keep seeing the same submodules during the deprecation
# window. Resource APIs consult the package loader rather than ``__path__``,
# so delegate those reads separately; this preserves access to top-level data
# such as aliases.json without shipping a second, drift-prone copy.
__path__ = list(_rapid_mlx.__path__)
if __spec__ is not None:
    __spec__.submodule_search_locations = list(__path__)
    if __loader__ is not None and _rapid_mlx.__loader__ is not None:
        __loader__ = _RapidMlxLegacyResourceLoader(__loader__, _rapid_mlx.__loader__)
        __spec__.loader = __loader__

# ``from vllm_mlx import *`` must keep yielding the same engine names as
# before the rename (the pre-rename package defined the identical
# ``__all__``). Star-import resolves each name through ``__getattr__``,
# triggering the same lazy loads the old package performed.
__all__ = list(_rapid_mlx.__all__)

_DEPRECATION_MESSAGE = (
    "The 'vllm_mlx' package has been renamed to 'rapid_mlx'; this import "
    "shim is deprecated and will be removed in a future release. "
    "Update your imports (e.g. 'from rapid_mlx import SamplingParams', "
    "'python -m rapid_mlx.server')."
)

_PREFIX = "vllm_mlx."
_TARGET_PREFIX = "rapid_mlx."


class _RapidMlxModuleAliasLoader:
    """Loader that makes ``vllm_mlx.<sub>`` resolve to ``rapid_mlx.<sub>``.

    ``create_module`` imports the target module and returns it, so the
    import system registers the *same* module instance under the legacy
    ``vllm_mlx.*`` name — no duplicate module objects, no double
    execution, and ``isinstance`` checks keep working across both names.
    ``get_code``/``get_source`` serve ``python -m vllm_mlx.<mod>`` (runpy
    pulls the code object through this loader) without importing the
    target first, so a legacy ``-m`` invocation executes the module file
    exactly once, as ``__main__`` — same as before the rename.
    """

    def __init__(self, target_name, target_spec):
        self._target_name = target_name
        self._target_spec = target_spec  # real spec of the rapid_mlx target

    def create_module(self, spec):
        module = importlib.import_module(self._target_name)
        self._target = module
        # Snapshot the target's real spec BEFORE the import machinery's
        # ``_init_module_attrs`` stamps our alias spec onto the shared
        # module object (see ``exec_module``).
        self._original_spec = module.__spec__
        return module

    def exec_module(self, module):
        # ``_init_module_attrs`` just stamped OUR alias spec onto the
        # shared module object. Restore the real metadata, otherwise:
        #   - ``importlib.reload(rapid_mlx.<sub>)`` would silently no-op
        #     (it re-uses ``__spec__``, whose exec_module is a no-op), and
        #   - ``__spec__.loader`` would be this alias loader, making
        #     ``get_code`` recurse into itself.
        # ``__name__``/``__file__`` survive on their own (already set on
        # the module); ``__spec__``/``__loader__``/``__package__`` are
        # unconditionally rewritten and must be restored here. A concurrent
        # ``reload()`` racing the brief clobber window (between
        # ``module_from_spec`` and this hook) is theoretically observable
        # but requires timing the first-ever alias import of that module;
        # the restore makes the steady state correct.
        original = self._original_spec
        if original is not None:
            module.__spec__ = original
            module.__loader__ = original.loader
            module.__package__ = original.parent
            search = original.submodule_search_locations
            if search is not None:
                module.__path__ = list(search)

    def get_code(self, fullname):
        loader = self._target_spec.loader if self._target_spec else None
        return None if loader is None else loader.get_code(self._target_name)

    def get_source(self, fullname):
        loader = self._target_spec.loader if self._target_spec else None
        return None if loader is None else loader.get_source(self._target_name)


class _RapidMlxModuleAliasFinder:
    """Resolve ``import vllm_mlx.<sub>`` to ``rapid_mlx.<sub>``.

    Registered at the FRONT of ``sys.meta_path``: nested names such as
    ``vllm_mlx.launch.cli`` resolve through the parent package's
    ``__path__`` — which, for an aliased parent, is the real
    ``rapid_mlx/launch`` directory — so the regular PathFinder would find
    the target FILE and execute it a second time as a duplicate
    ``vllm_mlx.launch.cli`` module. Intercepting first guarantees every
    ``vllm_mlx.*`` name aliases the ``rapid_mlx.*`` module object.

    Bare ``vllm_mlx`` (this package's own ``__init__``) and any name this
    finder declines are returned as ``None`` so the normal machinery
    handles them.
    """

    _rapid_mlx_alias_finder = True

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        target_name = _TARGET_PREFIX + fullname[len(_PREFIX) :]
        # Locate the target WITHOUT executing it. A missing target returns
        # None so the real ``ModuleNotFoundError: vllm_mlx.<sub>`` surfaces
        # untouched; a target whose own dependencies fail to import raises
        # through, so the honest error (e.g. ``No module named 'yaml'``)
        # propagates instead of being masked as a missing shim submodule.
        try:
            target_spec = importlib.util.find_spec(target_name)
        except ModuleNotFoundError as exc:
            # find_spec executes intermediate parent packages, so a broken
            # dependency of e.g. rapid_mlx.audio surfaces HERE as
            # ModuleNotFoundError('yaml'). Only a miss that names a
            # rapid_mlx module means "the target itself doesn't exist";
            # anything else is a real failure and must propagate.
            if not (
                exc.name
                and (exc.name == target_name or exc.name.startswith(_TARGET_PREFIX))
            ):
                raise
            return None
        except (AttributeError, ValueError):
            return None
        if target_spec is None or target_spec.loader is None:
            return None
        loader = _RapidMlxModuleAliasLoader(target_name, target_spec)
        is_package = target_spec.submodule_search_locations is not None
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=target_spec.origin,
            is_package=is_package,
        )
        if spec is not None and is_package:
            # Keep the alias spec introspectable (pkgutil.iter_modules and
            # friends walk submodule_search_locations). Safe to point at the
            # real directory: this finder intercepts all vllm_mlx.* names
            # before PathFinder can scan it and double-execute a file.
            spec.submodule_search_locations = list(
                target_spec.submodule_search_locations
            )
        return spec


def _install_alias_finder() -> None:
    for finder in sys.meta_path:
        # Class identity changes when the shim itself is reloaded, so an
        # isinstance check is not idempotent across importlib.reload().
        if getattr(finder, "_rapid_mlx_alias_finder", False):
            return  # idempotent (e.g. interpreter reload scenarios)
    sys.meta_path.insert(0, _RapidMlxModuleAliasFinder())


_install_alias_finder()


def __getattr__(name):
    """Forward attribute access to ``rapid_mlx`` (preserves its laziness)."""
    return getattr(_rapid_mlx, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_rapid_mlx)))


warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
