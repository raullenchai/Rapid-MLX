# SPDX-License-Identifier: Apache-2.0
"""SOP §10 complement: no out-of-band routing escape hatches.

``tests/test_no_mllm_flag.py`` gates the CLI + ``load_model`` surface.
This file gates every OTHER surface where a contributor could sneak a
routing decision in:

  - Per-request: request body fields, headers, middleware
  - Runtime mutation: setter methods, post-init writes to routing
    attributes
  - Environment: ``os.environ.get("RAPID_MLX_FORCE_*")`` etc.
  - Engine API: ``engine.set_force_*`` / ``engine.set_*_mllm`` setters

Round-4 red-team found 13 bypasses across these surfaces (5 per-request,
5 env/config, 3 trust-attack). The SOP-§10 gate caught zero — the
attacks didn't touch CLI or load_model at all. This file is the
companion gate. The invariant: **routing fields are write-only inside
their constructor (``__init__``) or the canonical load path
(``load_model`` / ``detect_model_config`` / ``enrich_model_config``)
from kwargs that are themselves registered in
``AUTO_ROUTING_FLAG_PAIRS``**.

When this fails: either move the routing decision into the registry
(register a new ``RoutingFlagPair`` + plumb a CLI flag) or remove the
shortcut. There is no third option — the whole point of the registry
is that routing changes are visible and reviewable.
"""

from __future__ import annotations

import ast
import importlib.resources
import pathlib
import re

# ---------------------------------------------------------------------------
# Constants — kept LOCAL to this file (not imported from test_no_mllm_flag)
# so the two gates can evolve independently and a failure in one doesn't
# cascade silently into the other.
# ---------------------------------------------------------------------------


# Attributes that ARE the routing decision. Any write to one of these is
# a routing decision. The whitelist below names the only legitimate
# write locations.
ROUTING_ATTRS: frozenset[str] = frozenset(
    {
        "_is_mllm",  # BatchedEngine
        "is_hybrid",  # ModelConfig
        "is_moe",  # ModelConfig / AliasProfile
        "supports_spec_decode",  # ModelConfig
        "is_multimodal",  # ModelConfig (auto-detection output)
        "supports_dflash",  # ModelConfig / AliasProfile
    }
)


# Functions that MAY write to routing attributes. Anything else is an
# escape hatch.
ROUTING_WRITE_ALLOWED_FUNCS: frozenset[str] = frozenset(
    {
        "__init__",  # Constructor — the canonical write site.
        "load_model",  # server.py public entry, hands kwargs to EngineCore.
        # Model-config detection / enrichment. These compute routing
        # fields from upstream sources (HF config.json, alias profile).
        "detect_model_config",
        "enrich_model_config",
        "_coerce",  # model_aliases.py — builds AliasProfile.
        "_load",  # model_aliases.py — file loader.
        # `model_post_init` is Pydantic v2's standard "after validation"
        # hook. Treated as constructor-equivalent.
        "model_post_init",
    }
)


# RAPID_MLX_* env vars that are allowed to exist. Routing-shaped env
# vars (``RAPID_MLX_FORCE_*`` etc.) are NEVER allowed — they bypass
# every CLI gate. Add a knob here only if it's a non-routing toggle
# (debug verbosity, version-check disable, etc.).
ALLOWED_RAPID_MLX_ENV_VARS: frozenset[str] = frozenset(
    {
        "RAPID_MLX_DISABLE_VERSION_CHECK",  # opt-out of version check
        "RAPID_MLX_PROFILE_VERBOSE",  # debug verbosity for profile logs
        # Test/integration helpers — server URL for integration suites,
        # not consulted at runtime by the engine.
        "RAPID_MLX_BASE_URL",
        # Telemetry consent toggle (off/on), not engine routing.
        "RAPID_MLX_TELEMETRY",
        # Port for doctor harness probe checks, not engine routing.
        "RAPID_MLX_PORT",
    }
)


# Per-request fields whose name happens to match the routing-shape
# regex but are documented non-routing knobs (UX, chat-template,
# behavior toggles). Same logic as ``NON_ROUTING_FLAGS_ALLOWLIST`` in
# ``test_no_mllm_flag.py`` — explicit allowlist forces a discussion
# in PR review.
NON_ROUTING_PYDANTIC_FIELDS_ALLOWLIST: frozenset[str] = frozenset(
    {
        # OpenAI-compatible chat-template toggle (mirrors --no-thinking
        # CLI flag, also non-routing-allowlisted there).
        "enable_thinking",
    }
)


ENV_VAR_ROUTING_PATTERN = re.compile(r"^RAPID_MLX_(?:FORCE|NO|ENABLE|DISABLE)_[A-Z_]+$")


PYDANTIC_FIELD_ROUTING_PATTERN = re.compile(r"^(force_|no_|enable_|disable_)")


SETTER_METHOD_ROUTING_PATTERN = re.compile(r"^set_(?:force_|no_|enable_|disable_|is_)")


def _pkg_root() -> pathlib.Path:
    return pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()


def _iter_module_files() -> list[pathlib.Path]:
    """Every .py file under vllm_mlx/, excluding __pycache__ and vendored
    upstream files. Vendored files (deepseek_v4.py) are explicitly
    excluded because they're upstream code held as-is for clean sync."""
    root = _pkg_root()
    _VENDORED = frozenset({"models/deepseek_v4.py"})
    out: list[pathlib.Path] = []
    for path in root.rglob("*.py"):
        if any(part.startswith("__") for part in path.parts[len(root.parts) :]):
            continue
        rel = path.relative_to(root).as_posix()
        if rel in _VENDORED:
            continue
        out.append(path)
    return out


def _find_enclosing_function(
    tree: ast.AST, target: ast.AST
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the innermost function that contains ``target`` in
    ``tree``, or ``None`` if ``target`` is at module level. Uses a
    parent walk built from the tree."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    cur: ast.AST | None = parents.get(id(target))
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = parents.get(id(cur))
    return None


def test_routing_fields_written_only_in_allowed_scopes():
    """Round-4 cat-3 (per-request routing) — all 5 attacks bypassed the
    SOP §10 gate by writing to ``engine._is_mllm`` / ``model_config.<routing
    field>`` from a request handler, middleware, sampling helper, admin
    endpoint, or env-var-driven branch — none of which the CLI/load_model
    gate watches.

    This test asserts the inverse invariant: routing fields are
    write-only inside constructors and the canonical load path. Any
    other write is an out-of-band escape hatch.

    Allowed functions: ``__init__``, ``load_model``,
    ``detect_model_config``, ``enrich_model_config``, ``_coerce``,
    ``_load``, ``model_post_init``. See ``ROUTING_WRITE_ALLOWED_FUNCS``.

    If you need to mutate a routing field from a new location, the
    answer is almost always "add a new ``RoutingFlagPair`` entry in
    ``test_no_mllm_flag.py`` and wire it through ``EngineCore.__init__``
    via a kwarg". Adding to the allowlist here requires a written
    justification (PR description) explaining why constructor mutation
    isn't sufficient.
    """
    offenders: list[str] = []
    pkg_root = _pkg_root()
    for path in _iter_module_files():
        rel = path.relative_to(pkg_root).as_posix()
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Match ``Assign`` to ``Attribute`` and ``AugAssign`` to
            # ``Attribute`` (e.g. ``self.x |= ...``).
            attr_targets: list[ast.Attribute] = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute):
                        attr_targets.append(t)
            elif (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Attribute)
                or isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
            ):
                attr_targets.append(node.target)

            for tgt in attr_targets:
                if tgt.attr not in ROUTING_ATTRS:
                    continue
                enclosing = _find_enclosing_function(tree, node)
                if enclosing is None:
                    # Module-level write — always forbidden for routing fields.
                    offenders.append(
                        f"{rel}:{node.lineno} module-level write to routing "
                        f"attribute `.{tgt.attr}` — not inside any function, "
                        "no review surface. Routing decisions must live in "
                        "AUTO_ROUTING_FLAG_PAIRS and be set by EngineCore.__init__."
                    )
                    continue
                if enclosing.name in ROUTING_WRITE_ALLOWED_FUNCS:
                    continue
                offenders.append(
                    f"{rel}:{node.lineno} writes routing attribute "
                    f"`.{tgt.attr}` inside function `{enclosing.name}` — "
                    f"only {sorted(ROUTING_WRITE_ALLOWED_FUNCS)} may write "
                    "routing fields (round-4 cat-3 per-request bypass). "
                    "If you need a new routing decision, add a "
                    "RoutingFlagPair entry in test_no_mllm_flag.py and "
                    "wire it through EngineCore.__init__."
                )

    assert not offenders, "\n".join(offenders)


def test_no_routing_shaped_rapid_mlx_env_vars():
    """Round-4 cat-4 (env-var/config routing) — 3 of 5 attacks read an
    env var (``RAPID_MLX_FORCE_MLLM``, ``RAPID_MLX_FORCE_TEXT_MODE``)
    at startup or request time and mutated routing without ever
    touching the CLI surface.

    This test forbids env var NAMES that match the routing-shape
    pattern (``RAPID_MLX_(FORCE|NO|ENABLE|DISABLE)_*``). The check is
    on the CONSTANT — even if the attacker reads the env var, the
    constant string has to appear somewhere in source, and that string
    can't be routing-shaped.

    Non-routing env vars (``RAPID_MLX_PROFILE_VERBOSE``,
    ``RAPID_MLX_DISABLE_VERSION_CHECK``) are allowlisted explicitly.
    Add new non-routing env vars to ``ALLOWED_RAPID_MLX_ENV_VARS``
    with a justification comment. NEVER add a routing-shaped name to
    the allowlist — register a CLI flag pair instead.
    """
    offenders: list[str] = []
    pkg_root = _pkg_root()
    for path in _iter_module_files():
        rel = path.relative_to(pkg_root).as_posix()
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            value = node.value
            if not value.startswith("RAPID_MLX_"):
                continue
            if value in ALLOWED_RAPID_MLX_ENV_VARS:
                continue
            if ENV_VAR_ROUTING_PATTERN.match(value):
                offenders.append(
                    f"{rel}:{node.lineno} references env var `{value}` — "
                    "matches the routing-shape pattern "
                    "RAPID_MLX_(FORCE|NO|ENABLE|DISABLE)_*. Routing decisions "
                    "must go through CLI flags, not env vars (round-4 cat-4 "
                    "bypass). Register a RoutingFlagPair instead."
                )
            else:
                offenders.append(
                    f"{rel}:{node.lineno} references env var `{value}` — "
                    "not in ALLOWED_RAPID_MLX_ENV_VARS. If this is a non-"
                    "routing knob, add it to the allowlist with a comment. "
                    "Routing env vars are forbidden."
                )

    assert not offenders, "\n".join(offenders)


def test_no_routing_shaped_pydantic_fields_in_api():
    """Round-4 cat-3 #1, #3 — attackers added ``force_mllm: bool`` to
    ``ChatCompletionRequest`` and ``routing_override: dict`` to a
    sampling-params helper. Both are per-request routing escape hatches
    invisible to SOP §10.

    This test forbids routing-shaped field names anywhere in
    ``vllm_mlx/api/`` (Pydantic request/response models) and
    ``vllm_mlx/routes/`` (handler-local fields). Catches the field
    declaration regardless of where the handler reads it.
    """
    pkg_root = _pkg_root()
    api_dir = pkg_root / "api"
    routes_dir = pkg_root / "routes"

    paths: list[pathlib.Path] = []
    if api_dir.exists():
        paths.extend(api_dir.rglob("*.py"))
    if routes_dir.exists():
        paths.extend(routes_dir.rglob("*.py"))

    offenders: list[str] = []
    for path in paths:
        if any(part.startswith("__") for part in path.parts):
            continue
        rel = path.relative_to(pkg_root).as_posix()
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Pydantic field declarations are AnnAssign at class body
            # level (e.g. ``force_mllm: bool = False``).
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if name in NON_ROUTING_PYDANTIC_FIELDS_ALLOWLIST:
                    continue
                if PYDANTIC_FIELD_ROUTING_PATTERN.match(name):
                    offenders.append(
                        f"{rel}:{node.lineno} declares field `{name}` matching "
                        "routing-shape pattern (force_/no_/enable_/disable_). "
                        "Per-request routing fields are an out-of-band escape "
                        "hatch (round-4 cat-3 #1/#3). Move the routing decision "
                        "to a CLI flag in AUTO_ROUTING_FLAG_PAIRS."
                    )

    assert not offenders, "\n".join(offenders)


def test_no_routing_setter_methods_on_engine():
    """Round-4 cat-3 #4 — attacker added ``def set_force_mllm(self, v)``
    to ``BatchedEngine`` and exposed it via an admin endpoint, flipping
    routing live. SOP §10 watches ``__init__`` signatures but not
    method names.

    This test forbids method names matching ``set_(force_|no_|enable_|
    disable_|is_)`` on any class in ``vllm_mlx/engine/``,
    ``vllm_mlx/engine_core.py``, or ``vllm_mlx/server.py``. The
    setter-method shape is the entire signal — read-only properties
    and getters are fine.
    """
    pkg_root = _pkg_root()
    targets = [
        pkg_root / "engine_core.py",
        pkg_root / "server.py",
    ]
    engine_dir = pkg_root / "engine"
    if engine_dir.exists():
        targets.extend(engine_dir.rglob("*.py"))

    offenders: list[str] = []
    for path in targets:
        if not path.exists():
            continue
        if any(part.startswith("__") for part in path.parts):
            continue
        rel = path.relative_to(pkg_root).as_posix()
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if SETTER_METHOD_ROUTING_PATTERN.match(node.name):
                offenders.append(
                    f"{rel}:{node.lineno} defines `{node.name}` — routing-"
                    "shaped setter method on engine/server. Routing fields "
                    "are write-once at construction; runtime setters are an "
                    "out-of-band escape hatch (round-4 cat-3 #4). Remove the "
                    "setter and require a process restart for routing changes."
                )

    assert not offenders, "\n".join(offenders)


def test_no_routing_shaped_request_headers():
    """Round-4 cat-3 #2 — attacker added ``X-Rapid-MLX-Force-MLLM``
    header read in middleware, mutating routing per-request. The
    header NAME has to appear as a string constant somewhere.

    Forbid any string constant matching ``X-Rapid-MLX-(Force|No|Enable|
    Disable)-*`` in ``vllm_mlx/middleware/`` and ``vllm_mlx/routes/``.
    The Rapid-MLX-prefixed header is our namespace, and the
    routing-shape inside it has no legitimate use.
    """
    pkg_root = _pkg_root()
    header_pattern = re.compile(
        r"^X-Rapid-MLX-(?:Force|No|Enable|Disable)-", re.IGNORECASE
    )

    paths: list[pathlib.Path] = []
    middleware_dir = pkg_root / "middleware"
    routes_dir = pkg_root / "routes"
    if middleware_dir.exists():
        paths.extend(middleware_dir.rglob("*.py"))
    if routes_dir.exists():
        paths.extend(routes_dir.rglob("*.py"))

    offenders: list[str] = []
    for path in paths:
        if any(part.startswith("__") for part in path.parts):
            continue
        rel = path.relative_to(pkg_root).as_posix()
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            if header_pattern.match(node.value):
                offenders.append(
                    f"{rel}:{node.lineno} references header `{node.value}` — "
                    "routing-shaped per-request header. Routing decisions "
                    "must come from process startup, not per-request "
                    "(round-4 cat-3 #2)."
                )

    assert not offenders, "\n".join(offenders)
