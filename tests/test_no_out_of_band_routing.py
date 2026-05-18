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


ENV_VAR_ROUTING_PATTERN = re.compile(
    # Round-5 subagent 3 broadened — drop RAPID_ prefix requirement so
    # MLX_FORCE_*/NO_*/ENABLE_*/DISABLE_* are caught regardless of
    # whether the contributor remembered the RAPID_ namespace.
    r"^(?:RAPID_)?MLX_(?:FORCE|NO|ENABLE|DISABLE)_[A-Z_]+$"
)


PYDANTIC_FIELD_ROUTING_PATTERN = re.compile(r"^(force_|no_|enable_|disable_)")


SETTER_METHOD_ROUTING_PATTERN = re.compile(
    # Round-5 subagent 2 broadened: also catch set_*mllm*, set_*hybrid*,
    # set_*routing*, set_*moe*, set_*dflash*, set_*multimodal*, plus
    # configure_*routing* — these all describe runtime routing flips
    # without using the strict force_/no_ prefix.
    r"^(?:set|configure)_(?:force_|no_|enable_|disable_|is_)"
    r"|^(?:set|configure)_.*(?:mllm|hybrid|routing|moe|dflash|multimodal|spec_decode)"
)


def _field_type_is_str(t: object) -> bool:
    """True iff ``t`` (a ``dataclasses.Field.type`` value) represents
    ``str``, ``Optional[str]``, or ``str | None`` — regardless of
    whether the annotation is a real type, a stringified annotation
    (PEP 563), or a PEP 604 ``types.UnionType``.

    Codex R1 (PR #409 review) flagged that the original ``f.type == "str |
    None"`` string match misses real ``types.UnionType`` objects, which
    is what ``dataclasses.fields()`` returns on Python 3.10+ when the
    module does NOT use ``from __future__ import annotations``. The
    consequence was that ``tool_call_parser: str | None`` slipped the
    allowlist check entirely — a future ``multimodal_mode: str | None``
    would have done the same.
    """
    import typing

    if t is str:
        return True
    # Stringified annotation (PEP 563 / from __future__ import annotations).
    if isinstance(t, str):
        try:
            tree = ast.parse(t, mode="eval")
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "str":
                return True
        return False
    # Real ``types.UnionType`` (PEP 604 ``str | None``) or
    # ``typing.Optional[str]`` / ``typing.Union[str, None]``.
    args = typing.get_args(t)
    if args:
        return any(a is str for a in args)
    return False


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


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _enclosing_function_chain(
    parents: dict[int, ast.AST], target: ast.AST
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return EVERY enclosing function from innermost to outermost.

    Round-5 hardening (subagent 2 bypass): a nested function NAMED
    ``__init__`` inside a real (non-allowed) method would pass the
    innermost-only check because the immediate enclosing function name
    matches the allowlist. By returning the full chain we let the
    caller require EVERY ancestor function to be allowlisted — a
    fake-nested-init slips past the innermost check but is caught
    when we see the outer real handler isn't allowed.
    """
    chain: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    cur: ast.AST | None = parents.get(id(target))
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chain.append(cur)
        cur = parents.get(id(cur))
    return chain


def _routing_attr_write_targets(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """Return ``(attr_name, source_node)`` pairs for every routing-attr
    write expressed by ``node``. Covers many indirect attribute-write
    forms beyond ``Attribute`` assignment:

      - ``self.x = ...`` (Assign target = Attribute)
      - ``self.x: int = ...`` (AnnAssign with Attribute target)
      - ``self.x |= ...`` (AugAssign with Attribute target)
      - ``setattr(self, "x", ...)`` (round-5 subagent 5 #1)
      - ``object.__setattr__(self, "x", ...)`` (#3)
      - ``type(self).__setattr__(self, "x", ...)`` (#4)
      - ``self.__dict__["x"] = ...`` (#5/6)
      - ``vars(self)["x"] = ...`` (#5/7)
      - ``del self.x`` / ``del vars(self)["x"]`` (also a routing mutation)

    Round-5 subagent 5 demonstrated 11/14 setattr-style bypasses on the
    previous Attribute-only scanner. This helper unifies all the shapes
    so the gate doesn't drift.
    """
    pairs: list[tuple[str, ast.AST]] = []

    # Codex R2 fix: destructuring assignment `self._is_mllm, other = ...`
    # nests the Attribute target inside a Tuple/List. Walk targets
    # recursively so destructuring forms are scanned the same as
    # direct assignment.
    def _flatten_targets(t: ast.AST) -> list[ast.AST]:
        if isinstance(t, (ast.Tuple, ast.List)):
            out: list[ast.AST] = []
            for elt in t.elts:
                out.extend(_flatten_targets(elt))
            return out
        # Unwrap Starred (e.g. `*rest, self._is_mllm = ...`).
        if isinstance(t, ast.Starred):
            return _flatten_targets(t.value)
        return [t]

    # 1. Direct Attribute assignment / augassign / annassign.
    attr_targets: list[ast.Attribute] = []
    flat_targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        for t in node.targets:
            flat_targets.extend(_flatten_targets(t))
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        flat_targets.append(node.target)

    for t in flat_targets:
        if isinstance(t, ast.Attribute):
            attr_targets.append(t)
    for tgt in attr_targets:
        pairs.append((tgt.attr, node))

    # 2. Subscript assignment to .__dict__["x"] or vars(obj)["x"].
    subscript_targets: list[ast.Subscript] = [
        t for t in flat_targets if isinstance(t, ast.Subscript)
    ]

    # ast.Delete: `del self.x` / `del vars(self)["x"]` — these clear a
    # routing decision the same way an assignment of False would.
    if isinstance(node, ast.Delete):
        for tgt in node.targets:
            if isinstance(tgt, ast.Attribute):
                pairs.append((tgt.attr, node))
            elif isinstance(tgt, ast.Subscript):
                subscript_targets.append(tgt)

    for sub in subscript_targets:
        # ``self.__dict__["x"] = ...`` — value is Attribute(attr="__dict__")
        if (isinstance(sub.value, ast.Attribute) and sub.value.attr == "__dict__") or (
            isinstance(sub.value, ast.Call)
            and isinstance(sub.value.func, ast.Name)
            and sub.value.func.id == "vars"
        ):
            key = sub.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                pairs.append((key.value, node))

    # 3. Call-form setattr / __setattr__ — check both possible arg
    # positions. Codex R2: bound `engine.__setattr__("x", v)` puts the
    # name at args[0]; unbound `object.__setattr__(engine, "x", v)`
    # puts it at args[1]. Rather than try to distinguish the binding
    # statically (fragile), inspect ALL leading string-Constant args
    # against ROUTING_ATTRS. If any matches, it's a routing write.
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
    elif isinstance(node, ast.Call):
        call = node
    else:
        call = None
    if call is not None:
        pairs.extend(_setattr_routing_writes(call, node))

    return pairs


def _setattr_routing_writes(
    call: ast.Call, owner_node: ast.AST
) -> list[tuple[str, ast.AST]]:
    """If ``call`` is a setattr-family invocation, return every
    string-constant arg paired with ``owner_node``. Cheap superset
    over ``args[0]`` and ``args[1]`` so both bound and unbound forms
    are covered:

    - ``setattr(obj, "x", v)`` — args[0]=obj, args[1]="x"
    - ``object.__setattr__(obj, "x", v)`` — args[0]=obj, args[1]="x"
    - ``engine.__setattr__("x", v)`` — args[0]="x"

    The downstream gate filters by ROUTING_ATTRS so non-routing
    setattr calls (``setattr(obj, "some_other_key", v)``) don't match.
    """
    out: list[tuple[str, ast.AST]] = []
    func = call.func
    is_setattr_name = isinstance(func, ast.Name) and func.id == "setattr"
    is_setattr_method = isinstance(func, ast.Attribute) and func.attr == "__setattr__"
    if not (is_setattr_name or is_setattr_method):
        return out
    # Inspect leading args 0 and 1 (covers both bound and unbound shapes).
    for arg in call.args[:2]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append((arg.value, owner_node))
    return out


def _all_routing_write_calls(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Walk every node in ``tree`` and return all
    ``(routing_attr, node)`` writes. Distinct from
    ``_routing_attr_write_targets`` (which is per-node) so the gate
    can also catch setattr() / __dict__[] writes that aren't the
    top-level statement node (e.g. inside walrus, comprehension)."""
    out: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        out.extend(_routing_attr_write_targets(node))
        # Standalone call inside any expression context — setattr()
        # via walrus or as a comprehension element. Uses the same
        # arg-position-agnostic helper as the per-node path.
        if isinstance(node, ast.Call):
            out.extend(_setattr_routing_writes(node, node))
    # Dedupe — Expr(Call(...)) and the inner Call both produce the same hit.
    seen: set[tuple[int, str]] = set()
    deduped: list[tuple[str, ast.AST]] = []
    for attr, n in out:
        key = (id(n), attr)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((attr, n))
    return deduped


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

        parents = _build_parent_map(tree)
        for attr_name, node in _all_routing_write_calls(tree):
            if attr_name not in ROUTING_ATTRS:
                continue
            chain = _enclosing_function_chain(parents, node)
            if not chain:
                offenders.append(
                    f"{rel}:{node.lineno} module-level write to routing "
                    f"attribute `.{attr_name}` — not inside any function, "
                    "no review surface. Routing decisions must live in "
                    "AUTO_ROUTING_FLAG_PAIRS and be set by EngineCore.__init__."
                )
                continue
            # Round-5 hardening: require EVERY ancestor function to be
            # allowlisted, not just the innermost. A real handler
            # `chat_completion` containing a nested `def __init__(): ...`
            # would pass the old innermost-only check; the all-ancestors
            # check catches it because `chat_completion` is not allowed.
            disallowed = [
                fn.name for fn in chain if fn.name not in ROUTING_WRITE_ALLOWED_FUNCS
            ]
            if disallowed:
                offenders.append(
                    f"{rel}:{node.lineno} writes routing attribute "
                    f"`.{attr_name}` inside function chain "
                    f"{[fn.name for fn in chain][::-1]} — function(s) "
                    f"{disallowed} not in "
                    f"{sorted(ROUTING_WRITE_ALLOWED_FUNCS)}. Round-5 subagent "
                    "found nested-fn-named-__init__ inside non-allowed methods "
                    "slips innermost-only checks; this gate now requires every "
                    "ancestor to be allowlisted. Detected shapes include "
                    "Assign, AnnAssign, AugAssign, Subscript (vars/__dict__), "
                    "setattr(), object.__setattr__(), and ast.Delete."
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
            # Round-5 subagent 3 #B: bytes literal env vars
            # (os.environb[b"RAPID_MLX_FORCE_MLLM"]) — decode and check
            # the same pattern.
            if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
                try:
                    value = node.value.decode("ascii")
                except UnicodeDecodeError:
                    continue
                _check_env_constant(value, rel, node.lineno, offenders)
                continue
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            _check_env_constant(node.value, rel, node.lineno, offenders)

        # Round-5 subagent 3 #B: ban os.environb references entirely.
        # No legitimate use, bytes-form env reads are a parallel surface.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "environb"
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                offenders.append(
                    f"{rel}:{node.lineno} uses `os.environb` — bytes-form env "
                    "var access bypasses the string-Constant scan. Use "
                    "os.environ.get() with a documented str key instead."
                )

    assert not offenders, "\n".join(offenders)


def _check_env_constant(
    value: str, rel: str, lineno: int, offenders: list[str]
) -> None:
    """Run the env-var routing checks against a single constant string."""
    # Strip RAPID_ prefix for the routing-shape check so both
    # RAPID_MLX_FORCE_* and MLX_FORCE_* are caught.
    if not (value.startswith("RAPID_MLX_") or value.startswith("MLX_")):
        return
    if value in ALLOWED_RAPID_MLX_ENV_VARS:
        return
    if ENV_VAR_ROUTING_PATTERN.match(value):
        offenders.append(
            f"{rel}:{lineno} references env var `{value}` — matches the "
            "routing-shape pattern (RAPID_)?MLX_(FORCE|NO|ENABLE|DISABLE)_*. "
            "Routing decisions must go through CLI flags, not env vars "
            "(round-4 cat-4 + round-5 subagent 3 #C). Register a "
            "RoutingFlagPair instead."
        )
    elif value.startswith("RAPID_MLX_"):
        # Only treat RAPID_MLX_ as our namespace; bare MLX_ may be from
        # upstream mlx-lm or transformers.
        offenders.append(
            f"{rel}:{lineno} references env var `{value}` — not in "
            "ALLOWED_RAPID_MLX_ENV_VARS. If this is a non-routing knob, "
            "add it to the allowlist with a comment. Routing env vars "
            "are forbidden."
        )


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

    Round-5 subagent 2 expanded scope: previously the scan was limited
    to ``middleware/`` and ``routes/``; an attacker writing the same
    header in ``server.py`` or ``api/`` slipped. Now scans every file
    under ``vllm_mlx/`` because ``X-Rapid-MLX-`` is OUR namespace and
    the routing-shape inside it has no legitimate use anywhere.
    """
    pkg_root = _pkg_root()
    header_pattern = re.compile(
        r"^X-Rapid-MLX-(?:Force|No|Enable|Disable)-", re.IGNORECASE
    )

    offenders: list[str] = []
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
            if header_pattern.match(node.value):
                offenders.append(
                    f"{rel}:{node.lineno} references header `{node.value}` — "
                    "routing-shaped per-request header. Routing decisions "
                    "must come from process startup, not per-request "
                    "(round-4 cat-3 #2)."
                )

    assert not offenders, "\n".join(offenders)


def test_routing_attr_write_detects_destructuring_and_bound_setattr():
    """Codex R2 regression-lock: two specific shapes that previously
    slipped the routing-attr write scanner.

    (a) Destructuring assignment: ``self._is_mllm, other = False, x``
        — outer target is Tuple, the Attribute is nested inside.
    (b) Bound ``engine.__setattr__("_is_mllm", False)`` — name is at
        args[0], not args[1] like the unbound classmethod form.

    Both forms must be detected by ``_routing_attr_write_targets``.
    If either regresses, this test fails loudly with a specific
    pointer to the bypass."""

    # (a) destructuring
    src_destruct = "self._is_mllm, other = False, x"
    tree = ast.parse(src_destruct)
    found = []
    for node in ast.walk(tree):
        found.extend(_routing_attr_write_targets(node))
    assert any(attr == "_is_mllm" for attr, _ in found), (
        f"Destructuring assignment `{src_destruct}` was NOT detected as a "
        "routing-attr write. The scanner must walk nested Tuple/List "
        "targets recursively (codex R2 fix)."
    )

    # (b) bound __setattr__ — args[0] is the name
    src_bound = 'engine.__setattr__("_is_mllm", False)'
    tree = ast.parse(src_bound)
    found = []
    for node in ast.walk(tree):
        found.extend(_routing_attr_write_targets(node))
    assert any(attr == "_is_mllm" for attr, _ in found), (
        f"Bound `{src_bound}` was NOT detected as a routing write. The "
        "scanner must check both args[0] (bound form) and args[1] (unbound "
        "form) — codex R2 fix."
    )

    # (b') unbound object.__setattr__ — args[1] is the name (must still work)
    src_unbound = 'object.__setattr__(engine, "_is_mllm", False)'
    tree = ast.parse(src_unbound)
    found = []
    for node in ast.walk(tree):
        found.extend(_routing_attr_write_targets(node))
    assert any(attr == "_is_mllm" for attr, _ in found), (
        f"Unbound `{src_unbound}` was NOT detected — original arg-1 path "
        "regressed when adding arg-0 coverage."
    )

    # Negative: non-routing setattr must NOT match.
    src_negative = 'setattr(engine, "some_unrelated_key", True)'
    tree = ast.parse(src_negative)
    found = []
    for node in ast.walk(tree):
        found.extend(_routing_attr_write_targets(node))
    # Found pairs may contain "some_unrelated_key" but should NOT
    # contain any ROUTING_ATTRS member.
    matching_routing = [attr for attr, _ in found if attr in ROUTING_ATTRS]
    assert not matching_routing, (
        f"Non-routing setattr `{src_negative}` falsely matched routing "
        f"attrs: {matching_routing}."
    )


def test_field_type_is_str_handles_pep604_unions():
    """Codex R1 regression: ``f.type`` from ``dataclasses.fields()`` is
    a real ``types.UnionType`` (not a string) on Python 3.10+ when the
    module doesn't use ``from __future__ import annotations``. The
    original ``f.type == "str | None"`` string compare missed every
    optional-str dataclass field, defeating the routing check.

    Lock the helper down: every flavor of "this annotation includes
    str" must return True, and obvious non-str annotations must return
    False. If anyone reverts the union-args path, this fails loudly.
    """
    import typing

    # Real annotations (without PEP 563).
    # noqa for UP045/UP007: this test verifies BOTH legacy typing forms
    # AND PEP 604 forms are detected — that's the point.
    assert _field_type_is_str(str)
    assert _field_type_is_str(str | None)
    assert _field_type_is_str(typing.Optional[str])  # noqa: UP045
    assert _field_type_is_str(typing.Union[str, int])  # noqa: UP007
    # Stringified (PEP 563) annotations.
    assert _field_type_is_str("str")
    assert _field_type_is_str("str | None")
    assert _field_type_is_str("Optional[str]")
    # Non-str annotations must NOT trigger.
    assert not _field_type_is_str(int)
    assert not _field_type_is_str(int | None)
    assert not _field_type_is_str("int")
    assert not _field_type_is_str("bool")
    assert not _field_type_is_str(typing.Optional[int])  # noqa: UP045
    assert not _field_type_is_str(tuple[tuple[str, float], ...] | None)


def test_alias_profile_str_fields_are_explicitly_listed():
    """Round-5 subagent 3 #D: an attacker adds ``multimodal_mode: str =
    "auto"`` to ``AliasProfile`` (passes round-4's name-shape check
    because the name doesn't start with ``force_``/``no_``) and code
    that branches on the value to flip routing. The previous gate
    looked only at name shape — string-enum routing fields slip.

    The defense: require an explicit per-field decision for every
    ``str`` field on ``AliasProfile``. The allowlist below names every
    legitimate string field with a one-line reason. Adding a new
    string field requires editing this allowlist AND explaining what
    the values mean — surfaces the routing-vs-data tradeoff at PR
    review.
    """
    import dataclasses

    from vllm_mlx.model_aliases import AliasProfile

    # Explicit allowlist of AliasProfile string-typed fields. Every
    # entry needs a 1-line reason describing the field's value space.
    # Strings whose value space is open-ended (HF paths, parser names)
    # are not routing decisions; strings whose value space is a small
    # closed enum are LIKELY routing and should be flagged.
    ALLOWED_STR_FIELDS: frozenset[str] = frozenset(
        {
            "hf_path",  # HF repo path, open-ended URL-like string
            "tool_call_parser",  # parser key, see PARSER_REGISTRY
            "reasoning_parser",  # parser key, see PARSER_REGISTRY
            "suffix_decoding_tier",  # one of VALID_SUFFIX_TIERS — non-routing data
            "dflash_draft_model",  # HF path for the spec-decode drafter
        }
    )

    str_fields = [
        f.name for f in dataclasses.fields(AliasProfile) if _field_type_is_str(f.type)
    ]
    unlisted = set(str_fields) - ALLOWED_STR_FIELDS
    assert not unlisted, (
        f"AliasProfile has unlisted string field(s): {sorted(unlisted)}. "
        "Every str field on AliasProfile must be in ALLOWED_STR_FIELDS with "
        "a 1-line reason. Round-5 subagent 3 #D showed that string-enum "
        "fields (e.g. multimodal_mode: str = 'auto') are silent routing "
        "escape hatches — name-shape regex misses them, value-space scans "
        "are unreliable. The fix is closed-set: a new str field requires "
        "explicit review."
    )
