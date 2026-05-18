# SPDX-License-Identifier: Apache-2.0
"""
Tests for the --no-mllm / --text-only escape hatch (#393).

Some HuggingFace model repos ship a `config.json` that declares
multimodal capabilities (e.g. `vision_config` block) but the actual
safetensors only contain text-model weights — a partial quant, a
text-only fork, or a checkpoint that was uploaded before vision shards
were finalized. Auto-detection (`is_mllm_model`) correctly identifies
the config as multimodal-capable, but the load path then crashes inside
mlx_vlm with `ValueError: Missing N parameters: vision_tower.*`.

`--no-mllm` (alias `--text-only`) is the user-facing escape hatch:
force the text path even when auto-detection would route to MLLM.

These tests verify:
1. BatchedEngine respects force_text=True (skips is_mllm_model probe).
2. force_text and force_mllm are not both honored — server.load_model
   raises ValueError if both are passed.
3. The friendly-error wrapper in MLLMModel.load() catches the
   missing-vision-tensor ValueError and re-raises as RuntimeError that
   mentions `--no-mllm`.
"""

from __future__ import annotations

import ast
import importlib.resources
import inspect
import pathlib
import re
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# SOP §10 routing registry — single source of truth.
#
# Every binary auto-routing decision in rapid-mlx has an entry here. Other
# gates in this file derive their checks from this registry instead of
# carrying parallel hand-maintained lists — adding a new pair only requires
# editing this one place. The 5-subagent red-team in PR #408 verified that
# every test below now catches every previously-discovered bypass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingFlagPair:
    """A binary auto-routing decision exposed via paired CLI flags.

    Fields:
        force_on: ``--force-*`` / ``--mllm``-style flag that forces the
            auto-detected behavior on.
        force_off: ``--no-*``-style flag that forces it off.
        desc: human-readable description used in test failure messages.
        required_files: source files (relative to ``vllm_mlx/``) where
            BOTH flags must appear in an ``add_argument()`` call. Every
            CLI entrypoint that takes a model name and runs the
            corresponding auto-detection is required. Adding a new
            model-taking entrypoint = adding it to every relevant pair.
        forwarded_kwargs: ``load_model(...)`` kwargs the CLI forwards
            when this pair is set. Empty tuple = the override is
            consumed at a higher layer (e.g. parser opt-outs short-
            circuit in cli.py before ``load_model``). When non-empty,
            ``test_routing_override_kwargs_are_forwarded_to_load_model``
            requires every ``load_model`` call site that forwards any
            one of these to forward all of them.
        model_config_field: ``ModelConfig`` attribute that
            ``EngineCore.__init__`` mutates when this pair's
            ``EngineConfig`` field is set. ``None`` = the override
            doesn't go through ModelConfig (e.g. ``--no-mllm`` mutates
            ``BatchedEngine._is_mllm``). When non-``None``,
            ``test_engine_core_applies_routing_overrides_from_registry``
            asserts the mutation actually happens.
    """

    force_on: str
    force_off: str
    desc: str
    required_files: tuple[str, ...]
    forwarded_kwargs: tuple[str, ...]
    model_config_field: str | None = None


AUTO_ROUTING_FLAG_PAIRS: tuple[RoutingFlagPair, ...] = (
    RoutingFlagPair(
        force_on="--mllm",
        force_off="--no-mllm",
        desc="MLLM vs text-only routing (#393)",
        required_files=("cli.py", "server.py", "benchmark.py"),
        forwarded_kwargs=("force_mllm", "force_text"),
        # --mllm acts on BatchedEngine._is_mllm, not ModelConfig.
        model_config_field=None,
    ),
    RoutingFlagPair(
        force_on="--tool-call-parser",
        force_off="--no-tool-call-parser",
        desc="AliasProfile tool-call parser auto-selection",
        required_files=("cli.py", "server.py"),
        # Parser opt-outs are consumed in cli.py / server.py main()
        # before load_model is ever called.
        forwarded_kwargs=(),
        model_config_field=None,
    ),
    RoutingFlagPair(
        force_on="--reasoning-parser",
        force_off="--no-reasoning-parser",
        desc="AliasProfile reasoning parser auto-selection",
        required_files=("cli.py", "server.py"),
        forwarded_kwargs=(),
        model_config_field=None,
    ),
    RoutingFlagPair(
        force_on="--force-hybrid",
        force_off="--no-hybrid",
        desc="ModelConfig.is_hybrid (gates spec/suffix decode)",
        required_files=("cli.py", "server.py"),
        forwarded_kwargs=("force_hybrid", "no_hybrid"),
        model_config_field="is_hybrid",
    ),
    RoutingFlagPair(
        force_on="--force-spec-decode",
        force_off="--no-spec-decode",
        desc="ModelConfig.supports_spec_decode (gates MTP/DFlash/suffix)",
        required_files=("cli.py", "server.py"),
        forwarded_kwargs=("force_spec_decode", "no_spec_decode"),
        model_config_field="supports_spec_decode",
    ),
)


# Flags whose name matches the routing-shape pattern (--force-*, --no-*,
# --enable-*, --disable-*) but are intentionally NOT auto-routing
# decisions. Feature toggles, prompt-template knobs, and runtime-perf
# opt-ins live here. Add to this list if and only if the flag is
# definitively not a binary auto-detection that could ever need a paired
# escape hatch — when in doubt, register the pair in
# AUTO_ROUTING_FLAG_PAIRS instead.
NON_ROUTING_FLAGS_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Auto-tool-choice is a behavior knob, not auto-detection.
        "--enable-auto-tool-choice",
        # Performance opt-in for jump-forward decoding bias.
        "--enable-tool-logits-bias",
        # Feature flags for speculative-decode backends. The routing
        # decision (which one is eligible) is gated by --force/no-spec-
        # decode (registered pair); these just enable the implementation.
        "--enable-mtp",
        "--enable-dflash",
        "--enable-suffix-decoding",
        "--enable-kv-cache-quantization",
        "--enable-kv-cache-turboquant",
        "--enable-prefix-cache",
        "--disable-prefix-cache",
        # Chat-template toggle, not engine routing.
        "--no-thinking",
        # CORS toggle.
        "--enable-cors",
        # Perf / UX toggles, not routing decisions.
        "--force-disk-check",  # forces eager disk-space check
        "--no-gc-control",  # disables Python GC tuning
        "--no-memory-aware-cache",  # disables memory-aware cache sizing
        # Privacy toggle.
        "--no-telemetry",
    }
)


# Derived from the registry — never edit these by hand. The whole point
# of the dataclass restructure is that adding a new RoutingFlagPair
# entry transparently extends every gate below.
KWARGS_THAT_MUST_BE_FORWARDED: frozenset[str] = frozenset(
    kw for p in AUTO_ROUTING_FLAG_PAIRS for kw in p.forwarded_kwargs
)


def _registered_flag_names() -> set[str]:
    """All flag strings (force-on + force-off) currently in the registry."""
    out: set[str] = set()
    for p in AUTO_ROUTING_FLAG_PAIRS:
        out.add(p.force_on)
        out.add(p.force_off)
    return out


def _pkg_root() -> pathlib.Path:
    return pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()


def test_force_text_overrides_auto_detection(monkeypatch):
    """When force_text=True, BatchedEngine._is_mllm is False even if
    is_mllm_model would return True. Verifies the probe is short-
    circuited (not just overridden later) by checking it isn't called."""
    from vllm_mlx.engine import batched as batched_mod

    probe_calls = []

    def _fake_is_mllm_model(name):
        probe_calls.append(name)
        return True  # would normally route to MLLM

    monkeypatch.setattr(batched_mod, "is_mllm_model", _fake_is_mllm_model)

    engine = batched_mod.BatchedEngine(
        model_name="fake/model-name",
        force_text=True,
    )

    assert engine._is_mllm is False, (
        "force_text=True must override auto-detection to False"
    )
    assert probe_calls == [], (
        "force_text=True should short-circuit the probe entirely; "
        f"is_mllm_model was called for: {probe_calls}"
    )


def test_force_mllm_still_works_when_force_text_is_false():
    """Regression: adding force_text must not break force_mllm."""
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        force_mllm=True,
        force_text=False,
    )
    assert engine._is_mllm is True


def test_force_text_is_keyword_only_in_load_model():
    """Regression: ``force_text`` must remain keyword-only so existing
    positional callers (e.g. ``load_model(name, None, 1, 32768, False,
    0.5)`` setting ``gpu_memory_utilization=0.5``) don't suddenly
    pass that float as a truthy ``force_text``. Codex R2 caught this
    on the original PR — the original placement after ``force_mllm``
    shifted every subsequent positional arg by one slot."""
    import inspect

    from vllm_mlx.server import load_model

    sig = inspect.signature(load_model)
    assert sig.parameters["force_text"].kind == inspect.Parameter.KEYWORD_ONLY, (
        "force_text must be KEYWORD_ONLY to preserve positional-arg "
        "compatibility for downstream callers — see codex R2 on PR #407."
    )

    from vllm_mlx.engine.batched import BatchedEngine

    sig = inspect.signature(BatchedEngine.__init__)
    assert sig.parameters["force_text"].kind == inspect.Parameter.KEYWORD_ONLY, (
        "BatchedEngine.__init__ force_text must be KEYWORD_ONLY too."
    )


def test_force_text_and_force_mllm_mutually_exclusive_in_load_model():
    """server.load_model raises ValueError if both flags are True. This
    is the second line of defense — CLI already rejects this via
    sys.exit(2), but load_model is also a public entry point so guard
    here too."""
    from vllm_mlx.server import load_model

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_model(
            "fake/model",
            force_mllm=True,
            force_text=True,
        )


def test_friendly_error_on_missing_vision_tensors(monkeypatch):
    """MLLMModel.load() must translate mlx_vlm's
    `ValueError: Missing N parameters: vision_tower.*` into a RuntimeError
    that mentions --no-mllm, so users find the escape hatch without
    grepping the source. Verifies the wrapper fires only on the
    vision-shaped missing-parameter signature."""
    import importlib
    import sys

    # mlx_vlm may not be installed (vision extra is opt-in). The wrapper
    # logic lives in MLLMModel.load, which doesn't need mlx_vlm to be
    # importable for the catch path. But we DO need mlx_vlm to satisfy
    # the `_require_mlx_vlm()` precondition. Skip cleanly if absent.
    try:
        importlib.import_module("mlx_vlm")
    except ImportError:
        pytest.skip("mlx_vlm not installed (vision extra)")

    from vllm_mlx.models import mllm as mllm_mod

    # Inject a fake mlx_vlm.load that raises the M5-style missing-tensor
    # ValueError. We poke sys.modules so the `from mlx_vlm import load`
    # inside MLLMModel.load() picks up our fake.
    real_mlx_vlm = sys.modules["mlx_vlm"]

    class _FakeMlxVlm:
        @staticmethod
        def load(_name):
            raise ValueError(
                "Missing 60 parameters: \n"
                "vision_tower.blocks.27.attn.proj.bias,\n"
                "vision_tower.blocks.27.attn.proj.weight,\n"
                "vision_tower.blocks.27.attn.qkv.bias."
            )

    class _FakeMlxVlmUtils:
        @staticmethod
        def load_config(_name):
            return {}

    monkeypatch.setitem(sys.modules, "mlx_vlm", _FakeMlxVlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", _FakeMlxVlmUtils)

    # Avoid the global instance count guard
    inst = mllm_mod.MLXMultimodalLM(model_name="fake/incomplete-vlm")

    try:
        with pytest.raises(RuntimeError) as excinfo:
            inst.load()

        msg = str(excinfo.value)
        assert "--no-mllm" in msg, (
            f"Friendly error must mention --no-mllm; got: {msg!r}"
        )
        assert "#393" in msg, "Friendly error must reference #393 for searchability"
        assert "60 vision tensors missing" in msg, (
            "Friendly error must surface the count from the underlying error"
        )
    finally:
        # Restore original mlx_vlm so subsequent tests aren't poisoned.
        sys.modules["mlx_vlm"] = real_mlx_vlm


def _flag_in_add_argument_calls(source: str, flag: str) -> bool:
    """True iff ``flag`` appears as a positional string literal to an
    ``add_argument`` call in ``source``. Uses AST so help text,
    comments, and unrelated string occurrences don't count.

    Strengthened version (codex R1 PR #407) of the previous
    "string-search" check that was a false-positive guard."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match both `parser.add_argument(...)` and `subparser.add_argument(...)`.
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == flag:
                return True
    return False


def _all_add_argument_flags(source: str) -> set[str]:
    """All positional string literals passed to any ``add_argument()``
    call in ``source``. Used by ``test_no_unregistered_routing_shaped_flags``
    to enumerate every argparse flag in an entrypoint without missing
    subparser blocks or argparse group calls."""
    tree = ast.parse(source)
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("-")
            ):
                flags.add(arg.value)
    return flags


def _add_argument_calls_with_custom_action(source: str) -> list[tuple[int, str]]:
    """Find ``add_argument(..., action=<non-stdlib>)`` calls — a
    contributor can sneak routing by writing a custom
    ``argparse.Action`` subclass whose ``__call__`` mutates the
    namespace bypassing every name/regex/dest check. Returns
    ``(lineno, action_repr)`` pairs.

    Allowed stdlib actions are the documented argparse strings:
    ``store``, ``store_const``, ``store_true``, ``store_false``,
    ``append``, ``append_const``, ``count``, ``help``, ``version``,
    ``extend``. Anything else — a Name, Attribute, or non-allowlisted
    string — is flagged for review. Closes round-4 cat-1 #A2 + cat-2
    #4 (custom Action subclass routing flips).
    """
    _STDLIB_ACTION_STRINGS = frozenset(
        {
            "store",
            "store_const",
            "store_true",
            "store_false",
            "append",
            "append_const",
            "count",
            "help",
            "version",
            "extend",
        }
    )
    # Stdlib Action classes that may appear via Attribute reference
    # (e.g. ``argparse.BooleanOptionalAction``). Trustworthy because they
    # come from the standard library and don't write to attributes
    # outside their declared dest.
    _STDLIB_ACTION_CLASSES = frozenset(
        {
            "BooleanOptionalAction",  # Python 3.9+
        }
    )
    tree = ast.parse(source)
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg != "action":
                continue
            if (
                isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
                and kw.value.value in _STDLIB_ACTION_STRINGS
            ):
                continue
            # `argparse.BooleanOptionalAction` (Attribute) or a bare
            # `BooleanOptionalAction` (Name) is also stdlib.
            if (
                isinstance(kw.value, ast.Attribute)
                and kw.value.attr in _STDLIB_ACTION_CLASSES
            ):
                continue
            if isinstance(kw.value, ast.Name) and kw.value.id in _STDLIB_ACTION_CLASSES:
                continue
            # Anything else — custom class, non-allowlisted string — is suspicious.
            try:
                repr_str = ast.unparse(kw.value)
            except Exception:
                repr_str = "<unparseable>"
            offenders.append((node.lineno, repr_str))
    return offenders


def _add_argument_calls_with_dict_kwarg_unpack(source: str) -> list[int]:
    """Find ``add_argument(..., **dict_literal_or_var)`` calls. Round-5
    subagent 1 #P2-3: dest aliasing via ``**{"dest": "force_mllm"}`` is
    invisible to the per-kwarg scan because the keyword arg has
    ``kw.arg is None``. Reject the unpack outright in entrypoint files
    — there's no legitimate use of ``**`` for add_argument, and helper
    dicts hide audit surface."""
    tree = ast.parse(source)
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg is None:
                offenders.append(node.lineno)
                break
    return offenders


def _getattr_add_argument_calls(source: str) -> list[int]:
    """Find ``getattr(p, "add_argument")(...)`` and string-concat
    variants. Round-5 subagent 1 #P2-5: a ``getattr`` indirection
    defeats the ``isinstance(func, ast.Attribute) and func.attr ==
    "add_argument"`` predicate at the heart of every prong. There's no
    legitimate use case in argparse code, so we ban the shape outright."""
    tree = ast.parse(source)
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Outer Call has Call as func (the result of getattr being called).
        if not isinstance(node.func, ast.Call):
            continue
        inner = node.func
        if not (isinstance(inner.func, ast.Name) and inner.func.id == "getattr"):
            continue
        if len(inner.args) < 2:
            continue
        name_arg = inner.args[1]
        # Constant "add_argument" or string concat resolving to it.
        if isinstance(name_arg, ast.Constant) and name_arg.value == "add_argument":
            offenders.append(node.lineno)
            continue
        # String concat shape: "add_" + "argument".
        try:
            value = ast.literal_eval(name_arg)
            if isinstance(value, str) and value == "add_argument":
                offenders.append(node.lineno)
        except (ValueError, TypeError):
            # Non-literal — also suspicious (could resolve to add_argument
            # at runtime). Flag the dynamic indirection.
            offenders.append(node.lineno)
    return offenders


def _add_argument_calls_with_non_literal_flag(source: str) -> list[int]:
    """Find ``add_argument(<non-Constant>, ...)`` calls where the first
    positional argument is a variable, attribute, function call, or
    subscript — i.e. the flag name is computed at runtime so the
    static scan can't see it. Closes round-4 cat-2 #3 (helper-function
    indirection like ``_add_routing_flag(p, "--force-text-mode", ...)``
    where the constant lives in the call site of the helper, not the
    ``add_argument`` itself)."""
    tree = ast.parse(source)
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant):
            continue
        # Allow `argparse.SUPPRESS` — the only legitimate non-literal
        # first arg in our codebase.
        if isinstance(first, ast.Attribute) and first.attr == "SUPPRESS":
            continue
        offenders.append(node.lineno)
    return offenders


def _routing_shaped_constants_in_module(source: str) -> set[str]:
    """Find string constants in a module that LOOK LIKE a single pure
    routing flag (``--force-*``, ``--no-*``, etc.), regardless of where
    they appear. Catches round-4 cat-2 #3 and #5 where the flag literal
    lives in a helper-function call site (``_add_routing_flag(p,
    "--force-text-mode", ...)``) or a plugin-registry call
    (``register_plugin("--no-vision-tower", ...)``) instead of an
    ``add_argument()`` call.

    Anchored regex requires the whole string to be a single flag —
    ``"--force-hybrid"`` matches, ``"--force-hybrid and --no-hybrid are
    mutually exclusive"`` (an error message) does not. Embedded flag
    names in error messages and docstrings are noise, not bypass."""
    routing_pattern = re.compile(r"^--(?:force|no|enable|disable)-[a-z0-9-]+$")
    tree = ast.parse(source)
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if routing_pattern.match(node.value):
                flags.add(node.value)
    return flags


# ---------------------------------------------------------------------------
# Auto-discovered entrypoint set (closes round-3 red-team bypass #1.4/#4.4).
#
# Previously the SOP checks hardcoded ("cli.py", "server.py", "benchmark.py")
# and never noticed routing-shape flags or load_model() callers added to
# new files. Now we walk every .py file under vllm_mlx/ and discover any
# file that calls add_argument() OR load_model() — that's the closure of
# "places a contributor could regress an SOP gate". If your new file
# starts appearing here, the gates automatically include it.
#
# Excluded: ``__pycache__``, ``__init__.py`` (re-exports only), test
# fixtures. We re-check this list against the static seed below so the
# discovery can't silently drop entrypoints either.
# ---------------------------------------------------------------------------


# Static seed (current known entrypoints). Discovery must produce a
# SUPERSET of this — if discovery returns fewer files, something
# downstream changed and we want a loud failure.
_KNOWN_ENTRYPOINTS_SEED: frozenset[str] = frozenset(
    {"cli.py", "server.py", "benchmark.py"}
)


def _discover_entrypoints() -> set[str]:
    """Discover every file under ``vllm_mlx/`` that either calls
    ``add_argument(...)`` or ``load_model(...)``. Returns paths
    relative to the package root (e.g. ``"cli.py"`` or
    ``"routes/audio_route.py"``).

    Closes round-3 bypass: contributor adds a new entrypoint file
    (e.g. ``vllm_mlx/serve.py`` with its own argparse) that the
    hardcoded gate-file list would never check."""
    root = _pkg_root()
    discovered: set[str] = set()

    for path in root.rglob("*.py"):
        # Skip __pycache__ and other dunder dirs.
        if any(part.startswith("__") for part in path.parts[len(root.parts) :]):
            continue
        if path.name == "__init__.py":
            # __init__ files are usually re-exports; if a real
            # entrypoint shows up there it's still discovered via the
            # AST checks below.
            pass

        try:
            source = path.read_text()
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        has_routing_shape = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else None)
            )
            if name == "add_argument":
                # Only flag this file if at least one add_argument call
                # is a routing-shape flag. Files that only register
                # config knobs (e.g. --max-tokens) don't need to be in
                # the parity/forwarding checks.
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.startswith("--")
                    ):
                        has_routing_shape = True
                        break
            elif name == "load_model":
                # Any caller of load_model is by definition an
                # entrypoint that needs the forwarding check.
                has_routing_shape = True

            if has_routing_shape:
                break

        if has_routing_shape:
            rel = path.relative_to(root).as_posix()
            discovered.add(rel)

    missing_seed = _KNOWN_ENTRYPOINTS_SEED - discovered
    assert not missing_seed, (
        f"Entrypoint discovery dropped known files: {sorted(missing_seed)}. "
        "Either a known entrypoint was deleted or the discovery logic "
        "regressed. If deletion was intentional, update "
        "_KNOWN_ENTRYPOINTS_SEED to match."
    )
    return discovered


def test_registry_invariants():
    """Round-3 hardening (PR #408): the registry IS the source of
    truth for every gate, so the registry itself must be sane.
    Catches red-team attacks where the registry is silently corrupted
    or duplicated to defeat the derived gates:

      - Bypass #2.1 (liar registry): a contributor sets
        ``model_config_field=None`` on a pair that REALLY does mutate
        ModelConfig. Parametrize then produces zero cases, no test
        complains. This invariant test demands proof: every pair
        whose ``forwarded_kwargs`` includes a ``force_*`` / ``no_*``
        kwarg pair AND whose semantics flow through ModelConfig must
        declare its field — we sanity-check by requiring any pair
        with non-empty ``forwarded_kwargs`` to either (a) have
        ``model_config_field`` set OR (b) be in
        ``KWARGS_FORWARDED_BUT_NOT_VIA_MODEL_CONFIG`` (explicit
        exception list, ``force_mllm``/``force_text`` go here).

      - Bypass #2.5 (duplicate entry): contributor inserts the same
        pair twice. We assert all ``(force_on, force_off)`` tuples
        are unique.

      - Hygiene: every ``model_config_field`` must actually exist on
        ``ModelConfig``. Bogus field names get caught here loudly
        instead of producing cryptic AttributeError at parametrize
        execution time.
    """
    from vllm_mlx.model_auto_config import ModelConfig

    # (a) No duplicate flag pairs.
    pair_keys = [(p.force_on, p.force_off) for p in AUTO_ROUTING_FLAG_PAIRS]
    assert len(pair_keys) == len(set(pair_keys)), (
        f"AUTO_ROUTING_FLAG_PAIRS has duplicate (force_on, force_off) entries: "
        f"{[pk for pk in pair_keys if pair_keys.count(pk) > 1]}. "
        "Round-3 red-team #2.5 — duplicates silently waste gate cycles "
        "without surfacing the issue."
    )

    # (b) Every model_config_field exists on ModelConfig.
    mc = ModelConfig()
    for pair in AUTO_ROUTING_FLAG_PAIRS:
        if pair.model_config_field is None:
            continue
        assert hasattr(mc, pair.model_config_field), (
            f"Registry pair {pair.force_on}/{pair.force_off} declares "
            f"model_config_field={pair.model_config_field!r} but that "
            "attribute doesn't exist on ModelConfig. Typo or stale field "
            "name — fix or remove."
        )

    # (c) Forwarded kwargs that don't go through ModelConfig must be
    # explicitly listed. Catches the round-3 "liar" registry bypass:
    # contributor sets model_config_field=None on a pair that really
    # does mutate ModelConfig to silently disable the derived gate.
    KWARGS_FORWARDED_BUT_NOT_VIA_MODEL_CONFIG = frozenset(
        {
            # --mllm / --no-mllm route through BatchedEngine._is_mllm,
            # not ModelConfig. Verified by test_force_text_overrides_auto_detection.
            "force_mllm",
            "force_text",
        }
    )
    for pair in AUTO_ROUTING_FLAG_PAIRS:
        if not pair.forwarded_kwargs:
            continue
        if pair.model_config_field is not None:
            continue
        # forwarded_kwargs non-empty AND model_config_field None — must
        # be an explicit exception.
        unexplained = set(pair.forwarded_kwargs) - (
            KWARGS_FORWARDED_BUT_NOT_VIA_MODEL_CONFIG
        )
        assert not unexplained, (
            f"Registry pair {pair.force_on}/{pair.force_off} forwards "
            f"kwargs {sorted(pair.forwarded_kwargs)} but has "
            "model_config_field=None and is not in "
            "KWARGS_FORWARDED_BUT_NOT_VIA_MODEL_CONFIG. Either set "
            "model_config_field (so the EngineCore mutation test parametrize "
            "covers it) or add these kwargs to the exception list with a "
            "comment explaining where the override actually takes effect "
            "(round-3 red-team #2.1 — liar-registry bypass)."
        )

    # (d) DeepSeek-V4 round 2 fix (PR #409): every forwarded_kwargs entry
    # must actually exist as a parameter on load_model AND
    # BatchedEngine.__init__. A typo like "force_hyrbid" silently
    # satisfies the invariants above, then fails downstream with a bare
    # KeyError in the keyword-only test. Catching it here gives a
    # descriptive failure with the offending pair name.
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.server import load_model

    load_params = set(inspect.signature(load_model).parameters)
    batched_params = set(inspect.signature(BatchedEngine.__init__).parameters)
    for pair in AUTO_ROUTING_FLAG_PAIRS:
        for kwarg in pair.forwarded_kwargs:
            assert kwarg in load_params, (
                f"Registry pair {pair.force_on}/{pair.force_off} forwards "
                f"`{kwarg}` but load_model has no such parameter — typo or "
                "stale refactor."
            )
            assert kwarg in batched_params, (
                f"Registry pair {pair.force_on}/{pair.force_off} forwards "
                f"`{kwarg}` but BatchedEngine.__init__ has no such "
                "parameter — typo or stale refactor."
            )

    # (e) DeepSeek-V4 round 2 fix (PR #409): for pairs with
    # ``model_config_field`` set, exactly 2 forwarded_kwargs are
    # required (force-on first, force-off second) — convention
    # enforced by ``_engine_core_override_cases()``. Previously this
    # check lived at parametrize-collection time, where a violation
    # aborts pytest with a collection error before any test runs.
    # Move it here so the failure is a normal, named test failure.
    for pair in AUTO_ROUTING_FLAG_PAIRS:
        if pair.model_config_field is None:
            continue
        assert len(pair.forwarded_kwargs) == 2, (
            f"Registry pair {pair.force_on}/{pair.force_off} declares "
            f"model_config_field={pair.model_config_field!r} but has "
            f"{len(pair.forwarded_kwargs)} forwarded kwargs; "
            "exactly 2 required (force-on first, force-off second)."
        )


def test_registry_is_not_runtime_mutated():
    """Round-4 cat-5 hardening: the registry is the SSOT for every gate,
    so it must equal what's literally written in source. Catches three
    classes of trust-attack on the registry:

      - Bypass #3 (conftest inject): a ``conftest.py`` mutates
        ``AUTO_ROUTING_FLAG_PAIRS`` at import time to absorb a new
        sneaky flag, making the parity check pass.
      - Bypass #4 (duck-typed entry): conftest replaces entries with
        non-dataclass duck types that satisfy attribute access but
        defeat the dataclass-aware invariants gate.
      - Bypass #5 (frozen-bypass): ``object.__setattr__`` mutates a
        frozen ``RoutingFlagPair`` to clear ``forwarded_kwargs`` and
        disable the forwarding gate.

    The defense: AST-parse THIS source file for literal
    ``RoutingFlagPair(...)`` calls, reconstruct the expected tuple of
    field values, then compare to the runtime tuple. Any divergence —
    in length, types, or field values — is a runtime mutation and
    fails loudly. This catches all three bypasses with one check
    because each of them produces a runtime tuple that disagrees with
    what's in source.
    """
    test_file = pathlib.Path(__file__).resolve()
    source = test_file.read_text()
    tree = ast.parse(source)

    # Find the AUTO_ROUTING_FLAG_PAIRS = (...) assignment and parse the
    # literal RoutingFlagPair(...) calls inside it. The assignment is
    # annotated (``: tuple[RoutingFlagPair, ...]``) so it parses as
    # ast.AnnAssign, not ast.Assign.
    #
    # DeepSeek-V4 round 2 fix (PR #409): walk ONLY direct children of
    # the module body, not arbitrary descendants. A local variable also
    # named ``AUTO_ROUTING_FLAG_PAIRS`` inside any function or class in
    # this file could be visited first by ``ast.walk`` (whose order is
    # unspecified) and replace ``expected_pairs`` with a different
    # Tuple — causing a spurious "runtime mutation detected" failure.
    # Restricting to ``tree.body`` makes the scan deterministic and
    # impossible to shadow.
    expected_pairs: list[dict[str, object]] = []
    for node in tree.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    target_name = t.id
                    break
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        else:
            continue
        if target_name != "AUTO_ROUTING_FLAG_PAIRS":
            continue
        if not isinstance(value_node, ast.Tuple):
            continue
        for elt in value_node.elts:
            if not (
                isinstance(elt, ast.Call)
                and isinstance(elt.func, ast.Name)
                and elt.func.id == "RoutingFlagPair"
            ):
                continue
            # Round-5 subagent 3 #E: RoutingFlagPair(*list) positional /
            # starred args bypass the kwargs check. Require every field
            # to be passed by keyword name so the source AST can verify.
            if elt.args:
                pytest.fail(
                    "RoutingFlagPair() call uses positional / starred args. "
                    "Every field must be passed by keyword name so the "
                    "source-AST scan can reconstruct it (round-5 subagent 3 "
                    "#E bypass)."
                )
            fields: dict[str, object] = {}
            for kw in elt.keywords:
                if kw.arg is None:
                    pytest.fail(
                        "RoutingFlagPair() call uses **unpack — disallowed "
                        "by registry-mutation gate (round-4 cat-5)."
                    )
                try:
                    fields[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:
                    pytest.fail(
                        f"RoutingFlagPair(...).{kw.arg} is not a literal value. "
                        "The registry-mutation gate requires every field to "
                        "be a literal so it can be reconstructed from source."
                    )
            expected_pairs.append(fields)
        break

    # Length must match — closes round-4 #3/#4 (added entries at runtime).
    assert len(AUTO_ROUTING_FLAG_PAIRS) == len(expected_pairs), (
        f"AUTO_ROUTING_FLAG_PAIRS has {len(AUTO_ROUTING_FLAG_PAIRS)} runtime "
        f"entries but source defines {len(expected_pairs)} literal entries. "
        "Runtime mutation detected (round-4 cat-5 #3/#4 — conftest inject "
        "or duck-type registry replacement)."
    )

    # Every runtime entry must be a real RoutingFlagPair (not a duck
    # type) AND match the source literal field-by-field.
    for i, (runtime_pair, expected) in enumerate(
        zip(AUTO_ROUTING_FLAG_PAIRS, expected_pairs)
    ):
        assert isinstance(runtime_pair, RoutingFlagPair), (
            f"AUTO_ROUTING_FLAG_PAIRS[{i}] is {type(runtime_pair).__name__}, "
            "not RoutingFlagPair — runtime mutation injected a duck type "
            "(round-4 cat-5 #4)."
        )
        for field_name, expected_value in expected.items():
            actual = getattr(runtime_pair, field_name)
            assert actual == expected_value, (
                f"AUTO_ROUTING_FLAG_PAIRS[{i}].{field_name} = {actual!r} "
                f"but source declares {expected_value!r}. Runtime mutation "
                "of a frozen dataclass detected (round-4 cat-5 #5 — "
                "object.__setattr__ bypass). If you intended to change the "
                "registry, edit AUTO_ROUTING_FLAG_PAIRS in source."
            )


def test_alias_profile_has_no_routing_shaped_fields():
    """Round-4 env-config #5: a contributor adds ``force_mllm: true``
    to an alias entry and a matching field on ``AliasProfile``, then
    consults the field inside load_model — a covert per-alias routing
    flip that bypasses CLI and load_model kwargs entirely.

    The ``aliases.json`` half is closed by ``_ALLOWED_PROFILE_KEYS``
    in ``model_aliases.py::_coerce`` (rejects unknown JSON keys). This
    test closes the dataclass half: no AliasProfile field may be
    routing-shaped (``force_*``, ``no_*``, ``enable_*``, ``disable_*``).
    Routing dimensions belong in ``AUTO_ROUTING_FLAG_PAIRS`` with full
    CLI surface, not as silent per-alias data.
    """
    import dataclasses

    from vllm_mlx.model_aliases import AliasProfile

    routing_shaped = [
        f.name
        for f in dataclasses.fields(AliasProfile)
        if _ROUTING_PARAM_NAME_RE.match(f.name)
    ]
    assert not routing_shaped, (
        f"AliasProfile has routing-shaped field(s): {routing_shaped}. "
        "Per-alias routing data is an out-of-band escape hatch "
        "(round-4 env-config #5). Routing dimensions must be registered "
        "in AUTO_ROUTING_FLAG_PAIRS with full CLI + load_model surface, "
        "not silently flipped by alias metadata."
    )


def test_aliases_json_has_no_routing_shaped_keys():
    """Companion check to ``test_alias_profile_has_no_routing_shaped_fields``:
    scan the actual ``aliases.json`` for any routing-shaped key. The
    closed-set check in ``_coerce`` already rejects unknown keys at
    load time, but this gate makes the failure visible at PR review
    time (test failure) rather than only at runtime."""
    import json

    pkg_root = _pkg_root()
    aliases_path = pkg_root / "aliases.json"
    # DeepSeek-V4 round 2 fix (PR #409): wrap read+parse in a
    # descriptive failure. Bare FileNotFoundError / JSONDecodeError on
    # this gate would confuse anyone investigating the failure — the
    # test isn't about file IO. Spell out the precondition.
    try:
        raw = json.loads(aliases_path.read_text())
    except FileNotFoundError as exc:
        pytest.fail(
            f"aliases.json missing at {aliases_path} — routing-shape "
            f"audit cannot run. Restore the file or update _pkg_root(). "
            f"Underlying error: {exc}"
        )
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"aliases.json at {aliases_path} is not valid JSON — "
            f"routing-shape audit cannot run. Fix the syntax error first. "
            f"Underlying error: {exc}"
        )

    offenders: list[str] = []
    for alias, value in raw.items():
        if not isinstance(value, dict):
            continue
        for key in value:
            if _ROUTING_PARAM_NAME_RE.match(key):
                offenders.append(f"{alias}.{key}")

    assert not offenders, (
        f"aliases.json contains routing-shaped key(s): {offenders}. "
        "Per-alias routing data is forbidden (round-4 env-config #5). "
        "Move the routing decision to AUTO_ROUTING_FLAG_PAIRS so it's "
        "visible at CLI + load_model."
    )


def test_conftests_do_not_deselect_or_replace_collection():
    """Round-4 cat-5 #6: a ``conftest.py`` deselects SOP gate tests via
    ``pytest_collection_modifyitems`` that calls ``items.remove(...)``,
    ``items[:] = ...``, or ``items.clear()``. The legitimate use of
    that hook is ``item.add_marker(skip_X)`` which leaves the items
    list intact; deselection is the attack pattern.

    Walks every conftest.py under ``tests/`` and AST-rejects calls
    that remove items from the collection list. Adding a new
    conftest that needs to filter the collection? Either justify in a
    PR-review comment and add an explicit allowlist entry, or use
    ``item.add_marker(pytest.mark.skip(...))`` instead.
    """
    pkg_root = _pkg_root()
    repo_root = pkg_root.parent
    tests_dir = repo_root / "tests"

    offenders: list[str] = []
    for conftest in tests_dir.rglob("conftest.py"):
        try:
            source = conftest.read_text()
        except UnicodeDecodeError:
            continue
        tree = ast.parse(source)

        # DeepSeek-V4 review fix (PR #409): only flag deselection-shaped
        # mutations INSIDE pytest_collection_modifyitems. A helper
        # function that happens to declare a local list named `items`
        # and calls items.remove(...) on it is unrelated to the
        # collection hook and shouldn't false-positive. Gate by
        # enclosing function name.
        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if func_node.name != "pytest_collection_modifyitems":
                continue
            for node in ast.walk(func_node):
                # Match `items.remove(...)`, `items.pop(...)`, `items.clear()`.
                if isinstance(node, ast.Call):
                    func = node.func
                    if (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "items"
                        and func.attr in {"remove", "pop", "clear"}
                    ):
                        offenders.append(
                            f"{conftest.relative_to(repo_root)}:{node.lineno} "
                            f"calls items.{func.attr}(...) inside "
                            "pytest_collection_modifyitems — drops tests from "
                            "collection (round-4 cat-5 #6 attack pattern). "
                            "Use item.add_marker(pytest.mark.skip(...)) to "
                            "skip without removing."
                        )
                # Match `items[:] = ...` (Assign with Subscript).
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "items"
                        ):
                            offenders.append(
                                f"{conftest.relative_to(repo_root)}:{node.lineno} "
                                "assigns to items[...] inside "
                                "pytest_collection_modifyitems — rewrites "
                                "collection list (round-4 cat-5 #6 attack "
                                "pattern). Use markers to skip tests instead."
                            )
                # Round-5 subagent 3 #H: `del items[i]` / `del items[:]`
                # achieves the same deselection via ast.Delete.
                if isinstance(node, ast.Delete):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "items"
                        ):
                            offenders.append(
                                f"{conftest.relative_to(repo_root)}:{node.lineno} "
                                "uses `del items[...]` inside "
                                "pytest_collection_modifyitems — drops tests "
                                "from collection (round-5 subagent 3 #H "
                                "bypass). Use item.add_marker(pytest.mark."
                                "skip(...)) instead."
                            )

    assert not offenders, "\n".join(offenders)


def test_auto_routing_flags_have_force_on_and_force_off_pair():
    """SOP gate (#393 lesson): every binary auto-routing decision must
    expose BOTH a force-on and a force-off CLI flag.

    The pattern that bit us with #393 — ``--mllm`` (force on) shipped
    without a paired ``--no-mllm`` (force off) — applies to every
    auto-detection path. False positives are inevitable (incomplete
    quants, custom forks, hardware-shaped edge cases); when we hit one
    the user needs an escape hatch *immediately*, not in a follow-up
    release that ships ~2 weeks later.

    Registry (module-level ``AUTO_ROUTING_FLAG_PAIRS``) is the single
    source of truth: every routing flag we expose must appear with
    both directions. Adding a new auto-detection *requires* adding
    both flags and registering them in the dataclass list above. Every
    other gate in this file derives from that same registry, so adding
    a new entry transparently extends keyword-only checks, kwarg
    forwarding checks, and EngineCore-mutation checks too.

    Past incidents this rule would have caught:
      - #393: ``--mllm`` had no inverse → Tylast had to wait for a
        patch release instead of overriding from his launchd plist.
      - #404 (related, hardware-side): no user override for MLX stream
        capability, only an internal probe. The bug went undetected on
        every chip family we don't own.

    Intentionally OUT OF SCOPE for the registry:
      - ``OutputRouter.from_tokenizer`` in vllm_mlx/output_router.py
        auto-detects Gemma 4 / Harmony channel formats by tokenizer
        vocabulary. Not a binary decision (3+ formats including None),
        already allowlisted to known-good tokens, and has a built-in
        legacy-parser fallback for any per-request failure. If a
        false-positive surfaces, add an override flag here.
    """
    pkg_root = _pkg_root()
    sources_by_file = {
        fname: (pkg_root / fname).read_text()
        for fname in ("cli.py", "server.py", "benchmark.py")
    }

    missing: list[str] = []
    for pair in AUTO_ROUTING_FLAG_PAIRS:
        for fname in pair.required_files:
            src = sources_by_file[fname]
            if not _flag_in_add_argument_calls(src, pair.force_on):
                missing.append(
                    f"force-on flag {pair.force_on} not registered via "
                    f"add_argument() in {fname} ({pair.desc}) — every "
                    "entrypoint that takes a model name needs the same "
                    "routing escape hatches (SOP §10)."
                )
            if not _flag_in_add_argument_calls(src, pair.force_off):
                missing.append(
                    f"force-off flag {pair.force_off} not registered via "
                    f"add_argument() in {fname} ({pair.desc}) — every binary "
                    "auto-routing decision needs an escape hatch in BOTH "
                    "directions; see #393 for the past incident this rule "
                    "encodes."
                )
    assert not missing, "\n".join(missing)


def test_no_unregistered_routing_shaped_flags():
    """SOP gate (red-team #1, PR #408): every CLI flag whose name
    matches the routing-shape pattern (``--force-*``, ``--no-*``,
    ``--enable-*``, ``--disable-*``) MUST be in
    ``AUTO_ROUTING_FLAG_PAIRS`` (registered as a binary routing
    decision) OR in ``NON_ROUTING_FLAGS_ALLOWLIST`` (intentionally a
    feature toggle, not a routing decision).

    The previous registry was a closed-set check — it only verified
    "known pairs are intact" and silently passed when a contributor
    added a new ``--audio`` / ``--enable-thinking`` flag without
    realizing they'd added an auto-routing decision. This test is the
    complement: enumerate every routing-shaped flag in the source and
    require a deliberate registration or allowlist decision.

    Scans every file discovered by ``_discover_entrypoints()`` — not
    a hardcoded list. Adding a new entrypoint file is automatically
    included (round-3 bypass #1.4 fix).

    Pick option 2 (allowlist) only when you're sure the flag is a UX
    knob, not a binary auto-detection that could ever go wrong. When
    in doubt, register the pair — the cost of an extra registry entry
    is zero, the cost of a missed escape hatch is a Tylast-style
    issue + patch release."""
    routing_pattern = re.compile(r"^--(?:force|no|enable|disable)-")

    discovered: set[str] = set()
    pkg_root = _pkg_root()
    registered_kwargs = KWARGS_THAT_MUST_BE_FORWARDED
    failures: list[str] = []
    for relpath in _discover_entrypoints():
        source = (pkg_root / relpath).read_text()

        # Prong 1: positional flag-string literals in add_argument calls
        # (original gate).
        for flag in _all_add_argument_flags(source):
            if routing_pattern.match(flag):
                discovered.add(flag)

        # Prong 2: routing-shaped string constants anywhere in the
        # module — catches helper-function indirection (round-4 cat-2
        # #3) and plugin-registry calls (round-4 cat-2 #5) where the
        # literal lives outside the add_argument() call.
        for flag in _routing_shaped_constants_in_module(source):
            discovered.add(flag)

        # Prong 3: `dest=` aliasing — innocent-looking flag with a
        # routing-kwarg dest is still a routing flag (round-4 cat-1).
        # Walk each add_argument call individually so we can pair the
        # dest= with its actual flag name. A registered flag using its
        # natural dest (e.g. `--force-hybrid` + `dest="force_hybrid"`)
        # is fine; an unregistered flag aliasing onto a routing kwarg
        # is the attack.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
                continue
            dest_val: str | None = None
            for kw in node.keywords:
                if (
                    kw.arg == "dest"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    dest_val = kw.value.value
            if dest_val is None or dest_val not in registered_kwargs:
                continue
            # dest IS a registered routing kwarg. Allow ONLY when at
            # least one positional flag name on this call matches a
            # registered routing flag — that's the legitimate "explicit
            # dest=" pattern. If every flag name on the call is
            # unregistered, this is dest= aliasing.
            flag_names_on_call = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            registered_flag_set = _registered_flag_names()
            if not any(fn in registered_flag_set for fn in flag_names_on_call):
                failures.append(
                    f"{relpath}:{node.lineno} add_argument({flag_names_on_call!r}, "
                    f"dest={dest_val!r}) — dest aliases a registered routing "
                    "kwarg under an unregistered flag name (round-4 cat-1). "
                    "Either rename the flag to a registered routing flag, "
                    "or remove dest= and route through cli.py main() with "
                    "an explicit name."
                )

        # Prong 4: custom argparse.Action subclasses — these can mutate
        # the namespace bypassing every name-based check (round-4
        # cat-1 #A2 + cat-2 #4).
        for lineno, action_repr in _add_argument_calls_with_custom_action(source):
            failures.append(
                f"{relpath}:{lineno} uses non-stdlib argparse action "
                f"{action_repr!r}. Custom Action subclasses can mutate the "
                "namespace bypassing every routing check — write the routing "
                "logic in cli.py main() with explicit kwargs to load_model "
                "instead (round-4 cat-1 #A2 + cat-2 #4). If you genuinely "
                "need a custom Action for non-routing reasons, add an "
                "allowlist branch to _add_argument_calls_with_custom_action "
                "with a comment."
            )

        # Prong 5: helper-function indirection — add_argument(<non-Constant>,
        # ...) where the flag name is computed at runtime (round-4 cat-2 #3).
        for lineno in _add_argument_calls_with_non_literal_flag(source):
            failures.append(
                f"{relpath}:{lineno} calls add_argument() with a non-literal "
                "first positional argument. Flag names must be string "
                "literals so the SOP gate can audit them. Helper functions "
                "that wrap add_argument() defeat AST scanning (round-4 cat-2 "
                "#3). Spell out the add_argument call site directly."
            )

        # Prong 6: add_argument(..., **dict-unpack) — round-5 subagent 1
        # #P2-3 (dest aliasing via unpacked dict literal). Reject the
        # ** shape outright in entrypoint files.
        for lineno in _add_argument_calls_with_dict_kwarg_unpack(source):
            failures.append(
                f"{relpath}:{lineno} calls add_argument() with **dict-unpack. "
                "Spell out every kwarg explicitly so the SOP gate can audit "
                "dest=/action=/etc (round-5 subagent 1 #P2-3 bypass)."
            )

        # Prong 7: getattr(p, "add_argument")(...) — round-5 subagent 1
        # #P2-5 (Attribute-access indirection). No legitimate use of
        # getattr on a parser for add_argument; ban the shape.
        for lineno in _getattr_add_argument_calls(source):
            failures.append(
                f"{relpath}:{lineno} uses getattr(...)('add_argument')(...) "
                "indirection — defeats every AST predicate that matches "
                "Attribute(attr='add_argument'). Use p.add_argument(...) "
                "directly (round-5 subagent 1 #P2-5 bypass)."
            )

    registered = _registered_flag_names()
    unregistered = discovered - registered - NON_ROUTING_FLAGS_ALLOWLIST

    if unregistered:
        failures.append(
            f"Found {len(unregistered)} routing-shaped flag(s) not in either "
            f"AUTO_ROUTING_FLAG_PAIRS or NON_ROUTING_FLAGS_ALLOWLIST:\n  "
            + "\n  ".join(sorted(unregistered))
            + "\n\nFor each, choose ONE:\n"
            "  (a) Register the pair in AUTO_ROUTING_FLAG_PAIRS (preferred — "
            "every binary auto-routing decision needs both directions per "
            "SOP §10, and this auto-extends every other gate in this file).\n"
            "  (b) Add to NON_ROUTING_FLAGS_ALLOWLIST if the flag is a feature "
            "toggle / UX knob, NOT a binary auto-detection.\n"
            "Don't pick (b) unless you're sure — the wrong choice lets the next "
            "#393 ship silently."
        )

    assert not failures, "\n\n".join(failures)


def test_routing_override_kwargs_are_forwarded_to_load_model():
    """Strengthened SOP gate (codex R1 PR #407 + red-team #4 PR #408 +
    round-3 hardening): catching the flag in argparse is not enough —
    it must also be forwarded to ``server.load_model`` so the
    override actually reaches EngineCore.

    Walks the AST of every ``load_model(...)`` call in every file
    discovered by ``_discover_entrypoints()`` (no longer hardcoded;
    closes round-3 bypass #4.4 where a new ``vllm_mlx/serve.py``
    entrypoint would never be scanned).

    Also REJECTS ``load_model(**expanded_dict)`` constructs (round-3
    bypass #4.1): with ``**`` the kwarg names are invisible to AST,
    so any call site using ``**`` to forward routing kwargs MUST
    instead spell them out so this gate can audit each one. Helper
    functions returning kwarg dicts are exactly the bypass shape we
    can't audit statically.

    The set of required kwargs is DERIVED from
    ``AUTO_ROUTING_FLAG_PAIRS[*].forwarded_kwargs`` so adding a new
    registry entry automatically extends this gate."""
    pkg_root = _pkg_root()

    def _find_load_model_calls(source: str) -> list[ast.Call]:
        tree = ast.parse(source)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match `load_model(...)` and `module.load_model(...)`.
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else None)
                )
                if name == "load_model":
                    calls.append(node)
        return calls

    failures: list[str] = []
    for relpath in _discover_entrypoints():
        source = (pkg_root / relpath).read_text()
        for call in _find_load_model_calls(source):
            # ``kw.arg is None`` indicates a `**unpack` construct. Allow
            # it ONLY if the call is empty of routing kwargs (i.e. no
            # explicit routing kwargs and no risk of partial forwarding
            # via the unpack). Otherwise reject — the unpacked dict is
            # opaque to AST audit.
            star_unpacks = [kw for kw in call.keywords if kw.arg is None]
            kwarg_names = {kw.arg for kw in call.keywords if kw.arg is not None}

            if star_unpacks:
                failures.append(
                    f"{relpath} load_model() call at line {call.lineno} uses "
                    "`**` unpacking which hides kwarg names from the SOP "
                    "audit. Spell out every routing kwarg explicitly "
                    f"({sorted(KWARGS_THAT_MUST_BE_FORWARDED)}) — otherwise a "
                    "helper that builds the dict can silently drop a routing "
                    "override (round-3 red-team bypass #4.1)."
                )
                continue

            # ``load_model`` defaults every routing override to False, so
            # omitting them all is fine for callers that never need to
            # override. But once a CALLER references one of these
            # kwargs, it must forward all of them — otherwise the
            # caller is half-wired and the override silently no-ops.
            overlap = kwarg_names & KWARGS_THAT_MUST_BE_FORWARDED
            if overlap and overlap != KWARGS_THAT_MUST_BE_FORWARDED:
                missing = KWARGS_THAT_MUST_BE_FORWARDED - overlap
                failures.append(
                    f"{relpath} load_model() call at line {call.lineno} forwards "
                    f"{sorted(overlap)} but omits {sorted(missing)} — every "
                    "routing-override kwarg must be forwarded from any caller "
                    "that forwards any one of them (SOP §10). This list is "
                    "auto-derived from AUTO_ROUTING_FLAG_PAIRS[*].forwarded_kwargs."
                )
    assert not failures, "\n".join(failures)


def test_load_model_has_no_unkeyworded_bool_or_routing_params_beyond_baseline():
    """SOP gate (red-team #3 PR #408, rounds 2 + 3): every positional
    parameter on ``load_model`` that is EITHER bool-typed OR
    routing-named OR has truthy-laundering shape MUST be keyword-only,
    except explicitly grandfathered pre-#407 entries.

    Three complementary detection prongs:

    1. ``_param_is_bool``: catches positional bools regardless of
       annotation style (real ``bool``, stringified ``"bool"`` or
       ``"Optional[bool]"``, or inferred from default value). Prevents
       the codex R2 bug shape — a new bool kwarg before ``*,`` shifts
       every downstream positional slot.

    2. ``_param_is_routing_shape``: catches positional params NAMED
       like routing overrides (``force_*``, ``no_*``, ``enable_*``,
       ``disable_*``) regardless of type. Closes the type-laundering
       bypass found in round 2.

    3. ``_param_default_has_custom_bool``: catches params whose
       default value is an instance of a class with a custom
       ``__bool__`` method. Closes round-3 bypass #3.5 where a
       ``_Flag`` class with ``__bool__`` returning False slipped all
       three previous prongs.

    Also: ``inspect.unwrap`` is called on ``load_model`` before
    introspection so decorators that don't use ``functools.wraps``
    can't hide the underlying signature (round-3 bypass #3.3).

    Also: rejects ``**var_keyword`` params on ``load_model``. A
    ``**routing_overrides: bool`` would hide every individual flag
    from the keyword-only check (round-3 bypass #3.6).

    The grandfather list is FROZEN. If you genuinely need to add a
    positional bool/routing param (almost never), update it with a
    comment explaining why — the explicit edit forces the discussion.
    """
    from vllm_mlx.server import load_model

    # Round-3 bypass #3.3: a non-functools.wraps decorator hides the
    # underlying signature. Unwrap explicitly.
    unwrapped = inspect.unwrap(load_model)
    sig = inspect.signature(unwrapped)

    # Pre-SOP positional bools/routing names. Do NOT extend casually
    # — see docstring. Every entry needs a 1-line reason.
    grandfathered = frozenset(
        {
            "force_mllm",  # original MLLM force-on flag, pre-#393
            "mtp",  # native MTP enable, pre-PR #407
        }
    )

    # Round-3 bypass #3.6: reject **var_keyword on load_model.
    for name, param in sig.parameters.items():
        assert param.kind != inspect.Parameter.VAR_KEYWORD, (
            f"load_model has VAR_KEYWORD param `**{name}` — this hides "
            "individual routing kwargs from SOP audit. Spell out every "
            "routing kwarg explicitly. See PR #408 round-3 red-team #3.6."
        )

    offenders: list[tuple[str, str]] = []
    for name, param in sig.parameters.items():
        if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            continue
        if name in grandfathered:
            continue
        if _param_is_bool(param):
            offenders.append((name, "bool-typed positional param"))
        elif _param_is_routing_shape(name):
            offenders.append((name, "routing-shape name (force_/no_/enable_/disable_)"))
        elif _param_default_has_custom_bool(param):
            offenders.append(
                (
                    name,
                    "positional param whose default's type defines __bool__ "
                    "(or inherits a non-object __bool__) — could be a "
                    "round-3 red-team #3.5 truthy-laundering bypass, but is "
                    "also triggered by harmless stdlib defaults like "
                    "pathlib.Path / enum.Enum / uuid.UUID. If your default "
                    "is one of those, move the param to keyword-only and "
                    "the gate will stop firing. The heuristic is "
                    "intentionally conservative — keyword-only is always "
                    "safe; positional + custom __bool__ shifts every later "
                    "positional caller silently",
                )
            )

    assert not offenders, (
        f"load_model has {len(offenders)} non-grandfathered positional "
        f"param(s) that should be keyword-only:\n  "
        + "\n  ".join(f"{n} — {reason}" for n, reason in offenders)
        + "\n\nMove each one AFTER the `*,` separator in load_model's "
        "signature. NEW bool/routing params must be keyword-only to avoid "
        "silently shifting downstream positional args (codex R2 lesson on "
        "PR #407) — a new positional changes the slot of every kwarg after "
        "it, so existing callers like "
        "`load_model(name, None, 1, 32768, False, 0.5)` start passing 0.5 "
        "as a truthy value to the wrong field. If you genuinely need a "
        "positional (almost never), update `grandfathered` in this test "
        "with the reason."
    )


def _param_is_bool(param: inspect.Parameter) -> bool:
    """Best-effort detection of bool-typed parameters across annotation
    styles. Strengthened in PR #408 round-2 red-team after subagent
    found 3 bypasses on the original ``annotation is bool`` check:

      - Stringified container annotations (``"Optional[bool]"``,
        ``"bool | None"``) defeated the literal ``annotation == "bool"``
        match. Now we AST-parse the annotation string and look for any
        ``Name(id="bool")`` node — catches arbitrary nesting.
      - Type laundering with ``int`` (``force_quant: int = 0``)
        semantically routes (truthy/falsy) but isn't a bool. NOT
        caught here — the complementary ``_param_is_routing_shape``
        name-pattern check handles that case in the test below.
      - No-annotation + non-bool falsy default (``force_quant=0``)
        also slips bool detection — same complementary name check
        catches it.
    """
    annotation = param.annotation
    if annotation is bool:
        return True
    if isinstance(annotation, str):
        # PEP 563 (``from __future__ import annotations``) stringifies
        # every annotation. AST-walk the string so wrappers don't hide
        # the bool reference.
        try:
            tree = ast.parse(annotation, mode="eval")
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "bool":
                return True
        return False
    if annotation is inspect.Parameter.empty:
        # No annotation — fall back to default. ``type(default) is bool``
        # is strict (won't match int).
        return type(param.default) is bool
    # PEP 604 ``bool | None`` / typing.Optional[bool] / typing.Union[...]
    # — these become real ``types.UnionType`` / ``typing.Union`` objects
    # when the module does NOT use ``from __future__ import annotations``.
    # Codex R1 (PR #409 review) flagged this gap: a positional
    # ``flag: bool | None = None`` annotated without PEP 563 slips the
    # bool detection above.
    import typing

    args = typing.get_args(annotation)
    if args and any(a is bool for a in args):
        return True
    return False


_ROUTING_PARAM_NAME_RE = re.compile(r"^(force_|no_|enable_|disable_)")


def _param_is_routing_shape(name: str) -> bool:
    """Does the parameter NAME look like a routing decision? Catches
    type-laundering bypasses (``force_quant: int = 0``) where the
    semantic role is "routing override" but the type annotation
    defeats ``_param_is_bool``. A parameter named in this shape MUST
    be keyword-only regardless of type — see PR #408 round-2
    red-team."""
    return bool(_ROUTING_PARAM_NAME_RE.match(name))


def _param_default_has_custom_bool(param: inspect.Parameter) -> bool:
    """Catches round-3 bypass #3.5: a custom class with ``__bool__``
    behaves as a routing toggle (``bool(default)`` returns False) but
    isn't a bool, isn't a routing-named param, and isn't any of the
    annotation shapes ``_param_is_bool`` checks.

    Heuristic: if the default value's type defines its own ``__bool__``
    method, treat it as a candidate truthy-laundering bypass.

    Codex R3 fix: previous version dereferenced ``object.__bool__``
    directly, but stock ``object`` does NOT define ``__bool__`` at all
    (Python's truthiness defaults are hardcoded in CPython without an
    explicit method on the type). That made ``object.__bool__`` raise
    AttributeError on any non-builtin default whose class didn't
    define ``__bool__`` — crashing the gate instead of returning
    False. Now uses ``getattr(..., None)`` for safety.

    False positives are rare — most real Python types either inherit
    no ``__bool__`` (default-True for non-empty) or are built-in
    types whose ``__bool__`` is at the type level. The check is
    intentionally conservative."""
    if param.default is inspect.Parameter.empty:
        return False
    if param.default is None:
        return False
    default_type = type(param.default)
    # Builtins (int, str, float, bool, list, dict, tuple, set, frozenset, None)
    # all have valid __bool__ — skip them. We're looking for user-defined
    # classes that override __bool__.
    if default_type.__module__ == "builtins":
        return False
    # Compare via getattr-with-default — both sides None means default
    # truthiness behavior (no custom __bool__); any difference means
    # the type has overridden it.
    object_bool = getattr(object, "__bool__", None)
    return getattr(default_type, "__bool__", None) is not object_bool


def test_param_default_has_custom_bool_does_not_crash():
    """Codex R3 regression: ``object`` does NOT define ``__bool__``
    on stock Python (truthiness is hardcoded in CPython). The previous
    implementation read ``object.__bool__`` directly, which raises
    AttributeError on any non-builtin default whose class also lacks
    a custom ``__bool__`` — crashing the gate. This test locks the
    getattr-with-default fix: ordinary user-defined classes return
    False, custom-__bool__ classes return True, builtins return False.
    """

    class _Plain:
        pass

    class _CustomBool:
        def __bool__(self):
            return False

    def _param_with_default(default) -> inspect.Parameter:
        return inspect.Parameter(
            "f",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default,
        )

    # Plain class — no custom __bool__, must NOT crash, must return False.
    assert _param_default_has_custom_bool(_param_with_default(_Plain())) is False
    # Custom __bool__ — must be detected.
    assert _param_default_has_custom_bool(_param_with_default(_CustomBool())) is True
    # Builtins must return False (early skip via __module__ == "builtins").
    assert _param_default_has_custom_bool(_param_with_default(0)) is False
    assert _param_default_has_custom_bool(_param_with_default("")) is False
    assert _param_default_has_custom_bool(_param_with_default([])) is False
    # None / empty defaults must return False.
    assert _param_default_has_custom_bool(_param_with_default(None)) is False
    assert (
        _param_default_has_custom_bool(
            inspect.Parameter("f", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        is False
    )


def test_param_is_bool_handles_pep604_unions():
    """Codex R1 regression: ``bool | None`` / ``Optional[bool]`` on a
    positional parameter slipped the previous detector because the
    annotation was a real ``types.UnionType``, not ``bool`` and not a
    string. Lock the union-aware branch so a future regression to the
    old behavior fails this test loudly.
    """
    import typing

    def _param_with(annotation, default=inspect.Parameter.empty) -> inspect.Parameter:
        return inspect.Parameter(
            "f",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
            default=default,
        )

    # Real PEP 604 / typing.Optional / typing.Union — must be detected.
    # noqa for UP045/UP007: this test verifies BOTH the legacy typing
    # forms AND the PEP 604 forms are detected — that's the point.
    assert _param_is_bool(_param_with(bool))
    assert _param_is_bool(_param_with(bool | None, default=None))
    assert _param_is_bool(_param_with(typing.Optional[bool], default=None))  # noqa: UP045
    assert _param_is_bool(_param_with(typing.Union[bool, str], default=False))  # noqa: UP007

    # Stringified.
    assert _param_is_bool(_param_with("bool"))
    assert _param_is_bool(_param_with("bool | None"))
    assert _param_is_bool(_param_with("Optional[bool]"))

    # No annotation, bool default.
    assert _param_is_bool(_param_with(inspect.Parameter.empty, default=False))
    assert _param_is_bool(_param_with(inspect.Parameter.empty, default=True))

    # Negative cases — must NOT trigger.
    assert not _param_is_bool(_param_with(int))
    assert not _param_is_bool(_param_with(str))
    assert not _param_is_bool(_param_with(int | None, default=None))
    assert not _param_is_bool(_param_with("int"))
    assert not _param_is_bool(_param_with("Optional[int]"))
    assert not _param_is_bool(_param_with(inspect.Parameter.empty, default=0))
    assert not _param_is_bool(_param_with(inspect.Parameter.empty, default=None))


def test_hybrid_overrides_mutually_exclusive_in_load_model():
    """server.load_model raises ValueError if both --force-hybrid and
    --no-hybrid are passed. Second line of defense — CLI also rejects
    via sys.exit(2), but load_model is a public entry point too."""
    from vllm_mlx.server import load_model

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_model(
            "fake/model",
            force_hybrid=True,
            no_hybrid=True,
        )


def test_spec_decode_overrides_mutually_exclusive_in_load_model():
    """server.load_model raises ValueError if both --force-spec-decode
    and --no-spec-decode are passed."""
    from vllm_mlx.server import load_model

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_model(
            "fake/model",
            force_spec_decode=True,
            no_spec_decode=True,
        )


def _post_sop_forwarded_kwargs() -> frozenset[str]:
    """Forwarded kwargs from the registry MINUS pre-SOP grandfathered
    ones. Used by the keyword-only test — pre-SOP positional bools
    (``force_mllm``) can't be retroactively moved without breaking
    callers, but every NEW kwarg added via the registry must be
    keyword-only."""
    return KWARGS_THAT_MUST_BE_FORWARDED - {"force_mllm"}


def test_routing_override_kwargs_are_keyword_only_in_load_model():
    """Every routing-override kwarg derived from the registry must be
    KEYWORD_ONLY in both ``load_model`` and ``BatchedEngine.__init__``,
    so existing positional callers don't silently get a True
    ``force_X`` etc. when they meant something else. See codex R2 on
    PR #407.

    Derived from ``AUTO_ROUTING_FLAG_PAIRS[*].forwarded_kwargs`` so
    adding a new pair to the registry automatically extends this
    check. The pre-SOP grandfathered ``force_mllm`` is excluded — its
    positional position is fixed for back-compat (verified by
    ``test_load_model_has_no_unkeyworded_bool_params_beyond_baseline``
    elsewhere)."""
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.server import load_model

    expected = _post_sop_forwarded_kwargs()
    assert expected, "Registry should produce at least one post-SOP kwarg"

    load_sig = inspect.signature(load_model)
    batched_sig = inspect.signature(BatchedEngine.__init__)

    for kwarg in expected:
        # DeepSeek-V4 round 2 fix (PR #409): assert the kwarg exists in
        # both signatures FIRST. Without this, a typo in
        # ``RoutingFlagPair.forwarded_kwargs`` (e.g. "force_hyrbid")
        # crashes with bare ``KeyError: 'force_hyrbid'`` — descriptive
        # failure beats cryptic crash.
        assert kwarg in load_sig.parameters, (
            f"Registry declares forwarded kwarg `{kwarg}` but it does not "
            "exist on load_model — typo in AUTO_ROUTING_FLAG_PAIRS or stale "
            "refactor. Fix the registry entry or add the kwarg to load_model."
        )
        assert kwarg in batched_sig.parameters, (
            f"Registry declares forwarded kwarg `{kwarg}` but it does not "
            "exist on BatchedEngine.__init__ — typo or stale refactor."
        )
        assert load_sig.parameters[kwarg].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"load_model({kwarg}=...) must be KEYWORD_ONLY to preserve "
            "positional-arg compatibility. See codex R2 on PR #407."
        )
        assert batched_sig.parameters[kwarg].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"BatchedEngine.__init__({kwarg}=...) must be KEYWORD_ONLY too."
        )


def _make_engine_core_for_override_test(monkeypatch, cfg, *, base=None):
    """Build an ``EngineCore`` with heavy dependencies stubbed so the
    routing-override block in ``__init__`` can be exercised in
    isolation. Returns the constructed core (or raises if __init__
    does).

    Round-3 fix #5.2: the stub ``enrich_model_config`` previously
    captured ``base`` from closure and ignored its ``_base`` argument.
    That allowed bypass #5.2 (pre-enrich mutation): a contributor
    could mutate ``base_cfg`` BEFORE enrich runs and the override
    would survive in the test, even though in production enrich
    overwrites with a fresh ModelConfig instance. Now the stub
    correctly clones from its argument so any pre-enrich mutation is
    discarded — matching production behavior.

    Round-3 fix #5.5: the default ``base`` was hardcoded
    ``ModelConfig(is_hybrid=True, supports_spec_decode=False)``. Tests
    that expected ``is_hybrid=False`` after a force-off override
    could pass even if the mutation never fired, because the base
    already happened to be ... wait, base was True for is_hybrid so
    that doesn't apply. The genuine #5.5 issue is for fields whose
    default matches the force-off expected value. We address this by
    making ``base`` parameter-overridable so callers can pre-set the
    field to the OPPOSITE of expected, forcing the mutation to be
    real."""
    from unittest.mock import MagicMock

    # Codex round-A fix (PR #409): in headless CI / sandboxed macOS
    # environments without a Metal device, importing engine_core
    # triggers a chain (scheduler.py → mlx_lm → mlx.core) that raises
    # RuntimeError at module-load time — BEFORE the monkeypatch below
    # can stub Scheduler. Catch that and skip; the test's purpose is
    # the EngineCore.__init__ routing-mutation block, which can't be
    # exercised without a usable MLX runtime anyway. On machines with
    # Metal (the SOP's intended audit surface), the import succeeds
    # and the gate fires.
    try:
        from vllm_mlx import engine_core as ec
        from vllm_mlx import model_auto_config as mac
        from vllm_mlx.model_auto_config import ModelConfig
    except RuntimeError as exc:
        pytest.skip(
            f"MLX runtime unavailable ({exc}) — EngineCore routing-mutation "
            "audit requires a working Metal device. Skipped on headless CI."
        )

    if base is None:
        base = ModelConfig(is_hybrid=True, supports_spec_decode=False)

    monkeypatch.setattr(mac, "detect_model_config", lambda _name: base)

    # Stub respects its argument (not closure). This means pre-enrich
    # mutation of ``base_cfg`` is invisible — matching production
    # behavior where enrich produces a fresh ModelConfig.
    def _stub_enrich(_base, _model):
        return ModelConfig(
            is_hybrid=_base.is_hybrid,
            supports_spec_decode=_base.supports_spec_decode,
        )

    monkeypatch.setattr(mac, "enrich_model_config", _stub_enrich)
    # Scheduler construction is heavy and pulls in MLX. Replace with
    # MagicMock so we only test the override block.
    monkeypatch.setattr(ec, "Scheduler", MagicMock())

    model = MagicMock()
    return ec.EngineCore(model=model, tokenizer=MagicMock(), config=cfg)


def _engine_core_override_cases() -> list[tuple[str, str, bool]]:
    """Build parametrize cases ``(model_config_field, kwarg, expected)``
    from the registry. Every routing pair whose ``model_config_field``
    is not None contributes 2 cases: one for the force-on kwarg
    (expected True) and one for the force-off kwarg (expected False).

    Convention enforced by ``_assert_registry_kwarg_convention``: for
    every ``model_config_field``-bearing pair, ``forwarded_kwargs[0]``
    is the force-on direction (sets field True) and
    ``forwarded_kwargs[1]`` is the force-off direction (sets False).
    """
    # DeepSeek-V4 round 2 fix (PR #409): the 2-kwarg invariant is
    # enforced by ``test_registry_invariants`` (execution-time test,
    # descriptive failure). Asserting it again here would fire at
    # parametrize-collection time — pytest aborts with a collection
    # error before any test runs, which is much harder to debug than a
    # named test failure. Here we just SKIP malformed entries; the
    # invariants test catches the malformation with full context.
    cases: list[tuple[str, str, bool]] = []
    for pair in AUTO_ROUTING_FLAG_PAIRS:
        if pair.model_config_field is None:
            continue
        if len(pair.forwarded_kwargs) != 2:
            continue
        on_kwarg, off_kwarg = pair.forwarded_kwargs
        cases.append((pair.model_config_field, on_kwarg, True))
        cases.append((pair.model_config_field, off_kwarg, False))
    return cases


@pytest.mark.parametrize(
    "model_config_field,kwarg,expected",
    _engine_core_override_cases(),
    ids=lambda v: str(v) if isinstance(v, (str, bool)) else "?",
)
def test_engine_core_applies_routing_overrides_from_registry(
    monkeypatch, model_config_field, kwarg, expected
):
    """For every routing pair in the registry with a non-None
    ``model_config_field``, ``EngineCore.__init__`` must mutate
    ``self.model_config.<field>`` when the corresponding kwarg is set
    on ``EngineConfig``. Catches red-team #5 from PR #408: a new
    EngineConfig field plumbed end-to-end but never actually applied
    to ModelConfig.

    Parametrize is DERIVED from the registry, so adding a new
    ``RoutingFlagPair`` with a ``model_config_field`` automatically
    adds new test cases. If your new pair's mutation block is missing
    from ``EngineCore.__init__``, this test fails immediately —
    nothing silently no-ops.

    DeepSeek-V4 review fix (PR #409): previously the stub default base
    happened to already match `expected` for two of four parametrized
    cases (force_hybrid=True / no_spec_decode=False), so those cases
    silently passed even if EngineCore stopped mutating ModelConfig.
    Now we force ``base.<field> = not expected`` so the mutation MUST
    fire to flip it back to ``expected`` — the only path to success
    is the actual mutation block running."""
    from vllm_mlx.engine_core import EngineConfig
    from vllm_mlx.model_auto_config import ModelConfig

    cfg = EngineConfig(model_name="fake/model", **{kwarg: True})
    # Force base to the OPPOSITE of expected. Now the only way for the
    # final config to equal expected is for EngineCore.__init__ to
    # actively mutate it — there's no "happens to match the default" path.
    base = ModelConfig(**{model_config_field: not expected})
    core = _make_engine_core_for_override_test(monkeypatch, cfg, base=base)

    actual = getattr(core.model_config, model_config_field)
    assert actual is expected, (
        f"Setting EngineConfig.{kwarg}=True must mutate "
        f"ModelConfig.{model_config_field} from {not expected} to {expected}, "
        f"but got {actual}. EngineCore.__init__ likely missing the mutation "
        f"block for this routing pair — add it after enrich_model_config() "
        f"and before `self.scheduler.model_config = self.model_config`."
    )


def test_engine_core_no_override_leaves_model_config_unchanged(monkeypatch):
    """Sanity: when no override kwarg is set, EngineCore leaves the
    enriched ModelConfig untouched. Pairs with the parametrized
    mutation test above — together they prove "fires when set, doesn't
    fire when not set"."""
    from vllm_mlx.engine_core import EngineConfig

    cfg = EngineConfig(model_name="fake/model")
    core = _make_engine_core_for_override_test(monkeypatch, cfg)
    # Stub returns is_hybrid=True, supports_spec_decode=False.
    assert core.model_config.is_hybrid is True
    assert core.model_config.supports_spec_decode is False


def _engine_core_mutex_cases() -> list[dict[str, bool]]:
    """Build mutex-conflict parametrize cases from the registry. For
    every pair with ``model_config_field`` not None, generate one
    conflict case where both directions are True."""
    cases: list[dict[str, bool]] = []
    for pair in AUTO_ROUTING_FLAG_PAIRS:
        if pair.model_config_field is None:
            continue
        on_kwarg, off_kwarg = pair.forwarded_kwargs
        cases.append({on_kwarg: True, off_kwarg: True})
    return cases


@pytest.mark.parametrize("flags", _engine_core_mutex_cases())
def test_engine_core_rejects_conflicting_routing_overrides(monkeypatch, flags):
    """Second line of defense: EngineCore raises ValueError if both
    directions of a registry-known routing-override pair are set on
    EngineConfig. Programmatic callers that bypass the CLI mutex still
    get caught. Derived from registry."""
    from vllm_mlx.engine_core import EngineConfig

    cfg = EngineConfig(model_name="fake/model", **flags)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_engine_core_for_override_test(monkeypatch, cfg)


def test_mtp_install_respects_supports_spec_decode():
    """Regression for codex R1 PR #407: MTP installer in scheduler.py
    must check ``self.model_config.supports_spec_decode`` (gated by
    --no-spec-decode). Pre-fix the gate only covered SuffixDecoding
    and DFlash, so --no-spec-decode silently let MTP run anyway."""
    import ast
    import importlib.resources
    import pathlib

    pkg_root = pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()
    source = (pkg_root / "scheduler.py").read_text()
    tree = ast.parse(source)

    # Find the block guarded by ``if self.config.enable_mtp:`` and
    # confirm it references ``supports_spec_decode`` somewhere within
    # its body. Coarse but catches the regression we care about
    # without coupling to the exact branch structure.
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # Match `if self.config.enable_mtp:` (Attribute chain).
        test = node.test
        if (
            isinstance(test, ast.Attribute)
            and test.attr == "enable_mtp"
            and isinstance(test.value, ast.Attribute)
            and test.value.attr == "config"
        ):
            body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
            if "supports_spec_decode" in body_src:
                found = True
                break
    assert found, (
        "scheduler.py's `if self.config.enable_mtp:` block must reference "
        "`supports_spec_decode` so --no-spec-decode (SOP §10) gates MTP "
        "the same way it gates SuffixDecoding/DFlash. Codex caught this "
        "as a silent override-no-op on PR #407 R1."
    )


def test_dflash_branch_rejects_no_spec_decode():
    """Regression for codex R1 PR #407: --enable-dflash + --no-spec-decode
    must be a mutex error. DFlash IS spec-decode; without this guard
    the user thinks they've disabled spec-decode but DFlash silently
    proceeds via its dedicated server (never touches EngineCore)."""
    import importlib.resources
    import pathlib

    pkg_root = pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()
    source = (pkg_root / "cli.py").read_text()

    # Substring check is enough — the mutex block is small and the
    # surrounding context is distinctive. We assert ordering: the
    # `no_spec_decode` check must come BEFORE `run_dflash_server` in
    # the same source file.
    no_spec_idx = source.find('"no_spec_decode"')
    dflash_idx = source.find("run_dflash_server(")
    assert no_spec_idx != -1, (
        "cli.py must reference no_spec_decode in the DFlash branch — "
        "DFlash is a spec-decode path and must honor --no-spec-decode."
    )
    assert dflash_idx != -1
    assert no_spec_idx < dflash_idx, (
        "no_spec_decode mutex check must come BEFORE run_dflash_server() "
        "call so the override actually rejects DFlash startup."
    )


def test_friendly_error_does_not_swallow_unrelated_valueerror(monkeypatch):
    """An unrelated ValueError (e.g. config parsing) must NOT trigger
    the friendly-error path — it should propagate as-is so genuine bugs
    surface and don't get misattributed to vision-tower issues."""
    import importlib.util
    import sys

    # Codex round-A fix (PR #409): in headless CI without a Metal
    # device, importlib.import_module("mlx_vlm") raises RuntimeError
    # from MLX init (not ImportError), bypassing the
    # ``except ImportError`` skip. Use find_spec to verify mlx_vlm is
    # installable without actually loading it — the test only needs
    # the module slot in sys.modules to exist before we replace it
    # with a fake. Skip if the package isn't installed at all.
    if importlib.util.find_spec("mlx_vlm") is None:
        pytest.skip("mlx_vlm not installed (vision extra)")

    from vllm_mlx.models import mllm as mllm_mod

    real_mlx_vlm = sys.modules.get("mlx_vlm")

    class _FakeMlxVlm:
        @staticmethod
        def load(_name):
            raise ValueError("config.json has an invalid model_type field")

    class _FakeMlxVlmUtils:
        @staticmethod
        def load_config(_name):
            return {}

    monkeypatch.setitem(sys.modules, "mlx_vlm", _FakeMlxVlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", _FakeMlxVlmUtils)

    inst = mllm_mod.MLXMultimodalLM(model_name="fake/bad-config")

    try:
        with pytest.raises(ValueError, match="invalid model_type"):
            inst.load()
    finally:
        # monkeypatch.setitem already restores on teardown, but the
        # original test had a paranoid manual restore. Preserve that
        # only when there's something to restore — find_spec-based
        # detection means we may not have imported it.
        if real_mlx_vlm is not None:
            sys.modules["mlx_vlm"] = real_mlx_vlm
