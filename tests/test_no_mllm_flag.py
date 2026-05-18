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

    Pick option 2 (allowlist) only when you're sure the flag is a UX
    knob, not a binary auto-detection that could ever go wrong. When
    in doubt, register the pair — the cost of an extra registry entry
    is zero, the cost of a missed escape hatch is a Tylast-style
    issue + patch release."""
    routing_pattern = re.compile(r"^--(?:force|no|enable|disable)-")

    discovered: set[str] = set()
    pkg_root = _pkg_root()
    for fname in ("cli.py", "server.py", "benchmark.py"):
        source = (pkg_root / fname).read_text()
        for flag in _all_add_argument_flags(source):
            if routing_pattern.match(flag):
                discovered.add(flag)

    registered = _registered_flag_names()
    unregistered = discovered - registered - NON_ROUTING_FLAGS_ALLOWLIST

    assert not unregistered, (
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


def test_routing_override_kwargs_are_forwarded_to_load_model():
    """Strengthened SOP gate (codex R1 PR #407 + red-team #4 PR #408):
    catching the flag in argparse is not enough — it must also be
    forwarded to ``server.load_model`` so the override actually
    reaches EngineCore.

    Walks the AST of every ``load_model(...)`` call in cli.py and
    server.py and confirms each kwarg in ``KWARGS_THAT_MUST_BE_FORWARDED``
    (derived from the registry) is either passed as a keyword arg OR
    not passed at all. The failure mode this catches: flag registered,
    mutex check present, but the kwarg is silently dropped between
    argparse and the load call → override no-ops.

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
    for fname in ("cli.py", "server.py"):
        source = (pkg_root / fname).read_text()
        for call in _find_load_model_calls(source):
            kwarg_names = {kw.arg for kw in call.keywords if kw.arg is not None}
            # ``load_model`` defaults every routing override to False, so
            # omitting them all is fine for callers that never need to
            # override. But once a CALLER references one of these
            # kwargs, it must forward all of them — otherwise the
            # caller is half-wired and the override silently no-ops.
            overlap = kwarg_names & KWARGS_THAT_MUST_BE_FORWARDED
            if overlap and overlap != KWARGS_THAT_MUST_BE_FORWARDED:
                missing = KWARGS_THAT_MUST_BE_FORWARDED - overlap
                failures.append(
                    f"{fname} load_model() call at line {call.lineno} forwards "
                    f"{sorted(overlap)} but omits {sorted(missing)} — every "
                    "routing-override kwarg must be forwarded from any caller "
                    "that forwards any one of them (SOP §10). This list is "
                    "auto-derived from AUTO_ROUTING_FLAG_PAIRS[*].forwarded_kwargs."
                )
    assert not failures, "\n".join(failures)


def test_load_model_has_no_unkeyworded_bool_params_beyond_baseline():
    """SOP gate (red-team #3, PR #408): every bool parameter on
    ``load_model`` must be keyword-only EXCEPT explicitly grandfathered
    pre-PR-#407 ones. Catches the codex R2 bug shape: a new bool kwarg
    inserted before the ``*,`` separator silently shifts every
    downstream positional arg by one slot, flipping the semantics of
    existing callers.

    The grandfather list is FROZEN. If you genuinely need to add a
    positional bool (almost never), update it with a comment
    explaining why — the explicit edit forces the discussion."""
    from vllm_mlx.server import load_model

    sig = inspect.signature(load_model)

    # Pre-SOP positional bools. Do NOT extend casually — see test
    # docstring. Every entry needs a 1-line reason.
    grandfathered = frozenset(
        {
            "force_mllm",  # original MLLM force-on flag, pre-#393
            "mtp",  # native MTP enable, pre-PR #407
        }
    )

    offenders: list[str] = []
    for name, param in sig.parameters.items():
        if not _param_is_bool(param):
            continue
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if name not in grandfathered:
                offenders.append(name)

    assert not offenders, (
        f"load_model has {len(offenders)} non-grandfathered POSITIONAL bool "
        f"param(s): {offenders}. NEW bool params must be keyword-only "
        f"(after the `*,` separator) to avoid silently shifting downstream "
        f"positional args. See codex R2 on PR #407 for the bug shape this "
        f"prevents — a new positional bool changes the slot of every kwarg "
        f"after it, so existing callers like "
        f"`load_model(name, None, 1, 32768, False, 0.5)` start passing 0.5 "
        f"as a truthy value to the wrong field. If you genuinely need a "
        f"positional bool (almost never), update `grandfathered` in this "
        f"test with the reason."
    )


def _param_is_bool(param: inspect.Parameter) -> bool:
    """Best-effort detection of bool-typed parameters across annotation
    styles (real type, stringified PEP 563, or just inferred from
    default value)."""
    if param.annotation is bool:
        return True
    if isinstance(param.annotation, str) and param.annotation == "bool":
        return True
    if param.annotation is inspect.Parameter.empty:
        # No annotation — fall back to default. ``type(default) is bool``
        # is strict (won't match int) which is exactly what we want.
        return type(param.default) is bool
    return False


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
        assert load_sig.parameters[kwarg].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"load_model({kwarg}=...) must be KEYWORD_ONLY to preserve "
            "positional-arg compatibility. See codex R2 on PR #407."
        )
        assert batched_sig.parameters[kwarg].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"BatchedEngine.__init__({kwarg}=...) must be KEYWORD_ONLY too."
        )


def _make_engine_core_for_override_test(monkeypatch, cfg):
    """Build an ``EngineCore`` with heavy dependencies stubbed so the
    routing-override block in ``__init__`` can be exercised in
    isolation. Returns the constructed core (or raises if __init__
    does)."""
    from unittest.mock import MagicMock

    from vllm_mlx import engine_core as ec
    from vllm_mlx import model_auto_config as mac
    from vllm_mlx.model_auto_config import ModelConfig

    base = ModelConfig(is_hybrid=True, supports_spec_decode=False)
    monkeypatch.setattr(mac, "detect_model_config", lambda _name: base)
    monkeypatch.setattr(
        mac,
        "enrich_model_config",
        lambda _base, _model: ModelConfig(
            is_hybrid=base.is_hybrid,
            supports_spec_decode=base.supports_spec_decode,
        ),
    )
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
    cases: list[tuple[str, str, bool]] = []
    for pair in AUTO_ROUTING_FLAG_PAIRS:
        if pair.model_config_field is None:
            continue
        assert len(pair.forwarded_kwargs) == 2, (
            f"Registry pair {pair.force_on}/{pair.force_off} declares "
            f"model_config_field={pair.model_config_field!r} but has "
            f"{len(pair.forwarded_kwargs)} forwarded kwargs; "
            "exactly 2 required (force-on first, force-off second)."
        )
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
    nothing silently no-ops."""
    from vllm_mlx.engine_core import EngineConfig

    cfg = EngineConfig(model_name="fake/model", **{kwarg: True})
    core = _make_engine_core_for_override_test(monkeypatch, cfg)

    actual = getattr(core.model_config, model_config_field)
    assert actual is expected, (
        f"Setting EngineConfig.{kwarg}=True must mutate "
        f"ModelConfig.{model_config_field} to {expected}, but got {actual}. "
        f"EngineCore.__init__ likely missing the mutation block for this "
        f"routing pair — add it after enrich_model_config() and before "
        f"`self.scheduler.model_config = self.model_config`."
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
    import importlib
    import sys

    try:
        importlib.import_module("mlx_vlm")
    except ImportError:
        pytest.skip("mlx_vlm not installed (vision extra)")

    from vllm_mlx.models import mllm as mllm_mod

    real_mlx_vlm = sys.modules["mlx_vlm"]

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
        sys.modules["mlx_vlm"] = real_mlx_vlm
