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

import pytest


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
    import ast

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


def test_auto_routing_flags_have_force_on_and_force_off_pair():
    """SOP gate (#393 lesson): every binary auto-routing decision must
    expose BOTH a force-on and a force-off CLI flag.

    The pattern that bit us with #393 — ``--mllm`` (force on) shipped
    without a paired ``--no-mllm`` (force off) — applies to every
    auto-detection path. False positives are inevitable (incomplete
    quants, custom forks, hardware-shaped edge cases); when we hit one
    the user needs an escape hatch *immediately*, not in a follow-up
    release that ships ~2 weeks later.

    Registry below is the source of truth: every routing flag we expose
    must appear with both directions. Adding a new auto-detection
    *requires* adding both flags and registering them here. If someone
    in the future removes one direction of the pair, CI fails with a
    pointer to this principle.

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
    import importlib.resources
    import pathlib

    pkg_root = pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()
    cli_source = (pkg_root / "cli.py").read_text()
    server_source = (pkg_root / "server.py").read_text()
    bench_source = (pkg_root / "benchmark.py").read_text()

    # Every entry: (force-on flag, force-off flag, what it routes,
    # entrypoint_files_required). The 4th element is a tuple of
    # entrypoint source files where BOTH directions must appear — this
    # is how we enforce that ``rapid-mlx serve`` and ``python -m
    # vllm_mlx.server`` (and any other CLI taking a model name) all
    # expose the same escape hatches. Adding a new entrypoint that
    # takes a model name means adding it to every pair's required list.
    AUTO_ROUTING_FLAG_PAIRS = [
        (
            "--mllm",
            "--no-mllm",
            "MLLM vs text-only routing (#393)",
            ("cli.py", "server.py", "benchmark.py"),
        ),
        (
            "--tool-call-parser",
            "--no-tool-call-parser",
            "AliasProfile tool-call parser auto-selection",
            ("cli.py", "server.py"),
        ),
        (
            "--reasoning-parser",
            "--no-reasoning-parser",
            "AliasProfile reasoning parser auto-selection",
            ("cli.py", "server.py"),
        ),
        (
            "--force-hybrid",
            "--no-hybrid",
            "ModelConfig.is_hybrid (gates spec/suffix decode)",
            ("cli.py", "server.py"),
        ),
        (
            "--force-spec-decode",
            "--no-spec-decode",
            "ModelConfig.supports_spec_decode (gates MTP/DFlash/suffix)",
            ("cli.py", "server.py"),
        ),
    ]

    sources_by_file = {
        "cli.py": cli_source,
        "server.py": server_source,
        "benchmark.py": bench_source,
    }

    missing = []
    for force_on, force_off, desc, required_files in AUTO_ROUTING_FLAG_PAIRS:
        # AST-based check: flag must appear as a positional literal in
        # an ``add_argument`` call (not just anywhere in the source).
        # Closes the false-positive gap codex caught in PR #407 R1.
        for fname in required_files:
            src = sources_by_file[fname]
            if not _flag_in_add_argument_calls(src, force_on):
                missing.append(
                    f"force-on flag {force_on} not registered via "
                    f"add_argument() in {fname} ({desc}) — every "
                    "entrypoint that takes a model name needs the same "
                    "routing escape hatches (SOP §10)."
                )
            if not _flag_in_add_argument_calls(src, force_off):
                missing.append(
                    f"force-off flag {force_off} not registered via "
                    f"add_argument() in {fname} ({desc}) — every binary "
                    "auto-routing decision needs an escape hatch in BOTH "
                    "directions; see #393 for the past incident this rule "
                    "encodes."
                )
        # Legacy substring guard: catch flags present only in cli.py or
        # only in server.py without explicit entrypoint registration.
        # Keep both checks; the per-file required_files loop above is
        # the stronger invariant.
        on_in_cli = _flag_in_add_argument_calls(cli_source, force_on)
        on_in_server = _flag_in_add_argument_calls(server_source, force_on)
        off_in_cli = _flag_in_add_argument_calls(cli_source, force_off)
        off_in_server = _flag_in_add_argument_calls(server_source, force_off)
        if (on_in_cli and not on_in_server) or (on_in_server and not on_in_cli):
            missing.append(
                f"force-on flag {force_on} present in only one entrypoint "
                f"({'cli.py' if on_in_cli else 'server.py'}); both CLI paths "
                "must expose every routing escape hatch (SOP §10)."
            )
        if (off_in_cli and not off_in_server) or (off_in_server and not off_in_cli):
            missing.append(
                f"force-off flag {force_off} present in only one entrypoint "
                f"({'cli.py' if off_in_cli else 'server.py'}); both CLI paths "
                "must expose every routing escape hatch (SOP §10)."
            )
    assert not missing, "\n".join(missing)


def test_routing_override_kwargs_are_forwarded_to_load_model():
    """Strengthened SOP gate (codex R1 PR #407): catching the flag
    appears in argparse is not enough — it must also be forwarded to
    ``server.load_model`` so the override actually reaches EngineCore.

    Walks the AST of every ``load_model(...)`` call in cli.py and
    server.py and confirms each of the 4 routing-override kwargs is
    either passed as a keyword arg OR not passed at all (the
    ``load_model`` default is False). The failure mode this catches:
    flag registered, mutex check present, but the kwarg is silently
    dropped between argparse and the load call → override no-ops."""
    import ast
    import importlib.resources
    import pathlib

    pkg_root = pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()

    KWARGS_THAT_MUST_BE_FORWARDED = {
        "force_text",
        "force_mllm",
        "force_hybrid",
        "no_hybrid",
        "force_spec_decode",
        "no_spec_decode",
    }

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

    failures = []
    for fname in ("cli.py", "server.py"):
        source = (pkg_root / fname).read_text()
        for call in _find_load_model_calls(source):
            kwarg_names = {kw.arg for kw in call.keywords if kw.arg is not None}
            # Note: load_model has defaults of False for every routing
            # override, so omitting them all is fine for callers that
            # never need to override. But once a CALLER references one
            # of these kwargs in argparse, it must forward it. We
            # approximate that by requiring the *full set* in any call
            # that forwards at least one of them.
            overlap = kwarg_names & KWARGS_THAT_MUST_BE_FORWARDED
            if overlap and overlap != KWARGS_THAT_MUST_BE_FORWARDED:
                missing = KWARGS_THAT_MUST_BE_FORWARDED - overlap
                failures.append(
                    f"{fname} load_model() call at line {call.lineno} forwards "
                    f"{sorted(overlap)} but omits {sorted(missing)} — every "
                    "routing-override kwarg must be forwarded from any caller "
                    "that forwards any one of them (SOP §10)."
                )
    assert not failures, "\n".join(failures)


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


def test_routing_override_kwargs_are_keyword_only_in_load_model():
    """Regression: same lesson as test_force_text_is_keyword_only — the
    4 new routing-override flags must be KEYWORD_ONLY so existing
    positional callers don't silently get a True ``force_hybrid`` etc.
    when they meant something else. See codex R2 on PR #407."""
    import inspect

    from vllm_mlx.server import load_model

    sig = inspect.signature(load_model)
    for kwarg in ("force_hybrid", "no_hybrid", "force_spec_decode", "no_spec_decode"):
        assert sig.parameters[kwarg].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{kwarg} must be KEYWORD_ONLY to preserve positional-arg "
            "compatibility. See codex R2 on PR #407."
        )

    from vllm_mlx.engine.batched import BatchedEngine

    sig = inspect.signature(BatchedEngine.__init__)
    for kwarg in ("force_hybrid", "no_hybrid", "force_spec_decode", "no_spec_decode"):
        assert sig.parameters[kwarg].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"BatchedEngine.__init__ {kwarg} must be KEYWORD_ONLY too."
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


@pytest.mark.parametrize(
    "flags,expected_hybrid,expected_spec",
    [
        ({"no_hybrid": True}, False, False),
        ({"force_hybrid": True}, True, False),
        ({"force_spec_decode": True}, True, True),
        ({"no_spec_decode": True}, True, False),
        ({"no_hybrid": True, "force_spec_decode": True}, False, True),
        ({}, True, False),  # no flags → auto-detection result unchanged
    ],
)
def test_engine_core_applies_routing_overrides_to_model_config(
    monkeypatch, flags, expected_hybrid, expected_spec
):
    """EngineCore.__init__ must mutate ``self.model_config.is_hybrid``
    and ``self.model_config.supports_spec_decode`` when the matching
    override flag is set on ``EngineConfig``. This is the *only* place
    in the call chain that talks to ModelConfig — if it stops working,
    every routing escape hatch silently no-ops and the user gets the
    old buggy behavior back."""
    from vllm_mlx.engine_core import EngineConfig

    cfg = EngineConfig(model_name="fake/model", **flags)
    core = _make_engine_core_for_override_test(monkeypatch, cfg)

    assert core.model_config.is_hybrid is expected_hybrid
    assert core.model_config.supports_spec_decode is expected_spec


@pytest.mark.parametrize(
    "flags",
    [
        {"force_hybrid": True, "no_hybrid": True},
        {"force_spec_decode": True, "no_spec_decode": True},
    ],
)
def test_engine_core_rejects_conflicting_routing_overrides(monkeypatch, flags):
    """Second line of defense: EngineCore raises ValueError if both
    directions of a routing-override pair are set on EngineConfig.
    Programmatic callers that bypass the CLI mutex still get caught."""
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
