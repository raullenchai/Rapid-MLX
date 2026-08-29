# SPDX-License-Identifier: Apache-2.0
"""The engine and the Desktop app must agree on serving-lane reason strings.

The engine reports *why* a model is serving text-only in
``serving_lane_reason``; the Desktop app turns that into the sentence a user
reads when the photo button is disabled. Nothing connects the two sides — the
engine emits bare string literals, the app matches them in a Swift ``switch`` —
so they drifted:

* the app matched ``operator_forced_text``, which the engine has never emitted
  (it emits ``text_lane_forced``), leaving that copy permanently unreachable;
* ``vision_memory_insufficient`` had no case at all, so a vision-capable model
  that merely did not fit in memory told the user to "choose a vision-capable
  model" — a remedy they had already applied.

Both survived because the Swift fixtures asserting this area used the ghost
strings too, so the tests agreed with the app instead of with the engine.

The engine now has a single source of truth: ``SERVING_LANE_REASONS`` /
``AUTO_TEXT_FALLBACK_REASONS`` in ``vllm_mlx/api/utils.py``, enforced at
construction time by ``ServingLaneDecision.__post_init__``. This test imports
that SSOT and, rather than trusting it to be right, re-scans the whole engine
tree for the literals actually emitted and proves the SSOT is both complete (no
emitted reason can fall outside it) and exactly matches the Swift ``case``
labels the user actually reads.

mlx-free: the ``vllm_mlx.api.utils`` import chain pulls in no MLX (verified —
the Linux CI leg runs this with no MLX installed), and reason collection is
pure text parsing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from vllm_mlx.api.utils import (
    AUTO_TEXT_FALLBACK_REASONS,
    SERVING_LANE_REASONS,
    VISION_SERVING_LANE_REASONS,
)

REPO = Path(__file__).resolve().parents[1]

# Scan the whole engine tree rather than naming files, so a reason introduced
# in a fourth module (or a literal seeded outside the decision function) cannot
# slip past this contract.
ENGINE = REPO / "vllm_mlx"

PROFILE_SWIFT = REPO / "apps/rapid-mac/Sources/Rapid/Server/ServerModelProfile.swift"


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Top-level string constants available to reason emitters in one module."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value.value
    return constants


def _string_value(node: ast.expr, constants: dict[str, str], *, context: str) -> str:
    """Resolve a string literal or a module-level string constant, else fail loud."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    raise AssertionError(
        f"{context} must use a string literal or module-level string constant; "
        f"AST scanner cannot validate {ast.unparse(node)!r}"
    )


def _bool_value(node: ast.expr, *, context: str) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    raise AssertionError(
        f"{context} must use a bool literal; AST scanner cannot validate "
        f"{ast.unparse(node)!r}"
    )


def _call_argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    for item in call.keywords:
        if item.arg == keyword:
            return item.value
    return call.args[position] if len(call.args) > position else None


def _decision_reasons() -> dict[str, tuple[bool, bool]]:
    """``{reason: (is_mllm, auto_fallback)}`` for every decision construction."""
    found: dict[str, tuple[bool, bool]] = {}
    for path in ENGINE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            constructor_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if constructor_name != "ServingLaneDecision":
                continue
            lane_node = _call_argument(node, 0, "is_mllm")
            reason_node = _call_argument(node, 1, "reason")
            auto_node = _call_argument(node, 2, "auto_text_fallback")
            assert lane_node is not None and reason_node is not None, (
                f"ServingLaneDecision at {path}:{node.lineno} omits lane or reason"
            )
            context = f"ServingLaneDecision at {path}:{node.lineno}"
            reason = _string_value(reason_node, constants, context=context)
            classification = (
                _bool_value(lane_node, context=context),
                _bool_value(auto_node, context=context)
                if auto_node is not None
                else False,
            )
            previous = found.setdefault(reason, classification)
            assert previous == classification, (
                f"{reason!r} has conflicting lane/fallback classifications: "
                f"{previous} and {classification}"
            )
    assert found, "no ServingLaneDecision constructions found — parser is stale"
    return found


def _assignment_targets(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        owner = ast.unparse(target.value)
        return [f"{owner}.{target.attr}"]
    if (
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    ):
        owner = ast.unparse(target.value)
        return [f"{owner}.{target.slice.value}"]
    if isinstance(target, (ast.List, ast.Tuple)):
        return [name for item in target.elts for name in _assignment_targets(item)]
    return []


# Dynamic assignments are allowed only when their exact source has already
# crossed a validated boundary. Keeping this roster exact makes the scanner
# fail closed on a new local variable, attribute, call, or destructuring site.
_VALIDATED_REASON_PASSTHROUGHS = frozenset(
    {
        (
            "api/utils.py",
            "error.serving_lane_reason",
            "serving_lane_reason",
        ),
        (
            "engine/batched.py",
            "self._serving_lane_reason",
            "serving_lane_reason",
        ),
        (
            "server.py",
            "_serving_lane_reason",
            "_serving_checkpoint.lane_reason",
        ),
        (
            "server.py",
            "serving_lane_reason",
            "serving_checkpoint.lane_reason",
        ),
        (
            "routes/models.py",
            "serving_lane_reason",
            "_served_lane_fields(model_id)",
        ),
    }
)

# Constructor/response keyword forwarding sites whose values have already
# crossed ``ServingLaneDecision`` or ``BatchedEngine`` validation. Any new
# dynamic keyword emission fails closed until its exact source is audited.
_VALIDATED_REASON_KEYWORD_PASSTHROUGHS = frozenset(
    {
        ("server.py", "BatchedEngine", "_serving_lane_reason"),
        ("server.py", "BatchedEngine", "serving_lane_reason"),
        (
            "routes/chat.py",
            "e.openai_detail",
            "getattr(engine, 'serving_lane_reason', None)",
        ),
        (
            "routes/responses.py",
            "e.openai_detail",
            "getattr(engine, 'serving_lane_reason', None)",
        ),
        ("routes/models.py", "ModelInfo", "serving_lane_reason"),
    }
)

# Dictionary-literal forwarding sites whose values come from a live engine
# that already validated its serving-lane reason at construction/ingestion.
_VALIDATED_REASON_DICT_PASSTHROUGHS = frozenset(
    {
        (
            "runtime/resident_models.py",
            "getattr(engine, 'serving_lane_reason', None)",
        ),
    }
)


def _literal_assignments(*, exact_target: str | None = None) -> set[str]:
    """Static reason values assigned to reason fields anywhere in the engine tree."""
    found: set[str] = set()
    seen_passthroughs: set[tuple[str, str, str]] = set()
    for path in ENGINE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [name for target in targets for name in _assignment_targets(target)]
            selected = [
                name
                for name in names
                if (
                    name == exact_target
                    if exact_target is not None
                    else name.endswith("serving_lane_reason")
                )
            ]
            if not selected:
                continue
            value = node.value
            if isinstance(value, ast.Constant) and value.value is None:
                # Optional response/schema fields are initialized empty; they
                # are not reason emitters.
                continue
            if not (
                isinstance(value, ast.Constant) and isinstance(value.value, str)
            ) and not (isinstance(value, ast.Name) and value.id in constants):
                relative_path = path.relative_to(ENGINE).as_posix()
                source = ast.unparse(value)
                passthroughs = {(relative_path, name, source) for name in selected}
                unknown = passthroughs - _VALIDATED_REASON_PASSTHROUGHS
                assert not unknown, (
                    "unvalidated dynamic serving-lane reason assignment(s): "
                    f"{sorted(unknown)}. Route the value through the shared "
                    "invariant, then add only that exact pass-through here."
                )
                seen_passthroughs.update(passthroughs)
                continue
            found.add(
                _string_value(
                    value,
                    constants,
                    context=f"reason assignment at {path}:{node.lineno}",
                )
            )
    if exact_target is None:
        assert seen_passthroughs == _VALIDATED_REASON_PASSTHROUGHS, (
            "validated serving-lane pass-through roster drifted:\n"
            f"missing: {sorted(_VALIDATED_REASON_PASSTHROUGHS - seen_passthroughs)}\n"
            f"unexpected: {sorted(seen_passthroughs - _VALIDATED_REASON_PASSTHROUGHS)}"
        )
    return found


def _keyword_reason_values() -> set[str]:
    """Static values passed through any ``serving_lane_reason=`` keyword."""
    found: set[str] = set()
    seen_passthroughs: set[tuple[str, str, str]] = set()
    for path in ENGINE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "serving_lane_reason":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and value.value is None:
                    continue
                if (
                    isinstance(value, ast.Constant) and isinstance(value.value, str)
                ) or (isinstance(value, ast.Name) and value.id in constants):
                    found.add(
                        _string_value(
                            value,
                            constants,
                            context=(
                                f"serving_lane_reason keyword at {path}:{node.lineno}"
                            ),
                        )
                    )
                    continue
                relative_path = path.relative_to(ENGINE).as_posix()
                passthrough = (
                    relative_path,
                    ast.unparse(node.func),
                    ast.unparse(value),
                )
                assert passthrough in _VALIDATED_REASON_KEYWORD_PASSTHROUGHS, (
                    "unvalidated dynamic serving_lane_reason keyword: "
                    f"{passthrough}. Route the value through the shared "
                    "invariant, then add only that exact pass-through here."
                )
                seen_passthroughs.add(passthrough)
    assert seen_passthroughs == _VALIDATED_REASON_KEYWORD_PASSTHROUGHS, (
        "validated serving_lane_reason keyword roster drifted:\n"
        f"missing: "
        f"{sorted(_VALIDATED_REASON_KEYWORD_PASSTHROUGHS - seen_passthroughs)}\n"
        f"unexpected: "
        f"{sorted(seen_passthroughs - _VALIDATED_REASON_KEYWORD_PASSTHROUGHS)}"
    )
    return found


def _dict_reason_values() -> set[str]:
    """Static values emitted under a dictionary ``serving_lane_reason`` key."""
    found: set[str] = set()
    seen_passthroughs: set[tuple[str, str]] = set()
    for path in ENGINE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                items = zip(node.keys, node.values, strict=True)
            elif isinstance(node, ast.DictComp):
                items = ((node.key, node.value),)
            else:
                continue
            for key, value in items:
                key_name = (
                    key.value
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    else constants.get(key.id)
                    if isinstance(key, ast.Name)
                    else None
                )
                if key_name != "serving_lane_reason":
                    continue
                if isinstance(value, ast.Constant) and value.value is None:
                    continue
                if (
                    isinstance(value, ast.Constant) and isinstance(value.value, str)
                ) or (isinstance(value, ast.Name) and value.id in constants):
                    found.add(
                        _string_value(
                            value,
                            constants,
                            context=(
                                f"serving_lane_reason dict value at "
                                f"{path}:{node.lineno}"
                            ),
                        )
                    )
                    continue
                passthrough = (
                    path.relative_to(ENGINE).as_posix(),
                    ast.unparse(value),
                )
                assert passthrough in _VALIDATED_REASON_DICT_PASSTHROUGHS, (
                    "unvalidated dynamic serving_lane_reason dictionary value: "
                    f"{passthrough}. Route the value through the shared "
                    "invariant, then add only that exact pass-through here."
                )
                seen_passthroughs.add(passthrough)
    assert seen_passthroughs == _VALIDATED_REASON_DICT_PASSTHROUGHS, (
        "validated serving_lane_reason dictionary roster drifted:\n"
        f"missing: "
        f"{sorted(_VALIDATED_REASON_DICT_PASSTHROUGHS - seen_passthroughs)}\n"
        f"unexpected: "
        f"{sorted(seen_passthroughs - _VALIDATED_REASON_DICT_PASSTHROUGHS)}"
    )
    return found


@pytest.mark.parametrize(
    "source",
    [
        "def assign(self, local_reason):\n    self._serving_lane_reason = local_reason\n",
        "def assign():\n    lane, serving_lane_reason = resolve()\n",
    ],
)
def test_dynamic_reason_assignment_scanner_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
):
    (tmp_path / "dynamic.py").write_text(source, encoding="utf-8")
    monkeypatch.setitem(globals(), "ENGINE", tmp_path)

    with pytest.raises(AssertionError, match="unvalidated dynamic"):
        _literal_assignments()


@pytest.mark.parametrize(
    "emission",
    [
        'payload["serving_lane_reason"] = "unknown_reason"',
        'payload = {"serving_lane_reason": "unknown_reason"}',
        ('payload = {"serving_lane_reason": "unknown_reason" for _ in range(1)}'),
        (
            'REASON_FIELD = "serving_lane_reason"\n'
            'payload = {REASON_FIELD: "unknown_reason"}'
        ),
    ],
)
def test_literal_reason_emission_reaches_exhaustive_ssot_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    emission: str,
):
    (tmp_path / "emission.py").write_text(
        f'decision = ServingLaneDecision(False, "text_checkpoint")\n{emission}\n',
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "ENGINE", tmp_path)
    monkeypatch.setitem(globals(), "_VALIDATED_REASON_PASSTHROUGHS", frozenset())
    monkeypatch.setitem(
        globals(),
        "_VALIDATED_REASON_KEYWORD_PASSTHROUGHS",
        frozenset(),
    )
    monkeypatch.setitem(
        globals(),
        "_VALIDATED_REASON_DICT_PASSTHROUGHS",
        frozenset(),
    )

    with pytest.raises(AssertionError, match="unknown_reason"):
        _assert_ssot_matches_engine()


def test_dynamic_dict_reason_scanner_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "dict_value.py").write_text(
        'payload = {"serving_lane_reason": local_reason}\n',
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "ENGINE", tmp_path)
    monkeypatch.setitem(
        globals(),
        "_VALIDATED_REASON_DICT_PASSTHROUGHS",
        frozenset(),
    )

    with pytest.raises(AssertionError, match="unvalidated dynamic"):
        _dict_reason_values()


def test_qualified_decision_constructor_is_scanned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "qualified.py").write_text(
        'decision = api.ServingLaneDecision(False, "qualified_reason")\n',
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "ENGINE", tmp_path)

    assert _decision_reasons() == {"qualified_reason": (False, False)}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('"literal_reason"', {"literal_reason"}),
        ("local_reason", None),
    ],
)
def test_reason_keyword_scanner_is_exhaustive_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: set[str] | None,
):
    (tmp_path / "keyword.py").write_text(
        f"response = Response(serving_lane_reason={value})\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "ENGINE", tmp_path)
    monkeypatch.setitem(
        globals(),
        "_VALIDATED_REASON_KEYWORD_PASSTHROUGHS",
        frozenset(),
    )

    if expected is None:
        with pytest.raises(AssertionError, match="unvalidated dynamic"):
            _keyword_reason_values()
    else:
        assert _keyword_reason_values() == expected


def _engine_reasons() -> dict[str, bool]:
    """``{reason: auto_text_fallback}`` across the whole engine tree.

    The flag answers "can a vision-capable model reach the user with this
    reason while serving text-only", which is what makes the generic copy
    wrong. ``ServingLaneDecision`` records that as ``auto_text_fallback``;
    assignments outside it have to be classified here.
    """
    reasons = {reason: auto for reason, (_, auto) in _decision_reasons().items()}
    # Plain assignments seed the field — startup does this for image-gen /
    # video-gen modalities and passes it straight through. Those models have
    # no vision lane to lose, so the generic copy is not misleading for them.
    for reason in _literal_assignments():
        reasons.setdefault(reason, False)
    # Response/model constructors can emit the field without assigning it to
    # a named target first. Include every static keyword literal in the same
    # exhaustive engine roster; audited dynamic forwarding sites add no new
    # value of their own.
    for reason in _keyword_reason_values():
        reasons.setdefault(reason, False)
    # Response dictionaries can emit the field without an assignment target
    # or call keyword. Scan their literal keys separately; audited dynamic
    # forwarding contributes no new reason value of its own.
    for reason in _dict_reason_values():
        reasons.setdefault(reason, False)
    # An assignment to the instance attribute is the engine downgrading
    # itself mid-load: after a failed vision load its own warning calls this
    # "auto-falling back to text-only serving" before starting the text lane.
    # Same silent downgrade the flag marks elsewhere, but it never passes
    # through ServingLaneDecision, so there is no flag to read. Runs after the
    # loop above, whose broader pattern also matches these lines.
    for reason in _literal_assignments(exact_target="self._serving_lane_reason"):
        reasons[reason] = True
    return reasons


def _swift_case_labels() -> set[str]:
    """Every string matched by ``ImageInputAvailability.photoHint(for:)``."""
    text = PROFILE_SWIFT.read_text()
    marker = "static func photoHint(for laneReason: String?) -> PhotoHint"
    start = text.find(marker)
    assert start != -1, (
        f"photoHint(for:) not found in {PROFILE_SWIFT} — parser is stale"
    )

    depth, body_start = 0, text.index("{", start)
    body = ""
    for i in range(body_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                body = text[body_start : i + 1]
                break
    assert body, f"unbalanced photoHint(for:) body in {PROFILE_SWIFT}"

    # Only quoted strings between `case` and its `:` are matched values; the
    # rest of the body is user-facing copy.
    labels: set[str] = set()
    for arm in re.finditer(r"case\s+((?:\"[^\"]+\"\s*,?\s*)+):", body):
        labels.update(re.findall(r'"([^"]+)"', arm.group(1)))
    assert labels, "no case labels parsed out of photoHint(for:) — parser is stale"
    return labels


def _assert_ssot_matches_engine() -> None:
    emitted = set(_engine_reasons())
    assert set(SERVING_LANE_REASONS) == emitted, (
        f"SERVING_LANE_REASONS and the engine tree disagree:\n"
        f"  in SSOT but never emitted by the engine: "
        f"{sorted(set(SERVING_LANE_REASONS) - emitted) or 'none'}\n"
        f"  emitted by the engine but missing from SSOT: "
        f"{sorted(emitted - set(SERVING_LANE_REASONS)) or 'none'}\n"
        f"Emitted (from tree scan): {sorted(emitted)}"
    )


def test_ssot_is_the_exhaustive_set_of_emitted_reasons():
    """The SSOT must both contain every reason the engine emits and not be dead.

    This is the mechanism-level check behind the issue: a stray ``serving_lane_reason``
    literal introduced anywhere in the engine (now or later) fails the ``<=`` half, and
    a stale SSOT entry that the engine stopped emitting fails the ``>=`` half. Either
    half means engine and copy have been allowed to drift again.
    """
    _assert_ssot_matches_engine()


def test_auto_text_fallback_ssot_matches_the_tree():
    """``AUTO_TEXT_FALLBACK_REASONS`` must exactly describe the decision function's flag.

    ``auto_text_fallback=True`` is a field on ``ServingLaneDecision``, so its
    authoritative enumeration is the set of ``ServingLaneDecision`` call sites
    that pass the flag (compared against the imported SSOT). ``__post_init__``
    enforces membership, so a reason that gains or loses the flag without the
    SSOT following is a construction-time error — this test is the belt that
    shows both stayed in lockstep.

    (This is deliberately distinct from the wider "silent downgrade" copy set —
    ``vision_weights_unavailable`` is a mid-load rewrite that never passes
    through ``ServingLaneDecision``, so it is not an ``AUTO_TEXT_FALLBACK``
    member even though it still needs dedicated copy; see
    ``test_every_silent_downgrade_has_dedicated_copy``.)
    """
    decision_auto = {
        reason for reason, (_, is_auto) in _decision_reasons().items() if is_auto
    }
    assert decision_auto, "no auto_text_fallback decisions parsed — parser is stale"
    assert set(AUTO_TEXT_FALLBACK_REASONS) == decision_auto, (
        f"AUTO_TEXT_FALLBACK_REASONS and the ServingLaneDecision call sites disagree:\n"
        f"  flagged by SSOT but no decision call site uses it: "
        f"{sorted(set(AUTO_TEXT_FALLBACK_REASONS) - decision_auto) or 'none'}\n"
        f"  decision call site flags it but SSOT misses it: "
        f"{sorted(decision_auto - set(AUTO_TEXT_FALLBACK_REASONS)) or 'none'}"
    )


def test_vision_lane_reason_ssot_matches_the_tree():
    decision_vision = {
        reason for reason, (is_mllm, _) in _decision_reasons().items() if is_mllm
    }
    assert set(VISION_SERVING_LANE_REASONS) == decision_vision, (
        "VISION_SERVING_LANE_REASONS and decision call sites disagree:\n"
        f"  in SSOT but no vision decision emits it: "
        f"{sorted(set(VISION_SERVING_LANE_REASONS) - decision_vision) or 'none'}\n"
        f"  emitted on vision lane but absent from SSOT: "
        f"{sorted(decision_vision - set(VISION_SERVING_LANE_REASONS)) or 'none'}"
    )


def test_desktop_matches_only_reasons_the_engine_emits():
    """No ghost cases.

    A ``case`` for a string the engine never sends is dead code that reads as
    handled. This is the exact shape of the ``operator_forced_text`` bug.
    """
    emitted = set(_engine_reasons())
    ghosts = _swift_case_labels() - emitted
    assert not ghosts, (
        f"ServerModelProfile.swift matches serving-lane reasons the engine "
        f"never emits: {sorted(ghosts)}. Either the engine renamed them (update "
        f"the Swift cases) or they were never real. Engine reasons: "
        f"{sorted(emitted)}"
    )


def test_every_silent_downgrade_has_dedicated_copy():
    """Silent downgrades must not fall through to the generic sentence.

    A silent downgrade means the checkpoint IS vision-capable but its vision
    lane was not admitted — recorded either as ``auto_text_fallback=True`` on a
    ``ServingLaneDecision`` or as the mid-load ``vision_weights_unavailable``
    rewrite (which never passes through the dataclass). The default arm tells
    the user to pick a vision-capable model, which is the model they are already
    running — so every silent downgrade needs copy naming the real cause. We use
    the full tree-derived downgrade set so both mechanisms are covered.
    """
    silent_downgrades = {r for r, is_auto in _engine_reasons().items() if is_auto}
    assert silent_downgrades, "no silent-downgrade reasons parsed — parser is stale"
    uncovered = silent_downgrades - _swift_case_labels()
    assert not uncovered, (
        f"these reasons downgrade a vision-capable model to text but have no "
        f"dedicated copy in ServerModelProfile.swift: {sorted(uncovered)}. They "
        f"currently render as 'Photos need a vision-capable model', which "
        f"names a remedy the user has already applied."
    )


def test_text_lane_forced_reason_has_dedicated_copy():
    """The deliberate text-lane pin still needs its own sentence.

    ``text_lane_forced`` is not an auto-fallback: the CLI reaches it through
    ``--no-mllm`` / ``--text-only``. The Desktop app exposes no such switch, so
    in-app it arrives only from an alias pinned ``is_text_only`` in the
    registry — a vision-config checkpoint served through the text lane. Either
    way the checkpoint looks vision-capable while photos are refused, which is
    exactly when the generic copy is at its most confusing.
    """
    assert "text_lane_forced" in SERVING_LANE_REASONS, (
        "the engine no longer emits text_lane_forced; update this test and the "
        "Swift case together"
    )
    assert "text_lane_forced" in _swift_case_labels(), (
        "ServerModelProfile.swift lost its text_lane_forced case — a "
        "registry-pinned text-only alias is back to the generic copy"
    )
