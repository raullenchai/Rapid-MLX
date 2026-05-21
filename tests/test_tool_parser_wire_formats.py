# SPDX-License-Identifier: Apache-2.0
"""
Declare-your-formats invariant for every registered ToolParser.

Forcing function for the bug class surfaced by #425 (jpcarranza94, PR #426)
and the v0.6.62 #429 meta-fix: when a new parser is added, the wire
format(s) it handles MUST be declared as a class attribute
``EXPECTED_WIRE_FORMATS``, drawn from the canonical
``WIRE_FORMAT_LABELS`` set in ``vllm_mlx/tool_parsers/abstract_tool_parser.py``.

The point is to make the parser ↔ format mapping *machine-checkable*
rather than buried in regex patterns. Three concrete consumers:

  1. ``tests/test_tool_call_streaming_parity.py`` (PR #430): parity
     fixtures use the same labels declared here.
  2. ``scripts/audit_tool_parser_coverage.py`` (PR #431): matrix-coverage
     audit can cross-reference declared formats.
  3. Future docs / CLI ``--help`` listing: can render the format-handled
     summary mechanically instead of free-form prose that rots.

Two layers, same shape as the audit gate:

  - Coverage assertion: every concrete (non-meta) parser declares at
    least one well-known format label.
  - Label-set integrity: every declared label exists in
    ``WIRE_FORMAT_LABELS`` — typo'd labels fail CI.
"""

from __future__ import annotations

import pytest

from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.abstract_tool_parser import WIRE_FORMAT_LABELS

# Meta-parsers that don't handle a wire format themselves — they route to
# concrete parsers. ``EXPECTED_WIRE_FORMATS = ()`` is the correct value;
# this set documents them explicitly so a future contributor adding a
# class to the meta tier can't accidentally bypass the declaration
# requirement by forgetting to set the attribute.
_META_PARSER_CLASSES: set[str] = {
    "AutoToolParser",  # "auto" / "generic" — routing meta-parser
}


def _registered_parser_classes() -> dict[type, set[str]]:
    """Return {parser_class: {registered_name, ...}}.

    Dedupes the registry — the same class can be registered under
    multiple alias names (e.g. ``hermes`` / ``nous`` / ``qwen3_coder`` all
    point at ``HermesToolParser``). We test each class once.
    """
    by_class: dict[type, set[str]] = {}
    for name, cls in ToolParserManager.tool_parsers.items():
        by_class.setdefault(cls, set()).add(name)
    return by_class


@pytest.mark.parametrize(
    "parser_cls, names",
    sorted(
        _registered_parser_classes().items(),
        key=lambda kv: kv[0].__name__,
    ),
    ids=lambda v: v.__name__ if isinstance(v, type) else "",
)
def test_every_parser_declares_wire_formats(parser_cls, names):
    """Every concrete (non-meta) ToolParser subclass must declare
    ``EXPECTED_WIRE_FORMATS`` with at least one label from
    ``WIRE_FORMAT_LABELS``.

    Forcing function: adding a new parser without this attribute fails
    CI here. The author is required to (a) think about what wire format
    they handle, (b) reuse an existing label or deliberately add a new
    one to ``WIRE_FORMAT_LABELS``. Both keep the audit/parity surfaces
    aligned with the actual parser landscape.

    Meta-parsers (``AutoToolParser``) are exempt — listed in
    ``_META_PARSER_CLASSES`` with a documented reason.
    """
    formats = parser_cls.EXPECTED_WIRE_FORMATS

    if parser_cls.__name__ in _META_PARSER_CLASSES:
        assert formats == (), (
            f"{parser_cls.__name__} is in _META_PARSER_CLASSES but declares "
            f"EXPECTED_WIRE_FORMATS={formats!r}. Meta-parsers route to other "
            f"parsers and should declare an empty tuple. Either remove from "
            f"_META_PARSER_CLASSES or unset the attribute."
        )
        return

    assert formats, (
        f"{parser_cls.__name__} (registered as {sorted(names)!r}) does not "
        f"declare EXPECTED_WIRE_FORMATS. Set it to a tuple of one or more "
        f"labels from WIRE_FORMAT_LABELS in "
        f"vllm_mlx/tool_parsers/abstract_tool_parser.py. If this is a "
        f"meta/router parser with no wire format of its own, add the "
        f"class name to _META_PARSER_CLASSES in this test file."
    )

    unknown = set(formats) - WIRE_FORMAT_LABELS
    assert not unknown, (
        f"{parser_cls.__name__} declares unknown wire-format label(s) "
        f"{sorted(unknown)!r}. Allowed labels are: "
        f"{sorted(WIRE_FORMAT_LABELS)!r}. If you intend to add a new label, "
        f"do so deliberately in WIRE_FORMAT_LABELS (with a docstring note "
        f"describing the format) and re-run this test."
    )


def test_wire_format_labels_have_consumers():
    """Every label in ``WIRE_FORMAT_LABELS`` should be claimed by at
    least one registered parser. Unclaimed labels are dead vocabulary
    and should be removed (or have a parser added that uses them).

    Soft-failure mode: report unclaimed labels but pass — adding a label
    for a planned-but-not-yet-implemented parser is legitimate (e.g.
    declaring ``llama_python_tag`` before the Llama parser is upgraded
    to handle it). This test exists so the unused-label drift surfaces
    in CI logs without blocking PRs.
    """
    claimed: set[str] = set()
    for cls in _registered_parser_classes():
        claimed.update(getattr(cls, "EXPECTED_WIRE_FORMATS", ()) or ())
    unclaimed = WIRE_FORMAT_LABELS - claimed
    # Print for CI visibility without failing — these are "TODO" labels.
    if unclaimed:
        print(
            f"NOTE: {len(unclaimed)} WIRE_FORMAT_LABELS not yet claimed by "
            f"any parser: {sorted(unclaimed)!r}. This is fine for "
            f"planned-but-unimplemented formats; remove the label if it's "
            f"truly dead."
        )


def test_meta_parser_classes_exist():
    """Sanity: every name in ``_META_PARSER_CLASSES`` resolves to an
    actually-registered class. Guards against typos that would silently
    bypass the assertion (a typo'd class name wouldn't match anything,
    so a real meta-parser would fall into the "must declare formats"
    branch and the audit would falsely complain)."""
    registered_class_names = {cls.__name__ for cls in _registered_parser_classes()}
    missing = _META_PARSER_CLASSES - registered_class_names
    assert not missing, (
        f"_META_PARSER_CLASSES references class names that aren't "
        f"registered: {sorted(missing)!r}. Either the class was renamed or "
        f"the name is a typo. Registered classes: "
        f"{sorted(registered_class_names)!r}"
    )
