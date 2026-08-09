# SPDX-License-Identifier: Apache-2.0
"""The fake sidecar's image row must stay parseable by the desktop catalog.

``gui-golden-flows.sh --flow image-generation`` only works because
``fake-rapid-mlx.sh models`` prints a row that ``ModelCatalog.parseImageRows``
recognises. Those two live in different languages, in different directories,
and neither imports the other — so the coupling is invisible and silent. When
it breaks, the flow fails as "Images.ModelPicker never resolved", which reads
as a product defect in the Images tab rather than as a drifted fixture.

The Swift side of the contract is pinned by ``ImageCatalogTests``. This is the
other side: it runs the fixture and checks the shape the parser depends on.
Stdlib + bash only, so it runs on the Linux CI lane that never sees a Mac.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FAKE = ROOT / "apps/rapid-mac/scripts/fake-rapid-mlx.sh"
GOLDEN_FLOWS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
IMAGE_TAG = "[image:gen]"


def run_fake(subcommand: str) -> str:
    return subprocess.run(
        [str(FAKE), subcommand],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


@pytest.fixture(scope="module")
def models_output() -> str:
    return run_fake("models")


def image_rows(output: str) -> list[list[str]]:
    """Re-implement ``ModelCatalog.parseImageRows``' field split.

    Deliberately a re-implementation and not a regex over the whole line: the
    Swift parser splits on whitespace and indexes, so testing anything looser
    here would pass on rows the app cannot read.
    """
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if IMAGE_TAG in fields:
            rows.append(fields)
    return rows


def test_models_emits_exactly_one_image_row(models_output):
    rows = image_rows(models_output)
    assert len(rows) == 1, f"expected one {IMAGE_TAG} row, got {rows}"


def test_image_row_has_the_shape_the_parser_indexes(models_output):
    (fields,) = image_rows(models_output)
    tag_index = fields.index(IMAGE_TAG)
    # alias = fields[0]; size = fields[1..<tag]; repo = fields[tag + 1]
    assert tag_index > 1, "no size column: parseImageRows would read size as empty"
    assert tag_index + 1 < len(fields), "no HF id after the tag: the repo would be nil"
    alias, repo = fields[0], fields[tag_index + 1]
    # ``isSafeAlias`` rejects anything outside this set, and a rejected alias
    # is dropped silently — the picker would simply stay empty.
    assert re.fullmatch(r"[A-Za-z0-9._-]+", alias), alias
    assert "/" in repo, f"HF id should be owner/name, got {repo!r}"


def test_image_alias_is_cached_so_the_tab_resolves_without_a_download():
    """``ImageGenViewModel.resolveAlias`` prefers a cached entry.

    Without a matching ``ls`` row the tab still lists the model but resolves to
    an uncached one, and the golden flow would drive a download path that has
    no business running against a fake.
    """
    (fields,) = image_rows(run_fake("models"))
    alias, repo = fields[0], fields[fields.index(IMAGE_TAG) + 1]
    cached = run_fake("ls")
    cached_rows = [line.split() for line in cached.splitlines()]
    assert any(row[:2] == [alias, repo] for row in cached_rows if len(row) >= 2), (
        f"{alias} -> {repo} is not in `ls`:\n{cached}"
    )


def _has_non_chat_kind_tag(line: str) -> bool:
    """Re-implement ``ModelCatalog.hasNonChatKindTag``.

    A whole ``[kind:subtype]`` token whose kind is a non-chat modality, not a
    bare substring — matching the parser exactly, so this test excludes a row
    only where the app would.
    """
    for field in line.split():
        if not (field.startswith("[") and field.endswith("]")):
            continue
        body = field[1:-1]
        kind, sep, subtype = body.partition(":")
        if not sep or kind not in {"audio", "video", "image"}:
            continue
        if subtype and all(ch.isalpha() or ch == "-" for ch in subtype):
            return True
    return False


def _chat_excluded_aliases(output: str) -> set[str]:
    """Re-implement ``ModelCatalog.parseExcludedAliases``: the first token of
    every non-chat-tagged line is an alias the chat catalog drops."""
    excluded = set()
    for raw in output.splitlines():
        line = raw.strip()
        if not _has_non_chat_kind_tag(line):
            continue
        first = line.split()[:1]
        if first and re.fullmatch(r"[A-Za-z0-9._-]+", first[0]):
            excluded.add(first[0])
    return excluded


def test_image_row_carries_a_kind_tag_so_chat_cannot_offer_it(models_output):
    """The chat catalog drops the image row, exercised through the real rule.

    One fixture, two opposite requirements: the Images tab must offer this
    model and the chat picker must refuse it. Asserting only that the tag is
    present would be tautological — ``image_rows`` already filters on it — so
    run the alias through the chat side's own exclusion rule and require it to
    fall out. If the tag were ever dropped, the Images tab would go empty AND a
    checkpoint that cannot answer a chat request would appear in chat (#1603).
    """
    (fields,) = image_rows(models_output)
    alias = fields[0]
    assert alias in _chat_excluded_aliases(models_output), (
        f"{alias} is not excluded from the chat catalog — its {IMAGE_TAG} tag "
        "no longer marks it as a non-chat modality"
    )


def test_flow_and_fixture_agree_on_the_alias(models_output):
    """The flow asserts against a literal; the fixture prints one."""
    (fields,) = image_rows(models_output)
    alias = fields[0]
    declared = GOLDEN_FLOWS.read_text(encoding="utf-8")
    assert f'FAKE_IMAGE_ALIAS="{alias}"' in declared, (
        f"gui-golden-flows.sh does not declare FAKE_IMAGE_ALIAS={alias!r}; "
        "the flow would assert against a model the fixture never prints"
    )
