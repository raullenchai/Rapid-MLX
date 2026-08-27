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

import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FAKE = ROOT / "apps/rapid-mac/scripts/fake-rapid-mlx.sh"
GOLDEN_FLOWS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
IMAGE_TAGS = {"[image:gen]", "[image:edit]", "[image:both]"}
FIXTURE_IMAGE_TAG = "[image:both]"


# --- Mirrors of the Swift gates, byte-for-byte ---------------------------
#
# These re-implement `ModelCatalog` rules that decide whether a fixture row is
# READ or SILENTLY DROPPED. Approximating them is worse than not testing: a
# looser rule is green on exactly the rows the app throws away, which is the
# failure this file exists to catch. Each mirror names its Swift source so the
# two can be diffed by hand when either moves.

MAX_ALIAS_BYTES = 128  # ModelCatalog.maxAliasBytes
MAX_HF_REPO_BYTES = 192  # ModelCatalog.maxHuggingFaceRepoBytes
_ALIAS_BYTES = set(b"-._") | set(
    bytes(range(48, 58)) + bytes(range(65, 91)) + bytes(range(97, 123))
)


def _is_safe_alias(alias: str) -> bool:
    """``ModelCatalog.isSafeAlias``.

    The first byte must be an ASCII letter or digit — a regex of
    ``[A-Za-z0-9._-]+`` accepts a leading ``.``/``_``/``-`` that Swift
    rejects — and the whole alias is capped in BYTES, not characters.
    """
    raw = alias.encode("utf-8")
    if not raw or len(raw) > MAX_ALIAS_BYTES:
        return False
    if not (48 <= raw[0] <= 57 or 65 <= raw[0] <= 90 or 97 <= raw[0] <= 122):
        return False
    return all(b in _ALIAS_BYTES for b in raw)


def _sanitized_hf_repo(repo: str) -> str | None:
    """``ModelCatalog.sanitizedHuggingFaceRepo``. ``None`` means "dropped"."""
    trimmed = repo.strip()
    if not trimmed or len(trimmed.encode("utf-8")) > MAX_HF_REPO_BYTES:
        return None
    if trimmed in ("-", "\u2014"):
        return None
    parts = trimmed.split("/")
    if not 1 <= len(parts) <= 2:
        return None
    for part in parts:
        if not part or part in (".", ".."):
            return None
        if not all(b in _ALIAS_BYTES for b in part.encode("utf-8")):
            return None
    return trimmed


def _split_on_multi_space(line: str) -> list[str]:
    """``ModelCatalog.splitOnMultiSpace``.

    Columns are separated by runs of TWO OR MORE spaces/tabs; a single space
    stays inside its column (``5d ago`` is one field). Python's ``str.split()``
    breaks on any run, so a single-spaced row parses cleanly here and is
    rejected by Swift — the fixture would then look fine while the app fell
    through to its download path.
    """
    result: list[str] = []
    current = ""
    space_run = 0
    for ch in line:
        if ch in (" ", "\t"):
            space_run += 1
            continue
        if space_run >= 2 and current:
            result.append(current)
            current = ""
        elif space_run == 1 and current:
            current += " "
        current += ch
        space_run = 0
    if current:
        result.append(current)
    return [field.strip() for field in result]


def run_fake(
    subcommand: str, *args: str, settings: dict[str, str] | None = None
) -> str:
    env = os.environ.copy()
    env.update(settings or {})
    return subprocess.run(
        [str(FAKE), subcommand, *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
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
        if IMAGE_TAGS.intersection(fields):
            rows.append(fields)
    return rows


def test_models_emits_exactly_one_image_row(models_output):
    rows = image_rows(models_output)
    assert len(rows) == 1, f"expected one image capability row, got {rows}"


def test_image_row_has_the_shape_the_parser_indexes(models_output):
    (fields,) = image_rows(models_output)
    tag_index = next(i for i, field in enumerate(fields) if field in IMAGE_TAGS)
    assert fields[tag_index] == FIXTURE_IMAGE_TAG, (
        "the golden image model must exercise generation and editing"
    )
    # alias = fields[0]; size = fields[1..<tag]; repo = fields[tag + 1]
    assert tag_index > 1, "no size column: parseImageRows would read size as empty"
    assert tag_index + 1 < len(fields), "no HF id after the tag: the repo would be nil"
    alias, repo = fields[0], fields[tag_index + 1]
    # A rejected alias is dropped SILENTLY and the picker simply stays empty,
    # so this has to be the app's own rule rather than something close to it.
    assert _is_safe_alias(alias), f"parseImageRows would drop this alias: {alias!r}"
    assert _sanitized_hf_repo(repo) is not None, (
        f"sanitizedHuggingFaceRepo would drop this repo: {repo!r}"
    )
    assert "/" in repo, f"HF id should be owner/name, got {repo!r}"


def test_image_alias_is_cached_so_the_tab_resolves_without_a_download():
    """``ImageGenViewModel.resolveAlias`` prefers a cached entry.

    Without a matching ``ls`` row the tab still lists the model but resolves to
    an uncached one, and the golden flow would drive a download path that has
    no business running against a fake.
    """
    (fields,) = image_rows(run_fake("models"))
    tag_index = next(i for i, field in enumerate(fields) if field in IMAGE_TAGS)
    alias, repo = fields[0], fields[tag_index + 1]
    cached = run_fake("ls")
    cached_rows = [_split_on_multi_space(line.strip()) for line in cached.splitlines()]
    assert any(row[:2] == [alias, repo] for row in cached_rows if len(row) >= 2), (
        f"{alias} -> {repo} is not in `ls` as 2+-space-separated columns:\n{cached}"
    )


def test_audio_pull_state_remains_independent_from_chat_pull_state(tmp_path):
    """The two GUI fixtures persist different lane selections.

    A configured audio state must always return a set; otherwise `ls` crashes
    while checking membership and the audio-readiness journey loses its fake
    catalog before it can exercise any Desktop behavior.
    """
    audio_state = tmp_path / "pulled-audio.txt"
    audio_state.write_text("fake-qwen3-tts\n")
    chat_state = tmp_path / "pulled-chat.txt"
    chat_state.write_text("lfm2.5-2.6b-4bit\n")

    cached = run_fake(
        "ls",
        settings={
            "FAKE_AUDIO_PULL_STATE": str(audio_state),
            "FAKE_PULL_STATE": str(chat_state),
        },
    )

    assert "fake-qwen3-tts         fake/qwen3-tts" in cached
    assert "lfm2.5-2.6b-4bit      fake-org/fake-repo" in cached


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
        # Swift gates this on ``isSafeAlias`` too, so the permissive regex
        # would have this mirror excluding aliases the app never excludes.
        if first and _is_safe_alias(first[0]):
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
        f"{alias} is not excluded from the chat catalog — its image tag "
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


def test_image_cancel_route_accepts_the_clients_model_query(tmp_path):
    """The fake must match HTTP paths independently from their query string.

    ``ImageClient.cancel`` identifies the active model with ``?model=...``.
    Production routing parses that query separately; treating the entire
    request target as a literal path makes the native cancellation journey get
    a false 404 even though its wire request is valid.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    events = tmp_path / "events.jsonl"
    env = os.environ.copy()
    env["FAKE_EVENT_LOG"] = str(events)
    process = subprocess.Popen(
        [
            str(FAKE),
            "serve",
            "fake-image-alias",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError("fake sidecar did not start")
                time.sleep(0.05)

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/images/cancel?model=fake-image-alias",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
            assert json.load(response) == {"cancelled": True}
        logged = [json.loads(line) for line in events.read_text().splitlines()]
        assert [entry["event"] for entry in logged].count("image_cancel") == 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_info_reports_each_aliass_own_repo(models_output):
    """``info <alias>`` must name the SAME repo the catalog rows do.

    ``ModelCatalog.parseInfoRepo`` reads ``Alias: <alias> -> <repo>`` to resolve
    a model; if ``info`` answered every alias with the chat repo, readiness and
    resolution for the image model would target the chat repository while
    ``models``/``ls`` pointed at the image one. Pin the image alias against the
    repo its own image-capability row declares.
    """
    (fields,) = image_rows(models_output)
    tag_index = next(i for i, field in enumerate(fields) if field in IMAGE_TAGS)
    alias, row_repo = fields[0], fields[tag_index + 1]
    info = run_fake("info", alias).strip()
    assert info == f"Alias: {alias} -> {row_repo}", (
        f"info {alias} said {info!r}, but its catalog row maps it to {row_repo!r}"
    )


def test_vision_fixture_uses_one_repo_across_models_ls_and_info():
    """A repo mismatch makes the cached vision model render as Download."""
    settings = {"FAKE_VISION_CHAT": "1"}
    catalog = json.loads(run_fake("models", "--json", settings=settings))
    vision = next(
        item for item in catalog["text"] if item["alias"] == "qwen3-vl-2b-4bit"
    )
    alias, repo = vision["alias"], vision["hf_path"]

    cached_rows = [
        _split_on_multi_space(line.strip())
        for line in run_fake("ls", settings=settings).splitlines()
    ]
    assert any(row[:2] == [alias, repo] for row in cached_rows if len(row) >= 2)
    assert any(
        row[:3] == [alias, repo, "256 MB"] for row in cached_rows if len(row) >= 3
    )
    assert run_fake("info", alias, settings=settings).strip() == (
        f"Alias: {alias} -> {repo}"
    )
