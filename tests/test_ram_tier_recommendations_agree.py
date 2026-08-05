# SPDX-License-Identifier: Apache-2.0
"""``install.sh`` and the desktop app must recommend the same model.

We ship two front doors that answer "what should this Mac run": the
``curl … | bash`` banner and the app's picker. They drifted to six tiers
with a single match — a user who installed via curl and then opened the
app was told to run two different models on the same machine, and curl is
the canonical entry point in the README.

This test is the thing that would have caught it. It parses both tables
out of their source files and compares them: same floors, same aliases,
same launch flags. Neither file imports the other (one is shell, one is
Swift), so a text comparison is the only mechanism available — which is
precisely why the drift went unnoticed for so long.

mlx-free: pure parsing, no engine import, runs on the Linux CI leg.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO / "install.sh"
APP_TIERS = REPO / "apps/rapid-mac/Sources/Rapid/Server/RAMBucketedDefault.swift"
README = REPO / "README.md"


def _parse_install_sh() -> list[tuple[int, str, list[str]]]:
    """``[(floor_gb, alias, flags)]`` from the RECOMMENDED_MODEL block."""
    text = INSTALL_SH.read_text()
    block = re.search(r"RECOMMENDED_FLAGS=\"\"\n(.*?)\nfi\n", text, re.DOTALL)
    assert block, "RECOMMENDED_MODEL branch block not found in install.sh"
    body = block.group(1)

    tiers: list[tuple[int, str, list[str]]] = []
    pending_floor: int | None = None
    pending_alias: str | None = None
    for line in body.splitlines():
        branch = re.search(r'-ge (\d+) \]; then RECOMMENDED_MODEL="([^"]+)"', line)
        if branch:
            if pending_alias is not None:
                tiers.append((pending_floor, pending_alias, []))
            pending_floor = int(branch.group(1))
            pending_alias = branch.group(2)
            continue
        fallback = re.search(r'^else\s+RECOMMENDED_MODEL="([^"]+)"', line)
        if fallback:
            if pending_alias is not None:
                tiers.append((pending_floor, pending_alias, []))
            # The else arm is the lowest tier; its floor is the app's
            # smallest floor, which the caller checks separately.
            pending_floor, pending_alias = -1, fallback.group(1)
            continue
        flags = re.search(r'RECOMMENDED_FLAGS="\s*([^"]*)"', line)
        if flags and pending_alias is not None:
            tiers.append((pending_floor, pending_alias, flags.group(1).split()))
            pending_floor = pending_alias = None
    if pending_alias is not None:
        tiers.append((pending_floor, pending_alias, []))
    return tiers


def _parse_app_tiers() -> list[tuple[int, str, list[str]]]:
    """``[(floor_gb, primary_alias, flags)]`` from RAMBucketedDefault."""
    text = APP_TIERS.read_text()
    body = re.search(r"static let tiers: \[Tier\] = \[(.*?)\n    \]", text, re.DOTALL)
    assert body, "tiers array not found in RAMBucketedDefault.swift"

    # Named picks declared above the array (lfm26Pick, lfm2FastPick, …).
    named: dict[str, tuple[str, list[str]]] = {}
    for m in re.finditer(
        r"static let (\w+) = Pick\(\s*alias: \"([^\"]+)\".*?launchFlags: \[([^\]]*)\]",
        text,
        re.DOTALL,
    ):
        named[m.group(1)] = (m.group(2), re.findall(r'"([^"]+)"', m.group(3)))

    tiers: list[tuple[int, str, list[str]]] = []
    for chunk in re.finditer(
        r"Tier\(\s*floorGB: (\d+),\s*primary: (.*?)(?:,\s*alt:|\s*\))",
        body.group(1),
        re.DOTALL,
    ):
        floor = int(chunk.group(1))
        primary = chunk.group(2)
        inline = re.search(
            r"Pick\(\s*alias: \"([^\"]+)\".*?launchFlags: \[([^\]]*)\]",
            primary,
            re.DOTALL,
        )
        if inline:
            tiers.append(
                (floor, inline.group(1), re.findall(r'"([^"]+)"', inline.group(2)))
            )
        else:
            key = primary.strip().rstrip(",").strip()
            assert key in named, f"unknown named pick {key!r}"
            alias, flags = named[key]
            tiers.append((floor, alias, flags))
    return tiers


def test_both_tables_parse():
    """A parser that silently matches nothing would make every assertion
    below vacuously true."""
    assert len(_parse_install_sh()) >= 5
    assert len(_parse_app_tiers()) >= 6


def test_same_alias_at_every_ram_size():
    """The comparison that matters: for a real Mac's RAM, both front doors
    name the same model. Compared by RAM size rather than by row so the
    app's 18 GB tier (which deliberately mirrors 16) doesn't register as a
    mismatch against install.sh's single 16-23 branch."""
    sh = sorted(_parse_install_sh(), key=lambda t: t[0], reverse=True)
    app = sorted(_parse_app_tiers(), key=lambda t: t[0], reverse=True)

    def pick(tiers, ram):
        """Both tables clamp below their lowest floor and must agree there
        too — ``RAMBucketedDefault.tier`` starts at ``tiers[0]`` and only
        moves up, install.sh's ``else`` arm catches everything. A 4 GB
        reading (a probe failure reports 0) must not fall off the end."""
        for floor, alias, flags in tiers:
            if ram >= floor:
                return alias, flags
        return tiers[-1][1], tiers[-1][2]

    # Every boundary in EITHER table, plus the value on each side of it.
    # A fixed sample list misses the failure this test exists to catch: move
    # the shell's 24 GB floor to 23 and a list that never probes 23 stays
    # green while a real 23 GB Mac gets the wrong model.
    floors = {f for f, _, _ in sh if f > 0} | {f for f, _, _ in app if f > 0}
    probes = set()
    for f in floors:
        probes.update({f - 1, f, f + 1})
    probes.update({4, 8, 256})  # below the floor, the floor, and a real Ultra
    probes = sorted(r for r in probes if r > 0)

    mismatches = []
    for ram in probes:
        sh_alias, sh_flags = pick(sh, ram)
        app_alias, app_flags = pick(app, ram)
        if (sh_alias, sh_flags) != (app_alias, app_flags):
            mismatches.append(
                f"{ram} GB: install.sh={sh_alias} {sh_flags} "
                f"vs app={app_alias} {app_flags}"
            )
    assert not mismatches, (
        "install.sh and the desktop app disagree about what to run:\n  "
        + "\n  ".join(mismatches)
        + "\n\nThe app's RAMBucketedDefault.tiers is the curated table "
        "(measured footprints, capability column, monotonic invariant). "
        "install.sh mirrors it. Update install.sh, not this test."
    )


def _render_banner(ram_gb: int) -> str:
    """Run install.sh's own tier block + quick-start line for ``ram_gb``.

    Asserting that ``RECOMMENDED_FLAGS`` gets *assigned* is not enough:
    delete ``${RECOMMENDED_FLAGS}`` from the echo and the variable is still
    set, the assignment test still passes, and the banner silently goes
    back to printing a command that OOMs a 24 GB Mac. So execute the real
    lines and read what a user would actually see.
    """
    text = INSTALL_SH.read_text()
    block = re.search(r"(RECOMMENDED_FLAGS=\"\"\n.*?\nfi)\n", text, re.DOTALL)
    assert block, "tier block not found"
    echo = [
        ln.strip()
        for ln in text.splitlines()
        if "rapid-mlx serve" in ln and ln.strip().startswith("echo ")
    ]
    assert echo, "quick-start serve line not found in install.sh"
    script = f"RAM_GB={ram_gb}\n{block.group(1)}\n" + "\n".join(echo)
    return subprocess.run(
        ["sh", "-c", script], capture_output=True, text=True, timeout=30
    ).stdout


def test_the_banner_actually_prints_the_launch_flags():
    """gemma-4-26b-4bit at 24 GB needs its vision tower dropped and its KV
    budget capped. That is not advice — without the flags the command does
    not fit on the Mac it is being handed to."""
    printed = _render_banner(24)
    assert "gemma-4-26b-4bit" in printed, printed
    for flag in ("--no-mllm", "--kv-cache-dtype", "bf16", "--cache-memory-mb", "512"):
        assert flag in printed, (
            f"{flag!r} missing from the quick-start line:\n{printed}"
        )


def test_the_banner_prints_a_bare_command_where_no_flags_are_needed():
    """Control: the flags must not leak onto tiers that do not want them."""
    for ram in (8, 16, 32, 64):
        printed = _render_banner(ram)
        assert "rapid-mlx serve" in printed
        assert "--no-mllm" not in printed, f"{ram} GB banner: {printed}"


def test_launch_flags_travel_with_the_recommendation():
    """The flags install.sh prints are the flags the app launches with."""
    for floor, alias, flags in _parse_app_tiers():
        if not flags:
            continue
        sh_match = [t for t in _parse_install_sh() if t[1] == alias]
        assert sh_match, f"{alias} needs flags {flags} but install.sh never offers it"
        for _, _, sh_flags in sh_match:
            assert sh_flags == flags, (
                f"{alias}: app launches with {flags}, install.sh prints {sh_flags}"
            )


def test_every_recommended_alias_exists():
    """A typo'd alias turns the install banner into a 404 at first run."""
    from vllm_mlx.model_aliases import list_aliases

    known = list_aliases()
    for source, tiers in (
        ("install.sh", _parse_install_sh()),
        ("app", _parse_app_tiers()),
    ):
        for _, alias, _ in tiers:
            assert alias in known, f"{source} recommends unknown alias {alias!r}"


@pytest.mark.parametrize(
    "ram,expected", [(8, "lfm2.5-2.6b-4bit"), (16, "bonsai-27b-2bit")]
)
def test_small_macs_get_something_that_fits(ram, expected):
    """Pinned literally: these two tiers are the ones that changed, and a
    regression here is the difference between an 8 GB Mac running a model
    and being told nothing fits."""
    sh = sorted(_parse_install_sh(), key=lambda t: t[0], reverse=True)
    for floor, alias, _ in sh:
        if ram >= floor:
            assert alias == expected
            return
    pytest.fail(f"no install.sh tier matched {ram} GB")


def _readme_table_tiers() -> list[tuple[int, str, list[str]]]:
    """``[(floor_gb, alias, one_shot_flags)]`` from the Choose Your Model table.

    Parsed from the tier ROWS, not by scanning the file: the README also
    names ``qwen3.5-4b-4bit`` as the ``rapid-mlx chat`` default, which is
    correct and is not a tier recommendation.

    Flags come from the One-shot code span specifically, not from anywhere
    in the row — a flag sitting in the prose column would satisfy a
    substring search while the command a reader actually pastes is still
    incomplete.
    """
    out = []
    for line in README.read_text().splitlines():
        m = re.match(
            r"\| \*\*(\d+)(?:[–-]\d+)? GB\+?\*\*[^|]*\| `([a-z0-9.\-]+)` \|[^|]*\| `([^`]+)` \|",
            line,
        )
        if not m:
            continue
        floor, alias, oneshot = int(m.group(1)), m.group(2), m.group(3).split()
        assert oneshot[:2] == ["rapid-mlx", "serve"], f"unexpected command: {oneshot}"
        assert oneshot[2] == alias, (
            f"README row recommends {alias} but its command runs {oneshot[2]}"
        )
        out.append((floor, alias, oneshot[3:]))
    return out


def _readme_prose_tiers() -> list[tuple[int, str]]:
    """``[(floor_gb, alias)]`` from the quick-start sentence.

    The README states the map twice. The prose copy is the one a reader
    hits first, and it can drift on its own — so it gets its own parse
    rather than being covered by the table's.
    """
    text = README.read_text()
    m = re.search(r"prints a serve command sized to your Mac \(([^)]*)\)", text)
    assert m, "quick-start tier sentence not found in README.md"
    return [
        (int(a), b)
        for a, b in re.findall(
            r"(\d+)(?:[–-]\d+)? GB\+? → `([a-z0-9.\-]+)`", m.group(1)
        )
    ]


def test_readme_table_matches_the_app_tier_for_tier():
    """Not just the same set of aliases — the same alias at the same RAM.

    Comparing sets would pass if two tiers swapped picks, or if a bucket
    boundary moved, while still reporting that the tables agree.
    """
    app = [(f, a) for f, a, _ in sorted(_parse_app_tiers())]
    # The app splits 16 and 18; the README covers both with one 16-23 row.
    app_collapsed = []
    for floor, alias in app:
        if app_collapsed and app_collapsed[-1][1] == alias:
            continue
        app_collapsed.append((floor, alias))
    readme = sorted(_readme_table_tiers())
    assert [(f, a) for f, a, _ in readme] == app_collapsed, (
        "README.md's tier table does not line up with RAMBucketedDefault.tiers:\n"
        f"  README: {[(f, a) for f, a, _ in readme]}\n"
        f"  app:    {app_collapsed}"
    )


def test_readme_prose_matches_the_readme_table():
    """The README states the map twice; both have to say the same thing."""
    prose = sorted(_readme_prose_tiers())
    table = sorted((f, a) for f, a, _ in _readme_table_tiers())
    assert prose == table, (
        "the README's quick-start sentence and its tier table disagree:\n"
        f"  prose: {prose}\n"
        f"  table: {table}"
    )


def test_readme_one_shot_commands_carry_the_exact_flags():
    """A README reader pastes the One-shot command verbatim. It must be the
    command the app runs — same flags, same order, nothing missing."""
    app = {alias: flags for _, alias, flags in _parse_app_tiers()}
    for floor, alias, oneshot_flags in _readme_table_tiers():
        assert alias in app, f"README recommends {alias}, which is not an app pick"
        assert oneshot_flags == app[alias], (
            f"README's one-shot command for {alias} has {oneshot_flags}, "
            f"the app launches it with {app[alias]}"
        )


# ---------------------------------------------------------------------------
# The Quickstart starter — the RAM-blind first-run pick
#
# The tier tables above answer "what should this Mac run". They do NOT
# cover the alias every brand-new user actually meets first: the
# Quickstart starter, which is deliberately the same on every Mac.
#
# Nothing checked it. The starter shipped as ``bonsai-1.7b-2bit`` on the
# strength of a tool-call eval (6/6 clean ``tool_calls``), and a community
# report found it degenerating on an ordinary chat question — reproduced
# 4/4, terminating 0/4, on plain-chat requests where the repetition guard
# is inactive by design (it gates on ``request.has_tools``). The eval that
# justified it measured a real capability that this slot is not judged on.
#
# These tests cannot judge output quality. What they can do is pin the
# mechanical contract that a swap must not break: the alias has to exist,
# its pinned HF repo has to be the one the registry resolves, and the
# downloaded and bundled (airgapped) paths must not drift apart — a
# divergence there means an offline build first-launches a model the
# online path already rejected.

QUICKSTART = REPO / "apps/rapid-mac/Sources/Rapid/UI/QuickstartView.swift"
BUNDLED = REPO / "apps/rapid-mac/Sources/Rapid/Server/BundledModel.swift"


def _parse_quickstart_default() -> tuple[str, str]:
    """``(alias, hfRepo)`` from ``QuickstartCoordinator.defaultChoice``."""
    text = QUICKSTART.read_text()
    block = re.search(
        r"static let defaultChoice = QuickstartModelChoice\((.*?)\n    \)",
        text,
        re.DOTALL,
    )
    assert block, "defaultChoice literal not found in QuickstartView.swift"
    body = block.group(1)
    alias = re.search(r'alias:\s*"([^"]+)"', body)
    repo = re.search(r'hfRepo:\s*"([^"]+)"', body)
    assert alias, "defaultChoice has no alias literal"
    assert repo, "defaultChoice must pin hfRepo — it drives the byte-progress monitor"
    return alias.group(1), repo.group(1)


def _parse_bundled() -> tuple[str, str]:
    """``(bundledAlias, bundledRepoID)`` from ``BundledModel.swift``."""
    text = BUNDLED.read_text()
    alias = re.search(r'static let bundledAlias: String = "([^"]+)"', text)
    repo = re.search(r'static let bundledRepoID: String = "([^"]+)"', text)
    assert alias and repo, (
        "bundledAlias / bundledRepoID not found in BundledModel.swift"
    )
    return alias.group(1), repo.group(1)


def test_quickstart_starter_exists():
    """An unknown starter alias breaks first run for every new user."""
    from vllm_mlx.model_aliases import list_aliases

    alias, _ = _parse_quickstart_default()
    assert alias in list_aliases(), f"Quickstart starter is unknown alias {alias!r}"


def test_quickstart_pinned_repo_matches_the_registry():
    """``hfRepo`` drives the bytes-on-disk progress bar. If it names a
    different repo than the alias resolves to, the download completes
    while the bar sits at 0% — the first-impression path, silently wrong."""
    from vllm_mlx.model_aliases import resolve_model

    alias, repo = _parse_quickstart_default()
    assert resolve_model(alias) == repo, (
        f"Quickstart pins hfRepo {repo!r} but {alias!r} resolves to "
        f"{resolve_model(alias)!r}"
    )


def test_bundled_starter_tracks_the_quickstart_starter():
    """``BundledModel`` is the airgapped twin of the Quickstart pick. The
    two are one product decision reached by two paths; letting them drift
    ships an offline build whose first launch uses the rejected model."""
    q_alias, q_repo = _parse_quickstart_default()
    b_alias, b_repo = _parse_bundled()
    assert b_alias == q_alias, (
        f"bundledAlias {b_alias!r} != Quickstart starter {q_alias!r}"
    )
    assert b_repo == q_repo, f"bundledRepoID {b_repo!r} != Quickstart hfRepo {q_repo!r}"


def test_build_script_stages_the_bundled_repo():
    """``BUNDLE_MODEL=1`` stages weights by repo id in build.sh. A stale
    default there bundles one model while the app asks for another, and
    the airgapped first launch falls through to a network pull it cannot
    make."""
    build_sh = (REPO / "apps/rapid-mac/scripts/build.sh").read_text()
    default = re.search(
        r'BUNDLED_MODEL_REPO="\$\{BUNDLED_MODEL_REPO:-([^}"]+)\}"', build_sh
    )
    assert default, "BUNDLED_MODEL_REPO default not found in build.sh"
    _, b_repo = _parse_bundled()
    assert default.group(1) == b_repo, (
        f"build.sh stages {default.group(1)!r} but BundledModel wants {b_repo!r}"
    )


def _parse_retired_starters() -> set[str]:
    text = QUICKSTART.read_text()
    block = re.search(
        r"static let retiredStarters: Set<String> = \[(.*?)\]", text, re.DOTALL
    )
    assert block, "retiredStarters literal not found in QuickstartView.swift"
    return set(re.findall(r'"([^"]+)"', block.group(1)))


def test_current_starter_is_not_itself_retired():
    """``retiredStarters`` re-opens onboarding for anyone whose last-served
    model is in it. Listing the *current* starter would re-show the wizard
    to every user who just completed it — an onboarding loop, and the one
    way this carve-out can strand people instead of rescuing them."""
    alias, _ = _parse_quickstart_default()
    assert alias not in _parse_retired_starters(), (
        f"current starter {alias!r} is listed in retiredStarters — "
        "every user who onboards onto it would be re-prompted forever"
    )


def test_retired_starters_are_real_aliases():
    """A typo here silently rescues nobody: the carve-out compares against
    the persisted ``rapid.serve.lastAlias``, so a misspelled entry just
    never matches and the stranded cohort stays stranded."""
    from vllm_mlx.model_aliases import list_aliases

    known = list_aliases()
    for alias in _parse_retired_starters():
        assert alias in known, f"retiredStarters names unknown alias {alias!r}"


VERIFY_SCRIPT = REPO / "apps/rapid-mac/scripts/verify-recommendation-tiers.swift"


def _normalise_swift(body: str) -> str:
    """Collapse whitespace and drop comments so a copy is compared on
    behaviour, not formatting."""
    body = re.sub(r"//[^\n]*", "", body)
    return re.sub(r"\s+", " ", body).strip()


def _extract_func_body(path: Path, signature: str) -> str:
    """Brace-matched body of the first ``func`` whose declaration starts
    with ``signature``, comments and whitespace normalised away."""
    text = path.read_text()
    start = text.index(signature)
    depth, i = 0, text.index("{", start)
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return _normalise_swift(text[i : j + 1])
    raise AssertionError(f"unbalanced {signature!r} body in {path}")


def _extract_retired_set(path: Path) -> set[str]:
    text = path.read_text()
    block = re.search(r"retiredStarters: Set<String> = \[(.*?)\]", text, re.DOTALL)
    assert block, f"retiredStarters literal not found in {path}"
    return set(re.findall(r'"([^"]+)"', block.group(1)))


QUICKSTART_PROD = REPO / "apps/rapid-mac/Sources/Rapid/UI/QuickstartView.swift"


def test_eligibility_check_script_has_not_drifted_from_production():
    """``verify-recommendation-tiers.swift`` re-declares the gate rather
    than importing it — the standalone-script pattern this repo uses
    because the SPM test target is stripped. That copy is the only thing
    that EXECUTES the gate, so if production changes and the copy does not,
    the check passes while testing yesterday's logic.

    The whole decision surface has to be pinned, not just the entry point:
    ``isEligible`` delegates to ``isStranded``, which reads
    ``retiredStarters``. Pinning only the first would let a new retired
    alias — the most likely future edit — land in production while the
    executable cases keep exercising the old set.

    Bodies are compared whole. An earlier version sliced from the first
    ``guard`` to skip signature differences, which meant anything inserted
    *above* that guard — an early return, a new precondition — diverged
    invisibly. The slice was never needed: ``_extract_func_body`` already
    returns brace-matched bodies without the signature, so the two sides
    are directly comparable."""
    for signature in ("func isEligible(", "func isStranded("):
        prod = _extract_func_body(QUICKSTART_PROD, signature)
        copy = _extract_func_body(VERIFY_SCRIPT, signature)
        assert prod == copy, (
            f"{signature.strip('func (')} drifted between "
            "QuickstartView.swift and verify-recommendation-tiers.swift — "
            "the executable check would be testing stale logic.\n"
            f"  production: {prod}\n  copy:       {copy}"
        )

    prod_set = _extract_retired_set(QUICKSTART_PROD)
    copy_set = _extract_retired_set(VERIFY_SCRIPT)
    assert prod_set == copy_set, (
        f"retiredStarters drifted: production {sorted(prod_set)} vs script "
        f"{sorted(copy_set)} — the rescued cohort differs from the tested one"
    )


def test_auto_start_skips_retired_starters():
    """Auto-start defaults to ON, so a stranded user's launch would resume
    the broken model and push ``serverState`` off ``.idle`` — which is
    exactly what Quickstart's third gate treats as "not a new user". The
    rescue card would then never render for the cohort it exists for.

    Pinned as source structure because ``AutoStartDecision.decide`` is not
    reachable from Python; the executable half lives in
    ``verify-recommendation-tiers.swift``."""
    decision = (
        REPO / "apps/rapid-mac/Sources/Rapid/Server/AutoStartDecision.swift"
    ).read_text()
    assert "case retiredStarter" in decision, (
        "AutoStartDecision lost its retiredStarter skip reason"
    )
    assert "if isRetiredStarter(alias) {" in decision, (
        "AutoStartDecision no longer guards against resuming a retired starter"
    )

    caller = (REPO / "apps/rapid-mac/Sources/Rapid/UI/ContentView.swift").read_text()
    assert (
        "!quickstart.done && QuickstartCoordinator.retiredStarters.contains" in caller
    ), (
        "the launch hook stopped passing the rescue-gated retired-starter "
        "predicate. Unconditional leaves a user who dismissed the rescue with "
        "neither auto-start nor a card; absent, the guard silently no-ops "
        "because the parameter defaults to 'never retired'"
    )

    # Presence is not enough: the guard has to come BEFORE the on-disk
    # check, or a cached retired starter returns .start and is resumed
    # before anything looks at whether it was retired. Order is what a
    # refactor moves silently, so pin it in both the production function
    # and the executable copy that models it.
    for path, label in (
        (
            REPO / "apps/rapid-mac/Sources/Rapid/Server/AutoStartDecision.swift",
            "AutoStartDecision.decide",
        ),
        (VERIFY_SCRIPT, "decideResume (the executable copy)"),
    ):
        text = path.read_text()
        guard_at = text.index("isRetiredStarter(alias)")
        disk_at = text.index("cachedAliases.contains(alias)")
        assert guard_at < disk_at, (
            f"{label}: the retired-starter guard moved after the on-disk "
            "check — a cached retired starter would be resumed before the "
            "guard ever runs"
        )
