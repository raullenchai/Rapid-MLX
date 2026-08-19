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

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO / "install.sh"
RECOMMENDATIONS = REPO / "vllm_mlx/model_recommendations.json"
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
    """``[(floor_gb, primary_alias, flags)]`` from the shared SSOT."""
    payload = json.loads(RECOMMENDATIONS.read_text())
    return [
        (tier["floor_gb"], tier["picks"][0]["alias"], tier["picks"][0]["launch_flags"])
        for tier in payload["tiers"]
    ]


def test_both_tables_parse():
    """A parser that silently matches nothing would make every assertion
    below vacuously true."""
    assert len(_parse_install_sh()) >= 5
    assert len(_parse_app_tiers()) >= 6


def test_same_alias_at_every_ram_size():
    """The comparison that matters: for a real Mac's RAM, both front doors
    name the same model. Compared by RAM size rather than by row so the
    app's explicit laptop tiers are checked at both sides of each boundary."""
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


def test_the_banner_prints_the_27b_bare_from_32_gb_up():
    """AA-Index policy (2026-08-18): every Mac from 32 GB up is handed
    qwen3.8-27b-4bit, and it needs no tier flags — MTP is baked into the
    alias and the measured 8K peak (20.0 GB) fits every one of these
    tiers bare. The gemma flag bundle left the table with gemma."""
    for ram in (32, 64, 96):
        printed = _render_banner(ram)
        assert "qwen3.8-27b-4bit" in printed, f"{ram} GB banner: {printed}"


def test_the_banner_prints_a_bare_command_where_no_flags_are_needed():
    """Control: launch flags must not leak onto tiers that do not want
    them — since the gemma pick retired, that is every tier."""
    for ram in (8, 16, 24, 32, 64, 96):
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
    "ram,expected",
    [
        (8, "lfm2.5-2.6b-4bit"),
        (16, "qwen3.5-4b-4bit"),
        (18, "qwen3.5-9b-4bit"),
    ],
)
def test_small_macs_get_something_that_fits(ram, expected):
    """Pinned literally: these laptop tiers are the ones that changed, and a
    regression here is the difference between an 8 GB Mac running a model
    and being told nothing fits."""
    sh = sorted(_parse_install_sh(), key=lambda t: t[0], reverse=True)
    for floor, alias, _ in sh:
        if ram >= floor:
            assert alias == expected
            return
    pytest.fail(f"no install.sh tier matched {ram} GB")


def _readme_table_tiers() -> list[tuple[int, str, str, list[str]]]:
    """``[(floor_gb, alias, peak_rss, one_shot_flags)]`` from the Choose Your
    Model table.

    Parsed from the tier ROWS, not by scanning the file: the README also
    names ``qwen3.5-4b-4bit`` as the ``rapid-mlx chat`` default, which is
    correct and is not a tier recommendation.

    Flags come from the One-shot code span specifically, not from anywhere
    in the row — a flag sitting in the prose column would satisfy a
    substring search while the command a reader actually pastes is still
    incomplete.

    The Peak RSS cell is captured (not skipped) so it can be pinned against
    the SSOT: it drifted on 2 of 5 rows precisely because the old regex
    swallowed it with ``[^|]*``.
    """
    out = []
    for line in README.read_text().splitlines():
        m = re.match(
            r"\| \*\*(\d+)(?:[–-]\d+)? GB\+?\*\*[^|]*"
            r"\| `([a-z0-9.\-]+)` "
            r"\|\s*([\d.]+) GB\s*"
            r"\| `([^`]+)` \|",
            line,
        )
        if not m:
            continue
        floor, alias, rss = int(m.group(1)), m.group(2), m.group(3)
        oneshot = m.group(4).split()
        assert oneshot[:2] == ["rapid-mlx", "serve"], f"unexpected command: {oneshot}"
        assert oneshot[2] == alias, (
            f"README row recommends {alias} but its command runs {oneshot[2]}"
        )
        out.append((floor, alias, rss, oneshot[3:]))
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
    # Collapse only genuinely identical adjacent picks, if a future table
    # expresses the same recommendation at two consecutive floors.
    app_collapsed = []
    for floor, alias in app:
        if app_collapsed and app_collapsed[-1][1] == alias:
            continue
        app_collapsed.append((floor, alias))
    readme = sorted(_readme_table_tiers())
    assert [(f, a) for f, a, *_ in readme] == app_collapsed, (
        "README.md's tier table does not line up with RAMBucketedDefault.tiers:\n"
        f"  README: {[(f, a) for f, a, *_ in readme]}\n"
        f"  app:    {app_collapsed}"
    )


def test_readme_prose_matches_the_readme_table():
    """The README states the map twice; both have to say the same thing."""
    prose = sorted(_readme_prose_tiers())
    table = sorted((f, a) for f, a, *_ in _readme_table_tiers())
    assert prose == table, (
        "the README's quick-start sentence and its tier table disagree:\n"
        f"  prose: {prose}\n"
        f"  table: {table}"
    )


def test_readme_one_shot_commands_carry_the_exact_flags():
    """A README reader pastes the One-shot command verbatim. It must be the
    command the app runs — same flags, same order, nothing missing."""
    app = {alias: flags for _, alias, flags in _parse_app_tiers()}
    for floor, alias, _rss, oneshot_flags in _readme_table_tiers():
        assert alias in app, f"README recommends {alias}, which is not an app pick"
        assert oneshot_flags == app[alias], (
            f"README's one-shot command for {alias} has {oneshot_flags}, "
            f"the app launches it with {app[alias]}"
        )


def test_readme_peak_rss_matches_the_ssot():
    """The Peak RSS column is the number a reader checks against their RAM
    before pasting the one-shot command. It sat outside this gate and drifted
    on 2 of 5 rows (3.2 vs SSOT 3.0, 5.8 vs SSOT 6.0) while the README right
    above the table promised "a CI test parses both files and fails if they
    drift apart" — so each cell is now pinned to the SSOT's ``footprint_gb``,
    rendered the way the README renders it (one decimal place)."""
    ssot: dict[str, float] = {}
    for tier in json.loads(RECOMMENDATIONS.read_text())["tiers"]:
        for pick in tier["picks"]:
            ssot[pick["alias"]] = pick["footprint_gb"]
    rows = _readme_table_tiers()
    assert rows, "no tier rows parsed from README.md"
    for floor, alias, rss, _flags in rows:
        assert alias in ssot, f"README recommends {alias}, which is not an app pick"
        expected = f"{ssot[alias]:.1f}"
        assert rss == expected, (
            f"README's {floor} GB row lists Peak RSS {rss} GB for {alias}, "
            f"the SSOT footprint_gb is {expected} GB"
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
    are directly comparable.

    ``onboardingOwed`` joined the list with #1589. It is now the predicate
    that actually decides "is this a new user" — ``isEligible`` is a thin
    wrapper over it, and the launch auto-start path consults it directly —
    so leaving it unpinned would let the load-bearing half drift while the
    wrapper above it stayed identical."""
    for signature in (
        "func onboardingOwed(",
        "func isEligible(",
        "func isStranded(",
    ):
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


def test_launch_auto_start_defers_to_first_run_surfaces():
    """#1589: the launch auto-start must not outrace the first-run surfaces.

    The Swift suite covers ``AutoStartDecision.decide`` thoroughly, but it
    calls the function directly. Both new gates default to ``false``, so
    deleting either argument from the ``ContentView`` call site silently
    restores the original bug with every unit test still green — the
    parameter is simply never supplied and the gate never fires. The wiring
    is the part that broke; it is the part that has to be pinned.

    Source structure is the available lever: ``ContentView`` is a SwiftUI
    view whose launch hook cannot be invoked from a test, and it is not
    reachable from Python at all."""
    caller = (REPO / "apps/rapid-mac/Sources/Rapid/UI/ContentView.swift").read_text()

    assert "firstRunDecisionPending: telemetryConsentPending" in caller, (
        "the launch hook stopped passing the telemetry-consent gate. The "
        "parameter defaults to 'not pending', so a model would once again "
        "load behind the modal first-run consent sheet"
    )
    assert "onboardingPending: QuickstartCoordinator.onboardingOwed(" in caller, (
        "the launch hook stopped asking whether onboarding is still owed. "
        "The parameter defaults to 'not pending', which is exactly the "
        "pre-#1589 behaviour: auto-start invents a first model, serverState "
        "leaves .idle, and the Quickstart wizard becomes unreachable"
    )
    # The gate is only worth anything if it consults the SAME predicate the
    # wizard presents on. A hand-rolled re-derivation here is how the two
    # halves drifted apart in the first place.
    assert "QuickstartCoordinator.onboardingOwed(" in caller, (
        "the launch hook must call QuickstartCoordinator.onboardingOwed "
        "rather than re-deriving 'is this a new user' locally"
    )

    # Deferring for the consent sheet is only correct if the hook runs again
    # once the answer lands. A bare `.task` fires once, which would strand a
    # returning user on an idle server for the whole session.
    assert (
        ".task(id: telemetryConsentPending) { await runLaunchAutoStart() }" in caller
    ), (
        "the launch task is no longer keyed on the consent decision. With a "
        "bare `.task` the auto-start that stood down for the consent sheet "
        "never gets its turn, and a returning user's model never loads"
    )

    # And the gates must sit above the serverState switch: below it they can
    # only observe the race, never prevent it.
    decision = (
        REPO / "apps/rapid-mac/Sources/Rapid/Server/AutoStartDecision.swift"
    ).read_text()
    for gate in ("firstRunDecisionPending {", "onboardingPending {"):
        assert gate in decision, f"AutoStartDecision lost its {gate} gate"
        assert decision.index(gate) < decision.index("switch serverState {"), (
            f"AutoStartDecision: the {gate} gate moved below the serverState "
            "switch. Auto-start is what MOVES serverState, so a gate placed "
            "after it can only observe the damage, never prevent it"
        )
