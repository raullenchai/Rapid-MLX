# SPDX-License-Identifier: Apache-2.0
"""Installer first-chat and Desktop recommendation source contracts.

The model browser answers "what is the smartest model this Mac can run";
the installer and Desktop Quickstart answer "what gets a new user to a good
first chat quickly". Those are deliberately different policies. This suite
keeps the browser table aligned with its JSON SSOT while executing the
installer's library-mode selector to pin the Quickstart baseline and the
RAM ceiling on cached promotions.

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


def _select_installer_starter(ram_gb: int, cached: tuple[str, ...] = ()) -> str:
    """Execute only install.sh's library-mode pure starter selector."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            'RAPID_INSTALL_LIB=1 source "$1"; select_starter_model "$2" "$3"',
            "selector",
            str(INSTALL_SH),
            str(ram_gb),
            "\n".join(cached),
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _installer_cached_order(ram_gb: int) -> list[str]:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'RAPID_INSTALL_LIB=1 source "$1"; starter_cached_order_for_ram "$2"',
            "order",
            str(INSTALL_SH),
            str(ram_gb),
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    return result.stdout.splitlines()


def _parse_app_tiers() -> list[tuple[int, str, list[str]]]:
    """``[(floor_gb, primary_alias, flags)]`` from the shared SSOT."""
    payload = json.loads(RECOMMENDATIONS.read_text())
    return [
        (tier["minimum_memory_mib"] // 1024, tier["picks"][0]["alias"], [])
        for tier in payload["tiers"]
    ]


def test_both_tables_parse():
    """Both the first-chat selector and model-browser SSOT are reachable."""
    assert _select_installer_starter(8) == "lfm2.5-1b-4bit"
    assert _select_installer_starter(16) == "qwen3.5-4b-4bit"
    assert len(_parse_app_tiers()) >= 6


def test_fresh_installer_uses_quickstart_baseline_at_every_ram_size():
    """First chat optimizes time-to-value, not the browser's largest pick."""
    # Desktop's sub-16 GB baseline is the safe 1.2B lowMemoryChoice (#2432);
    # at 16 GB and up it is the standard 4B defaultChoice.
    sub16_alias, _ = _parse_quickstart_choice("lowMemoryChoice")
    standard_alias, _ = _parse_quickstart_choice("defaultChoice")
    assert (
        "physicalRAMGB < 16 ? lowMemoryChoice : defaultChoice" in QUICKSTART.read_text()
    )
    for ram in (4, 8, 15):
        assert _select_installer_starter(ram) == sub16_alias
    for ram in (16, 18, 24, 32, 48, 64, 96, 256):
        assert _select_installer_starter(ram) == standard_alias


def test_cached_choice_is_preferred_only_when_it_fits_the_ram_tier():
    assert _select_installer_starter(32, ("qwen3.8-27b-4bit",)) == "qwen3.8-27b-4bit"
    assert _select_installer_starter(16, ("qwen3.8-27b-4bit",)) == "qwen3.5-4b-4bit"


def test_the_banner_prints_a_bare_command_where_no_flags_are_needed():
    text = INSTALL_SH.read_text()
    serve_line = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("echo ") and "rapid-mlx serve" in line
    )
    assert "${RECOMMENDED_MODEL}${RECOMMENDED_FLAGS}" in serve_line


def test_every_recommended_alias_exists():
    """A typo'd alias turns the install banner into a 404 at first run."""
    from vllm_mlx.model_aliases import list_aliases

    known = list_aliases()
    installer_aliases = set()
    for ram in (8, 16, 18, 24, 32, 48, 64, 96):
        installer_aliases.add(_select_installer_starter(ram))
        for cached in _installer_cached_order(ram):
            selected = _select_installer_starter(ram, (cached,))
            installer_aliases.add(selected)
            assert cached in known, (
                f"install.sh cached order for {ram} GB contains unknown alias {cached!r}"
            )
    for alias in installer_aliases:
        assert alias in known, f"install.sh recommends unknown alias {alias!r}"
    for _, alias, _ in _parse_app_tiers():
        assert alias in known, f"app recommends unknown alias {alias!r}"


@pytest.mark.parametrize(
    "ram,expected",
    [
        (8, "lfm2.5-1b-4bit"),
        (16, "qwen3.5-4b-4bit"),
        (18, "qwen3.5-4b-4bit"),
    ],
)
def test_small_macs_get_something_that_fits(ram, expected):
    """Pinned literally: these laptop tiers are the ones that changed, and a
    regression here is the difference between an 8 GB Mac running a model
    and being told nothing fits."""
    assert _select_installer_starter(ram) == expected


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
    """The quick-start prose pins the two first-chat baselines."""
    text = README.read_text()
    assert "prefers a runnable model already cached" in text
    assert "`lfm2.5-1b-4bit` below 16 GB" in text
    assert "`qwen3.5-4b-4bit` at 16 GB or above" in text
    return [(8, "lfm2.5-1b-4bit"), (16, "qwen3.5-4b-4bit")]


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
    """Installer starters are deliberately smaller than browser smart picks."""
    assert _readme_prose_tiers() == [
        (8, "lfm2.5-1b-4bit"),
        (16, "qwen3.5-4b-4bit"),
    ]


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
            ssot[pick["alias"]] = round(pick["footprint_mib"] / 1024, 1)
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


def _parse_quickstart_choice(name: str) -> tuple[str, str]:
    """``(alias, hfRepo)`` from one authored Quickstart choice."""
    text = QUICKSTART.read_text()
    block = re.search(
        rf"static let {name} = QuickstartModelChoice\((.*?)\n    \)",
        text,
        re.DOTALL,
    )
    assert block, f"{name} literal not found in QuickstartView.swift"
    body = block.group(1)
    alias = re.search(r'alias:\s*"([^"]+)"', body)
    repo = re.search(r'hfRepo:\s*"([^"]+)"', body)
    assert alias, f"{name} has no alias literal"
    assert repo, f"{name} must pin hfRepo — it drives the byte-progress monitor"
    return alias.group(1), repo.group(1)


def _parse_quickstart_default() -> tuple[str, str]:
    return _parse_quickstart_choice("defaultChoice")


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


def test_bundled_model_tracks_the_explicit_low_memory_choice():
    """Bundled weights remain a runnable offline/low-memory escape hatch.

    They are deliberately no longer the automatic starter: first run chooses
    the hardware-fit 2.6B/4B baseline, while the existing 1.2B bundle remains
    available without stranding airgapped or memory-constrained users.
    """
    q_alias, q_repo = _parse_quickstart_default()
    low_alias, low_repo = _parse_quickstart_choice("lowMemoryChoice")
    b_alias, b_repo = _parse_bundled()
    assert (b_alias, b_repo) == (low_alias, low_repo), (
        f"bundled model {(b_alias, b_repo)!r} != low-memory choice "
        f"{(low_alias, low_repo)!r}"
    )
    assert (b_alias, b_repo) != (q_alias, q_repo), (
        "the 1.2B bundle must not silently become the automatic starter again"
    )


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
    """#1589: auto-start defers to onboarding, never telemetry consent.

    The Swift suite covers ``AutoStartDecision.decide`` thoroughly, but it
    calls the function directly. The onboarding gate defaults to ``false``, so
    deleting the argument from the ``ContentView`` call site silently
    restores the original bug with every unit test still green — the
    parameter is simply never supplied and the gate never fires. The wiring
    is the part that broke; it is the part that has to be pinned. Telemetry
    consent is now deliberately post-value and must not gate launch work.

    Source structure is the available lever: ``ContentView`` is a SwiftUI
    view whose launch hook cannot be invoked from a test, and it is not
    reachable from Python at all."""
    caller = (REPO / "apps/rapid-mac/Sources/Rapid/UI/ContentView.swift").read_text()

    assert "telemetryConsentPending" not in caller, (
        "telemetry consent must not gate catalog refresh, onboarding, or "
        "launch auto-start; it is offered only after delivered product value"
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

    assert ".task { await restorePersistedSession() }" in caller, (
        "session restore should run once at launch without waiting for a "
        "telemetry choice"
    )
    restore = caller.split("private func restorePersistedSession(", 1)[1].split(
        "private func runLaunchAutoStart(", 1
    )[0]
    assert "runLaunchAutoStart(" in restore, (
        "session restore no longer reaches launch auto-start"
    )

    # And the gates must sit above the serverState switch: below it they can
    # only observe the race, never prevent it.
    decision = (
        REPO / "apps/rapid-mac/Sources/Rapid/Server/AutoStartDecision.swift"
    ).read_text()
    for gate in ("onboardingPending {",):
        assert gate in decision, f"AutoStartDecision lost its {gate} gate"
        assert decision.index(gate) < decision.index("switch serverState {"), (
            f"AutoStartDecision: the {gate} gate moved below the serverState "
            "switch. Auto-start is what MOVES serverState, so a gate placed "
            "after it can only observe the damage, never prevent it"
        )
