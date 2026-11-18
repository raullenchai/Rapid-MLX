# Research Loop visual system

This Operate surface keeps the scientific figure sheet: Public Sans, blue-grey
working surfaces, dark blue ink, and white plotting space. Projects, Discuss,
Experiments, and Results share the same quiet controls and type hierarchy.

| Role | Value |
| --- | --- |
| Canvas | `#eef2f5` |
| Surface | `#ffffff` |
| Discussion | `#e7edf2` |
| Ink | `#172b3b` |
| Muted | `#526476` |
| Selection | `#d8e5f0` |
| Supported / improved / running best | `#176345` |
| Refuted / worse / crashed | `#a33432` |
| No effect | `#795919` |
| Testing / focus | `#285e89` |
| Rule | `#ccd6de` |

Progress dots use stable hypothesis colours from a muted ink palette. The
running-best step remains green; a green outline/larger dot marks improvements,
a red outline marks worse measurements, and a neutral outline marks no effect.
Red crosses mean crashed. Unmeasured crashes sit at the bottom edge with an
explicit tooltip; their position is not a metric. Hover and accessible names say
“h3 · claim · experiment 2 of 2” plus the concrete change, value, and verdict.
Selection adds a blue ring. Every colour has a textual equivalent. The selected
experiment’s first configured curve follows its experiment verdict colour; other
series use neutral ink, and the primary parent comparison is dashed.

Public Sans is self-hosted from `fonts/PublicSans.ttf`, SIL OFL 1.1; see
[font provenance](fonts/README.md). All numerals are tabular. Code uses local
monospace. Main titles are 22px, plot title 23px, section titles 19px, change title
21px (19px at 1440), verdict 24px (23px at 1440), body 15px, and supporting
content 12–14px. Controls have 6px corners and 3px blue focus outlines.

## Structure

Projects remains a flat list with title, first research-goal content line,
domain chip, template, Running / Finished / No runs status on a separate line,
hypothesis count and completed experiment count across all runs (including an active run), best metric with
unit and direction arrow, and relative/short last activity. A compact step
sparkline tracks cumulative kept best across runs in creation order, including
each baseline and result event. Rejected or inconclusive changes hold the line;
lower-is-better metrics retain their natural downward orientation. No measured
baseline means no fabricated line. The native title link spans the row. New
project opens an inline panel with native radio template cards showing metric,
unit, direction, and editable file. An empty project list has a create action.
Domain filter chips include All domains and preserve descending last-activity
order. Missing domains remain visible under All domains. Small summary fields
come from the projects API; older or active server summaries use the existing
project/run detail APIs as a read-only fallback so completed rounds appear
before a run ends.

The project shell has a 64px app bar, 52px research strip, and 64px stage/run bar.
The research strip places step names over quiet “any tool” labels and pairs
Hypotheses / Experiments in one clearly active segment. Discuss, Experiments, and
Results remain the main navigation, with hover, current, and keyboard states. URLs retain run selection and replay state, while explicit ledger
selection adds the round number.

Discuss has three columns: a rendered markdown-lite goal with Edit/Preview and
Save, persistent discussion, and Hypotheses plus an experiments field and Start
experiments. Goal editing is disabled during a run
to keep its instructions fixed. The single discussion component moves into the
experiment rail, preserving its draft and model selector between stages. Stage
navigation refreshes saved messages and scrolls after mounting the visible panel.
An empty discussion starts at scroll position zero, including in the shorter
experiment rail. The empty discussion offers three generic, metric-aware starter prompts; choosing
one fills and focuses the composer without sending. User messages have a pale
surface; model messages show the local model name, a streaming caret while
pending, and a secondary Add as hypothesis action after completion. Hypotheses use separators, claim text,
status pills, experiment counts, and a count badge. The Add hypothesis form accepts
a claim. The empty state explains that a claim gets one or more experiments. A
quiet line explains the deterministic conclusion rule. The Start area names baseline,
experiment limit, actual configured seeds, and the fixed budget.
Default run selection prefers an active run, then the newest run with results by
creation time. Explicit run selections remain in the URL.

Experiments reserves a 330px hypothesis/discussion/control rail (295px at 1440) and gives
the remaining width to evidence. The main area divides 1.8:1 between the hero
progress figure and current round, with a 14px gap. Progress gets labelled kept
points, verdict-specific discarded markers, optional goal/reference lines, direction, and a
faint noise band around the running best (not a confidence interval). Percent
axes use 0/25/50/75/100% within [0, 1]; other units use 1–2–5 tick steps around
the baseline and kept range, including noise padding and any configured
goal/reference. Discarded values outside this range become small edge carets;
their accessible names and tooltips retain the actual values and explain clamping.
This remains a linear scale. Labels truncate to 28 characters with full SVG titles, move
above/below points, and turn inward at the right edge to stay inside the SVG.
Candidate label boxes alternate sides and avoid prior labels when space permits;
displaced labels have leader lines and a white text underpaint. It
plots recorded values at their natural numeric orientation: a minimum improves
downward. The step line advances only on improved verdicts. Dense runs scroll
horizontally within the figure; every point has a full tooltip and keyboard
inspection target. The rail groups hypotheses with nested selectable experiments, status, and counts.
The live heading identifies the hypothesis being tested. Discussion remains below
the internally scrolling groups. The lower area has current change, hypothesis, seeds, verdict,
a Results link, and a smaller curve. Curves use `task.curve.x`, `x_label`, and all configured series. Missing values
are skipped independently per series; fractional x values and extra metadata
do not affect aggregation. Series with `style: "points"` show dots, and an isolated
line-series sample also remains visible as a dot. Config labels form a wrapping
legend; the x label appears below the plot. Parent data participates in the extent. Round curves share nice 1–2–5 tick steps
with progress charts, reserve measured text width for signed y labels, and leave
42px at the right for rounded x labels. Rounded ticks do not change the recorded measurements.
Templates without live curves show per-seed final metrics instead of invented
training traces. Seed statuses are small inline chips; failed seeds say crashed.
Identical error summaries collapse across retries/seeds, with observed “on all seeds”
or “on seed 1” scope and one full traceback disclosure. Errors show the last exception type/message on one line with Show error opening
a bounded, scrollable monospace traceback. Both result-event and run-error
failures use this disclosure. Current change titles clamp to two lines, Results titles to three, with
hover text and explicit expansion; ledger changes clamp to two lines; code-like text uses monospace with a full hover title. A live
status names current round, configured round limit, and active seed. The latest
recorded point pulses during an active run; new baseline/result points arrive
over 200ms without restarting the animation on inspection. Current-round verdicts
include the signed delta. Each number and its unit stay together while the
verdict can break between comparisons, verdict, and delta. Curves use explicit colored line keys and a dashed
Parent key when comparison data exists.

Results shows a configured reference label/value and the first kept round within
its noise tolerance when the current best is within that tolerance. It also shows
best-code download and log links, a Hypotheses ledger (claim, status, best measured
result, experiment count), and a detail
column for the full hypothesis, why, prediction, curve, parent diff, and code.
Each hypothesis expands into experiments with change, value, Δ, experiment verdict,
and selection linking to code/diff. Cross-run experiments link to their own run.
The learned section groups supported, refuted, and inconclusive hypotheses
deterministically. Open claims remain untested/testing. The same quiet conclusion
rule appears above the ledger. The versioned Python reducer supplies statuses. Complete improvements support
a claim; conflicting complete results are inconclusive. At the experiment limit
or run end, worse evidence refutes, while only no-effect, crashed or incomplete
evidence is inconclusive. Failed edits count as crashes.
Legacy experiments dedupe by identical hypothesis text without altering artifacts. Missing artifacts
have explicit empty/error states. Code and diff start collapsed and scroll inside
their disclosures. Ledger headers stick within the scrolling table, verdicts
have semantic pills, delta colors follow metric direction, and the kept best row
has a pale surface. Code uses a local, handwritten Python
syntax tokenizer with escaped tokens, line numbers, and raw-source copy. Diff
lines have added/removed backgrounds, exact +added / −removed counts excluding
file headers, and a raw diff copy action. The filesystem destination stays
visible above the ledger.

## Desktop budgets (authored, not browser measurements)

At 1920×1080, the project shell has 880px of stage height after its app bar,
strip, stage bar, and bottom padding. At 1440×900 it has 704px. The plot figure is
about 557px / 444px high; the current round about 309px / 246px. The progress SVG
has a zero-minimum grid track; long runs scroll internally. The rail uses a
zero-minimum discussion track with an internally scrolling conversation. Goal,
queue, Results, and current-round copy have their own scroll regions. Document
scrolling is unnecessary on desktop Experiments.

Below 1100px, the layout stacks and document scrolling resumes. At narrow widths
the research strip drops auxiliary labels, Discuss stacks, the round curve moves
below its copy, and the ledger scrolls horizontally.

Focus is preserved when live progress and ledger markup rebuild. Stages are
native navigation links; point inspection supports Enter and Space. Selection,
caret, native scrollbar colours, placeholder contrast, loading, empty, error,
paused, finished, and replay states follow the shared palette. Point arrival has
a short state animation; reduced-motion disables it and all live pulses. Controls
use 180ms transitions. White panels share a 12px corner and one subtle, offset
shadow token; spacing follows a 4px base. Header and favicon share the authored
geometric loop mark.

## Verification boundary

Node tests validate generic/sparse curves and reference/outlier behavior,
domain filtering, active summary counts, compact errors, scroll reset, state, routing, metric directions, project summary fallback,
starter prompts, markdown/tokenizer escaping, diff counts, raw code copy, live
round/seed status, ledger selection, artifacts, replay controls, and available
local event streams. Python tests validate the small project-summary fields using
temporary fixtures, including cumulative kept best and completed-round counts
across finished and active runs. Available local domain streams cover all six
non-AI projects without writing their artifacts.
Text palette contrast was checked numerically (at least 4.5:1).

Desktop frame budgets above are authored CSS dimensions. Chrome aborted during
launch in this environment and the Chromium wrapper targets a missing executable;
rendered viewport fit, layout shift, font metrics, keyboard appearance, mobile,
and screen-reader behavior remain unverified. No application server or live run
was changed for verification.


## Protocol 1 artifact views

The selected experiment keeps the current copy/measurement split. Its measurement
column hosts Chart · Data · Code · Log · Notes tabs, only when present, using the
existing palette and controls. Tabs remain visible while their content scrolls;
arrow keys, Home and End navigate them. Live journal measurements remain visible
until immutable seed charts are finalized. Results includes the same browser,
run reports above the ledger, run-level attachments, and conclusion reports and
artifact links within expanded hypotheses. Public claim labels are H1, H2….
Hypothesis details have no top border/margin inside table cells. The Improved
legend marker has an explicit solid border and fill.

CSV/flat JSONL previews contain up to 500 sortable rows and show units/types.
Logs offer search and a last-200-lines toggle, with escaped monospace text.
Reports use the existing Markdown/KaTeX renderer; images use img. Chart roles
provide two-experiment comparison using declarative SVG marks, log scales, bands
and reference lines. Unknown or invalid envelopes become warned downloads.
Download all exports every revision, envelopes, records, journal and protocol.

Protocol interaction checks run in Node and a temporary DOM fixture. Chrome still
aborts at startup in this sandbox; the added artifact layouts have not received
browser screenshot or viewport verification. Existing screenshots, layout tokens,
responsive grid rules and craft guidance remain the visual reference.
