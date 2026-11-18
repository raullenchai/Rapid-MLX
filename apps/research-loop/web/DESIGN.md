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
| Supported / running best | `#176345` |
| Refuted / crashed | `#a33432` |
| No effect | `#795919` |
| Testing / focus | `#285e89` |
| Rule | `#ccd6de` |

Grey progress dots (`#b3bdc5`) mean no effect, light red (`#d9a19f`) means
refuted, and red crosses mean crashed. Unmeasured crashes sit at the bottom
edge with an explicit tooltip; their position is not a metric. Kept dots and the
step line are green. Selection adds a blue ring. Words, values, and accessible names accompany
colour. The selected round's curve follows its verdict colour; train measurements
use slate, and the parent comparison is dashed. SVG contains only real data.

Public Sans is self-hosted from `fonts/PublicSans.ttf`, SIL OFL 1.1; see
[font provenance](fonts/README.md). All numerals are tabular. Code uses local
monospace. Main titles are 22px, plot title 23px, section titles 19px, change title
21px (19px at 1440), verdict 26px (23px at 1440), body 15px, and supporting
content 12–14px. Controls have 4–5px corners and 3px blue focus outlines.

## Structure

Projects is a flat list with title, template, experiment count, best metric with
a direction arrow, and relative/short last activity. The native title link spans
the row. New project opens an inline form, with a template selector and
research-goal field. No promotional surface or fabricated sample results.

The project shell has a 64px app bar, 52px research strip, and 64px stage/run bar.
The five research steps stay small; Discuss, Experiments, and Results are the main
navigation. URLs retain run selection and replay state, while explicit ledger
selection adds the round number.

Discuss has three columns: a rendered markdown-lite goal with Edit/Preview and
Save, persistent discussion, and queued ideas plus a rounds field and Start
experiments. Goal editing is disabled during a run
to keep its instructions fixed. The single discussion component moves into the
experiment rail, preserving its draft and model selector between stages. Stage
navigation refreshes saved messages and scrolls after mounting the visible panel.
Default run selection prefers an active run, then the newest run with results by
creation time. Explicit run selections remain in the URL.

Experiments reserves a 330px discussion/control rail (295px at 1440) and gives
the remaining width to evidence. The main area divides 1.8:1 between the hero
progress figure and current round, with a 14px gap. Progress gets labelled kept
points, verdict-specific discarded markers, optional goal line, direction, and a
faint noise band around the running best (not a confidence interval). Percent
axes use 0/25/50/75/100% within [0, 1]; other units use 1–2–5 tick steps around
the observed range. Labels truncate to 28 characters with full SVG titles, move
above/below points, and turn inward at the right edge to stay inside the SVG. It
plots recorded values at their natural numeric orientation: a minimum improves
downward. The step line advances only on supported verdicts. Dense runs scroll
horizontally within the figure; every point has a full tooltip and keyboard
inspection target. The lower area has current change, hypothesis, seeds, verdict,
a Results link, and a smaller curve. Templates without live curves show per-seed
final metrics instead of invented training traces. Seed statuses are small inline
chips. Current change titles clamp to two lines, Results titles to three, with
hover text and explicit expansion; ledger changes clamp to two lines.

Results shows best-code download and log links, a selectable ledger, and a detail
column for the full hypothesis, why, prediction, curve, parent diff, and code.
The learned section groups event verdicts deterministically. Missing artifacts
have explicit empty/error states. Code and diff start collapsed and scroll inside
their disclosures. The filesystem destination stays visible above the ledger.

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
a short state animation; reduced-motion disables it.

## Verification boundary

Node tests validate state, routing, both metric directions, markdown escaping,
ledger selection, artifacts, replay controls, and both local real streams when
available. CSS sizing budgets have been reviewed at source level. Chrome and
Metal were unavailable for this task; actual desktop fit, zoom, touch/mobile,
font rendering, focus appearance, screen-reader behavior, and live model streaming
still need the user's browser and live review. No rendered approval is claimed.
