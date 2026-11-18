# Research Loop visual system

This Operate surface uses a scientific figure sheet: open white plotting space,
a quiet blue-grey discussion column, and one evidence-led reading path. The
parameter change, accuracy curve, and verdict lead; controls and ancestry support them.

## Colour

| Role | Value | Use |
| --- | --- | --- |
| Canvas | `#eef2f5` | Page and map |
| Surface | `#ffffff` | Experiments |
| Discussion | `#e7edf2` | Ideas and composer |
| Ink | `#172b3b` | Primary text and actions |
| Muted | `#526476` | Secondary text and baseline |
| Selection | `#d8e5f0` | Selected hypothesis |
| Supported | `#176345` | Supported verdict, curve, map node |
| Refuted | `#a33432` | Refuted/crashed verdict, curve, map node |
| No effect | `#795919` | No-effect verdict, curve, map node |
| Testing | `#285e89` | Measuring/testing, focus, selected map ring |
| Rule | `#ccd6de` | Structural separators |

State colours repeat between hypothesis outcomes, the selected test curve and
its legend, the verdict, and map nodes. State words accompany colour. The train
curve is slate (`#66798b`); the previous-best test curve is dashed (`#697684`).
Map node numbers identify rounds; a dark outer ring marks the best result and
a blue inner ring marks the selected result.

Text colours pass 4.5:1 against white, canvas, discussion, selection, and the diff
background. The lowest tested pairing is muted text on selection (4.76:1).

## Typography

Public Sans upright variable, weight 100–900, SIL OFL 1.1. See
[font source and delivery](fonts/README.md). All UI uses tabular numerals.
Code diffs use local monospace fonts; change titles and all metrics retain tabular numerals.

| Role | Desktop size | Weight |
| --- | --- | --- |
| Numeric verdict | 40px | 650 |
| Change title | 26px | 600 |
| Hypothesis | 17px | 400 |
| Task title | 22px (20px at 1440) | 650 |
| Idea row | 15px (14px at 1440) | 600 |
| Section heading | 18px | 650 |
| Body and composer | 15–16px | 400 |
| Supporting content | 12–14px | 400–600 |

The change title is a single line with ellipsis and full text on hover. The
hypothesis is clamped to two lines, with full text on hover and click to expand.
One compact 52px row per round shows its number, change, and verdict/value pill.
Clicking selects the evidence and toggles the full hypothesis, reason, and author.

## Structure and interaction

At both 1920×1080 and 1440×900, the desktop shell uses a 120px header and step
strip, followed by a fixed grid filling the remaining viewport. The task title
and run controls share one row. Main has 8px top / 12px bottom padding, an 8px
row gap, a flexible workspace, and a reserved 170px full-width ancestry strip
including run controls. The workspace is 30:70 ideas/evidence. Every flexible
track uses a zero minimum; long content cannot set the document height.

The ideas list and Discussion tab scroll independently inside the column. Chat
history occupies its own tab rather than displacing rounds. The composer is a
60px bottom region at rest and expands on focus to expose text and actions.
Collapsed rows budget for 11 visible at 1080px height and 8 at 900px height;
expanding a row or composer reduces that count and the list scrolls internally.

The evidence header shows the change, brief hypothesis, and seed statuses. The
chart uses the flexible middle track, with a large verdict underneath. The
figure is budgeted at 588px / 408px (including legend and verdict) and its SVG
at 500px / 320px for the two requested sizes. Axis labels redraw at actual
plot dimensions. The 170px map and supporting information constrain the plot
at the shorter viewport; its visibility takes precedence over a literal 55vh
plot height.

The diff starts collapsed as a file/count summary. Its full scrollable content,
full hypothesis, and error details expand over the plot without resizing it.
Switching rounds closes evidence disclosures; streaming updates preserve their
state. Map nodes and parent edges remain in the strip; longer runs scroll
horizontally. Below 1100px the workspace stacks and document scrolling resumes.

Controls have 5px corners, restrained state transitions, and visible 3px blue
focus outlines. History entries are flat selectable rows, with no nested card
shells. Round buttons and map nodes support Enter and Space; tabs support arrow keys, Home, and End. Follow
latest restores event following after inspection. Reduced motion disables
transitions. Empty, measuring, testing, finished, replay, error, and fix states
use the same component vocabulary.

## Verification boundary

Automated state checks cover both supplied real event streams, long content,
progressive disclosure, tab navigation, and diff summaries. Round 2 uses sizing
reasoning at 1920×1080 and 1440×900 as requested; no screenshots were captured.
Font rendering, focus appearance, actual fit, and video legibility await the
user’s rendered review. Source-level sizing is not screenshot evidence.
