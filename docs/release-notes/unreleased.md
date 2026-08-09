<!-- Scratch space for the next release's notes. Append as you land work; in the
     version-bump PR, `git mv` this to vX.Y.Z.md and recreate this file empty.
     Whole-line HTML comments like this one are stripped before publishing.
     See README.md in this directory for what good notes look like. -->

## Highlights

**The desktop app generates images.** A new Images tab renders locally through
the same one-model-per-process server the chat tab uses: pick an image model,
load it through the usual readiness gate, prompt, and refine. Renders land in a
filmstrip you can step back through, and selecting an older one restores the
prompt that produced it (#1705). Chat gained the other direction at the same
time — you can attach images to a message and ask about them (#1723).

**The Qwen tool-call parser handles awkward arguments correctly.** A series
of fixes to `qwen3_coder_xml` (#1730) addresses legacy raw string arguments
whose own content contains XML-like closing tags — the case where the parser
cannot tell an argument's text from the wrapper around it.
Different fixes in the series address different symptoms: some produced wrong
arguments, others leaked wrapper framing into the answer or dropped the text
that followed a call. `AutoToolParser`'s balanced-JSON scan was fixed alongside
them (#1726), and a replayed terminal chunk under `tool_choice: auto` no longer
duplicates content into the answer (#1711).

**Reasoning-plus-tools turns get a much larger default token budget.** The
desktop's floor for those turns was set to exactly the default budget, so the
`max()` meant to lift it never lifted anyone; it is now 16384 (#1722). This
applies to turns with a reasoning model and tools enabled, and only while Max
Tokens is still at its default. A non-default setting is respected; 4096 is
read as "untouched" even if you picked it deliberately. 16384 is a ceiling
too, so this makes a truncated answer much less likely rather than
impossible. Worth knowing because a short budget does not fail
loudly: it returns a cut-off answer, which reads as a model that "could not do
it".

**The first follow-up message no longer re-reads the opening context.** The
opening turn never saved a reusable cache boundary, so the second message paid
to re-read everything; from the third message on, reuse already worked. #1732
closes that one gap. Measured on `qwen3.6-27b-4bit` with a ~9.9K-token
document: the opening turn prefills 9922 tokens (32.4 s to first token), and
the follow-up prefills **34** instead of 9941 — 1.45 s, about 22x faster. The
longer the opening context, the more that first follow-up saves.

**The scheduler reclaims paged full-KV and free-block memory** instead of
wedging on a `D-METAL-CAP` 503 under sustained load (#1646).

**Claude Code has an agent profile** (#1720), and browsing approvals can be set
to always-allow rather than prompting every time (#1695).

## Fixes worth calling out

- The model picker's "Browse all models" now opens the catalogue instead of
  closing the wizard (#1662).
- A stored last-served alias is validated before the app tries to restore it,
  so a stale or removed model no longer produces a failed start on launch
  (#1729).
- `SIGHUP` termination semantics are preserved (#1703), and desktop tags no
  longer leak into the engine's update check (#1704).
- Fixed DeepSeek-R1 tool-result replays returning HTTP 500 when its official
  chat template expected JSON-string arguments, and removed native tool-wire
  residue from forced-call content channels. The 4B release lane now validates
  a deterministic forced call, streaming channel hygiene, and stream/non-stream
  tool-result replay without treating small-model knowledge errors as engine
  release failures (#1676, #1677).
- Multi-turn tool history now stays in each model family's trained wire format
  for Qwen, Gemma 4, MiniMax, Nemotron and xLAM instead of being rewritten into
  the generic `[Calling tool: ...]` transcript (#1593).

## Release engineering

Mostly invisible, but it is why the above is trustworthy: the app and engine
are now cut in one event instead of two that could drift (#1649); a release
gate that had not run for eleven releases was found dead and repaired (#1671);
a Codex review that is actually invoked now fails closed on backend, auth,
timeout or execution failure rather than passing silently (#1700) — a missing
Codex binary is still reported as a skip; and four AX-only GUI golden flows
run on every desktop PR — `chat-restore`, `slow-stream-stop` and
`model-crash-recovery` (#1721), plus `image-generation` (#1731) — driving the
app through the accessibility API with no screen recording (#1708) so they
work unattended in CI. Three more flows (`restored-tools`, `tool-loop-budget`,
`chat-depth`) are also Peekaboo-free but stay local-only; the rest still need
Peekaboo.
