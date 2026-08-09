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

**Tool calls on Qwen models stopped losing their arguments.** A long series of
fixes to the Qwen streaming parser: arguments split across chunks, wrapper
closers that arrive in pieces, truncated calls, and content that follows a
wrapper close were each mis-framed in ways that produced a call with the wrong
arguments rather than a visible failure. `AutoToolParser`'s balanced-JSON scan
was fixed alongside them (#1726), and a replayed terminal chunk under
`tool_choice: auto` no longer duplicates content into the answer (#1711).

**Tool-using turns are no longer cut off mid-thought.** The desktop's floor for
reasoning-plus-tools turns was set to exactly the default token budget, so the
`max()` that was supposed to lift it never lifted anyone. It is now 16384
(#1722). Short budgets do not fail loudly — they deliver a truncated answer, so
this showed up as models that "could not do it" rather than as an error.

**A second turn now reuses the first turn's prefix cache** (#1732), and the
scheduler reclaims paged full-KV and free-block memory instead of wedging on a
`D-METAL-CAP` 503 under sustained load (#1646).

**Claude Code has an agent profile** (#1720), and browsing approvals can be set
to always-allow rather than prompting every time (#1695).

## Fixes worth calling out

- The model picker's "Browse all models" opened the catalogue instead of
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

## Release engineering

Mostly invisible, but it is why the above is trustworthy: the app and engine
are now cut in one event instead of two that could drift (#1649); a release
gate that had not run for eleven releases was found dead and repaired (#1671);
the Codex review step fails closed when the reviewer is unavailable rather than
passing silently (#1700); and the GUI golden flows run on every desktop PR
(#1721), driving the app through the accessibility API with no screen recording
(#1708) so they work unattended in CI.
