<!-- Scratch space for the next release's notes. Append as you land work; in the
     version-bump PR, `git mv` this to vX.Y.Z.md and recreate this file empty.
     Whole-line HTML comments like this one are stripped before publishing.
     See README.md in this directory for what good notes look like. -->

- Fixed DeepSeek-R1 tool-result replays returning HTTP 500 when its official
  chat template expected JSON-string arguments, and removed native tool-wire
  residue from forced-call content channels. The 4B release lane now validates
  a deterministic forced call, streaming channel hygiene, and stream/non-stream
  tool-result replay without treating small-model knowledge errors as engine
  release failures (#1676, #1677).
