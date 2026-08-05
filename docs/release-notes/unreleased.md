<!-- Scratch space for the next release's notes. Append as you land work; in the
     version-bump PR, `git mv` this to vX.Y.Z.md and recreate this file empty.
     Whole-line HTML comments like this one are stripped before publishing.
     See README.md in this directory for what good notes look like. -->

- Removed the `ministral-3b-4bit` public alias. Its default multimodal route
  accepts the model and reports ready, but the first text completion hangs in
  the MLLM scheduler. Use another supported Mistral-family checkpoint; the raw
  Hugging Face path remains available for developers investigating the upstream
  compatibility issue tracked in #1367.
