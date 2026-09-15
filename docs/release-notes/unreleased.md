<!-- Scratch space for the next release's notes. Append as you land work; in the
     version-bump PR, `git mv` this to vX.Y.Z.md and recreate this file empty.
     Whole-line HTML comments like this one are stripped before publishing.
     See README.md in this directory for what good notes look like. -->

## Video generation

- LTX-2.5 now has a default-off, fail-closed experimental
  `generation_mode=fast` product path for operator-configured model packages.
  Capability discovery
  reports availability, qualification revision, experimental status, and the
  package's evaluation schedule. Standard mode remains the default and an
  unavailable explicit fast request returns 409 instead of silently falling
  back. The quality-first six-plus-one profile measured 301.40 seconds versus
  539.94 seconds standard on an M4 Pro 48 GB (1.791x), and 101.14 seconds versus
  175.65 seconds on an M3 Ultra (1.737x). The candidate MP4 digest matched
  across machines. The prior five-plus-one profile is rejected because decoded
  stress cases exposed texture and motion-smear artifacts. No fast artifact is
  release-qualified or bundled, standard remains
  the default, and capability appears only when an operator explicitly
  configures a valid package. The initial fast surface is text-to-video only;
  image conditioning keeps the standard path until separately qualified.
