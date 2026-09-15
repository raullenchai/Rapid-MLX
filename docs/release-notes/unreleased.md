<!-- Scratch space for the next release's notes. Append as you land work; in the
     version-bump PR, `git mv` this to vX.Y.Z.md and recreate this file empty.
     Whole-line HTML comments like this one are stripped before publishing.
     See README.md in this directory for what good notes look like. -->

## Video generation

- LTX-2.5 now has a default-off, fail-closed experimental
  `generation_mode=fast` product path for operator-configured model packages.
  Capability discovery
  reports availability, qualification revision, experimental status, and the
  five-plus-one evaluation schedule. Standard mode remains the default and an
  unavailable explicit fast request returns 409 instead of silently falling
  back. Four 10-second 768x512 qualification runs on an M4 Pro 48 GB measured
  268.97-269.76 seconds versus a 539.94-second standard reference—approximately
  2x—with zero swap. A Mac Studio M3 Ultra product-path pair measured
  175.65 seconds standard versus 90.71 seconds fast (1.936x), confirming the
  speed path is not tied to one Mac generation. That new prompt also exposed
  visible candidate artifacts, so no fast artifact is release-qualified or
  bundled, standard remains the default, and capability appears only when an
  operator explicitly configures a valid package. The initial fast surface is
  text-to-video only; image conditioning keeps the standard path until
  separately qualified.
