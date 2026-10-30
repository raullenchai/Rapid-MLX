<!-- Draft for the next release. Rename in the version-bump PR. -->

## Breaking: Python 3.12 or newer required

Rapid-MLX now requires Python 3.12+. Python 3.10 reaches end of life in October 2026, the chat harness, image and video lanes already needed a newer interpreter, and current numerical dependencies require 3.12. The official installer and the Desktop app already use 3.12.

pip on Python 3.10 or 3.11 keeps resolving to 1.0.0. Install Python 3.12 (or newer) and reinstall to get newer releases. Homebrew users are unaffected.
