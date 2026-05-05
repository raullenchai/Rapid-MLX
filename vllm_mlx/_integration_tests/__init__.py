# SPDX-License-Identifier: Apache-2.0
"""Bundled integration test scripts.

These are NOT run via pytest — they're loaded by
``vllm_mlx.agents.testing._run_specific_tests`` via
``importlib.util.spec_from_file_location`` when the user runs
``rapid-mlx agents <name> --test``.

The actual sources live at ``tests/integrations/`` in the repo; the
files here are symlinks to that single source of truth, included in
the wheel via ``package-data`` so pip/brew installs can run them
without a repo clone. setuptools dereferences the symlinks at
build time, so the wheel ships the actual content.
"""
