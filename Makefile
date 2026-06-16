.PHONY: help smoke check full benchmark update-baselines lint audit test stress soak release-smoke release-check-m3 clean

# Pick the interpreter:
#   1. Active venv ($VIRTUAL_ENV/bin/python) — wins so contributors using
#      a 3.10/3.11/3.13 venv get their venv's python regardless of PATH.
#   2. Versioned binaries that actually run a >=3.10 interpreter — we
#      must run --version because pyenv shims appear on PATH for *every*
#      version even if only one is installed, and macOS's bare 'python'
#      is often system 3.9 (below requires-python).
#   3. python3 last-resort fallback (lets the user see a clean error if
#      nothing on the system meets the version requirement).
# Override explicitly with: make smoke PY=python3.13
PY ?= $(shell \
  if [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python" ]; then \
    echo "$$VIRTUAL_ENV/bin/python"; exit 0; \
  fi; \
  for cand in python3.13 python3.12 python3.11 python3.10 python3; do \
    path=$$(command -v $$cand 2>/dev/null) || continue; \
    "$$path" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
      2>/dev/null && echo "$$path" && exit 0; \
  done; \
  echo python3)
HF_HUB_CACHE ?= $(shell echo $$HF_HUB_CACHE)
DOCTOR := $(PY) -m vllm_mlx.cli doctor

DEV_TEST := $(PY) scripts/dev_test.py

help:
	@echo "Rapid-MLX developer targets:"
	@echo ""
	@echo "  Dev testing (scripts/dev_test.py):"
	@echo "    make lint               ruff lint (~10s)"
	@echo "    make audit              CLI ↔ Config fidelity audit (~1s)"
	@echo "    make test               pytest unit suite (~30s)"
	@echo "    make smoke              lint + audit + unit (~1 min)"
	@echo "    make stress             8-scenario stress test (needs server)"
	@echo "    make soak               10-min agent soak test (needs server)"
	@echo ""
	@echo "  Doctor (regression harness — see harness/README.md):"
	@echo "    make check              ~10 min, qwen3.5-4b (auto starts server)"
	@echo "    make full               ~1-2 hr, 3 models + 12 agents"
	@echo "    make benchmark          overnight, all local models"
	@echo "    make update-baselines TIER=check  re-record baseline"
	@echo ""
	@echo "  Release (see docs/development/releasing.md):"
	@echo "    make release-smoke      clean-room install+import gate (~30s)"
	@echo ""
	@echo "  Env: HF_HUB_CACHE=$(HF_HUB_CACHE)"

# ---------- dev testing (scripts/dev_test.py) ----------
lint:
	$(DEV_TEST) lint

audit:
	$(DEV_TEST) audit

test:
	$(DEV_TEST) unit

smoke:
	$(DEV_TEST) smoke

stress:
	$(DEV_TEST) stress

soak:
	$(DEV_TEST) soak

# ---------- doctor tiers (regression harness) ----------
check:
	$(DOCTOR) check

full:
	$(DOCTOR) full

benchmark:
	$(DOCTOR) benchmark

update-baselines:
	@if [ -z "$(TIER)" ]; then \
		echo "error: TIER is required. Example: make update-baselines TIER=check"; \
		exit 2; \
	fi
	$(DOCTOR) $(TIER) --update-baselines

# ---------- release gate ----------
release-smoke:
	$(PY) scripts/release_smoke.py

# Full M3-only release gauntlet — runs every gate that needs a live
# `rapid-mlx serve` (G5/G6/G7/G8 end-to-end perf/G9). The CI-side gates
# (G1/G3/G10/G11/PF-1) run automatically on the bump PR via
# .github/workflows/release-preflight.yml; pr_validate runs on every PR
# via .github/workflows/pr-validate.yml. This target covers what CI
# cannot — every gate that boots a real server.
#
# Time budget: ~10-15 minutes on M3 Ultra with weights warm-cached.
# Cost: zero (your own machine + your own electricity).
#
# Override the test model: MODEL=qwen3.6-27b-4bit make release-check-m3
MODEL ?= qwen3.5-4b
release-check-m3:
	@MODEL=$(MODEL) PY=$(PY) bash scripts/release_check_m3.sh

clean:
	rm -rf harness/runs/*
	@echo "Cleared harness/runs/ — baselines kept."
