# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx start <agent>`` — one-command agent startup (#150).

Ties the pieces that already exist — agent profiles
(``vllm_mlx/agents/profiles/``), safe setup plans
(``vllm_mlx/agents/setup.py``), memory-fit model recommendations
(``vllm_mlx/recommendations.py``), and the canonical ``serve`` path — into a
single verb:

1. resolve the agent profile (or a generic OpenAI-compatible default);
2. pick a model: explicit ``--model``, else the first recommended alias
   that fits verified memory and is already cached, else a previewed download;
3. (unless ``--dry-run``) start ``serve`` for that model as a FOREGROUND child
   that this process owns — it forwards SIGINT/SIGTERM and leaves no orphan;
4. after the endpoint is ready, print the agent's setup instructions and
   optionally apply its config via the existing setup-plan machinery.

The public verb is ``start``, not the issue's literal ``run``: ``run`` is
already a shipped alias for ``chat`` (Ollama muscle-memory), so repurposing
it would break an established command. See the #150 plan.

This package is intentionally distinct from ``vllm_mlx/service`` (the engine's
helper/post-process layer) and ``vllm_mlx/headless_service`` (LaunchDaemon
lifecycle). It reuses both setups; it does not reimplement a config writer, a
model selector, or a server entrypoint.
"""
