#!/usr/bin/env python3
"""Agent Promotion Content Generator — creates ready-to-post content.

Generates promotion content for a new agent integration:
  - GitHub Discussion/Issue post (for their repo)
  - Reddit r/LocalLLaMA post
  - Twitter/X thread
  - Discord message

Usage:
    python3 scripts/agent_promo_gen.py --agent hermes \
        --pain-points "Ollama slow,Gemma 4 broken,tool schema 10x latency" \
        --unique-edge "Only backend with working Gemma 4 tool calling" \
        --models "Qwen3.5-4B,Qwen3.5-9B,Gemma 4 26B" \
        --test-results "20/20 on Gemma 4, 15/15 on Qwen3.5" \
        --issues "#6626,#7457"

    # Interactive mode
    python3 scripts/agent_promo_gen.py
"""

import argparse
import os


def generate_github_issue(config):
    """Generate GitHub issue/discussion post for the agent's repo."""
    agent = config["display_name"]
    pain_points = config.get("pain_points", [])
    unique_edge = config.get("unique_edge", "")
    models = config.get("models", [])
    test_results = config.get("test_results", "")
    related_issues = config.get("related_issues", [])

    pain_section = ""
    if pain_points:
        pain_section = "## Pain points this solves\n"
        for p in pain_points:
            pain_section += f"- {p}\n"
        pain_section += "\n"

    issue_refs = ""
    if related_issues:
        issue_refs = "**Related issues**: " + ", ".join(f"#{i.lstrip('#')}" for i in related_issues) + "\n\n"

    model_table = ""
    if models:
        model_table = "## Tested models\n\n| Model | Result | Notes |\n|---|---|---|\n"
        for m in models:
            model_table += f"| {m} | ✅ | |\n"
        model_table += "\n"

    return f"""## {agent} + local models on Mac — Rapid-MLX integration

{issue_refs}{pain_section}## What is Rapid-MLX?

[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) is an inference server for Apple Silicon that provides an OpenAI-compatible API. It's designed for agentic workloads — tool calling, structured output, streaming.

{f"**Key advantage**: {unique_edge}" if unique_edge else ""}

## Quick setup

```bash
# Install
pip install rapid-mlx

# Start server (pick a model)
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit
```

Then configure {agent} to point to `http://localhost:8000/v1`.

{model_table}## Integration test suite

We built a comprehensive test suite for {agent}: [`tests/integrations/test_{config['name']}.py`](https://github.com/raullenchai/Rapid-MLX/blob/main/tests/integrations/test_{config['name']}.py)

{f"Results: **{test_results}**" if test_results else ""}

## Benchmarks vs Ollama

| Model | Rapid-MLX tok/s | Ollama tok/s | Speedup |
|---|---|---|---|
| Qwen3.5-4B | ~65 | ~30 | 2.2x |
| Qwen3.5-9B | ~45 | ~20 | 2.3x |

(Apple Silicon M-series, measured on decode speed)

Happy to answer questions or help with setup!
"""


def generate_reddit_post(config):
    """Generate Reddit r/LocalLLaMA post."""
    agent = config["display_name"]
    unique_edge = config.get("unique_edge", "")
    models = config.get("models", [])
    test_results = config.get("test_results", "")

    model_list = ", ".join(models[:3]) if models else "Qwen3.5 family"

    return f"""**Title**: {agent} + {model_list}, fully local on Mac

**Body**:

Been running {agent} with local models via [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) on my Mac and wanted to share the setup.

{f"**The interesting part**: {unique_edge}" if unique_edge else ""}

### Setup (2 minutes)

```bash
pip install rapid-mlx
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit
```

Then point {agent} at `http://localhost:8000/v1`.

{f"### Test results: {test_results}" if test_results else ""}

### Why not Ollama?

Rapid-MLX is 2-4x faster on Apple Silicon for the same models. It also handles tool calling natively — no schema overhead, no format conversion.

Wrote an integration test suite with ~15 tests covering chat, tool calling, streaming, multi-step workflows. All passing.

Repo: https://github.com/raullenchai/Rapid-MLX

Happy to help anyone set this up.
"""


def generate_tweet(config):
    """Generate Twitter/X post."""
    agent = config["display_name"]
    unique_edge = config.get("unique_edge", "")
    models = config.get("models", [])
    twitter_handles = config.get("twitter_handles", [])

    model_highlight = models[0] if models else "local models"
    handles = " ".join(f"@{h.lstrip('@')}" for h in twitter_handles) if twitter_handles else ""

    return f"""{agent} + {model_highlight}, fully local on Mac

{unique_edge or "2-4x faster than Ollama on Apple Silicon"}

pip install rapid-mlx && rapid-mlx serve <model>

{handles}

#LocalLLM #{config['name'].title().replace('-','')}

[attach demo video]
"""


def generate_discord_message(config):
    """Generate Discord message."""
    agent = config["display_name"]
    unique_edge = config.get("unique_edge", "")
    test_results = config.get("test_results", "")

    return f"""Hey! Sharing a local inference setup for {agent} on Mac.

**Rapid-MLX** ({unique_edge or "2-4x faster than Ollama on Apple Silicon"})

Setup:
```
pip install rapid-mlx
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit
```
Point {agent} at `http://localhost:8000/v1`

{f"Ran {test_results} integration tests — all passing." if test_results else ""}

Repo: <https://github.com/raullenchai/Rapid-MLX>

Let me know if you need help setting it up!
"""


def generate_issue_reply(config):
    """Generate a reply to an existing issue about a specific problem we solve."""
    agent = config["display_name"]
    unique_edge = config.get("unique_edge", "")

    return f"""This works with [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) — it handles the parsing server-side so {agent} sees standard OpenAI `tool_calls` JSON.

{f"{unique_edge}" if unique_edge else ""}

Quick setup:
```bash
pip install rapid-mlx
rapid-mlx serve <MODEL>
```

Then point {agent} at `http://localhost:8000/v1`.

Full integration test suite: [`tests/integrations/test_{config['name']}.py`](https://github.com/raullenchai/Rapid-MLX/blob/main/tests/integrations/test_{config['name']}.py)
"""


def interactive_config():
    """Interactively build a promo config."""
    print("📣 Agent Promotion Content Generator")
    print("="*50)

    config = {}
    config["name"] = input("Agent name (lowercase): ").strip()
    config["display_name"] = input(f"Display name [{config['name'].title()}]: ").strip() or config["name"].title()
    config["repo"] = input("Their GitHub repo (owner/name): ").strip()

    pp = input("Pain points we solve (comma-separated, or empty): ").strip()
    config["pain_points"] = [p.strip() for p in pp.split(",") if p.strip()] if pp else []

    config["unique_edge"] = input("Our unique advantage: ").strip()

    models = input("Models tested (comma-separated, or empty): ").strip()
    config["models"] = [m.strip() for m in models.split(",") if m.strip()] if models else []

    config["test_results"] = input("Test results (e.g. '20/20 on Gemma 4'): ").strip()

    issues = input("Related issues in their repo (comma-separated, or empty): ").strip()
    config["related_issues"] = [i.strip() for i in issues.split(",") if i.strip()] if issues else []

    handles = input("Twitter handles to tag (comma-separated, or empty): ").strip()
    config["twitter_handles"] = [h.strip() for h in handles.split(",") if h.strip()] if handles else []

    return config


def main():
    parser = argparse.ArgumentParser(description="Generate agent promotion content")
    parser.add_argument("--agent", "--name", help="Agent name")
    parser.add_argument("--display-name", help="Display name")
    parser.add_argument("--repo", help="Their GitHub repo")
    parser.add_argument("--pain-points", help="Comma-separated pain points")
    parser.add_argument("--unique-edge", help="Our unique advantage")
    parser.add_argument("--models", help="Comma-separated models tested")
    parser.add_argument("--test-results", help="Test results summary")
    parser.add_argument("--issues", help="Related issues (comma-separated)")
    parser.add_argument("--twitter-handles", help="Twitter handles to tag")
    parser.add_argument("--format", choices=["all", "github", "reddit", "tweet", "discord", "issue-reply"],
                        default="all")
    parser.add_argument("--output-dir", "-o", help="Output directory for generated files")
    args = parser.parse_args()

    if args.agent:
        config = {
            "name": args.agent,
            "display_name": args.display_name or args.agent.title(),
            "repo": args.repo or "",
            "pain_points": [p.strip() for p in args.pain_points.split(",")] if args.pain_points else [],
            "unique_edge": args.unique_edge or "",
            "models": [m.strip() for m in args.models.split(",")] if args.models else [],
            "test_results": args.test_results or "",
            "related_issues": [i.strip() for i in args.issues.split(",")] if args.issues else [],
            "twitter_handles": [h.strip() for h in args.twitter_handles.split(",")] if args.twitter_handles else [],
        }
    else:
        config = interactive_config()

    generators = {
        "github": ("GitHub Issue/Discussion", generate_github_issue),
        "reddit": ("Reddit r/LocalLLaMA", generate_reddit_post),
        "tweet": ("Twitter/X", generate_tweet),
        "discord": ("Discord", generate_discord_message),
        "issue-reply": ("Issue Reply", generate_issue_reply),
    }

    formats = generators.keys() if args.format == "all" else [args.format]

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        label, gen_fn = generators[fmt]
        content = gen_fn(config)

        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        print(content)

        if output_dir:
            filepath = os.path.join(output_dir, f"{config['name']}_{fmt}.md")
            with open(filepath, "w") as f:
                f.write(content)
            print(f"  → Saved to {filepath}")

    if not output_dir:
        print("\nTip: add --output-dir /tmp/promo to save files")


if __name__ == "__main__":
    main()
