#!/usr/bin/env python3
"""Agent Discovery — monitor GitHub trending & HN for new AI coding agents.

Scans GitHub trending repos and Hacker News front page for AI coding agents
that use OpenAI-compatible APIs (our integration target). Outputs actionable
candidates sorted by urgency (stars, recency, compatibility signals).

Usage:
    python3 scripts/agent_discovery.py              # Full scan
    python3 scripts/agent_discovery.py --github-only # GitHub trending only
    python3 scripts/agent_discovery.py --hn-only     # Hacker News only
    python3 scripts/agent_discovery.py --json        # JSON output for piping

Requires: pip install httpx
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta

import httpx

# Agents we've already listed (skip these)
KNOWN_AGENTS = {
    "openclaude", "open-claude", "goose", "claw-code", "claude-code",
    "hermes-agent", "hermes", "aider", "cursor", "cline", "continue",
    "copilot", "opencode", "pydantic-ai", "pydanticai", "langchain",
    "smolagents", "librechat", "open-webui", "openwebui",
}

# Keywords that signal an AI coding agent
AGENT_KEYWORDS = [
    "ai agent", "coding agent", "ai coding", "code assistant",
    "terminal ai", "cli ai", "ai cli", "openai compatible",
    "openai api", "local llm", "ollama", "ai terminal",
    "ai ide", "ai editor", "agentic", "tool calling",
    "code generation", "ai pair", "ai programmer",
]

# Keywords that signal OpenAI API compatibility (our integration path)
COMPAT_KEYWORDS = [
    "openai", "base_url", "OPENAI_BASE_URL", "openai-compatible",
    "ollama", "lm studio", "local model", "self-hosted",
    "/v1/chat/completions", "openai_api_key", "api_key",
]


def fetch_github_trending(language=None, since="daily"):
    """Scrape GitHub trending page for repos."""
    url = "https://github.com/trending"
    if language:
        url += f"/{language}"
    url += f"?since={since}"

    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        resp.raise_for_status()
    except Exception as e:
        print(f"⚠️  GitHub trending fetch failed: {e}", file=sys.stderr)
        return []

    # Parse with regex (avoid bs4 hard dependency for simple case)
    repos = []
    # Pattern: /owner/repo in h2.lh-condensed a
    for match in re.finditer(
        r'<h2 class="h3 lh-condensed">\s*<a href="/([^"]+)"', resp.text
    ):
        full_name = match.group(1).strip().strip("/")
        repos.append(full_name)

    return repos


def fetch_github_search(query, sort="stars", per_page=30):
    """Search GitHub API for repos matching a query."""
    url = "https://api.github.com/search/repositories"
    # Only repos created in last 30 days or with recent pushes
    since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    params = {
        "q": f"{query} pushed:>{since}",
        "sort": sort,
        "order": "desc",
        "per_page": per_page,
    }
    try:
        resp = httpx.get(url, params=params, timeout=15, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "rapid-mlx-agent-discovery",
        })
        resp.raise_for_status()
        return resp.json().get("items", [])
    except Exception as e:
        print(f"⚠️  GitHub search failed: {e}", file=sys.stderr)
        return []


def fetch_hn_front_page(num_stories=30):
    """Fetch Hacker News top stories."""
    try:
        resp = httpx.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=10,
        )
        story_ids = resp.json()[:num_stories]
    except Exception as e:
        print(f"⚠️  HN fetch failed: {e}", file=sys.stderr)
        return []

    stories = []
    for sid in story_ids:
        try:
            r = httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                timeout=5,
            )
            item = r.json()
            if item and item.get("type") == "story":
                stories.append({
                    "id": sid,
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0),
                    "time": item.get("time", 0),
                })
        except Exception:
            continue

    return stories


def is_agent_candidate(name, description):
    """Check if a repo looks like an AI coding agent."""
    text = f"{name} {description}".lower()
    # Must match at least one agent keyword — compat signals alone are too broad
    return any(kw in text for kw in AGENT_KEYWORDS)


def check_openai_compat(owner_repo):
    """Quick check if a repo likely supports OpenAI-compatible API."""
    # Search README for compatibility signals
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{owner_repo}/readme",
            timeout=10,
            headers={
                "Accept": "application/vnd.github.v3.raw",
                "User-Agent": "rapid-mlx-agent-discovery",
            },
        )
        if resp.status_code == 200:
            readme = resp.text.lower()
            signals = []
            for kw in COMPAT_KEYWORDS:
                if kw.lower() in readme:
                    signals.append(kw)
            return signals
    except Exception:
        pass
    return []


def get_repo_info(owner_repo):
    """Get basic repo info from GitHub API."""
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{owner_repo}",
            timeout=10,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "rapid-mlx-agent-discovery",
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "stars": data.get("stargazers_count", 0),
                "description": data.get("description", ""),
                "language": data.get("language", ""),
                "created": data.get("created_at", ""),
                "pushed": data.get("pushed_at", ""),
                "topics": data.get("topics", []),
            }
    except Exception:
        pass
    return None


def score_candidate(name, info, compat_signals):
    """Score a candidate (higher = more urgent to integrate)."""
    score = 0
    stars = info.get("stars", 0)

    # Star tiers
    if stars >= 10000:
        score += 50
    elif stars >= 5000:
        score += 30
    elif stars >= 1000:
        score += 15
    elif stars >= 500:
        score += 5

    # Recency bonus (pushed in last 7 days)
    pushed = info.get("pushed", "")
    if pushed:
        try:
            push_date = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
            days_ago = (datetime.now(push_date.tzinfo) - push_date).days
            if days_ago <= 2:
                score += 20  # Very hot
            elif days_ago <= 7:
                score += 10
        except Exception:
            pass

    # OpenAI compatibility bonus
    score += len(compat_signals) * 5

    # Topic bonus
    topics = info.get("topics", [])
    agent_topics = {"ai", "agent", "coding-agent", "llm", "openai", "cli", "terminal"}
    score += len(set(topics) & agent_topics) * 3

    return score


def scan_github(verbose=True):
    """Full GitHub scan: trending + search."""
    candidates = {}

    # 1. Trending repos (Python, TypeScript — most agents)
    if verbose:
        print("🔍 Scanning GitHub trending (Python)...")
    for repo in fetch_github_trending("python", "daily"):
        name = repo.split("/")[-1].lower()
        if name not in KNOWN_AGENTS:
            candidates[repo] = {"source": "trending/python"}

    if verbose:
        print("🔍 Scanning GitHub trending (TypeScript)...")
    for repo in fetch_github_trending("typescript", "daily"):
        name = repo.split("/")[-1].lower()
        if name not in KNOWN_AGENTS:
            candidates[repo] = {"source": "trending/typescript"}

    # Give GitHub a breather
    time.sleep(1)

    # 2. Search for new AI agents
    if verbose:
        print("🔍 Searching GitHub for new AI agents...")
    for query in ["ai coding agent", "ai terminal assistant", "openai compatible cli"]:
        for item in fetch_github_search(query, per_page=10):
            full_name = item["full_name"]
            name = full_name.split("/")[-1].lower()
            if name not in KNOWN_AGENTS and full_name not in candidates:
                desc = item.get("description", "") or ""
                if is_agent_candidate(name, desc):
                    candidates[full_name] = {
                        "source": f"search/{query}",
                        "stars": item.get("stargazers_count", 0),
                        "description": desc,
                    }
        time.sleep(1)  # Rate limit

    # 3. Enrich candidates
    results = []
    for repo, meta in candidates.items():
        if verbose:
            print(f"  📦 Checking {repo}...")
        info = get_repo_info(repo)
        if not info:
            continue
        if info["stars"] < 100:  # Skip tiny repos
            continue

        # Check if it's actually an agent
        desc = info.get("description", "") or meta.get("description", "")
        name = repo.split("/")[-1]
        if not is_agent_candidate(name, desc):
            continue

        compat = check_openai_compat(repo)
        score = score_candidate(name, info, compat)

        results.append({
            "repo": repo,
            "stars": info["stars"],
            "description": desc,
            "language": info.get("language", ""),
            "pushed": info.get("pushed", ""),
            "compat_signals": compat,
            "score": score,
            "source": meta["source"],
        })
        time.sleep(0.5)  # Rate limit

    return sorted(results, key=lambda x: x["score"], reverse=True)


def scan_hn(verbose=True):
    """Scan Hacker News front page for AI agent launches."""
    if verbose:
        print("🔍 Scanning Hacker News front page...")

    stories = fetch_hn_front_page(50)
    results = []

    for story in stories:
        title = story["title"].lower()
        url = story.get("url", "")

        # Check if title mentions AI agents
        if not any(kw in title for kw in [
            "agent", "ai coding", "code assistant", "ai cli",
            "terminal ai", "local llm", "coding tool", "ai ide",
            "copilot", "ai pair", "ai programmer",
        ]):
            continue

        # Extract GitHub repo if URL points to one
        github_repo = None
        if "github.com" in url:
            parts = url.rstrip("/").split("github.com/")
            if len(parts) == 2:
                repo_path = parts[1].split("?")[0].split("#")[0]
                if "/" in repo_path and repo_path.count("/") == 1:
                    github_repo = repo_path

        results.append({
            "title": story["title"],
            "url": url,
            "score": story["score"],
            "hn_id": story["id"],
            "github_repo": github_repo,
            "hn_url": f"https://news.ycombinator.com/item?id={story['id']}",
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


def print_report(github_results, hn_results):
    """Print a human-readable report."""
    print(f"\n{'='*70}")
    print(f"  Agent Discovery Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    if github_results:
        print(f"\n📦 GitHub Candidates ({len(github_results)} found)")
        print(f"{'─'*70}")
        for r in github_results[:15]:
            stars = f"⭐{r['stars']:,}"
            compat = " 🔌" if r["compat_signals"] else ""
            score_bar = "█" * min(r["score"] // 5, 20)
            print(f"  {stars:>10}  {r['repo']:<40}{compat}")
            print(f"             {r['description'][:60]}")
            if r["compat_signals"]:
                print(f"             Signals: {', '.join(r['compat_signals'][:5])}")
            print(f"             Score: [{score_bar}] {r['score']}")
            print()
    else:
        print("\n📦 No new GitHub candidates found")

    if hn_results:
        print(f"\n📰 Hacker News Mentions ({len(hn_results)} found)")
        print(f"{'─'*70}")
        for r in hn_results[:10]:
            print(f"  🔥 {r['score']:>4} pts  {r['title']}")
            if r["github_repo"]:
                print(f"              → github.com/{r['github_repo']}")
            print(f"              {r['hn_url']}")
            print()
    else:
        print("\n📰 No AI agent mentions on HN front page")

    # Action items
    hot = [r for r in github_results if r["score"] >= 30]
    if hot:
        print(f"\n🚨 ACTION REQUIRED — {len(hot)} high-priority candidates:")
        print(f"{'─'*70}")
        for r in hot[:5]:
            print(f"  → {r['repo']} (⭐{r['stars']:,}, score={r['score']})")
            if r["compat_signals"]:
                print("    Already OpenAI-compatible! Run:")
                print(f"    python3 scripts/agent_test_gen.py {r['repo']}")
            print()
    else:
        print("\n✅ No urgent candidates — check back tomorrow")


def main():
    parser = argparse.ArgumentParser(description="Discover new AI coding agents")
    parser.add_argument("--github-only", action="store_true")
    parser.add_argument("--hn-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--min-stars", type=int, default=100)
    args = parser.parse_args()

    github_results = []
    hn_results = []

    if not args.hn_only:
        github_results = scan_github(verbose=not args.json)
        github_results = [r for r in github_results if r["stars"] >= args.min_stars]

    if not args.github_only:
        hn_results = scan_hn(verbose=not args.json)

    if args.json:
        print(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "github": github_results,
            "hn": hn_results,
        }, indent=2))
    else:
        print_report(github_results, hn_results)


if __name__ == "__main__":
    main()
