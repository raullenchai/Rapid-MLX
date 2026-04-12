#!/usr/bin/env python3
"""Rapid-MLX Usage Statistics — collects download and traffic data.

Combines:
  B) GitHub traffic (clones, views, stars, forks) — requires repo owner auth
  C) PyPI download stats — public API

Usage:
    python3 scripts/usage_stats.py              # Full report
    python3 scripts/usage_stats.py --json       # JSON output
    python3 scripts/usage_stats.py --save       # Append to docs/usage-stats.md
"""

import argparse
import json
import subprocess
from datetime import datetime

REPO = "raullenchai/Rapid-MLX"
PYPI_PACKAGE = "rapid-mlx"


def gh_api(path: str) -> dict | list | None:
    """Call GitHub API via gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "api", path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def get_github_stats() -> dict:
    """Get GitHub repo stats: stars, forks, watchers, issues."""
    data = gh_api(f"repos/{REPO}")
    if not data:
        return {}
    return {
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "watchers": data.get("subscribers_count", 0),
        "open_issues": data.get("open_issues_count", 0),
    }


def get_github_traffic() -> dict:
    """Get GitHub traffic: clones and views (last 14 days, owner only)."""
    clones = gh_api(f"repos/{REPO}/traffic/clones")
    views = gh_api(f"repos/{REPO}/traffic/views")
    referrers = gh_api(f"repos/{REPO}/traffic/popular/referrers")

    result = {}
    if clones:
        result["clones_14d"] = clones.get("count", 0)
        result["unique_cloners_14d"] = clones.get("uniques", 0)
        # Daily breakdown
        result["clone_daily"] = [
            {"date": c["timestamp"][:10], "count": c["count"], "unique": c["uniques"]}
            for c in clones.get("clones", [])
        ]
    if views:
        result["views_14d"] = views.get("count", 0)
        result["unique_visitors_14d"] = views.get("uniques", 0)
        result["view_daily"] = [
            {"date": v["timestamp"][:10], "count": v["count"], "unique": v["uniques"]}
            for v in views.get("views", [])
        ]
    if referrers:
        result["top_referrers"] = [
            {"source": r["referrer"], "count": r["count"], "unique": r["uniques"]}
            for r in referrers[:10]
        ]
    return result


def get_pypi_stats() -> dict:
    """Get PyPI download stats."""
    result = {}
    try:
        # Recent stats (flat dict: {"data": {"last_day": N, ...}})
        proc = subprocess.run(
            ["pypistats", "recent", PYPI_PACKAGE, "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout).get("data", {})
            if isinstance(data, dict):
                result["pypi_last_day"] = data.get("last_day", 0)
                result["pypi_last_week"] = data.get("last_week", 0)
                result["pypi_last_month"] = data.get("last_month", 0)

        # System breakdown (list of dicts)
        proc = subprocess.run(
            ["pypistats", "system", PYPI_PACKAGE, "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout).get("data", [])
            systems = {}
            if isinstance(data, list):
                for row in data:
                    cat = row.get("category") or "unknown"
                    systems[cat] = systems.get(cat, 0) + row.get("downloads", 0)
            result["pypi_by_system"] = systems

        # Python version breakdown (list of dicts)
        proc = subprocess.run(
            ["pypistats", "python_minor", PYPI_PACKAGE, "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout).get("data", [])
            versions = {}
            if isinstance(data, list):
                for row in data:
                    cat = row.get("category") or "unknown"
                    versions[cat] = versions.get(cat, 0) + row.get("downloads", 0)
            result["pypi_by_python"] = versions

    except FileNotFoundError:
        result["error"] = "pypistats not installed. Run: pip install pypistats"
    except Exception as e:
        result["error"] = str(e)

    return result


def print_report(github: dict, traffic: dict, pypi: dict):
    """Print a human-readable report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{'='*60}")
    print(f"  Rapid-MLX Usage Stats — {now}")
    print(f"{'='*60}")

    # GitHub
    print(f"\n  GitHub ({REPO})")
    print(f"  {'─'*50}")
    if github:
        print(f"  Stars:        {github.get('stars', '?'):>6,}")
        print(f"  Forks:        {github.get('forks', '?'):>6,}")
        print(f"  Watchers:     {github.get('watchers', '?'):>6,}")
        print(f"  Open Issues:  {github.get('open_issues', '?'):>6,}")

    # Traffic (14-day window)
    if traffic:
        print("\n  GitHub Traffic (last 14 days)")
        print(f"  {'─'*50}")
        print(f"  Page Views:   {traffic.get('views_14d', '?'):>6,}  ({traffic.get('unique_visitors_14d', '?')} unique)")
        print(f"  Git Clones:   {traffic.get('clones_14d', '?'):>6,}  ({traffic.get('unique_cloners_14d', '?')} unique)")

        if traffic.get("top_referrers"):
            print("\n  Top Referrers:")
            for ref in traffic["top_referrers"][:5]:
                print(f"    {ref['source']:30s} {ref['count']:>5} views ({ref['unique']} unique)")

        if traffic.get("clone_daily"):
            print("\n  Daily Clones:")
            for day in traffic["clone_daily"][-7:]:  # last 7 days
                bar = "█" * min(day["count"] // 5, 40)
                print(f"    {day['date']}  {day['count']:>4} ({day['unique']:>3} unique) {bar}")

    # PyPI
    print(f"\n  PyPI ({PYPI_PACKAGE})")
    print(f"  {'─'*50}")
    if "error" in pypi:
        print(f"  Error: {pypi['error']}")
    else:
        print(f"  Last day:     {pypi.get('pypi_last_day', '?'):>6,}")
        print(f"  Last week:    {pypi.get('pypi_last_week', '?'):>6,}")
        print(f"  Last month:   {pypi.get('pypi_last_month', '?'):>6,}")

        if pypi.get("pypi_by_system"):
            print("\n  By OS:")
            for os_name, count in sorted(pypi["pypi_by_system"].items(), key=lambda x: -x[1]):
                if os_name == "null" or os_name == "unknown":
                    continue
                print(f"    {os_name:15s} {count:>6,}")

        if pypi.get("pypi_by_python"):
            print("\n  By Python Version:")
            for ver, count in sorted(pypi["pypi_by_python"].items(), key=lambda x: -x[1]):
                if ver == "null" or ver == "unknown":
                    continue
                print(f"    {ver:15s} {count:>6,}")

    # Summary
    total_reach = (
        github.get("stars", 0)
        + traffic.get("unique_cloners_14d", 0)
        + pypi.get("pypi_last_month", 0)
    )
    print(f"\n  {'─'*50}")
    print(f"  Combined Reach Score: {total_reach:,}")
    print("  (stars + unique cloners + monthly PyPI downloads)")


def save_snapshot(github: dict, traffic: dict, pypi: dict, filepath: str):
    """Append a snapshot to a markdown file."""
    now = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"\n## {now}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| GitHub Stars | {github.get('stars', '?')} |",
        f"| GitHub Forks | {github.get('forks', '?')} |",
        f"| Views (14d) | {traffic.get('views_14d', '?')} ({traffic.get('unique_visitors_14d', '?')} unique) |",
        f"| Clones (14d) | {traffic.get('clones_14d', '?')} ({traffic.get('unique_cloners_14d', '?')} unique) |",
        f"| PyPI last day | {pypi.get('pypi_last_day', '?')} |",
        f"| PyPI last week | {pypi.get('pypi_last_week', '?')} |",
        f"| PyPI last month | {pypi.get('pypi_last_month', '?')} |",
        f"| PyPI Darwin | {pypi.get('pypi_by_system', {}).get('Darwin', '?')} |",
        f"| PyPI Linux | {pypi.get('pypi_by_system', {}).get('Linux', '?')} |",
        "",
    ]

    # Top referrers
    if traffic.get("top_referrers"):
        lines.append("**Top referrers:**")
        for ref in traffic["top_referrers"][:5]:
            lines.append(f"- {ref['source']}: {ref['count']} views")
        lines.append("")

    with open(filepath, "a") as f:
        f.write("\n".join(lines))
    print(f"\nSnapshot saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Rapid-MLX usage statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--save", action="store_true",
                        help="Append snapshot to docs/usage-stats.md")
    parser.add_argument("--output", default="docs/usage-stats.md",
                        help="Output file for --save")
    args = parser.parse_args()

    github = get_github_stats()
    traffic = get_github_traffic()
    pypi = get_pypi_stats()

    if args.json:
        print(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "github": github,
            "traffic": traffic,
            "pypi": pypi,
        }, indent=2))
    else:
        print_report(github, traffic, pypi)

    if args.save:
        save_snapshot(github, traffic, pypi, args.output)


if __name__ == "__main__":
    main()
