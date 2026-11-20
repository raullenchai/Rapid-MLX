# SPDX-License-Identifier: Apache-2.0
"""Read-only terminal monitor; deliberately render only allowlisted metadata."""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

import httpx


def _object(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _fmt(value: Any, *, count: bool = False) -> str:
    number = _number(value)
    return "-" if number is None else format(number, ".0f" if count else ".1f")


def _model(value: Any) -> str:
    # Local checkpoint paths and terminal control sequences must not appear.
    if not isinstance(value, str) or value.startswith(("/", "~", ".")) or "\\" in value:
        return "[local/unknown model]"
    return "".join(c for c in value[:80] if c.isprintable())


def _metrics(body: str) -> dict[str, float]:
    result = {}
    for line in body.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("rapid_mlx_"):
            try:
                value = float(parts[1])
            except ValueError:
                continue
            if math.isfinite(value):
                result[parts[0]] = value
    return result


def _fetch(client: httpx.Client, path: str) -> tuple[Any, str]:
    try:
        response = client.get(path)
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        return (response.text if path == "/metrics" else response.json()), ""
    except (httpx.HTTPError, ValueError):
        return None, "unavailable"


def snapshot(client: httpx.Client) -> dict:
    """Only GET the three monitoring surfaces; never print error bodies."""
    status, error = _fetch(client, "/v1/status")
    residency, residency_error = _fetch(client, "/v1/models/residency")
    metrics, metrics_error = _fetch(client, "/metrics")
    return {
        "status": _object(status),
        "error": error,
        "residency": _object(residency),
        "residency_error": residency_error,
        "metrics": _metrics(metrics) if isinstance(metrics, str) else {},
        "metrics_error": metrics_error,
    }


def _failure(current: dict, port: int) -> str:
    address = f"127.0.0.1:{port}"
    if "HTTP 401" in (
        current["error"],
        current["residency_error"],
        current["metrics_error"],
    ):
        return (
            f"Rapid-MLX server at {address}: HTTP 401 Unauthorized. "
            "Set RAPID_MLX_API_KEY to the server's API key."
        )
    if current["error"] == "unavailable":
        return (
            f"No Rapid-MLX server at {address}. Start one with: rapid-mlx serve <model>"
        )
    if current["error"]:
        return f"Rapid-MLX server at {address}: {current['error']}."
    return ""


def render(current: dict, previous: dict | None, port: int) -> str:
    failure = _failure(current, port)
    if failure:
        return failure
    status = current["status"]
    metrics = current["metrics"]
    state = status.get("status")
    state = state if state in ("idle", "generating", "not_loaded") else "unavailable"
    lines = [f"Rapid-MLX top | 127.0.0.1:{port} | {state}"]
    for name in ("residency", "metrics"):
        if current[f"{name}_error"]:
            lines.append(f"{name}: {current[f'{name}_error']}")
    rates: list[float | None] = [None, None, None]
    if previous and not current["error"] and not previous["error"]:
        before = previous["status"]
        uptime = _number(status.get("uptime_s"))
        old_uptime = _number(before.get("uptime_s"))
        if uptime is not None and old_uptime is not None and uptime > old_uptime:
            for i, key in enumerate(
                (
                    "total_requests_processed",
                    "total_prompt_tokens",
                    "total_completion_tokens",
                )
            ):
                new, old = _number(status.get(key)), _number(before.get(key))
                if new is not None and old is not None and new >= old:
                    rates[i] = (
                        (new - old) / (uptime - old_uptime) * (60 if i == 0 else 1)
                    )
    lines.append(
        f"Totals: requests/min {_fmt(rates[0])} | in tok/s {_fmt(rates[1])} | out tok/s {_fmt(rates[2])} | queue {_fmt(status.get('num_waiting'), count=True)}"
    )
    metal = _object(status.get("metal"))
    lines.append(
        f"Metal GiB: active {_fmt(metal.get('active_memory_gb'))} | peak {_fmt(metal.get('peak_memory_gb'))}"
    )
    cache = _object(status.get("cache"))
    hits, misses = _number(cache.get("hits")), _number(cache.get("misses"))
    hit_rate = (
        hits / (hits + misses) * 100
        if hits is not None and misses is not None and hits + misses > 0
        else None
    )
    cache_bytes = metrics.get("rapid_mlx_prefix_cache_current_bytes")
    cache_mib = (
        cache_bytes / 2**20
        if cache_bytes is not None
        else cache.get("current_memory_mb")
    )
    lines.append(
        f"Primary engine prefix cache: MiB {_fmt(cache_mib)} | hit % {_fmt(hit_rate)} | per-model KV size unavailable"
    )
    lines.append("Models: memory MiB (measured, otherwise estimated)")
    models = current["residency"].get("models")
    if isinstance(models, list):
        for value in models:
            model = _object(value)
            if model.get("state") != "resident":
                continue
            measured = _number(model.get("measured_bytes"))
            size = (
                measured
                if measured is not None
                else _number(model.get("estimated_bytes"))
            )
            lines.append(
                f"  {_model(model.get('id'))} | {_fmt(size / 2**20 if size is not None else None)} ({'measured' if measured is not None else 'estimated'})"
            )
    else:
        lines.append(f"  {_model(status.get('model'))} | -")
    lines.append("Requests: phase | prompt tokens | decode tok/s | elapsed s")
    requests = status.get("requests")
    if isinstance(requests, list):
        for index, value in enumerate(requests, 1):
            request = _object(value)
            phase = request.get("phase")
            phase = phase if phase in ("queued", "prefill", "generation") else "unknown"
            lines.append(
                f"  {index}: {phase} | {_fmt(request.get('prompt_tokens'), count=True)} | {_fmt(request.get('tokens_per_second'))} | {_fmt(request.get('elapsed_s'))}"
            )
    lines.append("- = unavailable; rates need two samples. Ctrl-C to quit.")
    return "\n".join(lines)


def top_command(args: Any) -> None:
    """Render once on pipes, refresh once a second on a terminal."""
    interactive = sys.stdout.isatty() and not args.once
    key = os.environ.get("RAPID_MLX_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    previous = None
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{args.port}",
            headers=headers,
            timeout=2,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            while True:
                current = snapshot(client)
                print(
                    ("\033[2J\033[H" if interactive else "")
                    + render(current, previous, args.port),
                    flush=True,
                )
                if not interactive:
                    if _failure(current, args.port):
                        raise SystemExit(1)
                    return
                previous = current
                time.sleep(1)
    except KeyboardInterrupt:
        return
