# SPDX-License-Identifier: Apache-2.0
"""Live TUI monitor for `rapid-mlx serve`.

Polls /health, /v1/status, and /v1/requests of a running rapid-mlx server and
renders a full-screen dashboard. Press `q` (or Ctrl-C) to exit.
"""

from __future__ import annotations

import json
import select
import shutil
import sys
import termios
import time
import tty
import urllib.error
import urllib.request


COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def _c(enabled: bool, name: str, text: str) -> str:
    if not enabled:
        return text
    return f"{COLORS.get(name, '')}{text}{COLORS['reset']}"


def _fetch_json(url: str, timeout: float = 2.0) -> tuple[dict, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except Exception as exc:
        return {}, str(exc)


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _integer(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _fmt_seconds(value) -> str:
    seconds = max(0.0, _num(value))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours = minutes // 60
    minutes %= 60
    return f"{hours}h{minutes:02d}m"


def _fmt_gb(value) -> str:
    return f"{_num(value):.2f} GB"


def _clamp(text: str, width: int) -> str:
    if width <= 0:
        return ""
    text = str(text)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _bar(value: float, limit: float, width: int = 18) -> str:
    if width <= 0:
        return ""
    ratio = 0.0 if limit <= 0 else max(0.0, min(1.0, value / limit))
    fill = int(round(ratio * width))
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def _line(width: int, char: str = "-") -> str:
    return char * max(0, width)


def _row(label: str, value: str, width: int, color: str, tty_on: bool) -> str:
    label_width = min(18, max(10, width // 4))
    value_width = max(0, width - label_width - 1)
    return (
        f"{_c(tty_on, 'dim', label.ljust(label_width))} "
        f"{_c(tty_on, color, _clamp(value, value_width))}"
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_screen(
    base_url: str,
    pid: int | str,
    interval: float,
    health: dict,
    status: dict,
    requests_data: dict,
    errors: list[str],
    tty_on: bool,
) -> str:
    width, height = shutil.get_terminal_size((110, 32))
    width = max(80, width)
    height = max(24, height)

    model = status.get("model") or health.get("model_name") or "n/a"
    engine_type = health.get("engine_type") or status.get("engine_type") or "n/a"

    state = str(status.get("status") or "unknown")
    loaded = bool(health.get("model_loaded"))

    running = _integer(status.get("num_running", 0))
    waiting = _integer(status.get("num_waiting", 0))
    total_done = _integer(status.get("total_requests_processed", 0))
    steps = _integer(status.get("steps_executed", 0))
    uptime = _num(status.get("uptime_s", 0.0))

    metal = status.get("metal") or {}
    active_gb = _num(metal.get("active_memory_gb", 0.0))
    cache_gb = _num(metal.get("cache_memory_gb", 0.0))
    peak_gb = _num(metal.get("peak_memory_gb", 0.0))

    prompt_toks = _integer(status.get("total_prompt_tokens", 0))
    out_toks = _integer(status.get("total_completion_tokens", 0))

    cache_info = status.get("cache") or {}
    cache_hits = _integer(cache_info.get("hits", 0))
    cache_misses = _integer(cache_info.get("misses", 0))
    cache_entries = _integer(cache_info.get("entries", 0))

    dflash_info = status.get("dflash") or {}

    running_requests = list(status.get("requests") or [])
    entries = list((requests_data or {}).get("entries") or [])
    active_request = (requests_data or {}).get("active") or {}

    # Active ticket age
    age = 0.0
    if active_request:
        started = _num(active_request.get("started_at"))
        if started:
            age = max(0.0, time.time() - started)

    if state == "generating" or running > 0:
        status_text = "RUNNING"
        status_color = "green"
    elif loaded:
        status_text = "IDLE"
        status_color = "cyan"
    else:
        status_text = "LOADING"
        status_color = "yellow"
    if errors:
        status_text = "DEGRADED"
        status_color = "red"

    left = 38
    mid = 38
    gap = "  "
    right = max(24, width - left - mid - len(gap) * 2)

    rows: list[str] = []
    title = " Rapid-MLX Monitor "
    subtitle = f"pid {pid} | {base_url} | refresh {interval:g}s | q quits"
    rows.append(_c(tty_on, "bold", title) + _c(tty_on, "dim", " " + subtitle))
    rows.append(_line(width))

    rows.append(
        _row("status", status_text, left, status_color, tty_on)
        + gap
        + _row(
            "active/queued",
            f"{running}/{waiting}  age {_fmt_seconds(age)}",
            mid,
            "white",
            tty_on,
        )
        + gap
        + _row("uptime", _fmt_seconds(uptime), right, "white", tty_on)
    )
    rows.append(
        _row("memory active", _fmt_gb(active_gb), left, "green", tty_on)
        + gap
        + _row("cache", _fmt_gb(cache_gb), mid, "yellow", tty_on)
        + gap
        + _row("peak", _fmt_gb(peak_gb), right, "magenta", tty_on)
    )
    rows.append(
        _row("model", str(model), left, "white", tty_on)
        + gap
        + _row("engine", str(engine_type), mid, "white", tty_on)
        + gap
        + _row("steps", str(steps), right, "white", tty_on)
    )
    rows.append(
        _row("prompt tokens", f"{prompt_toks}", left, "cyan", tty_on)
        + gap
        + _row("output tokens", f"{out_toks}", mid, "cyan", tty_on)
        + gap
        + _row("requests done", f"{total_done}", right, "white", tty_on)
    )
    if cache_info:
        hit_rate = (
            f"{cache_hits / max(1, cache_hits + cache_misses):.1%}"
            if (cache_hits or cache_misses)
            else "n/a"
        )
        rows.append(
            _row("prefix cache", f"{cache_entries} entries", left, "cyan", tty_on)
            + gap
            + _row(
                "hit/miss", f"{cache_hits}/{cache_misses}", mid, "cyan", tty_on
            )
            + gap
            + _row("hit rate", hit_rate, right, "white", tty_on)
        )
    if dflash_info:
        lifetime_ratio = _num(dflash_info.get("lifetime_acceptance_ratio", 0.0))
        cur_block = _integer(dflash_info.get("current_block_size", 0))
        adaptive_on = bool(dflash_info.get("adaptive_enabled"))
        adapt_min = _integer(dflash_info.get("adaptive_min", 0))
        adapt_max = _integer(dflash_info.get("adaptive_max", 0))
        obs_min = _integer(dflash_info.get("observed_block_min", 0))
        obs_max = _integer(dflash_info.get("observed_block_max", 0))
        adaptive_label = (
            f"{adapt_min}-{adapt_max} (obs {obs_min}-{obs_max})"
            if adaptive_on
            else "off"
        )
        rows.append(
            _row(
                "dflash accept",
                f"{lifetime_ratio:.1%} lifetime {_bar(lifetime_ratio, 1.0, 12)}",
                left,
                "magenta",
                tty_on,
            )
            + gap
            + _row("block size", f"{cur_block} cfg", mid, "magenta", tty_on)
            + gap
            + _row("adaptive", adaptive_label, right, "magenta", tty_on)
        )
    rows.append(_line(width))

    # Last request panel
    last = entries[-1] if entries else {}
    last_elapsed = _num(last.get("elapsed", 0.0))
    last_prompt_tps = _num(last.get("prompt_tps", 0.0))
    last_generation_tps = _num(last.get("generation_tps", 0.0))
    last_prompt_tokens = _integer(last.get("prompt_tokens", 0))
    last_generated_tokens = _integer(last.get("generated_tokens", 0))
    last_finish = last.get("finish_reason", "n/a")
    last_surface = last.get("surface", "n/a")
    last_ttft = last.get("ttft")
    last_accept = last.get("acceptance_ratio")
    last_block = last.get("block_size")

    rows.append(_c(tty_on, "bold", "Last request"))
    if not last:
        rows.append(_c(tty_on, "dim", "  no completed requests yet"))
    else:
        rows.append(
            _row(
                "tokens",
                f"{last_generated_tokens} out / {last_prompt_tokens} prompt",
                left,
                "white",
                tty_on,
            )
            + gap
            + _row("elapsed", _fmt_seconds(last_elapsed), mid, "white", tty_on)
            + gap
            + _row("finish", str(last_finish), right, "white", tty_on)
        )
        rows.append(
            _row(
                "generation",
                f"{last_generation_tps:.1f} tok/s {_bar(last_generation_tps, 60, 12)}",
                left,
                "green",
                tty_on,
            )
            + gap
            + _row(
                "prefill",
                f"{last_prompt_tps:.1f} tok/s {_bar(last_prompt_tps, 2500, 12)}",
                mid,
                "cyan",
                tty_on,
            )
            + gap
            + _row(
                "ttft",
                f"{_num(last_ttft):.2f}s" if last_ttft is not None else "n/a",
                right,
                "yellow",
                tty_on,
            )
        )
        accept_text = (
            f"{_num(last_accept):.0%} {_bar(_num(last_accept), 1.0, 12)}"
            if last_accept is not None
            else "n/a"
        )
        block_text = str(last_block) if last_block is not None else "n/a"
        rows.append(
            _row("surface", str(last_surface), left, "white", tty_on)
            + gap
            + _row("dflash accept", accept_text, mid, "magenta", tty_on)
            + gap
            + _row("block size", block_text, right, "magenta", tty_on)
        )
    rows.append(_line(width))

    # Averages so far
    rows.append(_c(tty_on, "bold", f"Averages so far ({len(entries)} requests)"))
    if not entries:
        rows.append(_c(tty_on, "dim", "  no completed request metrics yet"))
    else:
        avg_out = _mean([_num(item.get("generated_tokens", 0)) for item in entries])
        avg_prompt = _mean([_num(item.get("prompt_tokens", 0)) for item in entries])
        avg_gen_tps = _mean([_num(item.get("generation_tps", 0.0)) for item in entries])
        avg_pre_tps = _mean([_num(item.get("prompt_tps", 0.0)) for item in entries])
        avg_elapsed = _mean([_num(item.get("elapsed", 0.0)) for item in entries])
        accept_values = [
            _num(item.get("acceptance_ratio"))
            for item in entries
            if item.get("acceptance_ratio") is not None
        ]
        avg_accept = _mean(accept_values) if accept_values else None
        if avg_accept is not None:
            header = "  out  prompt  gen tok/s  prefill tok/s  elapsed   accept"
            row = (
                f"  {avg_out:>5.1f} "
                f"{avg_prompt:>7.1f} "
                f"{avg_gen_tps:>9.1f} "
                f"{avg_pre_tps:>13.1f} "
                f"{avg_elapsed:>8.1f}s "
                f"{avg_accept:>7.0%}"
            )
        else:
            header = "  out  prompt  gen tok/s  prefill tok/s  elapsed"
            row = (
                f"  {avg_out:>5.1f} "
                f"{avg_prompt:>7.1f} "
                f"{avg_gen_tps:>9.1f} "
                f"{avg_pre_tps:>13.1f} "
                f"{avg_elapsed:>8.1f}s"
            )
        rows.append(_c(tty_on, "dim", _clamp(header, width)))
        rows.append(_clamp(row, width))
    rows.append(_line(width))

    # Recent requests
    rows.append(_c(tty_on, "bold", "Recent requests"))
    last_message_reserved_rows = 14 + (5 if errors else 0)
    recent_limit = max(1, min(8, height - len(rows) - last_message_reserved_rows - 1))
    recent_entries = entries[-recent_limit:]
    if not recent_entries:
        rows.append(_c(tty_on, "dim", "  no completed request metrics yet"))
    else:
        any_accept = any(
            item.get("acceptance_ratio") is not None for item in recent_entries
        )
        if any_accept:
            header = (
                "  time      surface              out  prompt  gen tok/s  prefill tok/s  accept  block  finish"
            )
        else:
            header = (
                "  time      surface              out  prompt  gen tok/s  prefill tok/s  finish"
            )
        rows.append(_c(tty_on, "dim", _clamp(header, width)))
        for item in reversed(recent_entries):
            ts = item.get("finished_at") or 0
            try:
                when = time.strftime("%H:%M:%S", time.localtime(float(ts)))
            except Exception:
                when = "--:--:--"
            surface = str(item.get("surface", "n/a"))[-18:].ljust(18)
            base = (
                f"  {when}  "
                f"{surface} "
                f"{_integer(item.get('generated_tokens', 0)):>5} "
                f"{_integer(item.get('prompt_tokens', 0)):>7} "
                f"{_num(item.get('generation_tps', 0.0)):>9.1f} "
                f"{_num(item.get('prompt_tps', 0.0)):>13.1f}  "
            )
            if any_accept:
                accept = item.get("acceptance_ratio")
                accept_s = (
                    f"{_num(accept):>5.0%}" if accept is not None else "  -  "
                )
                block = item.get("block_size")
                block_s = f"{_integer(block):>4}" if block is not None else "  - "
                row = base + f"{accept_s}  {block_s}  {str(item.get('finish_reason', 'n/a'))[:8]}"
            else:
                row = base + str(item.get("finish_reason", "n/a"))[:12]
            rows.append(_clamp(row, width))
    rows.append(_line(width))

    # Last messages
    rows.append(_c(tty_on, "bold", "Last messages"))
    now_ts = time.time()
    message_rows: list[str] = []
    if active_request:
        started_at = _num(active_request.get("started_at", 0.0))
        updated_at = _num(active_request.get("updated_at", 0.0))
        a_age = now_ts - started_at if started_at else 0.0
        stale = now_ts - updated_at if updated_at else 0.0
        message_preview = str(active_request.get("message_preview") or "")
        text = message_preview if message_preview else "no model text yet"
        message_rows.append(
            "  * active "
            f"{active_request.get('surface', 'n/a')} "
            f"{active_request.get('phase', 'active')} "
            f"age {_fmt_seconds(a_age)} stale {_fmt_seconds(stale)} "
            f"{_integer(active_request.get('generated_tokens', 0))} tok | {text}"
        )

    for item in reversed(entries):
        if len(message_rows) >= 10:
            break
        message_preview = str(item.get("message_preview") or "")
        if not message_preview:
            continue
        finished_at = _num(item.get("finished_at", 0.0))
        m_age = now_ts - finished_at if finished_at else 0.0
        message_rows.append(
            "  - "
            f"{item.get('surface', 'n/a')} "
            f"{_fmt_seconds(m_age)} ago "
            f"{_integer(item.get('generated_tokens', 0))} tok "
            f"{item.get('finish_reason', 'n/a')} | {message_preview}"
        )

    if message_rows:
        for row in message_rows:
            rows.append(_clamp(row, width))
    else:
        rows.append(_c(tty_on, "dim", "  no model messages yet"))

    # Active running requests (engine view)
    if running_requests:
        rows.append(_line(width))
        rows.append(_c(tty_on, "bold", f"Active requests ({len(running_requests)})"))
        any_dflash = any(
            ("acceptance_ratio" in r) or ("block_size" in r) for r in running_requests
        )
        if any_dflash:
            header = (
                "  id            phase       elapsed   prompt   out   tok/s   ttft   accept  block"
            )
        else:
            header = (
                "  id            phase       elapsed   prompt   out   tok/s   ttft   max"
            )
        rows.append(_c(tty_on, "dim", _clamp(header, width)))
        for item in running_requests[:4]:
            rid = str(item.get("request_id") or "")[-12:].ljust(12)
            phase = str(item.get("phase") or item.get("status") or "")[:10].ljust(10)
            elapsed = _fmt_seconds(item.get("elapsed_s", 0.0))
            ptoks = _integer(item.get("prompt_tokens", 0))
            otoks = _integer(item.get("completion_tokens", 0))
            tps = item.get("tokens_per_second")
            tps_s = "  -  " if tps is None else f"{_num(tps):>5.1f}"
            ttft = item.get("ttft_s")
            ttft_s = "  -  " if ttft is None else f"{_num(ttft):>4.2f}"
            if any_dflash:
                accept = _num(item.get("acceptance_ratio", 0.0))
                bs = _integer(item.get("block_size", 0))
                row = (
                    f"  {rid} {phase} {elapsed:>7} "
                    f"{ptoks:>7} {otoks:>5} {tps_s} {ttft_s} "
                    f"{accept:>6.2f}  {bs:>4}"
                )
            else:
                mx = _integer(item.get("max_tokens", 0))
                row = (
                    f"  {rid} {phase} {elapsed:>7} "
                    f"{ptoks:>7} {otoks:>5} {tps_s} {ttft_s} {mx:>5}"
                )
            rows.append(_clamp(row, width))

    if errors:
        rows.append(_line(width))
        rows.append(_c(tty_on, "red", "Errors"))
        for error in errors[-3:]:
            rows.append(_c(tty_on, "red", "  " + _clamp(error, width - 2)))

    rows.append("")
    rows.append(
        _c(
            tty_on,
            "dim",
            "Tip: send a request to /v1/chat/completions in another terminal; metrics update here.",
        )
    )

    return "\n".join(rows[:height])


def _read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    try:
        return sys.stdin.read(1)
    except Exception:
        return None


def run_monitor(base_url: str, interval: float = 1.0, pid: int | str = "?") -> int:
    """Run the full-screen TUI loop."""
    health_url = base_url.rstrip("/") + "/health"
    status_url = base_url.rstrip("/") + "/v1/status"
    requests_url = base_url.rstrip("/") + "/v1/requests?limit=50"
    interval = max(0.1, float(interval))
    tty_on = sys.stdout.isatty()

    old_term = None
    if tty_on and sys.stdin.isatty():
        try:
            old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
        except Exception:
            old_term = None

    try:
        if tty_on:
            sys.stdout.write("\033[?1049h\033[?25l")
            sys.stdout.flush()
        while True:
            health, herr = _fetch_json(health_url)
            status, serr = _fetch_json(status_url)
            requests_data, rerr = _fetch_json(requests_url)
            errors = [e for e in (herr, serr, rerr) if e]
            screen = _build_screen(
                base_url,
                pid,
                interval,
                health,
                status,
                requests_data,
                errors,
                tty_on,
            )
            if tty_on:
                sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(screen + "\n")
            sys.stdout.flush()
            deadline = time.time() + interval
            while time.time() < deadline:
                key = _read_key()
                if key in {"q", "Q", "\x03"}:
                    return 0
                time.sleep(0.05)
            if not tty_on:
                sys.stdout.write("\n")
    except KeyboardInterrupt:
        return 0
    finally:
        if old_term is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
            except Exception:
                pass
        if tty_on:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
    return 0
