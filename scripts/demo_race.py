#!/usr/bin/env python3
"""
Side-by-side speed comparison: Rapid-MLX vs Ollama
Records a visual terminal demo for social media.

Usage:
    # Start Rapid-MLX first:
    #   rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --port 8000
    # Then run:
    python3 scripts/demo_race.py
"""

import asyncio
import json
import sys
import time

import aiohttp

# ── Config ───────────────────────────────────────────────────────────
PROMPT = "Write a Python function to find the longest palindromic substring. Be concise."
MAX_TOKENS = 200

ENGINES = [
    {
        "name": "Rapid-MLX",
        "url": "http://localhost:8000/v1/chat/completions",
        "model": "default",
        "color": "\033[38;5;208m",  # orange
        "api": "openai",
    },
    {
        "name": "Ollama",
        "url": "http://localhost:11434/api/chat",
        "model": "qwen3.5:9b",
        "color": "\033[38;5;245m",  # gray
        "api": "ollama",
    },
]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[38;5;82m"
CYAN = "\033[38;5;45m"
WHITE = "\033[38;5;255m"

# Terminal layout
COL_WIDTH = 58
DIVIDER = "│"


def clear_screen():
    print("\033[2J\033[H", end="")


def move_to(row, col):
    print(f"\033[{row};{col}H", end="")


def print_at(row, col, text, max_width=None):
    move_to(row, col)
    if max_width:
        # Truncate visible characters (strip ANSI for counting)
        import re
        visible = re.sub(r'\033\[[0-9;]*m', '', text)
        if len(visible) > max_width:
            text = text[:max_width - 1] + "…"
    print(text, end="", flush=True)


def draw_header():
    clear_screen()
    title = f"{BOLD}{WHITE}  ⚡ Rapid-MLX vs Ollama — Same Model, Same Prompt{RESET}"
    print_at(1, 1, title)
    print_at(2, 1, f"{DIM}  Model: Qwen3.5-9B · Prompt: \"{PROMPT[:50]}…\"{RESET}")
    print_at(3, 1, f"  {'─' * COL_WIDTH}{DIVIDER}{'─' * COL_WIDTH}")

    # Column headers
    e1, e2 = ENGINES[0], ENGINES[1]
    print_at(4, 1, f"  {e1['color']}{BOLD}{e1['name']}{RESET}")
    print_at(4, COL_WIDTH + 4, f"{e2['color']}{BOLD}{e2['name']}{RESET}")
    print_at(5, 1, f"  {'─' * COL_WIDTH}{DIVIDER}{'─' * COL_WIDTH}")


class StreamState:
    def __init__(self, col_start, color, start_row=6):
        self.col_start = col_start
        self.color = color
        self.start_row = start_row
        self.tokens = 0
        self.text = ""
        self.t0 = None
        self.ttft = None
        self.elapsed = 0
        self.lines = []
        self.done = False

    def count_hidden_token(self):
        """Count a reasoning/thinking token for speed calculation without displaying it."""
        if self.t0 is None:
            self.t0 = time.monotonic()
        self.tokens += 1
        self.elapsed = time.monotonic() - self.t0
        self._render_status_only()

    def _render_status_only(self):
        """Update just the status line without redrawing text."""
        max_rows = 18
        status_row = self.start_row + max_rows + 1
        tok_s = self.tokens / self.elapsed if self.elapsed > 0.05 else 0
        if not self.text:
            status = f"{self.color}{tok_s:.0f} tok/s{RESET} {DIM}· thinking...{RESET}"
        else:
            ttft_str = f"{self.ttft:.2f}s" if self.ttft else "..."
            status = f"{self.color}{tok_s:.0f} tok/s{RESET} {DIM}· {self.tokens} tokens · TTFT {ttft_str}{RESET}"
        move_to(status_row, self.col_start)
        print(status + " " * 20, end="", flush=True)

    def add_token(self, token_text):
        if self.t0 is None:
            self.t0 = time.monotonic()
        if self.ttft is None and token_text.strip():
            self.ttft = time.monotonic() - self.t0

        self.tokens += 1
        self.text += token_text
        self.elapsed = time.monotonic() - self.t0

        # Word-wrap into lines
        self._rewrap()
        self._render()

    def _rewrap(self):
        max_w = COL_WIDTH - 2
        self.lines = []
        line = ""
        for ch in self.text:
            if ch == "\n":
                self.lines.append(line)
                line = ""
            else:
                line += ch
                if len(line) >= max_w:
                    self.lines.append(line)
                    line = ""
        self.lines.append(line)

    def _render(self):
        max_rows = 18
        display_lines = self.lines[-max_rows:]
        for i, line in enumerate(display_lines):
            row = self.start_row + i
            move_to(row, self.col_start)
            print(f"{self.color}{line}{RESET}" + " " * (COL_WIDTH - len(line)), end="", flush=True)

        # Clear remaining rows
        for i in range(len(display_lines), max_rows):
            row = self.start_row + i
            move_to(row, self.col_start)
            print(" " * COL_WIDTH, end="")

        # Status line
        status_row = self.start_row + max_rows + 1
        tok_s = self.tokens / self.elapsed if self.elapsed > 0.1 and self.tokens > 3 else 0
        ttft_str = f"{self.ttft:.2f}s" if self.ttft else "..."

        if self.done:
            status = f"{GREEN}{BOLD}{tok_s:.0f} tok/s{RESET} {DIM}· {self.tokens} tokens · TTFT {ttft_str}{RESET}"
        else:
            status = f"{self.color}{tok_s:.0f} tok/s{RESET} {DIM}· {self.tokens} tokens · TTFT {ttft_str}{RESET}"

        move_to(status_row, self.col_start)
        print(status + " " * 20, end="", flush=True)

    def finish(self):
        self.done = True
        if self.t0:
            self.elapsed = time.monotonic() - self.t0
        self._render()


async def stream_engine(session, engine, state):
    """Stream from one engine and update state."""
    if engine["api"] == "ollama":
        # Ollama native API — supports think: false
        payload = {
            "model": engine["model"],
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "think": False,
            "options": {"num_predict": MAX_TOKENS, "temperature": 0.6},
        }
    else:
        # OpenAI-compatible API (no reasoning parser — all tokens as content)
        payload = {
            "model": engine["model"],
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "temperature": 0.6,
        }

    try:
        async with session.post(
            engine["url"],
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if engine["api"] == "ollama":
                # Ollama native: newline-delimited JSON
                async for line in resp.content:
                    text = line.decode().strip()
                    if not text:
                        continue
                    try:
                        chunk = json.loads(text)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            state.add_token(content)
                            await asyncio.sleep(0)
                    except (json.JSONDecodeError, KeyError):
                        pass
            else:
                # OpenAI SSE format
                async for line in resp.content:
                    text = line.decode().strip()
                    if not text.startswith("data: ") or text == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(text[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            state.add_token(content)
                            await asyncio.sleep(0)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
    except Exception as e:
        move_to(28, state.col_start)
        print(f"\033[31mError: {e}{RESET}", end="")

    state.finish()


async def run_race():
    """Run both engines simultaneously."""
    draw_header()

    state_left = StreamState(col_start=3, color=ENGINES[0]["color"])
    state_right = StreamState(col_start=COL_WIDTH + 5, color=ENGINES[1]["color"])

    # Draw divider
    for row in range(5, 28):
        move_to(row, COL_WIDTH + 3)
        print(f"{DIM}{DIVIDER}{RESET}", end="")

    async with aiohttp.ClientSession() as session:
        # Small delay so header renders
        await asyncio.sleep(0.5)

        # Race!
        await asyncio.gather(
            stream_engine(session, ENGINES[0], state_left),
            stream_engine(session, ENGINES[1], state_right),
        )

    # Final summary
    summary_row = 27
    move_to(summary_row, 1, )
    print(f"  {'─' * COL_WIDTH}{DIVIDER}{'─' * COL_WIDTH}")

    left_tps = state_left.tokens / state_left.elapsed if state_left.elapsed > 0 else 0
    right_tps = state_right.tokens / state_right.elapsed if state_right.elapsed > 0 else 0

    if left_tps > 0 and right_tps > 0:
        speedup = left_tps / right_tps
        move_to(summary_row + 2, 1)
        winner = ENGINES[0]["name"] if speedup > 1 else ENGINES[1]["name"]
        ratio = speedup if speedup > 1 else 1 / speedup
        print(f"  {GREEN}{BOLD}⚡ {winner} is {ratio:.1f}x faster{RESET}")

    move_to(summary_row + 3, 1)
    print(f"  {DIM}github.com/raullenchai/Rapid-MLX{RESET}")
    move_to(summary_row + 4, 1)
    print()


async def check_engines():
    """Verify both engines are reachable."""
    async with aiohttp.ClientSession() as session:
        for engine in ENGINES:
            if engine["api"] == "ollama":
                check_url = "http://localhost:11434/api/tags"
            else:
                check_url = engine["url"].rsplit("/", 2)[0] + "/models"
            try:
                async with session.get(
                    check_url,
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        print(f"  ✓ {engine['name']} OK")
                    else:
                        print(f"  ✗ {engine['name']} returned {resp.status}")
                        return False
            except Exception:
                print(f"  ✗ {engine['name']} not reachable at {check_url}")
                return False
    return True


async def main():
    print(f"\n{BOLD}Checking engines...{RESET}")
    if not await check_engines():
        print(f"\n{BOLD}Please start both engines:{RESET}")
        print("  1. rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --port 8000")
        print("  2. ollama serve  (should already be running)")
        print("  3. ollama pull qwen3.5:9b")
        sys.exit(1)

    # Warmup both engines (primes cache, JIT, etc.)
    print(f"\n{BOLD}Warming up engines...{RESET}")
    async with aiohttp.ClientSession() as session:
        warmup_tasks = []
        for engine in ENGINES:
            if engine["api"] == "ollama":
                payload = {
                    "model": engine["model"],
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                    "think": False,
                    "options": {"num_predict": 5},
                }
            else:
                payload = {
                    "model": engine["model"],
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                    "stream": False,
                }
            warmup_tasks.append(session.post(engine["url"], json=payload))
        responses = await asyncio.gather(*warmup_tasks, return_exceptions=True)
        for r in responses:
            if not isinstance(r, Exception):
                await r.read()
                r.close()
    print(f"  ✓ Both engines warmed up")

    print(f"\n{BOLD}Starting race in 2 seconds...{RESET}")
    await asyncio.sleep(2)
    await run_race()


if __name__ == "__main__":
    asyncio.run(main())
