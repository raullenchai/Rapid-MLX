#!/usr/bin/env python3
"""Side-by-side: autoregressive baseline vs DFlash speculative decoding.

Records a visual terminal demo of DFlash speedup for social/launch.
Runs sequentially (baseline first, then DFlash) — parallel runs cause
GPU contention on shared Metal device and hide the speedup.

Setup (two servers, same alias, different ports):

    # terminal 1 — baseline
    rapid-mlx serve qwen3.5-27b-8bit --port 8000

    # terminal 2 — DFlash
    pip install 'rapid-mlx[dflash]'
    rapid-mlx serve qwen3.5-27b-8bit --enable-dflash --port 8001

    # terminal 3 — run the demo
    python3.12 scripts/demo_dflash.py
"""

import asyncio
import json
import sys
import time

import aiohttp

PROMPT = (
    "Write a one-line Python function that returns the n-th Fibonacci number "
    "using recursion. Add a one-line docstring. No tests. /no_think"
)
MAX_TOKENS = 120

BASELINE_URL = "http://localhost:8000/v1/chat/completions"
DFLASH_URL = "http://localhost:8001/v1/chat/completions"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[38;5;82m"
ORANGE = "\033[38;5;208m"
GRAY = "\033[38;5;245m"
WHITE = "\033[38;5;255m"

COL_WIDTH = 58
DIVIDER = "│"


def clear_screen():
    print("\033[2J\033[H", end="")


def move_to(row, col):
    print(f"\033[{row};{col}H", end="")


def print_at(row, col, text):
    move_to(row, col)
    print(text, end="", flush=True)


def draw_chrome():
    clear_screen()
    print_at(1, 1, f"{BOLD}{WHITE}  ⚡ Qwen3.5-27B-8bit · same prompt · DFlash on/off{RESET}")
    print_at(2, 1, f"{DIM}  Block-diffusion drafter via mlx-vlm · z-lab/Qwen3.5-27B-DFlash{RESET}")
    print_at(3, 1, f"  {'─' * COL_WIDTH}{DIVIDER}{'─' * COL_WIDTH}")
    print_at(4, 3, f"{GRAY}{BOLD}Baseline (autoregressive){RESET}")
    print_at(4, COL_WIDTH + 5, f"{ORANGE}{BOLD}DFlash speculative decoding{RESET}")
    print_at(5, 1, f"  {'─' * COL_WIDTH}{DIVIDER}{'─' * COL_WIDTH}")
    for row in range(5, 28):
        move_to(row, COL_WIDTH + 3)
        print(f"{DIM}{DIVIDER}{RESET}", end="")


class Panel:
    def __init__(self, col_start, color):
        self.col_start = col_start
        self.color = color
        self.start_row = 6
        self.tokens = 0
        self.text = ""
        self.t0 = None
        self.ttft = None
        self.elapsed = 0.0
        self.lines = []
        self.done = False

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
        for i in range(len(display_lines), max_rows):
            row = self.start_row + i
            move_to(row, self.col_start)
            print(" " * COL_WIDTH, end="")

        status_row = self.start_row + max_rows + 1
        tok_s = self.tokens / self.elapsed if self.elapsed > 0.1 and self.tokens > 3 else 0
        ttft_str = f"{self.ttft:.2f}s" if self.ttft else "..."
        color = GREEN if self.done else self.color
        weight = BOLD if self.done else ""
        status = f"{color}{weight}{tok_s:.0f} tok/s{RESET} {DIM}· {self.tokens} tokens · TTFT {ttft_str}{RESET}"
        move_to(status_row, self.col_start)
        print(status + " " * 25, end="", flush=True)

    def show_pending(self):
        status_row = self.start_row + 19
        move_to(status_row, self.col_start)
        print(f"{DIM}waiting...{RESET}" + " " * 25, end="", flush=True)

    def add_token(self, token_text):
        if self.t0 is None:
            self.t0 = time.monotonic()
        if self.ttft is None and token_text.strip():
            self.ttft = time.monotonic() - self.t0
        self.tokens += 1
        self.text += token_text
        self.elapsed = time.monotonic() - self.t0
        self._rewrap()
        self._render()

    def finish(self):
        self.done = True
        if self.t0:
            self.elapsed = time.monotonic() - self.t0
        self._render()


async def stream(session, url, panel):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=180)
        ) as resp:
            async for line in resp.content:
                text = line.decode().strip()
                if not text.startswith("data: ") or text == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(text[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        panel.add_token(content)
                        await asyncio.sleep(0)
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
    except Exception as e:
        move_to(28, panel.col_start)
        print(f"\033[31mError: {e}{RESET}", end="")
    panel.finish()


async def run():
    draw_chrome()
    baseline = Panel(col_start=3, color=GRAY)
    dflash = Panel(col_start=COL_WIDTH + 5, color=ORANGE)
    dflash.show_pending()

    async with aiohttp.ClientSession() as session:
        await asyncio.sleep(0.3)
        await stream(session, BASELINE_URL, baseline)
        await asyncio.sleep(0.5)
        await stream(session, DFLASH_URL, dflash)

    final_row = 27
    if baseline.tokens > 0 and dflash.tokens > 0 and baseline.elapsed > 0:
        tps_b = baseline.tokens / baseline.elapsed
        tps_d = dflash.tokens / dflash.elapsed
        speedup = tps_d / tps_b if tps_b else 0
        msg = (
            f"  {BOLD}{GREEN}⚡ DFlash speedup: {speedup:.2f}×{RESET}  "
            f"{DIM}({tps_b:.0f} → {tps_d:.0f} tok/s){RESET}"
        )
        print_at(final_row, 1, msg)
    print_at(final_row + 2, 1, "")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print(f"\n{RESET}Interrupted.")
        sys.exit(130)
