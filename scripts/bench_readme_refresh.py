#!/usr/bin/env python3.12
# SPDX-License-Identifier: Apache-2.0
"""README benchmark refresh — B=4 concurrent throughput sweep.

Re-measures the headline numbers shown in README.md for the post-v0.6.80
codebase against current upstream engines.

Workload (per model, per engine):
  1 warmup request (sequential, discarded)
  3 measured rounds of B=4 concurrent streaming requests
  Aggregate metric: sum(output_tokens) / wall_clock  (tok/s, higher = better)

Engines:
  rapid-mlx           — this project
  mlx-lm serve        — same MLX weights, upstream Apple
  ollama              — closest GGUF arch; arch_note flagged when mismatched

Each engine is the ONLY one running at a time (Metal contention destroys
otherwise-clean numbers on the same Mac). An 8 s cooldown is inserted
between engine swaps (see ``COOLDOWN_S``) to let MTL free unified-memory
buffers.

Usage:
    python3.12 scripts/bench_readme_refresh.py                     # full sweep
    python3.12 scripts/bench_readme_refresh.py --models qwen3.5-4b # one model
    python3.12 scripts/bench_readme_refresh.py --engines rapid-mlx,mlx-lm
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "reports" / "benchmarks" / "readme-refresh"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "You are a senior software engineer writing a blog post. "
    "Explain the difference between threads and processes in operating systems, "
    "covering address space, scheduling, context switch cost, IPC, and a worked "
    "example of when each is appropriate. Be concrete and specific."
)
MAX_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.95
DEFAULT_CONCURRENCY = 4
ROUNDS = 3
SERVER_READY_TIMEOUT = 600
KILL_TIMEOUT = 30
COOLDOWN_S = 8


# ============================================================
# Model registry: (alias, mlx_hf_path, ollama_tag, ollama_note)
# ============================================================


@dataclass
class ModelSpec:
    alias: str
    mlx_path: str
    ollama_tag: str | None
    ollama_note: str  # "same arch" or "closest available — Qwen3 vs Qwen3.5" etc.


MODELS: list[ModelSpec] = [
    ModelSpec(
        "qwen3.5-4b",
        "mlx-community/Qwen3.5-4B-MLX-4bit",
        "qwen3:4b",
        "Ollama Qwen3 (not Qwen3.5; DeltaNet arch unavailable on llama.cpp)",
    ),
    ModelSpec(
        "qwen3.5-9b",
        "mlx-community/Qwen3.5-9B-4bit",
        "qwen3:8b",
        "Ollama Qwen3 8B (not Qwen3.5 9B; closest available)",
    ),
    ModelSpec(
        "qwen3.5-27b",
        "mlx-community/Qwen3.5-27B-4bit",
        "qwen3:32b",
        "Ollama Qwen3 32B Q4_K_M (closest dense 27-32B; Qwen3.5 DeltaNet not on llama.cpp; Unsloth Qwen3.6-27B GGUF fails to load in Ollama 0.24)",
    ),
    ModelSpec(
        "gemma-4-12b",
        "mlx-community/gemma-4-12B-it-4bit",
        "gemma3:12b",
        "Ollama Gemma 3 12B (Gemma 4 not yet on llama.cpp)",
    ),
    ModelSpec(
        "gpt-oss-20b", "mlx-community/gpt-oss-20b-MXFP4-Q8", "gpt-oss:20b", "Same arch"
    ),
    ModelSpec(
        "qwen3.6-35b",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "qwen3:30b-a3b",
        "Ollama Qwen3 30B-A3B (not Qwen3.6; closest MoE A3B)",
    ),
    ModelSpec(
        "qwen3.5-35b",
        "mlx-community/Qwen3.5-35B-A3B-8bit",
        "qwen3:30b-a3b",
        "Ollama Qwen3 30B-A3B 4bit (not Qwen3.5-35B 8bit; closest MoE)",
    ),
]


# ============================================================
# Engine adapters
# ============================================================


class Engine:
    name: str
    port: int
    process: subprocess.Popen | None = None

    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self.process = None

    def start(self, model: ModelSpec) -> None:
        raise NotImplementedError

    def model_id(self, model: ModelSpec) -> str:
        raise NotImplementedError

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=KILL_TIMEOUT)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    self.process.wait(timeout=5)
            except (ProcessLookupError, PermissionError):
                pass
        self.process = None

    def ready_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/models"

    def chat_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/chat/completions"

    def expected_model_id(self) -> str | None:
        """The model id this engine claims to be serving, for cross-check.

        Overridden per subclass once `start(model)` has set internal state.
        Returning None disables the cross-check (Ollama daemon serves many
        tags from one process, so we verify presence via `ollama list`
        rather than `/v1/models`).
        """
        return None

    def wait_ready(self) -> None:
        deadline = time.time() + SERVER_READY_TIMEOUT
        last_err = "no response"
        while time.time() < deadline:
            # Codex round 1 BLOCKING #2 — process can crash mid-boot while
            # the port stays occupied by a stale server. Honour the actual
            # subprocess state before trusting any HTTP probe.
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.name} process exited during startup "
                    f"(returncode={self.process.returncode})"
                )
            try:
                r = requests.get(self.ready_url(), timeout=2)
                if r.status_code == 200:
                    expected = self.expected_model_id()
                    if expected is None:
                        return
                    # Validate the right model is loaded — a stale server
                    # left on the same port from a prior alias would
                    # silently pollute the new alias's numbers otherwise.
                    try:
                        body = r.json()
                    except ValueError:
                        last_err = "non-JSON /v1/models body"
                    else:
                        served = {
                            item.get("id")
                            for item in (body.get("data") or [])
                            if isinstance(item, dict)
                        }
                        if expected in served:
                            return
                        last_err = (
                            f"/v1/models reports {sorted(served)!r}, "
                            f"expected {expected!r}"
                        )
                else:
                    last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = repr(e)
            time.sleep(1.0)
        raise RuntimeError(
            f"{self.name} not ready in {SERVER_READY_TIMEOUT}s: {last_err}"
        )


class RapidMLXEngine(Engine):
    _expected: str | None = None

    def start(self, model: ModelSpec) -> None:
        env = os.environ.copy()
        env["RAPID_MLX_TELEMETRY"] = "0"
        self.process = subprocess.Popen(
            [
                "python3.12",
                "-m",
                "vllm_mlx.cli",
                "serve",
                model.mlx_path,
                "--port",
                str(self.port),
                "--no-thinking",
            ],
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self._expected = model.mlx_path

    def expected_model_id(self) -> str | None:
        return self._expected

    def model_id(self, model: ModelSpec) -> str:
        return model.mlx_path


class MlxLmEngine(Engine):
    _expected: str | None = None

    def start(self, model: ModelSpec) -> None:
        self.process = subprocess.Popen(
            [
                "python3.12",
                "-m",
                "mlx_lm",
                "server",
                "--model",
                model.mlx_path,
                "--port",
                str(self.port),
            ],
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._expected = model.mlx_path

    def expected_model_id(self) -> str | None:
        return self._expected

    def model_id(self, model: ModelSpec) -> str:
        return model.mlx_path


class OllamaEngine(Engine):
    """Reuses the system ollama daemon (port 11434).

    Ollama doesn't support multi-instance on alt ports cleanly. We point
    to the system daemon and use the requested tag in `model` field.
    Start = noop (system service is already running); we just verify
    the tag exists in `ollama list`.
    """

    _benched_tag: str | None = None

    def __init__(self, name: str, port: int):
        super().__init__(name, port=11434)  # ignore caller port

    def start(self, model: ModelSpec) -> None:
        if model.ollama_tag is None:
            raise RuntimeError(f"{model.alias} has no Ollama mapping")
        # Codex round 2 BLOCKING #1 — substring match on the family name
        # (``qwen3``) would happily green-light any ``qwen3:*`` tag,
        # silently swapping in the wrong comparator. Parse rows from
        # ``ollama list`` (first whitespace-delimited column) and require
        # the exact ``name:tag`` literal.
        r = subprocess.run(
            ["ollama", "list"], check=True, capture_output=True, text=True
        )
        installed = set()
        for line in r.stdout.splitlines()[1:]:  # skip header row
            head = line.split()
            if head:
                installed.add(head[0])
        if model.ollama_tag not in installed:
            raise RuntimeError(
                f"ollama tag {model.ollama_tag!r} not in `ollama list` "
                f"(found {sorted(installed)!r}). Run: "
                f"ollama pull {model.ollama_tag}"
            )
        # No process to spawn — Ollama runs as a launchd service.
        self.process = None
        self._benched_tag = model.ollama_tag

    def stop(self) -> None:
        # Codex round 1 BLOCKING #4 — only unload the specific tag we
        # benched, never `ollama stop all` (would evict the user's own
        # in-flight LM Studio / chat work). Surface failures instead of
        # swallowing them, since a stuck model holds Metal buffers that
        # poison the next engine's numbers.
        tag = self._benched_tag
        if tag is None:
            return
        try:
            r = subprocess.run(
                ["ollama", "stop", tag],
                check=False,
                timeout=10,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(
                    f"    [ollama stop {tag}] non-zero exit "
                    f"({r.returncode}): {(r.stderr or r.stdout).strip()}",
                    flush=True,
                )
        except subprocess.TimeoutExpired:
            print(f"    [ollama stop {tag}] timed out after 10s", flush=True)
        finally:
            self._benched_tag = None

    def model_id(self, model: ModelSpec) -> str:
        return model.ollama_tag or ""

    def expected_model_id(self) -> str | None:
        # Daemon serves many tags from one process; we already verified
        # the tag via `ollama list` in start(). /v1/models cross-check
        # is a noop here.
        return None

    def wait_ready(self) -> None:
        deadline = time.time() + 20
        last_err = "no response"
        while time.time() < deadline:
            try:
                r = requests.get(self.ready_url(), timeout=2)
                if r.status_code == 200:
                    return
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = repr(e)
            time.sleep(0.5)
        raise RuntimeError(f"ollama daemon not ready: {last_err}")


ENGINE_FACTORIES: dict[str, type[Engine]] = {
    "rapid-mlx": RapidMLXEngine,
    "mlx-lm": MlxLmEngine,
    "ollama": OllamaEngine,
}

ENGINE_PORTS: dict[str, int] = {
    "rapid-mlx": 8101,
    "mlx-lm": 8102,
    "ollama": 11434,
}


# ============================================================
# Workload
# ============================================================


@dataclass
class RoundResult:
    wall_s: float
    total_output_tokens: int
    aggregate_tps: float
    per_request_tps: list[float] = field(default_factory=list)


@dataclass
class ModelEngineResult:
    model: str
    engine: str
    arch_note: str
    rounds: list[RoundResult] = field(default_factory=list)
    error: str | None = None

    def median_aggregate_tps(self) -> float | None:
        vals = [r.aggregate_tps for r in self.rounds]
        return statistics.median(vals) if vals else None

    def median_per_stream_tps(self) -> float | None:
        flat = [tps for r in self.rounds for tps in r.per_request_tps]
        return statistics.median(flat) if flat else None


def make_payload(model_id: str, stream: bool) -> dict:
    # `chat_template_kwargs.enable_thinking=False` is the standard hook
    # for Qwen3.x on rapid-mlx / mlx-lm / mlx-vlm. Ollama 0.24 ignores
    # this OpenAI-compat extension and keeps emitting `delta.reasoning`
    # chunks (= the chain-of-thought stream), but those chunks come out
    # at the same model decode rate as content tokens, so counting them
    # is the honest throughput measurement. We don't claim "thinking off"
    # in headlines — see README + run_one_stream() comment.
    payload: dict = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        # Force Ollama (and any other server that defaults usage off) to
        # emit a streaming `usage` chunk so we measure real tokens, not
        # transport frames.
        payload["stream_options"] = {"include_usage": True}
    return payload


def run_one_stream(chat_url: str, model_id: str) -> tuple[float, int, float]:
    """Single streaming request. Returns (e2e_s, output_tokens, decode_tps)."""
    payload = make_payload(model_id, stream=True)
    t0 = time.perf_counter()
    ttft = None
    output_tokens = 0
    chunks_seen = 0
    with requests.post(chat_url, json=payload, stream=True, timeout=300) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        for line in resp.iter_lines():
            if not line:
                continue
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                if obj.get("usage"):
                    usage = obj["usage"]
                    if isinstance(usage, dict) and usage.get("completion_tokens"):
                        output_tokens = usage["completion_tokens"]
                continue
            delta = choices[0].get("delta") or {}
            # Reasoning routing varies across engines: rapid-mlx and the
            # OpenAI spec use `delta.reasoning_content`; Ollama 0.24
            # emits `delta.reasoning`; some servers stuff thinking back
            # into `delta.content`. They all exit the model at the same
            # decode rate, so for a throughput benchmark we count any of
            # them as a generated chunk — and ALSO trust the streaming
            # `usage` chunk for the authoritative token count.
            content = (
                (delta.get("content") or "")
                + (delta.get("reasoning_content") or "")
                + (delta.get("reasoning") or "")
            )
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks_seen += 1
            if obj.get("usage"):
                usage = obj["usage"]
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    output_tokens = usage["completion_tokens"]
        # Codex round 1 BLOCKING #1 — SSE chunks are transport frames,
        # not tokens. Servers can collapse multiple tokens into one
        # chunk or split one token across two, so silently substituting
        # chunks_seen for the authoritative completion_tokens count
        # publishes a wrong tok/s while the run still "succeeds".
        if output_tokens == 0:
            raise RuntimeError(
                "stream finished without a usage chunk; rerun with "
                "`stream_options.include_usage=true` support enabled "
                "on the server, or capture completion_tokens out-of-band. "
                f"(saw {chunks_seen} content/reasoning chunk(s))"
            )
    e2e = time.perf_counter() - t0
    if ttft is None:
        raise RuntimeError("no content chunk")
    decode = max(e2e - ttft, 1e-6)
    return e2e, output_tokens, output_tokens / decode


def run_concurrent_round(chat_url: str, model_id: str, concurrency: int) -> RoundResult:
    t0 = time.perf_counter()
    per_request = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(run_one_stream, chat_url, model_id) for _ in range(concurrency)
        ]
        for f in concurrent.futures.as_completed(futures):
            per_request.append(f.result())
    wall = time.perf_counter() - t0
    total_tokens = sum(t[1] for t in per_request)
    return RoundResult(
        wall_s=wall,
        total_output_tokens=total_tokens,
        aggregate_tps=total_tokens / max(wall, 1e-6),
        per_request_tps=[t[2] for t in per_request],
    )


_HOME_PREFIX = str(Path.home())


def _sanitize_error(msg: str) -> str:
    """Redact the operator's home path from error strings before we
    write them into a committed JSON artifact (codex round 2 NIT #4).
    Diagnostic content (HTTP status, error type, blob suffix) is kept;
    just the personally-identifying prefix is stripped."""
    return msg.replace(_HOME_PREFIX, "<redacted-home>")


def bench_model_engine(
    model: ModelSpec, engine_name: str, concurrency: int
) -> ModelEngineResult:
    print(f"\n  [{engine_name}] starting {model.alias}…", flush=True)
    res = ModelEngineResult(
        model=model.alias,
        engine=engine_name,
        arch_note=model.ollama_note if engine_name == "ollama" else "Same MLX weights",
    )
    engine_cls = ENGINE_FACTORIES[engine_name]
    engine = engine_cls(engine_name, port=ENGINE_PORTS[engine_name])
    try:
        engine.start(model)
        engine.wait_ready()
        print("    ready", flush=True)

        # Warmup
        try:
            run_one_stream(engine.chat_url(), engine.model_id(model))
            print("    warmup ok", flush=True)
        except Exception as e:
            print(f"    warmup FAIL: {e}", flush=True)
            res.error = _sanitize_error(f"warmup: {e}")
            return res

        for i in range(ROUNDS):
            print(f"    round {i + 1}/{ROUNDS}…", flush=True, end=" ")
            try:
                r = run_concurrent_round(
                    engine.chat_url(), engine.model_id(model), concurrency
                )
                res.rounds.append(r)
                print(
                    f"agg={r.aggregate_tps:.1f}tok/s wall={r.wall_s:.1f}s", flush=True
                )
            except Exception as e:
                print(f"FAIL: {e}", flush=True)
                res.error = _sanitize_error(f"round {i + 1}: {e}")
                break
            time.sleep(2)
    except Exception as e:
        res.error = _sanitize_error(f"setup: {e}")
        print(f"    SETUP FAIL: {e}", flush=True)
    finally:
        engine.stop()
        time.sleep(COOLDOWN_S)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(m.alias for m in MODELS))
    ap.add_argument("--engines", default="rapid-mlx,mlx-lm,ollama")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    selected_aliases = {m.strip() for m in args.models.split(",") if m.strip()}
    selected_engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    selected_models = [m for m in MODELS if m.alias in selected_aliases]
    if not selected_models:
        sys.exit(f"no models matched: {args.models}")

    print("=== README refresh sweep ===", flush=True)
    print(f"models:  {[m.alias for m in selected_models]}", flush=True)
    print(f"engines: {selected_engines}", flush=True)
    print(
        f"workload: B={args.concurrency}, rounds={ROUNDS}, max_tokens={MAX_TOKENS}",
        flush=True,
    )

    all_results: list[ModelEngineResult] = []
    for model in selected_models:
        print(f"\n=== {model.alias} ({model.mlx_path}) ===", flush=True)
        for engine_name in selected_engines:
            if engine_name == "ollama" and model.ollama_tag is None:
                print(f"  [ollama] skipping {model.alias} — no tag", flush=True)
                continue
            r = bench_model_engine(model, engine_name, args.concurrency)
            all_results.append(r)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = (
        Path(args.output) if args.output else RESULTS_DIR / f"results-{stamp}.json"
    )
    out_path.write_text(
        json.dumps(
            {
                "stamp": stamp,
                "concurrency": args.concurrency,
                "rounds": ROUNDS,
                "max_tokens": MAX_TOKENS,
                "results": [
                    {
                        "model": r.model,
                        "engine": r.engine,
                        "arch_note": r.arch_note,
                        "median_aggregate_tps": r.median_aggregate_tps(),
                        "median_per_stream_tps": r.median_per_stream_tps(),
                        "error": r.error,
                        "rounds": [asdict(rnd) for rnd in r.rounds],
                    }
                    for r in all_results
                ],
            },
            indent=2,
        )
    )
    print(f"\n=== Results saved: {out_path} ===", flush=True)

    # Markdown summary
    print("\n=== Summary ===")
    print(
        f"\n| Model | rapid-mlx (B={args.concurrency} tok/s) | mlx-lm | Ollama | Speedup vs mlx-lm | Speedup vs Ollama |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for model in selected_models:
        cells = [model.alias]
        scores: dict[str, float | None] = {}
        for eng in ["rapid-mlx", "mlx-lm", "ollama"]:
            r = next(
                (x for x in all_results if x.model == model.alias and x.engine == eng),
                None,
            )
            if r is None or r.error:
                scores[eng] = None
                cells.append("—")
            else:
                v = r.median_aggregate_tps()
                scores[eng] = v
                cells.append(f"{v:.1f}" if v is not None else "—")
        rapid = scores.get("rapid-mlx")
        mlxlm = scores.get("mlx-lm")
        oll = scores.get("ollama")
        if rapid and mlxlm:
            cells.append(f"{rapid / mlxlm:.2f}x")
        else:
            cells.append("—")
        if rapid and oll:
            cells.append(f"{rapid / oll:.2f}x")
        else:
            cells.append("—")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
