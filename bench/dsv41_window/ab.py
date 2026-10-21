"""Relative A/B harness for DeepSeek V4.1 speed branches on a truncated checkpoint.

usage: ab.py MODEL_DIR LABEL OUT.jsonl PHASE [PHASE...]
phases: decode prefill batched draft
Imports rapid_mlx from PYTHONPATH (one source tree per run).
"""
import io
import json
import os
import statistics
import sys
import time

os.environ.setdefault("DSV41_LAZY", "0")
import mlx.core as mx

path, label, out_path, *phases = sys.argv[1:]
out = open(out_path, "a")
import rapid_mlx

SRC = os.path.dirname(os.path.dirname(rapid_mlx.__file__))


def emit(**row):
    row = dict(label=label, src=SRC, device=str(mx.default_device()), **row)
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n")
    out.flush()


FILL = ("The history of cartography spans many civilizations, from clay tablets in "
        "Mesopotamia to portolan charts of medieval sailors and satellite surveys. ")


def prompt_ids(tok, n, seed=0):
    ids = tok.encode(f"Request {seed}. " + FILL * (n // 20 + 8), add_special_tokens=False)
    return ids[:n]


def median_ms(fn, warm, reps):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t = time.perf_counter()
        fn()
        mx.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), ts


def count_nodes(*arrays):
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrays)
    text = buf.getvalue()
    return sum(1 for line in text.splitlines() if "label" in line and "->" not in line)


def load_target():
    from rapid_mlx.models.deepseek_v41_native.load import load, supports_engram_ssd_offload
    from rapid_mlx.models.deepseek_v41_native.serving import install_target_qmv
    from transformers import PreTrainedTokenizerFast

    t = time.time()
    model, args = load(path, lazy=False, engram_ssd_offload=supports_engram_ssd_offload(path))
    from rapid_mlx.models.deepseek_v41_native import serving as _serving
    # each tree's own product interval (40 before the async-boundary branch)
    model.eval_interval = int(
        os.environ.get("EVAL_INTERVAL") or getattr(_serving, "TARGET_EVAL_INTERVAL", 40)
    )
    install_target_qmv(model)
    tok = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
    emit(phase="load", kind="target", load_s=round(time.time() - t, 1), layers=len(model.layers), eval_interval=model.eval_interval)
    return model, tok


def prefill_cache(model, ids, extra=96):
    cache = model.make_cache(max_seq_len=len(ids) + extra)
    logits = None
    for s in range(0, len(ids), 512):
        logits = model(mx.array([ids[s:s + 512]]), cache, last_logit_only=True)
        mx.eval(logits)
    return cache, int(mx.argmax(logits[:, -1]).item())


DECODE_CTX = int(os.environ.get("DECODE_CTX", "1024"))


def phase_decode(model, tok):
    ids = prompt_ids(tok, DECODE_CTX)
    cache, nxt = prefill_cache(model, ids, extra=256)
    state = {"tok": nxt}

    def step():
        logits = model(mx.array([[state["tok"]]]), cache, last_logit_only=True)
        y = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(y)
        state["tok"] = int(y.item())

    med, ts = median_ms(step, 10, 150)
    logits = model(mx.array([[state["tok"]]]), cache, last_logit_only=True)
    nodes = count_nodes(logits)
    mx.eval(logits)
    h = mx.random.normal((1, 1, model.args.dim)).astype(mx.bfloat16)
    head_ms, _ = median_ms(lambda: mx.eval(model.head(h.astype(mx.float32))), 3, 20)
    n = len(model.layers)
    emit(phase="decode", ctx=DECODE_CTX, step_ms=round(med, 2), step_ms_p10=round(sorted(ts)[len(ts) // 10], 2),
         head_ms=round(head_ms, 2), per_layer_ms=round((med - head_ms) / n, 2),
         graph_nodes=nodes, nodes_per_layer=round(nodes / n, 1))


def phase_tokens(model, tok):
    ids = prompt_ids(tok, DECODE_CTX, seed=11)
    cache, nxt = prefill_cache(model, ids, extra=96)
    out = [nxt]
    for _ in range(63):
        logits = model(mx.array([[out[-1]]]), cache, last_logit_only=True)
        out.append(int(mx.argmax(logits[:, -1], axis=-1).item()))
    emit(phase="tokens", ctx=DECODE_CTX, n=len(out), tokens=out)



def phase_tf(model, tok):
    """Teacher-forced decode: feed the prompt's own continuation, save per-step logits."""
    import numpy as np
    ids = prompt_ids(tok, DECODE_CTX + 64, seed=11)
    cache, _ = prefill_cache(model, ids[:DECODE_CTX], extra=96)
    rows = []
    for t in ids[DECODE_CTX:]:
        logits = model(mx.array([[t]]), cache, last_logit_only=True)
        rows.append(np.array(logits[0, -1].astype(mx.float32)))
    os.makedirs(os.path.join(os.path.dirname(out_path), "tf"), exist_ok=True)
    np.save(os.path.join(os.path.dirname(out_path), "tf", f"{label}.npy"), np.stack(rows))
    emit(phase="tf", ctx=DECODE_CTX, n=len(rows))

def phase_stream_prefill(model, tok):
    """TTFT through serving.stream_generate with a no-op drafter (target-side prefill only)."""
    from types import SimpleNamespace

    from rapid_mlx.models.deepseek_v41_native import serving

    class StubDrafter:
        position = -1

        def reset(self):
            self.position = -1

        def observe(self, hidden, position):
            mx.eval(hidden)
            self.position = position

        def observe_many(self, hidden, position):
            mx.eval(hidden)
            self.position = position + hidden.shape[1] - 1

        def propose(self, seed):
            return [seed], None

    drafter = StubDrafter()
    runtime = SimpleNamespace(drafter=drafter, reset=drafter.reset)
    for n in (1024, 4096):
        ids = prompt_ids(tok, n, seed=7 + n)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _chunk in serving.stream_generate(model, tok, ids, runtime=runtime, max_tokens=1, ignore_eos=True):
                break
            ts.append(time.perf_counter() - t0)
        emit(phase="stream_prefill", tokens=len(ids), ttft_s=round(min(ts), 3), prefill_tok_s=round(len(ids) / min(ts), 1))


def phase_prefill(model, tok):
    for n in (1024, 4096):
        ids = prompt_ids(tok, n, seed=n)

        def run():
            prefill_cache(model, ids, extra=8)

        med, ts = median_ms(run, 1, 3)
        emit(phase="prefill_raw", tokens=len(ids), ms=round(med, 1), tok_s=round(len(ids) / med * 1e3, 1))


def phase_batched(model, tok):
    try:
        from rapid_mlx.models.deepseek_v41_native.batch_scheduler import BatchBackend
    except ImportError:
        emit(phase="batched", status="n/a (no batch scheduler in this tree)")
        return
    for lanes in (1, 4, 8):
        backend = BatchBackend(model, tok, lanes=lanes if lanes in (1, 2, 4, 8) else 1)
        try:
            prompts = [prompt_ids(tok, 1024, seed=100 + i) for i in range(lanes)]
            t0 = time.perf_counter()
            streams = [backend.submit(p, max_tokens=64) for p in prompts]
            firsts, counts = [None] * lanes, [0] * lanes
            import threading

            def drain(i):
                for _ in streams[i]:
                    if firsts[i] is None:
                        firsts[i] = time.perf_counter() - t0
                    counts[i] += 1

            threads = [threading.Thread(target=drain, args=(i,)) for i in range(lanes)]
            [th.start() for th in threads]
            [th.join() for th in threads]
            wall = time.perf_counter() - t0
            ttft = sorted(firsts)
            emit(phase="batched", lanes=lanes, requests=lanes, tokens=sum(counts),
                 wall_s=round(wall, 2), aggregate_tok_s=round(sum(counts) / wall, 2),
                 ttft_p50_s=round(ttft[len(ttft) // 2], 2), ttft_max_s=round(ttft[-1], 2))
        finally:
            backend.close()


def phase_draft():
    from rapid_mlx.models.deepseek_v41_native import serving

    from rapid_mlx.models.deepseek_v41_native import dspark as _dspark

    full = os.environ.get("FULL") == "1"
    if not full:
        _pin = _dspark.DSparkWeights.pin_mtp
        # 36 GB host: the 8 GB product reserve leaves no room beside a 4-layer target.
        _dspark.DSparkWeights.pin_mtp = lambda self, reserve_gb=1.0: _pin(self, reserve_gb)
    mtp_path = os.environ.get("DRAFT_MTP") or os.path.join(path, "dspark")
    t = time.time()
    try:
        model, tok, runtime = serving.load_product_runtime(
            path, mtp_path,
            target_revision=os.environ.get("TARGET_REV", "trunc"),
            mtp_revision=os.environ.get("MTP_REV", "trunc"),
            mixed_target_precision=True)
    except Exception as exc:  # truncated checkpoint may not satisfy draft validation
        emit(phase="draft", status=f"n/a: {type(exc).__name__}: {str(exc)[:200]}")
        return None
    emit(phase="load", kind="runtime", load_s=round(time.time() - t, 1))
    rounds = {"n": 0, "accepted": 0}
    orig = serving._match

    def counting(candidate, *a, **k):
        res = orig(candidate, *a, **k)
        rounds["n"] += 1
        rounds["accepted"] += len(res[0])
        return res

    serving._match = counting
    for n in (1024, 4096):
        ids = prompt_ids(tok, n, seed=7 + n)
        ts = []
        for _ in range(2):
            t0 = time.perf_counter()
            for _chunk in serving.stream_generate(model, tok, ids, runtime=runtime, max_tokens=1, ignore_eos=True):
                break
            ts.append(time.perf_counter() - t0)
        emit(phase="ttft_stream", tokens=len(ids), ttft_s=round(min(ts), 2), prefill_tok_s=round(len(ids) / min(ts), 1))
    ids = prompt_ids(tok, 1024, seed=3)
    rounds.update(n=0, accepted=0)
    t0 = time.perf_counter()
    first = None
    count = 0
    for _chunk in serving.stream_generate(model, tok, ids, runtime=runtime, max_tokens=160, ignore_eos=True):
        count += 1
        if first is None:
            first = time.perf_counter()
    end = time.perf_counter()
    emit(phase="draft", tokens=count, decode_tok_s=round((count - 1) / (end - first), 2),
         rounds=rounds["n"], mean_tokens_per_round=round(rounds["accepted"] / max(rounds["n"], 1), 3),
         note=("full model" if full else "truncated 4-layer target: acceptance is not representative of the full model"))
    return model


if any(p in phases for p in ("decode", "prefill", "batched", "tokens", "tf", "stream_prefill")):
    model, tok = load_target()
    if "decode" in phases:
        phase_decode(model, tok)
    if "prefill" in phases:
        phase_prefill(model, tok)
    if "tokens" in phases:
        phase_tokens(model, tok)
    if "tf" in phases:
        phase_tf(model, tok)
    if "stream_prefill" in phases:
        phase_stream_prefill(model, tok)
    if "batched" in phases:
        phase_batched(model, tok)
    del model
    import gc

    gc.collect()
    mx.clear_cache()
if "draft" in phases:
    phase_draft()
