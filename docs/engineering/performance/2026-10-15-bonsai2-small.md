# Bonsai 2 small model qualification — 2026-10-15

## Scope and environment

Qualify `bonsai2-{2,4,9}b-2bit` on Rapid-MLX's existing native Hadamard
multimodal lane. No drafter integration or speculative decoding is enabled.

- Apple M3 Ultra, 256 GiB unified memory; macOS 26.5.2 (25F84).
- Python 3.12.14; fresh worktree `.venv`, installed with `pip install -e '.[dev]'`.
- Rapid-MLX base `a1cbf627be27295ddd0f954f360851d62e48a4e0` (`origin/main`).
- MLX 0.32.4, mlx-metal 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2,
  transformers 5.15.1, tokenizers 0.22.2, torch 2.14.1.
- Repositories downloaded in full through the default `~/.cache/huggingface`
  cache. No model-snapshot Python is imported or executed.

| Model | HF revision | Layers / width | Tied embeddings | Root weights (bytes) | Catalog footprint (bytes) |
|-------|-------------|----------------|-----------------|----------------------|---------------------------|
| 2B | `1c6cbae21f8c38830ce337972b46fdae7b18eaaa` | 24 / 2048 | Yes | 1,200,284,784 | 1,213,224,808 |
| 4B | `dff2caaa23fa8706164c44c10f1b58feb405f76a` | 32 / 2560 | Yes | 1,871,365,704 | 1,952,250,684 |
| 9B | `f4e28a59531731c41b6dfac88cf271bd58f0a287` | 32 / 4096 | No | 3,462,149,260 | 3,612,063,766 |

The size manifest uses `scripts.gen_model_sizes._size_one` and `_render` for
only the three new repositories, preserving unrelated entries. This counts
weight and tokenizer/config payloads, including optional drafter files, rather
than only the root target weights. All generation configs specify temperature
1.0, top-p 0.95, top-k 20, matching the 27B alias. The image/tool/reasoning,
hybrid, and speculative-decoding flags mirror the existing ordinary 27B alias.
Desktop reads the existing CLI/atomic catalog, so no separate UI list is added.
The architecture-based startup and residency lane guards already apply.

## Reproduction

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for size in (2, 4, 9):
    snapshot_download(f"prism-ml/Ternary-Bonsai-2-{size}B-mlx-2bit")
PY

# Run one server at a time; wait for the lock, and stop only your own server.
lockf -k /var/tmp/rapid-mlx-large-model.lock \
  .venv/bin/rapid-mlx serve bonsai2-2b-2bit \
  --host 127.0.0.1 --port 18762 --disable-prefix-cache --disable-disk-caches
# Repeat with bonsai2-4b-2bit on 18764 and bonsai2-9b-2bit on 18769.
```

Requests use `/v1/chat/completions` with `model` set to the served alias.
Unless specified otherwise: seed 17, max_tokens 128, temperature 0.7,
top_p 0.8, top_k 20, presence_penalty 1.5, and
`chat_template_kwargs: {"enable_thinking": false}`.

| Gate | Request and acceptance |
|------|------------------------|
| Multi-turn | Say “My name is Mira and my favorite number is 17. Reply with a short greeting.” Then include the assistant response and ask “What is my name and favorite number?” Require both Mira and 17. |
| Thinking off | “What is 1 + 1? Answer briefly.” Require final answer 2 and no reasoning content. |
| Thinking on | Same prompt, enable_thinking true, max_tokens 8192, temperature 1.0, top_p 0.95, presence_penalty 1.0. Require separate reasoning and final answer 2. |
| Tool, non-stream | “Use get_weather to check the weather in Paris.” Expose function get_weather with required string city. max_tokens 256. Require exactly one structured call with city Paris. |
| Tool, stream | Same tool request with stream true; assemble tool_calls deltas and parse arguments as JSON. Require one call to get_weather for Paris. |
| Image | Generate a white 224×224 PNG with a red ellipse spanning (40,40) to (184,184). Send a base64 image_url and “What color and shape are shown? Answer briefly.” Require red and circle/circular. |
| Anthropic | POST `/v1/messages` with max_tokens 256, temperature 0.7, thinking type disabled, and “Say hello in one short sentence.” Require a message with a nonempty text block. |

Every OpenAI final-content gate rejects leaked `<think>`, `</think>`,
`<tool_call>`, `</tool_call>`, `<function=`, and `<parameter=` tags.

Benchmark request construction (first request after fresh server startup):

```python
import requests
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

size, port = 2, 18762  # repeat for (4, 18764), (9, 18769)
alias = f"bonsai2-{size}b-2bit"
tokenizer = AutoTokenizer.from_pretrained(snapshot_download(
    f"prism-ml/Ternary-Bonsai-2-{size}B-mlx-2bit", local_files_only=True))
text = ("Continue writing the sequence of positive integers separated by commas. "
        "Do not stop until told. Context: " + "word " * 993)
messages = [{"role": "user", "content": text}]
assert len(tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_dict=False)) == 1024
base = f"http://127.0.0.1:{port}"
response = requests.post(base + "/v1/chat/completions", json={
    "model": alias, "messages": messages, "max_tokens": 128,
    "temperature": 0.0, "presence_penalty": 0.0, "seed": 17,
    "top_p": 0.8, "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}, timeout=360)
response.raise_for_status()
print(response.json()["usage"])
print(requests.get(base + "/v1/status").json())
```

Read `prompt_tps`, `generation_tps`, and `metal.peak_memory_gb` before running
other gates. These are server counters for the first request, with prefix and
disk caches disabled. Peak memory includes loading and that request; it is MLX
allocator peak in decimal GB, not total system memory. Fresh-process first-request timing can include compilation
overhead; this is a single-run qualification, not a steady-state or comparative
benchmark.

## Loader correction and unchanged-model check

All three targets loaded without converting or editing their snapshots.
Before the source fix, 9B answered text and correctly identified the red circle.
Both 2B and 4B returned HTTP 503 (`engine_aborted`) for text and image requests:
the model worker raised `AttributeError: 'Packed' object has no attribute
'as_linear'` at the tied output projection. Their packed input embeddings also
serve as the output head, unlike the untied 9B/27B targets.

`Packed.as_linear` now exposes the existing forward Hadamard activation
transform and affine 2-bit matmul. Ordinary packed linear calls delegate to the
same method. No new loading mechanism, weight conversion, dense head copy,
vision change, or configuration workaround is required. The regression compares
tied-head logits against the effective embedding matrix with and without the
rotation; it failed before this fix and passes afterward.

The full 4B and 9B snapshots include `mtp/config.json` (`qwen3_5_mtp`) and draft
weights. They load cleanly while the loader reads only the root
`model.safetensors`; speculative/native MTP capabilities remain disabled.
A loader regression also places invalid drafter weights in `mtp/` and proves
that they are ignored.

## API gate results

All listed requests returned HTTP 200 after the fix. “Tools” means one call to
`get_weather` with JSON arguments `{"city":"Paris"}` in each mode.

| Model | Multi-turn | Thinking off / on | Tool non-stream | Tool stream | Image | Anthropic |
|-------|------------|-------------------|-----------------|-------------|-------|-----------|
| 2B | Pass | Pass / Pass | Pass | Pass | Pass | Pass |
| 4B | Pass | Pass / Pass | Pass | Pass | Pass | Pass |
| 9B | Pass | Pass / Pass | Pass | Pass | Pass | Pass |

Each model remembered “Mira” and “17” on the second chat turn. Thinking-off
returned `2` without reasoning content; thinking-on returned `2` with separate
reasoning content (285 / 419 / 428 characters for 2B / 4B / 9B). Image answers
were respectively “The image shows a **red** circle.”, “Red, circle”, and
“Red and circle”. Final content contained no raw reasoning or tool markup;
reasoning fields were also checked for leaked tags. Anthropic responses were
ordinary message objects with nonempty text blocks.

## Performance

One request per fresh server, after loading and before the other gates.
Server counters were zero before the request and showed exactly one processed
request afterward. Every response reported exactly 1,024 prompt tokens and
128 completion tokens. Numbers below are the server's instrumented prefill and
decode rates; they are not calculated from HTTP wall time.

| Alias | Prompt / output tokens | Prefill tok/s | Decode tok/s | Peak MLX memory (GB) |
|-------|------------------------|---------------|--------------|----------------------|
| `bonsai2-2b-2bit` | 1024 / 128 | 1076.92 | 222.25 | 3.60 |
| `bonsai2-4b-2bit` | 1024 / 128 | 792.96 | 145.76 | 4.64 |
| `bonsai2-9b-2bit` | 1024 / 128 | 463.54 | 103.97 | 6.04 |

These short, synthetic repeated-word-context measurements qualify the serving
path; they do not measure instruction-following quality or establish long-context,
concurrency, sustained-throughput, or drafter performance. The ordinary hybrid
multimodal compatibility lane serializes active media requests. The 4B and 9B
optional drafters remain outside this qualification.

## Automated verification

All **4,438 tests passed** in the combined suite. New tests are in
`tests/test_prism_hadamard_qwen35.py`, already included in the Apple list in
`.github/workflows/ci.yml`. Coverage includes tied-head numerics, ignoring MTP
files, each new CLI/Desktop catalog entry, and startup/preflight/residency lane
guards across 2B, 4B, 9B, and 27B.

```bash
mkdir -p /private/tmp/bonsai2-small-validation
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/pytest -q \
  tests/test_prism_hadamard_qwen35.py \
  tests/test_model_aliases.py tests/test_model_auto_config.py \
  tests/test_model_sizes.py tests/test_atomic_model_catalog.py \
  tests/test_readme_image_matrix.py tests/test_cli_models.py \
  tests/test_aliases_contract.py tests/test_alias_hybrid_classification.py \
  tests/test_alias_recommended_sampling.py \
  --basetemp=/private/tmp/bonsai2-small-validation/pytest-final \
  --cov=rapid_mlx \
  --cov-report=xml:/private/tmp/bonsai2-small-validation/coverage.xml
.venv/bin/diff-cover /private/tmp/bonsai2-small-validation/coverage.xml \
  --compare-branch=origin/main --fail-under=100
.venv/bin/ruff check .
.venv/bin/ruff format --check rapid_mlx/models/prism_hadamard_qwen35.py \
  tests/test_prism_hadamard_qwen35.py
git diff --check
```

Changed-line coverage against `origin/main`: **100% (2/2 executable production
lines, none missing)**. JSON and Markdown additions are validated by the
alias/catalog/size and README-total consistency tests rather than runtime line
coverage. Repository-wide Ruff, changed-file formatting, and whitespace checks
pass.
