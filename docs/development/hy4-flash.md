# Hy4 Flash: phase-one text runtime

The `hy_v4_flash` architecture loads through the text runtime, including
checkpoints whose outer configuration still contains a vision tower. There is
no public model alias or qualification claim. Vision and `dfly_layers` weights
are omitted with an informational log message.

The text adapter implements per-layer MLA and sliding attention, learnable
softmax sinks, coarse gated IndexPool selection, latent routed experts,
independent identity residual streams, and frozen Engram token maps. The
checkpoint's rotary channels use adjacent pairs. Text positions are shared
across the positional axes. Dense and shared experts use ordinary SwiGLU;
only routed experts use the configured clamp. Router bias affects expert
selection, and the unbiased sigmoid scores determine the mixture weights.

The implementation uses portable array operations and the shared expert and
cache interfaces. Existing stream-mixing and KV-compression modules implement
different equations: this architecture has identity residual connections and
compresses selection keys rather than attention values. It does not use their
model-specific kernels. IndexPool's normalized Hadamard transform and scaled
E4M3 rounding are expressed numerically without requiring an FP8 array dtype.
Engram hashing runs on the CPU using the checkpoint's frozen lookup table and
seeded signed-int64 arithmetic. Affine Engram tables can remain on SSD.

Each layer retains latent KV, raw selection keys, and original token IDs in
three ordinary caches. Full history is retained even for sliding attention;
the window is enforced by the attention mask. This favors correctness over
memory efficiency. All three states are trimmed together. Prefix-cache reuse
is disabled automatically for this model because the server's reuse stores
have not been qualified for this layout.
Speculative decoding is disabled in the detected model profile.

The `hy_v4` parsers consume the native suffixed structural tags. Tool arguments
follow the supplied JSON Schema, including nullable types and local references.
The parser preserves argument-key whitespace. Text deltas use the reference
string-marker path, including partial-marker withholding. Rapid's terminal
flush hook releases any ordinary partial marker at end of stream. Reasoning
assumes the prompt injected the opening think tag; `no_think` returns content
directly. Use `chat_template_kwargs={"reasoning_effort": "high"}` or
`{"reasoning_effort": "no_think"}`. The supplied checkpoint template remains
authoritative; the adapter does not substitute another template.

## Local checkpoint conversion

Use an existing local source directory. These commands do not fetch weights.
Set `HY4_FP8` to the source directory and `HY4_MLX` to a new output directory;
activate the contributor environment first.

```sh
export RAPID_MLX_TELEMETRY=0 DO_NOT_TRACK=1
export HY4_FP8=/path/to/local/Hy4-flash-FP8
export HY4_MLX=/path/to/new/Hy4-flash-4bit
PYTHONPATH=. python - <<'PY'
import os
from rapid_mlx.byom._import_worker import convert
convert(os.environ['HY4_FP8'], os.environ['HY4_MLX'], bits=4, group_size=64)
PY
```

The standard import worker dispatches block-FP8 sources before the ordinary
array loader sees unsupported FP8 storage. It decodes E4M3 bytes, multiplies
`weight_scale_inv` per block (or embedding row), sanitizes the names and splits
packed expert gate/up tensors, then quantizes each weight and uses the standard
MLX saver. Ordinary weights use the requested affine group size and Engram
embeddings use group 32. The router, residual gates, index pooling projections,
and output head retain their required dense precision. NaN bytes and malformed
scale shapes fail conversion. No source model Python is executed.

The regular `rapid-mlx import` command also accepts this source architecture,
including its block-FP8 quantization metadata. Conversion can require substantial
host memory, especially for a complete Engram table; peak memory has not been
measured with release weights.

## Engram SSD offload on 128 GB Macs

Product loading (`rapid-mlx serve <local-quantized-directory>`) enables Engram
SSD offload when affine tables and a safetensors index are available. For the
approximately 131 GB hy4-gptq-v1 layout, keeping the four approximately 10 GB
Engram tables on SSD targets approximately 91 GB of resident weights, leaving
room for KV state on a 128 GB Mac. This is a memory budget target, not a measured
peak or throughput claim; context length and concurrent requests still matter.

Selected packed rows, scales and biases are read from read-only mapped shards
and dequantized using the checkpoint's affine settings. Hashes are prefetched
before decoder execution using each layer's current token history. A mismatched
prefetch is discarded and the requested rows are read again. No weights are
re-quantized. Other parameters are evaluated in bounded groups during loading
to avoid a single cold-disk, whole-model Metal command buffer.

Set `HY4_ENGRAM_SSD_OFFLOAD=0` before serving to keep tables resident. Python
callers can use `rapid_mlx.models.hy_v4_flash_load.load(path,
engram_ssd_offload=False)`; `True` requires a valid indexed affine layout and
fails on missing tensors, invalid shapes/dtypes, or escaping shard paths.
The default `None` selects supported offload automatically.

## Real-weight smoke

On an Apple Silicon host with sufficient memory, point `HY4_MLX` at the
converted checkpoint and run the following from the repository root. This
loads local files and does not start a server. A failed assertion is a failed
qualification check, even when the tiny tests pass.

```sh
export RAPID_MLX_TELEMETRY=0 DO_NOT_TRACK=1
PYTHONPATH=. python - <<'PY'
import os
from rapid_mlx.utils.tokenizer import _register_vendored_archs, load_model_with_fallback
_register_vendored_archs()
from mlx_lm import generate, stream_generate
from mlx_lm.sample_utils import make_sampler
from rapid_mlx.utils.chat_template import apply_chat_template
from rapid_mlx.reasoning.hy4_parser import HyV4ReasoningParser
from rapid_mlx.tool_parsers.hy_v4_tool_parser import HyV4ToolParser

model, tokenizer = load_model_with_fallback(os.environ['HY4_MLX'])[:2]
for effort in ('high', 'no_think'):
    prompt = apply_chat_template(
        tokenizer, [{'role': 'user', 'content': 'What is 2 + 2?'}],
        chat_template_kwargs={'reasoning_effort': effort},
    )
    chunks = list(stream_generate(model, tokenizer, prompt=prompt,
                                 max_tokens=256, sampler=make_sampler(temp=0)))
    text = ''.join(chunk.text for chunk in chunks)
    buffered = generate(model, tokenizer, prompt=prompt, max_tokens=256,
                        sampler=make_sampler(temp=0), verbose=False)
    assert text and text == buffered, (effort, text, buffered)
    reasoning, content = HyV4ReasoningParser(tokenizer).extract_reasoning(
        text, enable_thinking=effort != 'no_think')
    print(effort, {'reasoning': reasoning, 'content': content})

tools = [{'type': 'function', 'function': {
    'name': 'weather', 'description': 'Look up the weather for a city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                   'required': ['city']},
}}]
prompt = apply_chat_template(
    tokenizer, [{'role': 'user', 'content': 'Call weather for Paris.'}], tools=tools,
    chat_template_kwargs={'reasoning_effort': 'no_think'},
)
text = generate(model, tokenizer, prompt=prompt, max_tokens=256,
                sampler=make_sampler(temp=0), verbose=False)
result = HyV4ToolParser(tokenizer).extract_tool_calls(text, {'tools': tools})
assert result.tools_called, text
assert result.tool_calls[0]['name'] == 'weather', result
print(result)
PY
```

Before a public alias is added, record hardware, memory, dependency versions,
checkpoint revision, the conversion command, streamed/buffered parity, reasoning
and tool results, and peak memory. Native tests also cover affine checkpoint
reload and cache trim/replay. CPU tests cover the same graph with a NumPy test
backend and independently check the core equations; that is not Metal or
real-weight inference evidence.
