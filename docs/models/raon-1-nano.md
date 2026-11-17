# Raon-1 nano

```bash
rapid-mlx serve raon-1-nano
```

The catalog alias selects the pinned KRAFTON/Raon-1-nano-mlx checkpoint
(Apache-2.0, 3.2B dense, 4-bit GPTQ). Its architecture ships inside Rapid;
checkpoint Python is neither downloaded nor executed. The same catalog entry
is available to Desktop without a UI change.

The release template enables thinking by default. Responses separate thinking
into `reasoning_content` and XML function calls into OpenAI `tool_calls`.
Set `chat_template_kwargs: {"enable_thinking": false}` in a request to disable
thinking. Declare tools with their JSON schemas so argument types can be
recovered from the template's text representation.

The model defaults to temperature 0.1 and presence penalty 1.5. For agentic
workloads, the vendor recommends explicitly requesting temperature 1.0.
Request sampling settings override the catalog defaults.

The architecture mixes 18 recurrent RDA layers and six MLA layers. Rapid's
existing hybrid cache policy prevents trimming recurrent state to an earlier
prefix. Bounded exact-prefix/extension reuse follows the same snapshot rules
as other hybrid models. MTP is unsupported; the `mtp/` weights are excluded
from downloads.

Run local correctness and timing checks on a Mac with Metal access:

```bash
RAPID_MLX_TELEMETRY=0 DO_NOT_TRACK=1 \
  .venv/bin/python -m scripts.qualify_raon_1_nano
```

This uses the existing default Hugging Face cache. It compares production-loader
logits with the vendored release's direct weight loader, greedily decodes three
prompts, and prints decode throughput and an exactly 2,048-token prefill timing.
Measurements are local diagnostics; they are not catalog performance claims.
