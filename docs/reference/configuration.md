# Configuration Reference

Install Rapid-MLX on an Apple Silicon Mac and restart Terminal. Start a server with its defaults:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-4b-4bit
```

Use `rapid-mlx serve --help` for the flags and defaults in your installed version. The [site CLI reference](https://rapidmlx.com/docs/cli.html) explains common choices. Add only the setting your workload needs.

## Server configuration

- **Network:** `--host`, `--port`, and `--api-key` control where the API listens and who can call it. The default host is loopback. See the [server guide](../guides/server.md).
- **Model:** Pass an alias to `serve`; use `rapid-mlx recipe` to pick one that fits this Mac and `rapid-mlx info <alias>` to inspect it.
- **Memory and cache:** Inspect the current `--kv-cache-*`, `--prefix-cache-*`, and `--disable-disk-caches` flags with `rapid-mlx serve --help` before tuning them.
- **Multimodal:** The [image and video guide](../guides/multimodal.md) shows model selection and request formats.

## Speculative decoding options

Speculative modes depend on the model and its matching draft checkpoint. Check the alias profile with `rapid-mlx info <alias>` and the current `--speculative-*` options with `rapid-mlx serve --help` before enabling one.

## Environment variables

Use `rapid-mlx serve --help` for settings with environment equivalents. Keep API keys outside scripts and check the [server guide](../guides/server.md) before exposing a server beyond loopback.

### Rejected tool-call diagnostics

Chat Completions keeps undeclared tool names out of executable `tool_calls`.
The marker-delimited call scanner and GLM-style parser also report complete
undeclared calls in an optional top-level `rejected_tool_calls` response field:

```json
{
  "rejected_tool_calls": [
    {
      "name": "write_file",
      "raw_arguments": "<arg_key>path</arg_key><arg_value>hello.py</arg_value>",
      "reason": "undeclared_tool"
    }
  ]
}
```

This is a Rapid-MLX extension, separate from executable calls. The argument
string is the rejected parser input, not necessarily JSON; it is untrusted model
output, including any claimed results. Never execute these records. Streaming
responses attach the field to the terminal completion chunk. The field is absent
when there are no recorded rejections. Normal content, accepted calls and finish
reasons retain their existing behavior. Incomplete or otherwise unparseable
markup is outside this diagnostic; absence of metadata is not proof that the
model attempted no call.

Warnings include the rejected name and reason, but omit arguments. The
process-local `/metrics` counter `rapid_mlx_rejected_tool_calls_total` counts
observations at these rejection sites. Within a chat request, identical
`(name, raw_arguments)` observations are deduplicated across streaming reparses
and fallbacks; this counts unique rejected records, not execution attempts.
Outside a request, each parser observation counts separately. The counter resets
on process restart and has no model- or tool-name labels.

Tool grammar admission logs `armed` or `skipped:<cause>` with a per-request
identifier and model. Causes distinguish operator opt-out, absent tools/parser,
unsupported parser or tokenizer, tool choice, unsafe auto grammar, schema bounds,
native-wire mismatch, missing optional dependencies, builder refusal, compile
failure and saturated/failed offload admission. `armed` means a usable processor
was built; a subsequently cancelled request need not reach generation. `auto`
is already eligible for supported optional-call grammars. Extending that behavior
to additional parser families is a separate decision; diagnostics do not change
which requests receive a grammar.
