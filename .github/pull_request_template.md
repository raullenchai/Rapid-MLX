## Summary

<!-- What changed and why? Link the related issue if any. -->

## Type of change

<!-- Keep all that apply, remove the rest. -->

- Bug fix
- New feature
- Performance
- Security
- Documentation
- Refactor or internal cleanup

## Surface touched

<!-- Keep all that apply, remove the rest. -->

- CLI / serve command
- OpenAI API endpoints
- Anthropic API endpoints
- Scheduler / batching
- Sampling
- KV cache / prefix cache
- Multimodal pipeline
- MCP / tool calling
- Reasoning models
- Metal kernels / v1 engine
- Build / packaging
- Tests only
- Docs only

## Test plan

<!-- Exact commands a reviewer can run to verify this PR. Mention the model used. -->

## Test results

<!-- Paste relevant test output. If this PR fixes a failing test, include before and after. -->

## Performance impact

<!-- Skip with "N/A, docs or tests only" if it does not apply. Otherwise: -->
<!-- - Hardware (chip, RAM): -->
<!-- - Model and quantization: -->
<!-- - Benchmark command: -->
<!-- - Metrics measured (TTFT, decode tok/s at batch=1 and batch=8, peak RSS): -->
<!-- - Delta vs main: -->
<!-- A perf-sensitive PR must be equal or faster on every metric it claims to improve. If a metric regresses, justify the trade-off here. -->

## Output parity

<!-- For changes that touch generation (sampling, scheduler, kernels, models, tokenization), confirm that greedy output matches main byte-for-byte on a fixed prompt set, or describe the expected divergence. Skip otherwise. -->

## Risk notes

<!-- Runtime, parser, scheduler, memory, security, or compatibility risk. Call out anything reviewers should look at closely. -->
