<!-- Draft for 0.16.9. Rename in the version-bump PR. -->

Rapid-MLX 0.16.9 improves Desktop responsiveness, chat instruction handling, and reuse of long prompts. It also strengthens model loading and bundled-runtime installation, and adds explicit concurrency controls for supported backends.

## Highlights

- **Responsive model picker**: opening or dismissing the Desktop picker during a streamed reply no longer repeatedly rebuilds unchanged model rows. Model switching still asks before interrupting an active response. ([#5176](https://github.com/raullenchai/Rapid-MLX/pull/5176))
- **More predictable chat**: Gemma chats stop at their turn terminators even when generation configuration omits them. Custom system instructions remain verbatim when tools are offered in auto mode; required and named tool constraints remain enforced. ([#5185](https://github.com/raullenchai/Rapid-MLX/pull/5185), [#5235](https://github.com/raullenchai/Rapid-MLX/pull/5235))
- **Faster repeated long prompts under cache pressure**: hybrid caches retain useful restored boundaries instead of displacing them with redundant entries. In a measured Qwen3.6-35B workload with a 2 GiB prefix budget, repeat-request time to first token fell from 18.622 s to 0.390 s. This is a workload-specific result; cache residency can change output tokens. See the performance section for conditions and limits. ([#4459](https://github.com/raullenchai/Rapid-MLX/pull/4459))
- **Clearer authenticated connections**: connection discovery and generated snippets use the configured key, and `start` distinguishes authentication requirements from an unhealthy server. Explicitly blank keys now fail startup. Credentials are withheld from remote plaintext connections unless the supported explicit opt-in is used. ([#5174](https://github.com/raullenchai/Rapid-MLX/pull/5174), [#5188](https://github.com/raullenchai/Rapid-MLX/pull/5188), [#5207](https://github.com/raullenchai/Rapid-MLX/pull/5207))

## Compatibility and model loading

- Unauthenticated network listeners now require explicit consent; ordinary loopback serving remains available. Checkpoint-provided Python also requires explicit consent, and legacy tensor loading uses restricted deserialization where supported. ([#5157](https://github.com/raullenchai/Rapid-MLX/pull/5157), [#5155](https://github.com/raullenchai/Rapid-MLX/pull/5155), [#5151](https://github.com/raullenchai/Rapid-MLX/pull/5151))
- Desktop verifies its bundled engine before launch and provides full-app recovery guidance when it cannot be verified. Full-app upgrades preserve existing model caches and older managed-engine directories. ([#5164](https://github.com/raullenchai/Rapid-MLX/pull/5164))
- Bundled Desktop runtime dependencies now use exact versions and SHA-256-checked package artifacts. Public pip installs retain compatible version ranges; this does not make every build input hermetic. ([#5238](https://github.com/raullenchai/Rapid-MLX/pull/5238))
- Explicit model pulls verify downloaded content before reporting completion. Mirror attribution follows transferred weights, so metadata fallback no longer mislabels a mirror weight download. ([#5154](https://github.com/raullenchai/Rapid-MLX/pull/5154), [#5210](https://github.com/raullenchai/Rapid-MLX/pull/5210))
- Pinned four-bit and eight-bit Nemotron 3.5 Bolt E4B aliases add qualified text support with tool calls and explicit thinking controls. Multimodal and draft acceleration are not included. Cached MTP aliases avoid unnecessary sidecar fetches. ([#5138](https://github.com/raullenchai/Rapid-MLX/pull/5138), [#5143](https://github.com/raullenchai/Rapid-MLX/pull/5143), [#5095](https://github.com/raullenchai/Rapid-MLX/pull/5095))
- The experimental LTX-2.5 video runtime is bundled. Complete cached flat-component checkpoints are recognized for offline serving, avoiding false missing-weight notices. ([#5089](https://github.com/raullenchai/Rapid-MLX/pull/5089), [#5187](https://github.com/raullenchai/Rapid-MLX/pull/5187))

## Performance and serving controls

The hybrid-cache measurement above used an M3 Ultra with 256 GiB unified memory, macOS 26.5.2, Python 3.12.14, MLX 0.32.4, mlx-lm 0.31.3, Transformers 5.15.1, Qwen3.6-35B, approximately 32K-token prompts, 32-token completions, and a 2 GiB prefix-cache budget. Shared CPU activity overlapped the measurements; these are descriptive results, not a quiet-machine confidence bound or a general performance guarantee. Repeated larger-cache controls did not reproduce the earlier regression, and output variation also occurred between identical baseline runs. [Methodology and limitations](../engineering/performance/hybrid-boundary-retention.md).

| Measured workload | Before | After |
| --- | ---: | ---: |
| Repeat-request median TTFT | 18.622 s | 0.390 s |
| Four-request cohort completion | 63.441 s | 43.441 s |

- Single-lane Qwen27 can resume late edits from an optional interior checkpoint when memory permits. Historical measurements reduced 3–8K-token late-edit TTFT from 8.98–23.88 s to 3.12–5.84 s, with peak memory increasing from 20.12 to 20.38 GiB. These are limited workload measurements, not a fresh benchmark of this release. ([#5140](https://github.com/raullenchai/Rapid-MLX/pull/5140))
- `serve --tensorfold-lanes {1,2,4,8}` exposes concurrency for qualified Qwen3.8-27B and Nemotron profiles. Standalone serving stays serial by default. The Qwen27 pool defaults to four matching slots and lanes; existing registrations require `--reregister` to adopt changed capacity. Four maximum-length contexts on 48 GiB hardware are not qualified. ([#5231](https://github.com/raullenchai/Rapid-MLX/pull/5231))
- `serve --dsv41-lanes {1,2,4,8}` and `--no-dsv41-lanes` expose native DeepSeek V4.1 decode controls. **Migration required:** unset `RAPID_MLX_DSV41_LANES` and use the CLI option; any set value of the retired variable now produces migration guidance. The default remains one serial lane. ([#5230](https://github.com/raullenchai/Rapid-MLX/pull/5230))
- DeepSeek V4.1 adds offline build, calibration, qualification, and streamed GPTQ conversion tools. These are preparation tools, not a claim that a new converted checkpoint is qualified for serving. ([#5046](https://github.com/raullenchai/Rapid-MLX/pull/5046), [#5142](https://github.com/raullenchai/Rapid-MLX/pull/5142), [#5163](https://github.com/raullenchai/Rapid-MLX/pull/5163), [#5167](https://github.com/raullenchai/Rapid-MLX/pull/5167))

## Everyday use and diagnostics

- Interactive `serve` offers a quiet, one-time newsletter link after readiness. Desktop coordinates newsletter and GitHub-star cards around successful use, with a shared cooldown and at most one card per launch, and adds a permanent Newsletter link in About. ([#5145](https://github.com/raullenchai/Rapid-MLX/pull/5145), [#5139](https://github.com/raullenchai/Rapid-MLX/pull/5139))
- `doctor` retains useful diagnostics when optional imports are slow. Unknown or retired model names receive actionable errors, and cached audio models map to catalog aliases. ([#5175](https://github.com/raullenchai/Rapid-MLX/pull/5175), [#5173](https://github.com/raullenchai/Rapid-MLX/pull/5173), [#5194](https://github.com/raullenchai/Rapid-MLX/pull/5194))
- Unclassified serve failures gain a fixed `failure_subtype` category in telemetry. This field contains no exception text, paths, or model names. Desktop diagnostic exports also redact account paths more consistently. ([#5213](https://github.com/raullenchai/Rapid-MLX/pull/5213), [#5184](https://github.com/raullenchai/Rapid-MLX/pull/5184))
- Repeated client launches preserve selected models, honor requested ports, and distinguish saved configuration from server readiness. ([#5106](https://github.com/raullenchai/Rapid-MLX/pull/5106), [#5129](https://github.com/raullenchai/Rapid-MLX/pull/5129), [#5104](https://github.com/raullenchai/Rapid-MLX/pull/5104))
