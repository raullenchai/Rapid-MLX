# Startup failure telemetry

`model_serve_failed` emits `failure_subtype` only when `error_class=other`.
This applies to every startup stage, including `preflight` and `bind`; a known
error class still takes precedence. The subtype prefers an annotated operation over
an exception type, scanning the bounded explicit cause chain. Implicit exception
context is deliberately ignored. No messages, paths, model identifiers, or
traceback frames are read by the subtype classifier.

The ten startup boundaries are fixed in `rapid_mlx/engine_startup_errors.py`:

| Boundary | Owned operation |
| --- | --- |
| `checkpoint_resolve` | Alias/local checkpoint resolution, subfolder selection, native/VLM snapshot resolution and serving-lane selection. Works before engine construction too. |
| `model_prepare` | Server `load_model` preparation and engine construction; more precise inner annotations win. |
| `config_load` | Local text config validation and explicit native/multimodal config loading. |
| `tokenizer_load` | Shared checked tokenizer loader, including text fallback loaders. |
| `chat_template` | Profile/explicit chat-template resolution and checkpoint template sidecar loading. |
| `vision_load` | Vision runtime initialization/availability gate. The standard upstream loader combines tower weights and processor loading, so failures inside that atomic operation remain `weights_load`. |
| `weights_load` | Text, disk-stream, and multimodal worker loader results (existing boundary). |
| `draft_attach` | Explicit MTP attachment dispatch on the model worker (existing boundary). |
| `cache_setup` | Text and multimodal scheduler construction and the multimodal KV-cache compatibility probe. |
| `warmup` | Dynamic residency warmup calls. Eager shader/grammar warmups already catch their errors and remain nonfatal; they do not create failed-serve events. |

Annotations keep the original exception object and preserve the innermost
operation through worker futures. Read-only exceptions retain type fallback.
The synchronous boundary decorator is reusable; async operations use `with`
around the awaited operation instead. Cancellation, `SystemExit`, and keyboard
interrupts bypass annotation and keep existing control flow.

Type fallback adds `subprocess_error`, `timeout_error`, `not_implemented`,
`system_exit`, `template_error`, `safetensors_error`, `hub_error`, `network_error`,
`cache_layout`, and `cancelled_error`. Timeout checks precede `OSError`, and
`NotImplementedError` precedes `RuntimeError`. Python 3.12 asyncio and concurrent
future timeouts alias the builtin `TimeoutError`. Subprocess timeout is separate
from builtin timeout and is checked before `SubprocessError`.

Optional exception families use a fixed allowlist of module/qualified-name
pairs matched against the MRO, with fixed output tokens. Telemetry never imports
optional runtimes to classify them. Jinja template errors and Rust safetensors
errors can directly subclass `Exception`; tokenizers can raise plain
`Exception`, which cannot safely be distinguished by type and needs the
`tokenizer_load` boundary. Other Transformers and MLX failures normally inherit
`ValueError`, `RuntimeError`, `ImportError`, or `OSError` and keep those fallbacks.
Rapid's typed config/tokenizer/weight failures already have dedicated error
classes; its `PagedCacheUnsupportedLayoutError` directly subclasses `Exception`
and gains `cache_layout`. Request-only backpressure/context exceptions are not
startup-specific and retain `unknown` unless an operation annotated them.
Unknown identities, generic `Exception`, and other unlisted `BaseException`
subclasses still return `unknown`. The query counts from older releases cannot
identify which of these types caused historical `unknown` values.

Desktop launches use the same Python prepare/start paths after process handoff.
Subprocess failures reaching Python telemetry now have a fixed type token;
a sidecar that exits before Python emits an event cannot gain a Python startup
boundary. No Swift handoff or event contract changes are made here. Optional
post-load attachment/fusion failures outside the explicit MTP dispatcher retain
their surrounding operation/type; this change does not make optional fallbacks
fatal or change readiness policy.

`events.json`, the Desktop privacy disclosure, and registry drift tests contain
the same closed enum. CPU tests cover every token, actual owned call sites,
worker identity, explicit causes, unknown fallback, frozen exceptions, and
emitted payload validation at all three stages. Headless startup tests use the
repository's inert MLX import seam and execute no tensor operations.

## Accepted starts must describe their own failure

Every emitted `server_start_state(state=failed)` carries `exit_reason` and the
existing `serve_error_class` as `error_class`. When that class is `other`, it also
carries the existing `serve_failure_subtype`. All three fields are closed enums;
no new enum values, identifiers, messages, argument values, or paths are added.
The registry keeps the new fields optional for older producers. Ready and
attempted events never carry them. The original model reference is used only
in memory to classify local-file failures, through the existing classifier.

Exit reasons remain `user_interrupt`, `declined_download`, `usage_error`,
`port_in_use`, `memory_refused`, and `exception`. Validation owners bind
`usage_error` explicitly; exit code 2 alone is not evidence of a usage error.
A download disabled by policy is `declined_download`. Actual allocation failures
remain exceptions; a deliberate memory admission refusal is `memory_refused`.
Missing dependencies use `missing_extra`; missing model weights do not.

The CLI and standalone entrypoint guards supply the missing
`model_serve_failed` fallback, including unknown exceptions and nonzero
`SystemExit`. Explicit stage contexts use the same fallback. Inner owners run
first and retain their model/recovery details through the existing one-shot
latch. Usage errors with no known model error, port conflicts, download declines,
and interrupts need only the failed-start event. A successful `SystemExit(0)`
or `SystemExit(None)` is not a failed start. Errors after readiness do not
create startup failures. Unhandled exceptions now print an installation-check
and bug-report next step before preserving the original traceback.

Consent, upload gates, and existing restart-loop suppression still apply.
A suppressed or rejected model-failure event cannot remove cause information
from a failed-start event. Existing restart-loop keys remain stage/explicit-port
based; these counts are sampled outcomes, not a record of every retry.

## First-install triage (source audit)

The ordering below is a qualitative prior for first-time CLI use, not an
attribution of historical installs. The reported Oct 20–Nov 16 counts are not
mutually exclusive stage totals, and absent fields cannot recover their causes.
This audit describes the checked-out source; older releases can differ. In
particular, this checkout already defaulted missing exit reasons to `exception`
before this fix. Its main gap was missing cause detail and incomplete fallback
coverage, rather than a way to infer the historical null reasons.

In the table, **S** means `server_start_state(state=failed)` and **M** means
`model_serve_failed`. Before the fix S has no error class/subtype; its reason is
`exception` unless specified. Afterwards every S has its closed reason and
class/subtype even if M is suppressed. Templates below omit leading whitespace;
brace expressions are substituted locally on screen, never sent as text.

| Priority / path | Trigger and exact diagnostic template | Before → after |
| --- | --- | --- |
| High: unknown alias; `cli.main`, model-resolution block | A nonlocal bare name resolves to no catalog/audio alias: `Error: '{args.model}' is not a known alias or HuggingFace path.` Followed by `Did you mean: {suggestions}?` or `Try one of: {POPULAR_ALIASES}`, ``Run `rapid-mlx models` to see all available aliases,``, and `or pass a full path like: mlx-community/Qwen3.5-9B-4bit`. | resolve S, no M → `usage_error / invalid_model_ref`, M fallback |
| High: local path/config/checkpoint validation; `cli.main` | Missing path: `Cause: Local model path {root} does not exist.` / ``Next step: Check the path, then run `rapid-mlx serve /path/to/model`.`` Other exact cause/remedy pairs are in `local_model_path.local_serve_preflight_lines` and `local_serve_diagnostics.startup_error_lines`: non-directory, unreadable file, missing config/manifest/component/weights, malformed config. | resolve S; M already emitted for local validation → original typed cause retained on S too (`local_path_missing`, `invalid_config`, or other existing local-load class) |
| High: HF authentication/gating or nonexistent repo; `_fail_hub_resolution` | `Error: access to '{rendered_model_id}' is gated on Hugging Face.` / `Accept the licence or request access at https://huggingface.co/{rendered_model_id}` / ``Then run `hf auth login` or set `HF_TOKEN`, and try again.`` 401 has `Error: Hugging Face returned 401 for '{rendered_model_id}'. The repo may be private, gated, or misspelled.` 404 has `Error: model repository '{rendered_model_id}' was not found on Hugging Face.` and exact-ID/catalog guidance. | resolve S and M already present → S also has `exception / download_failed`; no invented auth enum |
| High: explicit busy port; `_exit_for_port_collision` | Socket probe finds EADDRINUSE: `Error: Port {port} is already in use on {collision_host}.` / `Try a different port: rapid-mlx serve {model} --port {port + 1}`. | bind S with `port_in_use`, no M → same reason, `other / unknown` on S |
| High: implicit port scan exhausted; `_exit_for_port_scan_exhaustion` | All candidates occupied: `Ports {scan_base}-{scan_end} are all in use; pass --port with a free port outside that range.` | bind S with `port_in_use`, no M → same reason, `other / unknown` on S |
| Medium: runtime extra absent/broken/incompatible; `handle_optional_runtime_missing` | Audio/vision/image/video guard raises `OptionalRuntimeMissing`: its lane-specific `detail` plus install-method-aware `install_hint`, and `RAPID-MLX-STARTUP-FAILURE: {marker_reason} extra={extra}`. In a noninteractive installable session: `Non-interactive session: rerun with --yes to {action}`. | preflight S + M already present → S also has `exception / missing_extra`; M keeps extra/recovery outcome |
| Medium: embedding/native embedding dependency; `embedding.require_mlx_embeddings_or_exit` | `error: --embedding-model requires the [embeddings] extra. {EMBEDDINGS_EXTRA_INSTALL_HINT}` or `error: EmbeddingGemma 2 requires the mlx-vlm native runtime. {optional_extra_install_hint('vision')}`. | preflight S, no M → `exception / missing_extra`, M fallback; console install instructions preserved |
| Medium: catalog/BYOM memory admission; `byom.preflight` | BYOM: `✗ {shown} needs ~{estimated_memory} of memory; this Mac has {ram}.` / `Its weights alone are {weights}.` / `Pick a smaller or more quantized build: rapid-mlx models`. Catalog refusal renders its measured memory-floor or GPU-budget explanation plus a fitting-model suggestion. Both print `(Think this is wrong? Re-run with --no-preflight to skip this check.)`. | preflight S with `memory_refused`, M already present → typed refusal retained on S as `insufficient_memory` |
| Medium: enforced alias memory floor; `_check_alias_min_memory` | `Error: '{user_typed}' requires at least {floor_gb:.0f} GB of unified memory; this Mac reports {total_ram_gb:.1f} GB.` / `Rapid-MLX is refusing the load because this checkpoint is not qualified below its memory floor and an undersized host may become unresponsive.` | preflight S, no M → `memory_refused / insufficient_memory`, M fallback; adds `Pick a smaller model: rapid-mlx models` |
| Medium: offline uncached / downloads disabled | Offline: `Error: {model_name} is not cached and the network is unavailable (offline mode is enabled).` followed by connectivity/offline-variable and pull guidance. Disabled: `Error: {model_name} is not cached and model downloads are disabled by {source}.` / ``Download it with `rapid-mlx pull {model_name}`, then try again.`` | offline resolve S+M already present → S cause retained; disabled resolve S only → `declined_download / other` |
| Medium: unsupported BYOM format/architecture | `! {shown} only has GGUF files{quants}.` / `Rapid-MLX runs MLX and safetensors weights, not GGUF.`; or `! {shown} only has PyTorch .bin weights.` / `Look for a safetensors or MLX build of the same model.`; or `✗ Architecture {identity} is not supported yet.` with loader version and catalog/BYOM guide. Remote refusals also say `Nothing was downloaded.` | preflight S+M already present → explicit cause retained on S as `unsupported_format` / `unsupported_architecture` |
| Lower: malformed host/port/listen fd or flag combinations | Exact per-guard diagnostics are enumerated below. Host adds `Use --host 127.0.0.1 for a local server.`; argument errors identify the invalid flag/value or incompatible options. Standalone parser errors render `usage: ...` then `{prog}: error: {message}`. | preflight S, generally no M → `usage_error / other`; no loader failure needed |
| Lower: retired/user alias or draft-only checkpoint | Alias exception renders `Error: {exc}` with its replacement/repair guidance. Draft-only exception renders its supported-primary guidance. | alias: resolve S, no M → `usage_error / invalid_model_ref`, M fallback; draft: M already present → S retains `unsupported_architecture` |
| Lower: disk admission / partial image checkpoint | `Error: Insufficient disk space for download.` plus download/free/required sizes and Suggestions. Partial image: `Error: {_shown} is only partially downloaded — {len(_missing)} required file(s) missing, starting with {_missing[0]}.` / ``Finish the download with `rapid-mlx pull {_shown}`, then serve again.`` | preflight/download S, no M → disk `exception / other / system_exit`; partial image `exception / corrupt_weights`, M fallback |
| Lower: unexpected dependency/config/runtime exception | Previously only the original exception traceback (or a callee's own message). Now: `Error: Server startup failed: {exc}` / ``Next step: run `rapid-mlx doctor` to check this installation. If it passes, report this failure with the traceback below.`` The original exception and exit status are preserved. | S; CLI only emitted M for known preflight classes, standalone had no generic M → S class/type plus M fallback at every stage |
| Bind race / TLS / socket failures; `_uvicorn.run_uvicorn` | EADDRINUSE: `Error: Port {port} already in use on {host}. Stop the existing server or pass --port <n> (lsof -i :{port}).` Other Uvicorn errors retain its exact logged exception, e.g. `[Errno ...] ...`, or TLS-file traceback. | bind S with reason but no class/M → original exception on S (`port_in_use` or `exception`, class/subtype); no misleading loader event after S is terminal |
| Cancellation / upgrade | Ctrl+C is `user_interrupt`; refusing a large download is `declined_download`, with the existing cancellation/pull instructions. An accepted upgrade can end startup with exit 0. | interrupts/declines already have explicit S reason; successful upgrade wrongly recorded failed → successful exit no longer emits failed |
| Excluded: missing model, parse errors, Python/platform failures before startup | With no model and no reusable local model: `rapid-mlx serve: a model is required.` / `Pick one for this Mac:  rapid-mlx recipe` / `Or start a small one:   rapid-mlx serve qwen3.5-4b-4bit`. This and initial argparse errors occur before `attempted`. Python <3.12 is rejected by package metadata; unsupported platform/native import failures can occur before either entrypoint's telemetry starts. | no accepted-start event in this checkout; cannot explain its recorded failed-start cohort. No telemetry is initialized solely to report invalid invocations. |

The inventory below lists the individual deliberate-exit sites reachable from
`serve_command`, plus the resolution and standalone boundaries. It uses exact
source predicates and print arguments so variable text is not mistaken for a
constant diagnostic. Multiple predicates joined with `and` describe enclosing
branches. Callee-produced text remains owned by that callee (linked above).

| Exit site | Trigger predicate / exception | Exact print argument(s) in its block | Telemetry before → after |
| --- | --- | --- | --- |
| [preflight.py:1428](../../../rapid_mlx/byom/preflight.py#L1428) `_check_catalog_memory_fit` | failure in _check_catalog_memory_fit | '  Nothing was downloaded.'; '    (To download anyway, for example to move the files to a bigger Mac, re-run with --no-preflight.)\n' if args.command == 'pull' else '    (Think this is wrong? Re-run with --no-preflight to skip this check.)\n' | S + typed M already emitted → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [preflight.py:1480](../../../rapid_mlx/byom/preflight.py#L1480) `run_cli_preflight` | args.command == 'pull' and gguf_format_requested(getattr(args, 'format', None)) | f'\n  Error: {GGUF_FORMAT_MESSAGE}' | S + typed M already emitted → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [preflight.py:1548](../../../rapid_mlx/byom/preflight.py#L1548) `run_cli_preflight` | verdict.failure is not None | ''; '    (Think this is wrong? Re-run with --no-preflight to skip this check.)\n' | S + typed M already emitted → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:219](../../../rapid_mlx/cli.py#L219) `_resolve_serve_log_level` | level not in _SERVE_LOG_LEVELS | f"error: {LOG_LEVEL_ENV_VAR}={raw!r} is not one of {', '.join(_SERVE_LOG_LEVELS)} (case-insensitive)" | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:401](../../../rapid_mlx/cli.py#L401) `_exit_for_port_collision` | failure in _exit_for_port_collision | f'\n  Error: Port {port} is already in use on {collision_host}.'; f'  Try a different port: rapid-mlx serve {model} --port {port + 1}' | S with `port_in_use`; no M → `port_in_use / other` (authoritative bind owner) |
| [cli.py:415](../../../rapid_mlx/cli.py#L415) `_exit_for_port_scan_exhaustion` | failure in _exit_for_port_scan_exhaustion | f'Ports {scan_base}-{scan_end} are all in use; pass --port with a free port outside that range.' | S with `port_in_use`; no M → `port_in_use / other` (authoritative bind owner) |
| [cli.py:427](../../../rapid_mlx/cli.py#L427) `_exit_for_host_bind_error` | failure in _exit_for_host_bind_error | f'Invalid --host {display_host!r}: could not bind that address ({exc}).'; 'Use --host 127.0.0.1 for a local server.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:471](../../../rapid_mlx/cli.py#L471) `_port_preflight_or_die` | not 0 <= port <= 65535 | f'\n  Error: --port {port} is out of range. Ports must be 0-65535.'; f'  Try a valid port: rapid-mlx serve {model} --port 8000' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:599](../../../rapid_mlx/cli.py#L599) `_reject_unsupported_listen_fd_lane` | lane is not None | f'--listen-fd is not supported with the {lane} lane; omit --listen-fd or pass --host/--port.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:624](../../../rapid_mlx/cli.py#L624) `_resolve_serve_port` | listen_fd is not None and except OSError | f'Invalid --listen-fd {listen_fd}: {exc}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:696](../../../rapid_mlx/cli.py#L696) `_print_port_collision_and_exit` | failure in _print_port_collision_and_exit | Caller/callee diagnostic; see ranked table and source. | S only; generic reason → `port_in_use / other` (authoritative bind owner) |
| [cli.py:952](../../../rapid_mlx/cli.py#L952) `_validate_primary_lifecycle_args` | not math.isfinite(idle_unload_seconds) or idle_unload_seconds < 0 | 'Error: --idle-unload-seconds must be finite and >= 0' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:956](../../../rapid_mlx/cli.py#L956) `_validate_primary_lifecycle_args` | not math.isfinite(audio_ttl) or audio_ttl < 0 | 'Error: --audio-lane-idle-ttl must be finite and >= 0' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:1586](../../../rapid_mlx/cli.py#L1586) `_load_embedding_model_or_exit` | except ValueError | f'error: --embedding-max-length {exc}' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:1613](../../../rapid_mlx/cli.py#L1613) `_load_embedding_model_or_exit` | except not_found_exc_classes | f"\n  Error: --embedding-model '{original_embed}' could not be loaded ({type(exc).__name__}: {exc})."; '  Tip: use a registered embedding alias (see ``rapid-mlx ls`` for the list — e.g. ``embeddinggemma-300m-6bit``) or pass the full HuggingFace id (e.g. ``mlx-community/embeddinggemma-300m-6bit``).\n' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:1630](../../../rapid_mlx/cli.py#L1630) `_load_rerank_model_or_exit` | except ValueError | f'error: --rerank-max-length {exc}' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:1650](../../../rapid_mlx/cli.py#L1650) `_load_rerank_model_or_exit` | except UnsupportedRerankModelError | f'\n  Error: --rerank-model {exc}\n' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:1661](../../../rapid_mlx/cli.py#L1661) `_load_rerank_model_or_exit` | except _embedding_not_found_exception_classes() | f"\n  Error: --rerank-model '{original}' could not be loaded ({type(exc).__name__}: {exc})."; '  Tip: pass the full HuggingFace id of a supported reranker (e.g. ``mlx-community/Qwen3-Reranker-0.6B-mxfp8``) or a local model directory.\n' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:1893](../../../rapid_mlx/cli.py#L1893) `_check_download_disk_space` | not force | '    - Bypass this check (download will likely fail mid-way): --force-disk-check' | S only; generic reason → `exception / other / system_exit`, M |
| [cli.py:2108](../../../rapid_mlx/cli.py#L2108) `_apply_mtp_cli_model_type_reconciliation` | _eligibility_model_type is None | 'error: MTP speculative-config eligibility passed but the CLI could not extract config.json::model_type — this is a plumbing skew between detect_mtp_eligibility (accepted the config) and this CLI reconciliation block. Refusing to boot: the engine\'s dispatch requires the CLI-vetted model_type to hard-fail on non-attached dispatch results (round-E BLOCKER #2 contract). File an issue with the output of `python -c "import json; print(json.load(open(\'config.json\')).get(\'model_type\'))" run in the model directory.' | S only; generic reason → `exception / other / system_exit`, M fallback |
| [cli.py:2136](../../../rapid_mlx/cli.py#L2136) `_apply_mtp_cli_model_type_reconciliation` | except ValueError | f'error: {exc}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:2234](../../../rapid_mlx/cli.py#L2234) `_check_alias_min_memory` | _enforced_memory_floor(profile, total_ram_bytes) | f"\n  Error: '{user_typed}' requires at least {floor_gb:.0f} GB of unified memory; this Mac reports {total_ram_gb:.1f} GB.\n  Rapid-MLX is refusing the load because this checkpoint is not qualified below its memory floor and an undersized host may become unresponsive.\n"; '  Pick a smaller model: rapid-mlx models' | S only; generic reason → `memory_refused / insufficient_memory`, M |
| [cli.py:2748](../../../rapid_mlx/cli.py#L2748) `_refuse_model_download` | failure in _refuse_model_download | f'\n  Error: {model_name} is not cached and model downloads are disabled by {model_downloads.source()}.\n  Download it with `rapid-mlx pull {model_name}`, then try again.\n' | S only; generic reason → `declined_download / other` |
| [cli.py:3010](../../../rapid_mlx/cli.py#L3010) `_fail_hub_resolution` | failure in _fail_hub_resolution | f'\n{rendered}\n' | S; M owner exists for typed model failures → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:3166](../../../rapid_mlx/cli.py#L3166) `_ensure_target_model_downloaded` | except Exception and isinstance(exc, MirrorOnlyPullError) or (isinstance(exc, OSError) and exc.errno == errno.ENOSPC) | f'\n{render_hub_error(exc, model_name)}\n' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:3736](../../../rapid_mlx/cli.py#L3736) `_reject_tensorfold_family_opt_out_or_exit` | getattr(args, 'no_spec_decode', False) or getattr(args, 'mllm', False) | f'error: {family.profile_id} has no ordinary serving mode. Use {family.fallback_model} instead.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:3766](../../../rapid_mlx/cli.py#L3766) `_require_tensorfold_family_lane_or_exit` | not on_lane | f'error: {family.profile_id} serves only through its TensorFold profile. Use {family.fallback_model} for other configurations.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:3802](../../../rapid_mlx/cli.py#L3802) `_reject_tensorfold_lanes_off_lane_or_exit` | explicit is not None and except ValueError | f'error: {exc}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:3833](../../../rapid_mlx/cli.py#L3833) `_reject_tensorfold_lanes_off_lane_or_exit` | failure in _reject_tensorfold_lanes_off_lane_or_exit | f'error: {message}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4079](../../../rapid_mlx/cli.py#L4079) `_normalize_speculative_config_or_exit` | failure in _normalize_speculative_config_or_exit | f'error: --no-spec-decode is mutually exclusive with {joined}.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4137](../../../rapid_mlx/cli.py#L4137) `_normalize_speculative_config_or_exit` | failure in _normalize_speculative_config_or_exit | f'error: legacy speculative decoding knob {knob} requires {selector}.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4177](../../../rapid_mlx/cli.py#L4177) `_normalize_speculative_config_or_exit` | mtp_max_k is not None and mtp_num_draft_tokens != 1 and (mtp_max_k != mtp_num_draft_tokens) | 'error: legacy MTP aliases mtp_max_k and mtp_num_draft_tokens conflict; pass only one token-count value.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4205](../../../rapid_mlx/cli.py#L4205) `_normalize_speculative_config_or_exit` | getattr(args, 'mtp_optimistic', False) | 'error: legacy speculative decoding knob mtp_optimistic is not supported under the unified spec-decode interface; the vendored MTP installer does not implement optimistic mode. Remove --mtp-optimistic.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4236](../../../rapid_mlx/cli.py#L4236) `_normalize_speculative_config_or_exit` | len(distinct) > 1 | f'error: legacy speculative decoding aliases select multiple methods ({joined}); use one --speculative-config payload.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4255](../../../rapid_mlx/cli.py#L4255) `_normalize_speculative_config_or_exit` | raw_config_was_explicit and legacy_fields | f'error: --speculative-config is mutually exclusive with legacy speculative decoding aliases: {joined}.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4389](../../../rapid_mlx/cli.py#L4389) `_normalize_speculative_config_or_exit` | raw_config is not None and except SpeculativeConfigError | f'error: {exc}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4430](../../../rapid_mlx/cli.py#L4430) `_normalize_speculative_config_or_exit` | raw_config is not None and config is not None and getattr(args, 'mllm', False) and (not companion_dspark) | 'error: --mllm is mutually exclusive with an explicit speculative-decoding request because the vision lane cannot honour speculative decoding. Remove the speculative option to serve images, or remove --mllm to use the accelerated text lane.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4437](../../../rapid_mlx/cli.py#L4437) `_normalize_speculative_config_or_exit` | raw_config is not None and getattr(args, 'no_spec_decode', False) | 'error: --speculative-config is mutually exclusive with --no-spec-decode.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4510](../../../rapid_mlx/cli.py#L4510) `_normalize_speculative_config_or_exit` | not (config.method == 'ddtree') and not (config.method == 'dflash') and not (config.method == 'dspark') and config.method == 'mtp' and continuous_was_explicit and args.mtp_continuous_batching and (continuous_tier != 'verified') and (not getattr(args, 'force_spec_decode', False)) | f'error: continuous MTP is not verified for {args.model!r}: this target {state}. Use ordinary MTP, or pass --force-spec-decode only for an operator-controlled experiment. (Qualification tiers are recorded per catalog alias; unknown local paths or repo ids remain unqualified unless they resolve to a catalog profile.)' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4600](../../../rapid_mlx/cli.py#L4600) `_preflight_native_mtp_or_exit` | unsupported | f"error: native MTP uses a text-only serial server and does not support: {', '.join(unsupported)}" | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4619](../../../rapid_mlx/cli.py#L4619) `_preflight_native_mtp_or_exit` | except NativeMTPUnavailableError | f'error: {exc}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4634](../../../rapid_mlx/cli.py#L4634) `_preflight_native_mtp_or_exit` | not have_runtime() | f"\n  Error: native MTP requires the qualified mlx-vlm {QUALIFIED_MLX_VLM_VERSION} runtime.\n\n  {optional_extra_install_hint('mtp')}\n" | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4642](../../../rapid_mlx/cli.py#L4642) `_preflight_native_mtp_or_exit` | pair.drafter_model_type == 'glm5_next_mtp' and (not have_glm_cache_runtime()) | '\n  Error: GLM-5.3 native MTP requires the complete cache-owned GLM runtime; the installed mlx-vlm build is structurally incompatible.\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4746](../../../rapid_mlx/cli.py#L4746) `_serve_tensorfold_mtp_if_requested` | qualified is None or requested not in (None, qualified) | "error: backend='tensorfold' requires a catalog alias qualified for that method" | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:4816](../../../rapid_mlx/cli.py#L4816) `_preflight_companion_dspark_or_exit` | except CompanionDSparkError | f'error: {exc}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4836](../../../rapid_mlx/cli.py#L4836) `_preflight_companion_dspark_or_exit` | unsupported | f'error: the qualified companion DSpark serial server does not support {joined}.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:4843](../../../rapid_mlx/cli.py#L4843) `_preflight_companion_dspark_or_exit` | not have_runtime() | f'error: LFM companion DSpark requires exactly mlx-vlm {QUALIFIED_MLX_VLM_VERSION}; install the qualified vision runtime.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:5022](../../../rapid_mlx/cli.py#L5022) `_resolve_dflash_launch_or_exit` | except DFlashUnavailable | f'\n  Error: {e}\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:5091](../../../rapid_mlx/cli.py#L5091) `_preflight_dflash_mutexes_or_exit` | spec_config is not None and getattr(spec_config, 'method', None) == 'dflash' and getattr(args, 'suffix_decoding', False) | '\n  Error: DFlash cannot combine with other spec-decode methods. DFlash runs a dedicated single-user server that bypasses BatchedEngine; other spec-decode methods only apply to the BatchedEngine path.\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:5105](../../../rapid_mlx/cli.py#L5105) `_preflight_dflash_mutexes_or_exit` | getattr(args, 'suffix_decoding', False) or getattr(args, 'enable_mtp', False) or getattr(args, 'spec_decode', 'none') != 'none' | '\n  Error: DFlash cannot combine with other spec-decode methods. DFlash runs a dedicated single-user server that bypasses BatchedEngine; other spec-decode methods only apply to the BatchedEngine path.\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:5112](../../../rapid_mlx/cli.py#L5112) `_preflight_dflash_mutexes_or_exit` | getattr(args, 'no_spec_decode', False) | 'error: DFlash and --no-spec-decode are mutually exclusive — DFlash is a speculative-decode mode.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:5194](../../../rapid_mlx/cli.py#L5194) `_preflight_tensorfold_qwen27_or_exit` | issue is not None | f'\n  Error: {issue}\n' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:5207](../../../rapid_mlx/cli.py#L5207) `_preflight_ddtree_or_exit` | spec_config is None or spec_config.method != 'ddtree' | 'error: DDTree requires --speculative-config \'{"method":"ddtree"}\'.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:5214](../../../rapid_mlx/cli.py#L5214) `_preflight_ddtree_or_exit` | getattr(args, 'enable_dflash', False) | '\n  Error: DDTree cannot combine with DFlash. Pick one block-diffusion speculative-decoding server.\n' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:5225](../../../rapid_mlx/cli.py#L5225) `_preflight_ddtree_or_exit` | getattr(args, 'suffix_decoding', False) or getattr(args, 'spec_decode', 'none') != 'none' | '\n  Error: DDTree cannot combine with other spec-decode methods. DDTree runs a dedicated single-user server that bypasses BatchedEngine; other spec-decode methods only apply to the BatchedEngine path.\n' | S only; generic reason → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:5232](../../../rapid_mlx/cli.py#L5232) `_preflight_ddtree_or_exit` | getattr(args, 'no_spec_decode', False) | 'error: DDTree and --no-spec-decode are mutually exclusive — DDTree is a speculative-decode mode.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:5286](../../../rapid_mlx/cli.py#L5286) `_preflight_ddtree_or_exit` | except DDTreeUnavailable | f'\n  Error: {e}\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:5298](../../../rapid_mlx/cli.py#L5298) `_preflight_ddtree_or_exit` | not have_runtime() | f'\n  Error: DDTree requires the experimental dtree-mlx runtime. Install with: `{DDTREE_INSTALL_COMMAND}`.{detail}\n' | S only; generic reason → `exception / missing_extra`, M |
| [cli.py:5776](../../../rapid_mlx/cli.py#L5776) `_reject_embedding_alias_serve` | failure in _reject_embedding_alias_serve | f"error: '{model_name}' is a sentence-embedding alias and has no chat surface, so it cannot be served as the main model.\nServe it as the embeddings backend instead, alone or next to a chat model:\n    rapid-mlx serve --embedding-model {model_name}\n    rapid-mlx serve <chat-alias> --embedding-model {model_name}" | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6264](../../../rapid_mlx/cli.py#L6264) `_validate_v41_product_spec_flags` | getattr(args, '_speculative_config', None) is not None or getattr(args, 'no_spec_decode', False) | 'error: the DeepSeek V4.1 product lane always uses its qualified DSpark K4 runtime; do not pass --speculative-config or --no-spec-decode.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6494](../../../rapid_mlx/cli.py#L6494) `_serve_cua_only_mode` | not configured_api_key | 'rapid-mlx serve --cua-only requires an API key via --api-key or RAPID_MLX_API_KEY.' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6507](../../../rapid_mlx/cli.py#L6507) `_serve_cua_only_mode` | incompatible | 'rapid-mlx serve --cua-only cannot be combined with ' + ', '.join(incompatible) | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6616](../../../rapid_mlx/cli.py#L6616) `_validate_local_search_exposure_or_exit` | except ValueError | f'error: {exc}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6855](../../../rapid_mlx/cli.py#L6855) `resolve_serve_pflash_config` | except ValueError | f'Error: {e}' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6878](../../../rapid_mlx/cli.py#L6878) `resolve_serve_kv_cache` | _continuous_cache_conflict is not None | f'\n  Error: {_continuous_cache_conflict}\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6893](../../../rapid_mlx/cli.py#L6893) `resolve_serve_kv_cache` | _kv_conflict is not None | f'\n  Error: {_kv_conflict}\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6940](../../../rapid_mlx/cli.py#L6940) `resolve_serve_kv_cache` | not args.kv_cache_turboquant and (not args.kv_cache_quantization) and except KVCacheQuantizationUnsupportedError | f'\n  Error: {e}\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:6994](../../../rapid_mlx/cli.py#L6994) `resolve_serve_kv_cache` | not (not args.kv_cache_turboquant and (not args.kv_cache_quantization)) and args.kv_cache_quantization and except KVCacheQuantizationUnsupportedError | f'\n  Error: {e}\n' | S only; generic reason → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7282](../../../rapid_mlx/cli.py#L7282) `serve_command` | except ValueError | f'error: {exc}' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7290](../../../rapid_mlx/cli.py#L7290) `serve_command` | except (ValueError, OSError) | f'error: {exc}' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7301](../../../rapid_mlx/cli.py#L7301) `serve_command` | except log_file.LogFileError | f'error: {exc}' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7322](../../../rapid_mlx/cli.py#L7322) `serve_command` | except ValueError | f'error: {exc}' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7353](../../../rapid_mlx/cli.py#L7353) `serve_command` | retrieval_only and incompatible | 'rapid-mlx serve without a model (retrieval-only) cannot be combined with ' + ', '.join(incompatible) | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7367](../../../rapid_mlx/cli.py#L7367) `serve_command` | (bounds_error := _vision_pixel_bounds_error(getattr(args, 'vision_min_pixels', 0), getattr(args, 'vision_max_pixels', 0))) | f'error: {bounds_error}' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7383](../../../rapid_mlx/cli.py#L7383) `serve_command` | except (OSError, RuntimeError, ValueError) | f'error: cannot configure video output directory: {exc}' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7485](../../../rapid_mlx/cli.py#L7485) `serve_command` | _owns_v41_product_download and (not 1 <= effective_max_tokens <= 4096) | 'error: the experimental DeepSeek V4.1 DSpark K4 server currently supports 1 <= --max-tokens <= 4096.' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7592](../../../rapid_mlx/cli.py#L7592) `serve_command` | _owns_v41_product_download and args.mcp_config | message | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:7642](../../../rapid_mlx/cli.py#L7642) `serve_command` | _wants_dflash and not have_runtime() | f"""\n  Error: DFlash speculative decoding (--speculative-config '{{"method":"dflash"}}') requires mlx-vlm 0.5.0+ for the DFlash drafter hooks.\n\n  {optional_extra_install_hint('dflash')}\n""" | S only; no M on this validation branch → `exception / missing_extra`, M |
| [cli.py:7796](../../../rapid_mlx/cli.py#L7796) `serve_command` | prompt_upgrade_if_available() | Caller/callee diagnostic; see ranked table and source. | S only; no M on this validation branch → No failed event (clean exit) |
| [cli.py:7999](../../../rapid_mlx/cli.py#L7999) `serve_command` | _serve_profile is not None and _serve_profile.modality == 'image-gen' and _missing | f'\n  Error: {_shown} is only partially downloaded — {len(_missing)} required file(s) missing, starting with {_missing[0]}.\n  Finish the download with `rapid-mlx pull {_shown}`, then serve again.\n' | S only; no M on this validation branch → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:8021](../../../rapid_mlx/cli.py#L8021) `serve_command` | args.enable_auto_tool_choice and (not args.tool_call_parser) | 'Error: --enable-auto-tool-choice requires --tool-call-parser'; 'Example: --enable-auto-tool-choice --tool-call-parser mistral' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8056](../../../rapid_mlx/cli.py#L8056) `serve_command` | args.tool_call_parser and valid and args.tool_call_parser not in valid | f"error: argument --tool-call-parser: invalid choice: {args.tool_call_parser!r} (choose from: {', '.join(valid)})" | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8065](../../../rapid_mlx/cli.py#L8065) `serve_command` | args.gpu_memory_utilization is not None and (not 0.0 < args.gpu_memory_utilization <= 1.0) | 'Error: --gpu-memory-utilization must be between 0.0 (exclusive) and 1.0 (inclusive)' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8068](../../../rapid_mlx/cli.py#L8068) `serve_command` | getattr(args, 'resident_memory_limit_gb', 0.0) < 0 | 'Error: --resident-memory-limit-gb must be >= 0' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8071](../../../rapid_mlx/cli.py#L8071) `serve_command` | getattr(args, 'resident_model_idle_ttl', 0.0) < 0 | 'Error: --resident-model-idle-ttl must be >= 0' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8078](../../../rapid_mlx/cli.py#L8078) `serve_command` | idle_cache_clear_seconds is not None and not math.isfinite(idle_cache_clear_seconds) or idle_cache_clear_seconds < 0 | 'Error: --idle-cache-clear-seconds must be finite and >= 0' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8149](../../../rapid_mlx/cli.py#L8149) `serve_command` | args.tool_call_parser and _opt_out_tool | 'error: --tool-call-parser and --no-tool-call-parser are mutually exclusive — pick one to override auto-detection.' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8156](../../../rapid_mlx/cli.py#L8156) `serve_command` | args.reasoning_parser and _opt_out_reasoning | 'error: --reasoning-parser and --no-reasoning-parser are mutually exclusive — pick one to override auto-detection.' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8435](../../../rapid_mlx/cli.py#L8435) `serve_command` | args.reasoning_parser and except KeyError | f'Error: {e}' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8439](../../../rapid_mlx/cli.py#L8439) `serve_command` | args.reasoning_parser and except ImportError | f'Error: Failed to import reasoning module: {e}'; 'Run `rapid-mlx doctor` to check this installation.' | S only; no M on this validation branch → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:8446](../../../rapid_mlx/cli.py#L8446) `serve_command` | args.reasoning_parser and except Exception | f"Error: Failed to initialize reasoning parser '{args.reasoning_parser}': {e}"; 'Retry without --reasoning-parser, or run `rapid-mlx doctor`.' | S only; no M on this validation branch → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:8462](../../../rapid_mlx/cli.py#L8462) `serve_command` | args.enable_dflash and args.suffix_decoding | '\n  Error: DFlash cannot combine with other spec-decode methods. DFlash runs a dedicated single-user server that bypasses BatchedEngine; other spec-decode methods only apply to the BatchedEngine path.\n' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8676](../../../rapid_mlx/cli.py#L8676) `serve_command` | args.enable_dflash and getattr(args, 'no_spec_decode', False) | 'error: DFlash and --no-spec-decode are mutually exclusive — DFlash is a speculative-decode mode.' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:8991](../../../rapid_mlx/cli.py#L8991) `serve_command` | getattr(args, 'spec_decode', 'none') == 'dspark' and metadata is None | 'error: DSpark requires a complete DeepSeek V4 DSpark checkpoint (3 mtp stages plus Markov heads).' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:9001](../../../rapid_mlx/cli.py#L9001) `serve_command` | getattr(args, 'spec_decode', 'none') == 'dspark' and requested_k != metadata.block_size | f"error: requested DSpark num_speculative_tokens={requested_k} does not match checkpoint block_size={metadata.block_size}. Rapid-MLX currently requires the checkpoint's complete DSpark block." | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:9049](../../../rapid_mlx/cli.py#L9049) `serve_command` | getattr(args, 'spec_decode', 'none') == 'mtp' and eligibility is MTPEligibility.NONE | '  🐆 Rapid-MLX'; '  ─────────'; f'  Model: {args.model}' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:9162](../../../rapid_mlx/cli.py#L9162) `serve_command` | args.mllm and args.no_mllm | 'error: --mllm and --no-mllm are mutually exclusive — pick one to override auto-detection.' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:9169](../../../rapid_mlx/cli.py#L9169) `serve_command` | getattr(args, 'force_hybrid', False) and getattr(args, 'no_hybrid', False) | 'error: --force-hybrid and --no-hybrid are mutually exclusive — pick one to override auto-detection.' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:9178](../../../rapid_mlx/cli.py#L9178) `serve_command` | getattr(args, 'force_spec_decode', False) and getattr(args, 'no_spec_decode', False) | 'error: --force-spec-decode and --no-spec-decode are mutually exclusive — pick one to override auto-detection.' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:9188](../../../rapid_mlx/cli.py#L9188) `serve_command` | getattr(args, 'force_openai_harmony_streaming', False) and getattr(args, 'no_openai_harmony_streaming', False) | 'error: --force-openai-harmony-streaming and --no-openai-harmony-streaming are mutually exclusive — pick one to override the HarmonyStreamingRouter auto-upgrade gate (#516).' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:9238](../../../rapid_mlx/cli.py#L9238) `serve_command` | except KVCacheQuantizationUnsupportedError | f'\n  Error: {e}\n' | S only; no M on this validation branch → `usage_error / other`; M only for a known typed model cause |
| [cli.py:9253](../../../rapid_mlx/cli.py#L9253) `serve_command` | except Exception | '  🐆 Rapid-MLX'; '  ─────────'; f'  Model: {args.model}' | S only; no M on this validation branch → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:17656](../../../rapid_mlx/cli.py#L17656) `main` | hasattr(args, 'model') and args.model and (_local_audio is None) and (getattr(args, 'command', None) not in ('doctor', 'service', 'system-one')) and except (OSError, InvalidModelConfig) and not validating_local_model and (not isinstance(exc, (FileNotFoundError, NotADirectoryError, PermissionError))) | f'\n  Error: Model resolution failed: {exc}' | S; M owner exists for typed model failures → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:17674](../../../rapid_mlx/cli.py#L17674) `main` | hasattr(args, 'model') and args.model and (_local_audio is None) and (getattr(args, 'command', None) not in ('doctor', 'service', 'system-one')) and except (OSError, InvalidModelConfig) and getattr(args, 'command', None) == 'serve' and except log_file.LogFileError | f'error: {log_exc}' | S; local-model M can already exist → `usage_error / other` |
| [cli.py:17708](../../../rapid_mlx/cli.py#L17708) `main` | hasattr(args, 'model') and args.model and (_local_audio is None) and (getattr(args, 'command', None) not in ('doctor', 'service', 'system-one')) and except (OSError, InvalidModelConfig) | Caller/callee diagnostic; see ranked table and source. | S; M owner exists for typed model failures → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [cli.py:17713](../../../rapid_mlx/cli.py#L17713) `main` | hasattr(args, 'model') and args.model and (_local_audio is None) and (getattr(args, 'command', None) not in ('doctor', 'service', 'system-one')) and except (RetiredModelAliasError, UserAliasError) | f'\n  Error: {exc}' | S only; no M → `usage_error / invalid_model_ref`, M fallback |
| [cli.py:17775](../../../rapid_mlx/cli.py#L17775) `main` | hasattr(args, 'model') and args.model and (_local_audio is None) and (getattr(args, 'command', None) not in ('doctor', 'service', 'system-one')) and not (resolved != args.model) and '/' not in args.model and (not os.path.exists(args.model)) and not is_audio_model_alias(args.model) and (not quicksilver_catalog) | f"\n  Error: '{args.model}' is not a known alias or HuggingFace path." | S only; no M → `usage_error / invalid_model_ref`, M fallback |
| [cli.py:17838](../../../rapid_mlx/cli.py#L17838) `main` | hasattr(args, 'model') and args.model and (_local_audio is None) and (getattr(args, 'command', None) not in ('doctor', 'service', 'system-one')) and getattr(args, 'command', None) in ('serve', 'bench', 'chat', 'run') and (not (getattr(args, 'base_url', None) or (getattr(args, 'command', None) in ('bench', 'chat', 'run') and getattr(args, 'port', None) is not None))) and except DraftModelNotServableError | f'\n  Error: {exc}' | S; M owner exists for typed model failures → Closed reason + `class(exc)` / subtype; M fallback unless already owned |
| [embedding.py:124](../../../rapid_mlx/embedding.py#L124) `require_mlx_embeddings_or_exit` | is_native_embedding_model(model_name) | 'error: EmbeddingGemma 2 requires the mlx-vlm native runtime. ' + optional_extra_install_hint('vision') | S only; generic reason → `exception / missing_extra`, M |
| [embedding.py:132](../../../rapid_mlx/embedding.py#L132) `require_mlx_embeddings_or_exit` | failure in require_mlx_embeddings_or_exit | 'error: --embedding-model requires the [embeddings] extra. ' + EMBEDDINGS_EXTRA_INSTALL_HINT | S only; generic reason → `exception / missing_extra`, M |
| [server.py:4558](../../../rapid_mlx/server.py#L4558) `main` | except ValueError | `usage: ...`, then `{prog}: error: ` + str(exc) | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:4568](../../../rapid_mlx/server.py#L4568) `main` | except ValueError | `usage: ...`, then `{prog}: error: ` + str(exc) | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:4611](../../../rapid_mlx/server.py#L4611) `main` | except (OSError, RuntimeError, ValueError) | `usage: ...`, then `{prog}: error: ` + f'cannot configure video output directory: {exc}' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:4771](../../../rapid_mlx/server.py#L4771) `main` | args.tool_call_parser and _opt_out_tool | `usage: ...`, then `{prog}: error: ` + '--tool-call-parser and --no-tool-call-parser are mutually exclusive' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:4775](../../../rapid_mlx/server.py#L4775) `main` | args.reasoning_parser and _opt_out_reasoning | `usage: ...`, then `{prog}: error: ` + '--reasoning-parser and --no-reasoning-parser are mutually exclusive' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:4919](../../../rapid_mlx/server.py#L4919) `main` | except ValueError | `usage: ...`, then `{prog}: error: ` + str(e) | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:4947](../../../rapid_mlx/server.py#L4947) `main` | args.vision_min_pixels < 0 or args.vision_max_pixels < 0 | `usage: ...`, then `{prog}: error: ` + 'vision pixel bounds must be non-negative' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:4953](../../../rapid_mlx/server.py#L4953) `main` | args.vision_min_pixels and args.vision_max_pixels and (args.vision_min_pixels > args.vision_max_pixels) | `usage: ...`, then `{prog}: error: ` + '--vision-min-pixels must not exceed --vision-max-pixels' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:4966](../../../rapid_mlx/server.py#L4966) `main` | vision_prefill_token_budget <= 0 | `usage: ...`, then `{prog}: error: ` + '--vision-prefill-token-budget must be positive' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:5011](../../../rapid_mlx/server.py#L5011) `main` | args.mllm and args.no_mllm | `usage: ...`, then `{prog}: error: ` + '--mllm and --no-mllm are mutually exclusive' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:5013](../../../rapid_mlx/server.py#L5013) `main` | getattr(args, 'force_hybrid', False) and getattr(args, 'no_hybrid', False) | `usage: ...`, then `{prog}: error: ` + '--force-hybrid and --no-hybrid are mutually exclusive' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:5017](../../../rapid_mlx/server.py#L5017) `main` | getattr(args, 'force_spec_decode', False) and getattr(args, 'no_spec_decode', False) | `usage: ...`, then `{prog}: error: ` + '--force-spec-decode and --no-spec-decode are mutually exclusive' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |
| [server.py:5021](../../../rapid_mlx/server.py#L5021) `main` | getattr(args, 'force_openai_harmony_streaming', False) and getattr(args, 'no_openai_harmony_streaming', False) | `usage: ...`, then `{prog}: error: ` + '--force-openai-harmony-streaming and --no-openai-harmony-streaming are mutually exclusive' | S only at standalone preflight; generic reason → `usage_error / other`; M only for a known typed model cause |


The generic exception guards in `cli._capture_start_failures`,
`server._capture_start_failures`, `server_start.failure_stage`, and
`_uvicorn.run_uvicorn` also cover exceptions escaping any called preflight,
metadata, configuration, filesystem, dependency, or bind operation. Their
message is the original callee diagnostic/traceback plus the new recovery hint
for unhandled exceptions. A finite table cannot enumerate arbitrary exceptions
from optional third-party runtimes; the closed `other` + subtype fallback makes
those paths observable without extracting free text.

Soft alias memory warnings, BYOM metadata timeouts/no-verdict outcomes, and
transient prefetch warnings that say `server will retry` are not failed-start
paths by themselves. A later failing loader is reported at its actual stage.
A native abort or forced kill cannot run Python exception reporting; the
existing inflight marker can only report `previous_run_unterminated` on a later
attempt. No inferred cause is attached to those terminations.

An unexplained nonzero `SystemExit` without an explicit cause or bound refusal
reason now also prints `Error: Server startup exited before listening (exit
{code}).` and ``Next step: run `rapid-mlx doctor` to check this installation. If
it passes, report the startup output.`` Already diagnosed typed exits retain their
specific diagnostic, and successful exits remain silent here.

## Verification environment

The reporting tests run under Python 3.12 via `uv run --extra test`, with
`--no-sync` after removing native MLX packages from the worktree virtualenv:
Metal initialization aborts in this filesystem/process sandbox. The uv cache is
in `/private/tmp/website-poshog-preflight-uv`, per Studio scratch policy. No model
weights or production telemetry were used. Existing tests inject telemetry or
post only to loopback.

The main reporting/startup selection passed 1,310 tests with 9 skips; the
pre-reboot process-marker test was deselected after reproducing its failure
using the original `server_start.py` loaded from HEAD. A focused follow-up for
optional-runtime recovery ownership also passed. The additional native/listener
selection passed 285 tests with 40 skips and four environment failures. Those
same four failures reproduce using HEAD's original CLI: the installed vision
runtime cannot load without MLX, and the qualified native adapter test requires
that runtime. These are disclosed limits, not passing inference checks.

The negative-control reporting regression against HEAD fails because
`server_start_state.error_class` is absent. Current regressions check every
startup stage, typed refusals through `SystemExit` causes, explicit validation
reasons, privacy and schema rejection, generic and silent-exit recovery messages,
optional-runtime recovery ownership, and suppression of a separate model event.
Ruff check, Ruff format check, and `git diff --check` pass.

The main selection command (the final diagnostic-only change was subsequently
checked by the 196-pass startup file run):

```bash
UV_CACHE_DIR=/private/tmp/website-poshog-preflight-uv \
uv run --no-sync --extra test python -m pytest \
  tests/test_server_start_state.py tests/test_telemetry_model_events.py \
  tests/test_telemetry_model_id.py tests/test_telemetry_cli.py \
  tests/test_telemetry_loop_dedupe.py tests/test_telemetry_registry.py \
  tests/test_telemetry_registry_drift.py tests/test_telemetry_track.py \
  tests/test_optional_runtime.py tests/test_serve_ready_after_bind.py \
  tests/test_byom_preflight.py tests/test_catalog_memory_preflight.py \
  tests/test_engine_startup_errors.py tests/test_serve_port_collision.py \
  tests/test_serve_local_preflight.py tests/test_local_serve_diagnostics.py \
  tests/test_remaining_serve_diagnostics.py -q \
  --deselect tests/test_server_start_state.py::test_pid_reuse_does_not_hide_pre_reboot_marker

UV_CACHE_DIR=/private/tmp/website-poshog-preflight-uv uv run --no-sync --extra test ruff check .
UV_CACHE_DIR=/private/tmp/website-poshog-preflight-uv uv run --no-sync --extra test ruff format --check .
git diff --check
```
