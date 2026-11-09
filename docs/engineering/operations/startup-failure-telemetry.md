# Startup failure telemetry

`model_serve_failed` emits `failure_subtype` only when `error_class=other`.
This applies equally to `resolve`, `prepare`, and `engine_start`; a known error
class still takes precedence. The subtype prefers an annotated operation over
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
