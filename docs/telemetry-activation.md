# Telemetry activation / engagement semantics

**Spec version: `2`** (`vllm_mlx.telemetry.activation_spec.ACTIVATION_SPEC_VERSION`)
· Status: **active** · Owner: growth

This document is the single, versioned source of truth for what counts as
an **engaged** / **activated** install. The growth dashboard and the
repository's regression tests both encode these rules; when the rules
change, bump `ACTIVATION_SPEC_VERSION` and update both sides in lockstep.

## Why this exists

DAU / WK1-retention / announcement-recall experiments are only trustworthy
if "an active user" means *someone who did real work*, not *a process that
started and phoned home*. Without a fixed definition, a rise in DAU could be
nothing but more `serve` boots, update polls, or health checks. This spec
pins the behavioral definition so the baseline can't drift.

## Definitions

**Engaged** — the install completed at least one *successful inference*.
This is the load-bearing signal for engaged-DAU and retention cohorts.

**Activation** — the install crossed a funnel milestone for the *first time
ever*. Each milestone represents one install reaching that step, keyed on the
persistent `client_id`. Activation is a superset of "engaged": the
`first_inference` milestone is the moment an install first becomes engaged.

Emission is **at-least-once**, not exactly-once (see "Delivery semantics"
below): the client suppresses repeats locally on a best-effort basis, and the
**collector is the authoritative de-duplicator** — it counts *distinct*
`client_id`, so a rare duplicate for the same install collapses to one.

## What counts

| Trigger | Activation? | `activation_kind` | Notes |
|---|---|---|---|
| First successful inference request (HTTP API) | ✅ engaged | `first_inference` | `surface = api`. Emitted after the response is built (non-streaming) or the stream completes normally (streaming). **Implemented.** |
| CLI `chat` stream completes normally with ≥1 completion token | ✅ engaged | `first_inference` | `surface = cli`. Emitted after the stream drains normally, not at the first token — a partially consumed / client-cancelled stream is conservatively not counted. **Implemented.** |
| `rapid-mlx pull <alias>` completes successfully | ✅ activation | `model_pull` | `surface = cli`; NOT inference-engaged. **Implemented.** |
| Agent integration setup completes AND its connection check passes | ✅ activation | `agent_setup` | `surface = cli`; integration activation. **Spec'd; wiring deferred** — `rapid-mlx launch` has no engine connection check today; that check must land first (see below). |
| Desktop receives its first successful text-only assistant reply | ✅ Desktop activation | `first_chat_reply` | `surface = desktop`. **Implemented.** |
| Desktop receives its first successful assistant reply to a vision turn | ✅ Desktop activation | `first_vision_reply` | `surface = desktop`. **Spec'd; wiring deferred.** |
| Desktop delivers its first successful dictation transcript | ✅ Desktop activation | `first_dictation` | `surface = desktop`. **Implemented.** |
| Desktop generates its first image | ✅ Desktop activation | `first_image` | `surface = desktop`. **Implemented.** Editing an existing image does not count. |

## What does NOT count

- **Server startup** (`serve` boot / `session_start`) — a running process is
  not engagement.
- **`/health`, `/models`, and the update/version ping** — liveness and
  metadata probes, never engagement.
- **Error responses** — any non-`2xx` inference request.
- **Empty generation** — a `2xx` request that produced **zero** completion
  tokens (`completion_tokens == 0`). "It ran" is not "it worked."
- **Model load / warmup** by itself — loading weights is not inference.

Degenerate-but-non-empty output (repetitive / low-quality) **does** count as
engaged: the user got tokens. Quality is tracked separately via the
`request` event's `output_degenerate` canary, not here.

## Success criteria, precisely

A request is a **successful inference** iff **both**:

1. HTTP status is `2xx`, and
2. `completion_tokens > 0` (non-empty generation).

Only generative endpoints are inference for this purpose. Today the
**active, instrumented** endpoint is `/v1/chat/completions` (streaming and
non-streaming). `/v1/completions` and `/v1/messages` are generative too but
are **not wired yet** — they are excluded from `INFERENCE_ENDPOINTS` until
their `request`/activation instrumentation lands, so the spec never claims
coverage the code doesn't deliver.

## The `activation` event

A dedicated low-frequency telemetry event (opt-in, consent-gated, **never
sampled** — unlike `request`). Envelope is the standard telemetry envelope
(`client_id`, `session_id`, `rapid_mlx_version`, `platform`, `timestamp`);
the event-specific payload is:

```json
{ "activation_kind": "first_inference" | "model_pull" | "agent_setup"
                   | "first_chat_reply" | "first_vision_reply"
                   | "first_dictation" | "first_image",
  "surface":         "cli" | "api" | "desktop" }
```

No prompt, no completion, no content of any kind — two enums only. This is
the same privacy class as the `first_session` / `auto_selected` booleans.
Kinds and surfaces are a closed pair contract, not independent allowlists: only
the combinations listed in the milestone table above are valid. Engine
`first_inference` may use `cli` or `api`; `model_pull` and `agent_setup` use
`cli`; every Desktop kind uses `desktop`. Producers drop any other pairing
before creating a marker or event.

### Why a dedicated event and not derivation from `request`

The spec's preference is to **derive** engagement in the collector from
existing `session` / `request` events and avoid new high-frequency events.
That is not possible for `first_inference`: `request` events are sampled at
10% (`RAPID_MLX_TELEMETRY_REQUEST_SAMPLE`), so ~90% of installs' *first*
successful inference is never emitted — a first-touch milestone cannot be
reconstructed from a 1-in-10 sample. `activation` is therefore emitted
**unsampled** at the success chokepoint. It stays low-frequency by
construction: the local marker suppresses it to ~one emission per install per
`activation_kind` (see "Delivery semantics" for the at-least-once caveat), so
it can never become a high-volume stream.

### Delivery semantics — at-least-once, deduped by `client_id`

Each engine `activation_kind` is guarded by a local marker file
`~/.rapid-mlx/activation_seen_<kind>`; Desktop uses
`activation_seen_desktop_<kind>` in the same shared directory. Both are
created with exclusive (`O_CREAT | O_EXCL`) semantics — the same primitive as
`mark_first_session`. In the common case the first accepted emission claims
the marker and every later call (this process, via an in-memory latch; other
processes, via the marker) is a silent no-op.

The engine client deliberately does **not** attempt exactly-once across processes.
It **enqueues first, then claims the marker**, so a transient envelope/enqueue
failure leaves the marker unclaimed and the next successful inference simply
retries — an install is never permanently dropped from the funnel by one
queue hiccup. The tradeoff is that two processes racing the marker check can
both enqueue; this is acceptable because the **collector de-duplicates by
`client_id`** (it counts distinct installs, and — per the worker's
aggregation — `activation` events are excluded from the per-session
engaged-ratio accumulation, so a duplicate cannot inflate any metric). Losing
an install (at-most-once) is a strictly worse error for a growth signal than a
harmless, dedup-able duplicate (at-least-once). The marker is a local empty
file; only the derived enum pair ever leaves the machine, and only when
telemetry is enabled.

Desktop uses the same at-least-once contract with one stricter transport
boundary: it sends first and claims its marker only after the collector returns
`2xx`. Transport errors, `408`, `429`, and `5xx` remain retryable; a permanent
non-retryable rejection is not reported as accepted. Consent is re-checked
before the marker is claimed, so an opt-out racing an accepted send can produce
only a de-duplicable retry, never a burned milestone.

`kind` is validated against the allowlist before it is ever interpolated into
the marker filename, so no caller-controlled string can escape `~/.rapid-mlx`.

### `surface` resolution

`surface` distinguishes the CLI REPL, HTTP API, and native Desktop app.
Desktop always emits the literal `desktop`; it never derives this label from a
model, feature name, or request. Because
`rapid-mlx chat` runs inference by spawning its own ephemeral `serve` and
looping through `/v1/chat/completions`, the `first_inference` milestone is
emitted at the **server-side** success chokepoint for both surfaces; the
surface is derived from the marker `rapid-mlx chat` **already** sets on the
server it spawns — `RAPID_MLX_CHAT_SPAWN=1`. A chat-spawned server is the
`cli` surface; a standalone `serve` is `api`. Reusing the existing marker
means no new env var, and a single emission site (no double-counting across
the chat front-end and its spawned server).

Edge: `rapid-mlx chat --base-url` / `--port` connects to a pre-existing
server instead of spawning one, so that server attributes the inference to
its own surface (`api` for a standalone `serve`). The auto-spawn path — the
common case — attributes to `cli`.

**Surface under the at-least-once duplicate.** Because emission is
at-least-once (see "Delivery semantics"), one pathological interleaving can
send two `first_inference` events for the *same* install with *different*
`surface`: a chat-spawned server (`cli`) and a standalone `serve` (`api`) both
running their very first successful inference for the same `client_id` before
either claims the marker. This is deliberately treated as a best-effort
*secondary* label: the load-bearing **engaged** metric counts distinct
`client_id` and is unaffected (the install is engaged exactly once either way);
only the cli-vs-api attribution of that one install may be order-dependent, and
only in this rare same-install double-server race. Making the surface itself
deterministic would require claiming before enqueue (reintroducing the
permanent-suppression-on-failure hazard this design exists to avoid) or a
collector-side tie-break; both are out of scope for a secondary dimension whose
skew is bounded to at most one install per genuinely-concurrent double-serve.

### `agent_setup` is deferred

The spec defines `agent_setup` as "integration setup **and** a passing
engine connection check." Today `rapid-mlx launch <agent>` writes the
integration config but performs **no** connectivity probe against the local
engine, so there is nothing to gate the milestone on. Wiring `agent_setup`
is therefore blocked on first adding that connection check to `launch`;
until then the enum value is reserved and never emitted.

## Dashboard contract

- **Engaged install**: a `client_id` with ≥1 `activation` event where
  `activation_kind = first_inference`.
- **Engaged-DAU / WK1**: build activation cohorts from the `activation`
  event's `timestamp`, then measure return with the existing `session_start`
  stream. Do **not** define "active" as "has a `session_start`" — that
  counts boots and polls.
- **Activation funnel**: `model_pull` → `first_inference`, and
  `agent_setup` as the integration path, all keyed on the shared
  `client_id`.
- **Desktop activation**: count only events with `surface = desktop`, grouped
  by the four Desktop kinds. Never add them to engine `first_inference`
  (`surface = cli|api`): the shared consent/client ID means one install can
  legitimately emit both.

Both this document and the repository tests (`tests/test_telemetry_activation.py`)
reference `ACTIVATION_SPEC_VERSION`. Any change to the rules above bumps
that constant.
