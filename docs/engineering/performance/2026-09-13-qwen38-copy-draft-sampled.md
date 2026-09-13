# Qwen3.8 copy-draft for sampled and cache-reusing requests

Date: 2026-09-13

## Decision

Let the Qwen3.5/3.8 family copy-draft for `temperature > 0` requests, and
index the request's own prompt rather than the batch row so a prefix-cache hit
keeps copying. Both are the same bug from the user's side: copy-draft shipped
in #3388/#3398 and a desktop chat never once used it.

Two independent gates had to fall:

1. **Temperature.** `generator.py` and `scheduler.py` both gated prompt lookup
   on `temp == 0`. The API's default temperature is `0.0`, so OpenAI-compatible
   callers who omit the field did get the speedup; the desktop app's persisted
   default is `SamplingConfig.temperatureDefault = 0.7` and it always sends a
   value, so every GUI chat ran with the feature off. Greedy was never a
   correctness requirement — see *Correctness* below — so this is now a
   per-family declaration (`PromptLookupPolicy.enabled_under_sampling`), not a
   global flip.
2. **Prefix-cache reuse.** The scheduler indexed
   `list(gb.tokens[0]) + [first_tok]`, and a reused prefix never passes through
   the batch: on a hit the row begins at the *unprocessed tail*. So the copy
   index was built over a stub exactly when the prompt was longest. Measured
   below: the same 236-token request proposes 12 copies cold and **zero** on
   the byte-identical next request. For a chat that is every turn after the
   first, which is to say all the turns that have something to copy from.

## Environment

- Mac mini (Mac16,11), Apple M4 Pro, 48 GB unified memory
- Python 3.13, MLX 0.32.2
- Target `mlx-community/Qwen3.8-27B-4bit`, sidecar
  `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` (alias `qwen3.8-27b-4bit`)
- Base `7a665912c` (both #3388 and #3398 merged), `max_k=3`, hybrid GDN target
- Shared machine: wall-clock has roughly ±15% of noise on it, so every
  comparison below is interleaved or repeated, and the engine-level table is a
  mean of three interleaved reps per cell

The arms are the operator knob, not a source edit. `RAPID_MLX_MTP_PROMPT_LOOKUP_SAMPLED=0`
reproduces shipped behaviour for a sampled request; unset is this branch. The
knob is read per request, but the server owns the environment, so each arm is
its own server process with everything else identical. It is disable-only by
design: it can put a qualified family's sampled requests back on the greedy
route, and cannot put an unqualified family onto the sampled one, because the
declaration is the family's to make.

## Engine-level A/B (`mtp_generate_step` directly, 1200 tokens)

Interleaved off/on, three reps per cell, mean tok/s:

| Task | temp | copy off | copy on | Delta | Accepted of copied |
|---|---:|---:|---:|---:|---:|
| rename | 0.0 | 21.21 | 25.81 | +21.7% | 54.5–54.7% |
| rename | 0.7 | 21.74 | 26.45 | **+21.7%** | 52.5–57.7% |
| bugfix | 0.0 | 20.80 | 29.34 | +41.1% | 81.6% |
| bugfix | 0.7 | 20.55 | 27.53 | **+34.0%** | 71.4–78.0% |

Acceptance is the number that decides whether sampling is worth qualifying,
and it barely moves: a model copying from its own prompt is confident exactly
where it is copying, so `p(token)` — which *is* the acceptance probability for
a point-mass proposal — stays close to the argmax match rate.

## Served A/B (HTTP, streaming, temperature 0.7)

`decode tok/s` is measured over the token-to-token window, after the first
chunk, because a prefix-cache hit removes prefill and end-to-end tok/s then
mostly reports that. Three reps per task: one cold, two on the same prompt
(hits). Server counters are read per request from `GET /v1/status`.

| Task | copy off (3 reps) | copy on (3 reps) | Delta | Accepted of copied |
|---|---:|---:|---:|---:|
| rename | 26.84 / 26.73 / 26.13 | 25.56 / 27.97 / 25.82 | −0.5% | 84 of 171 (49%) |
| bugfix | 26.89 / 27.15 / 24.80 | 41.23 / 40.57 / 41.65 | **+56.6%** | 131 of 173 (76%) |
| doc | 25.88 / 23.39 / 23.71 | 28.21 / 24.64 / 23.79 | +5.0% | 21–125 of 30–230 |

The rename row is the useful floor: at 49% acceptance the copies pay for
themselves and no more, and the sizing gate from #3398 is what keeps that from
going negative. The spread within a task is the machine, not the feature.

## The multi-turn case, which is what a chat is

Three turns on one conversation: paste a module and ask for an edit, then ask
for a second edit, then a third. Turns 2 and 3 hit the prefix cache, and what
they want to copy lives in the reused part of the prompt.

| Turn | copy off | copy on | Delta | Accepted of copied (on) |
|---|---:|---:|---:|---:|
| 1 (cold) | 25.30 | 25.41 | +0.4% | 84 of 171 (49%) |
| 2 (cache hit) | 23.63 | 47.14 | **+99.5%** | 141 of 142 (99.3%) |
| 3 (cache hit) | 22.83 | 45.40 | **+98.9%** | 141 of 144 (97.9%) |

Later turns accept nearly everything, because the module the model is asked to
re-emit is now in its own prompt. This is the case the second fix unlocks: on
the same build with the old history construction, those turns proposed zero
copies — cold 12 proposals, then 0 and 0 on the two hits — while the first
turn kept working, which is why the temperature fix alone would have looked
fine in a single-turn benchmark and done nothing for a real chat.

## Correctness

**Why greedy was never required.** A copied token is a point-mass proposal:
`q(d) = 1`. Speculative sampling accepts it with probability
`min(1, p(d)/q(d)) = p(d)` under the target's own tempered distribution, and
on rejection emits a draw from that distribution with the proposed token
removed and renormalised. Composed, the two branches emit exactly `p`:
`p(d)` for the proposal, and `(1 - p(d)) · p(x)/(1 - p(d)) = p(x)` for
anything else. The generator has implemented both halves for every non-greedy
request since #2911 (`_point_mass_residual_distribution`); only the admission
gate was greedy-only.

That is now pinned by a distribution test rather than by the argument alone.
A scripted target emits `temp · log p` at the verified position for
`p = {A: 0.5, B: 0.3, C: 0.2}`, and 400 seeded single-copy turns deliver
`{A: 0.508, B: 0.320, C: 0.173}` — total variation 0.028 — with the draft
rate 0.508 against the predicted acceptance `p(A) = 0.5`. An acceptance rule
that ignored the proposal's own probability, or a residual that left the
refused token in play, moves both numbers well outside the tolerance.

**What stays family-specific.** Not the sampler: what a family has to qualify
is which rows its caches can roll back when a proposal is refused, and how
much of the win survives when acceptance stops being an argmax match. Hence
`enabled_under_sampling` per family. Qwen4 Flash-Next keeps its greedy-only
policy and its own qualification document unchanged.

**Output parity.** Not a contract under sampling, and not claimed as one, but
worth recording: on the rename and bugfix tasks every arm and every rep
returned byte-identical text (`afa7fd955bb3`, `c897870669fa`) at temperature
0.7 — a copy task is peaked enough that sampling rarely diverges from the
argmax. The free-form `doc` task did vary between reps, as sampling should,
and the prose control (same prompt, temperature 1.3, three reps) produced three
different outputs, which is what establishes that the server is really
sampling and these are not greedy runs in disguise.

**Model-free tests.** Copy engagement at temperature 0.7 on a qualified
family; no engagement on an unqualified one; the emitted-distribution test
above; the served path admitting a sampled request and carrying the
qualification through `_effective_prompt_lookup_policy`'s override rebuild;
the operator knob taking sampled requests off the route while leaving greedy
alone; and the full prompt reaching the index when the batch row holds only an
uncached tail.

## Scope boundary

This qualifies the sampled copy route for the Qwen3.5/3.8 family on the
measured artifact pair, and the prompt-history fix for every family already on
the route. It does not qualify other families for sampled copying, or change
any family's greedy behaviour, or touch continuous multi-request speculation.

## Real desktop app (Qwen3.5-4B, temperature 1.0)

Rapid-MLX Desktop 0.14.1 on an M2 Pro 32 GB, pointed at this branch through
`RAPID_BIN` (the bundled sidecar runtime with `vllm_mlx/` replaced by the
branch tree). The app's own defaults were left alone: temperature 1.0,
top_p 0.95, max_tokens 4096. `qwen3.5-4b-4bit` ships MTP default-off
(#3115), so MTP was opted in the way a user does it, through the
Performance panel's persisted per-model preset; the sidecar then launched
with `--speculative-config {"method":"mtp",...}`. The 27B alias was refused
by the app's memory guard on 32 GB, so the 4B stands in for the family.
Counters are `GET /v1/status` → `mtp_prompt_lookup`, read after each turn.

| turn (same chat) | completion tok | proposals | drafted | accepted | accept | server gen tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 add type hints + docstring | 950 | 11 | 48 | 34 | 71% | 18.9 |
| 2 rename `merged` → `combined` | 949 | 39 | 1032 | 783 | 76% | 61.9 |
| 3 fix two bugs, return full code | 3022 | 102 | 2027 | 1432 | 71% | 31.7 |

Every turn was a prompt-cache miss (hybrid family, 4266-token prompt by
turn 3), so the copy index came from the full prompt rather than the
uncached tail; `prompt_lookup_cache_fallthroughs` stayed 0. Five steps
fell through on `ft_batch_size` while the app's title request shared the
batch, which is the documented batch>1 boundary, not a regression. Turn 1
has almost nothing to copy and shows the 4B MTP floor from #3115; turns 2
and 3 are what a GUI user editing code now gets, at the app's default
sampling temperature.
