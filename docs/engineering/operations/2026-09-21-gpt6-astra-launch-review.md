# GPT-6-Astra — third review: launch readiness (2026-09-21, session 3)

Scope: what is missing before/at seed-user launch. Prompt: /tmp/subagent_brief3.md
(state of 2026-09-21 night). Executed follow-ups are checked off below.

## P0 (pre-launch) — astra's list

1. [x] Stop logging full prompts by default — implemented: log carries
   request_id/prompt_chars/decision/confidence/disposition/latency; full
   prompt only with `MARVIN_LOG_PROMPTS=1` (see serve/PRIVACY.md).
2. [x] Hard input boundaries — payload ≤256KB / prompt ≤8,000 chars /
   candidates 2..8 → 400/413; auth before parsing; request_id +
   `Cache-Control: no-store` on all responses; `Retry-After` on 429/503.
   (astra suggested 64KiB body; 256KB kept as outer cap with the 8,000-char
   prompt limit doing the real work.)
3. [x] Per-user tokens + loopback bind — bind default 127.0.0.1
   (`MARVIN_BIND` override); `CF-Connecting-IP` trusted for rate limiting
   behind the tunnel (socket addr is loopback there). Token hashing and
   per-token rate buckets: still shared-token admin + manual seed tokens —
   P1 this week.
4. [x] Disposition semantics — response now has BOTH `recommended_action`
   (accept/review/abstain, kept for compat) and `disposition`
   (auto_decide/review/abstain); docs must state timeout/5xx/review/abstain
   ≠ allow; thresholds labeled beta policy.
5. [x] Process supervision — serve/com.marvingarden.api.plist (launchd
   KeepAlive, ThrottleInterval=10) + caffeinate in launch runbook; canary
   loop + /livez vs /readyz split: P1.
6. [x] Named tunnel — documented in launch runbook (Quick Tunnel only for
   temporary previews, URL changes every run).

## P1 (within a week) — astra's list

1. CORS: currently `*` (open for adoption); restrict origins + playground
   CSP/nosniff/no-store when seeds are stable.
2. Legal pages: /notice + /privacy live (serve/NOTICE.md, PRIVACY.md);
   /terms page still to add.
3. Observability: stats endpoint + alert thresholds (canary >10 s, 5xx >1%,
   503 >5% for 2 min, queue ≥6 for 2 min, RSS 2×, disk <50 GB) + external
   pinger on /healthz.
4. 24 h soak test (mixed valid/oversize/invalid/disconnect/queue-full/
   restart traffic; watch Metal RSS + latency + log growth).
5. Comparison statistics: paired McNemar vs Jev before any "significantly
   better" claim (point estimates only until then).

## Multi-worker guidance (astra)

Do NOT treat 8 model replicas as throughput scaling — they share one GPU.
If process-level failover is needed, test 2 active-passive workers with
Caddy health checks first; a homemade round-robin proxy is banned. Watchdog
+ host-reboot path remains mandatory (Metal driver faults kill all replicas).

## Soak & statistics (still open before claiming production)

- 24 h soak with chaos mix (astra's list above).
- Paired McNemar + Wilson/bootstrap intervals for the Marvin-vs-Jev claim.

## Launch-day runbook

Distilled into `docs/engineering/operations/launch-runbook.md` (T-60
freeze, T-45 golden gate, T-30 failure drills, wave-based opening,
incident table). Astra's single biggest risk: **downstream fail-open** —
callers proceeding on review/abstain/timeout turn a good classifier into a
security bypass; every example client needs an explicit fail-closed policy.
