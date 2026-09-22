# Launch-day runbook — seed-user beta (Mac Studio target)

Owner: Atlas · Host: Mac Studio M3 Ultra (512 GB) · Reviewed by GPT-6-Astra
(session 3, see `docs/engineering/community/2026-09-21-gpt6-astra-launch-review.md`)

## T−60 · freeze & start

1. Record git SHA, model/adapter/policy versions (response echoes them).
2. Start the API: `TOKEN=$(openssl rand -hex 16) bash serve/deploy_local.sh`
   (binds localhost:8123; idempotent; logs → `/tmp/demo_server.log`).
3. Decide exposure: `cloudflared tunnel --url http://localhost:8123`
   (Quick Tunnel = new URL each run; named tunnel for a stable hostname).
4. Issue **per-user tokens** for seeds (edit deploy to loop; shared token is
   admin-only emergency). Keep `MARVIN_RATE_PER_MIN=30`, `QUEUE_CAP=8`.
5. Confirm anti-sleep: `caffeinate -dims &` and System Settings → prevent
   automatic sleep on power.

## T−45 · functional acceptance (must be ALL PASS)

```bash
python scripts/run_golden.py --base http://localhost:8123 --token $TOKEN
```

Covers: 4 golden decisions with confidence floors, 413 oversize, 400 schema,
401 no-auth. Additionally verify: `curl /healthz` → {"ok": true}; the
decision log has **no prompt text** (`MARVIN_LOG_PROMPTS` unset); responses
carry `request_id`, `policy_version`, `adapter_version`.

## T−30 · failure acceptance

1. `pkill -f demo_server.py` → restart via deploy_local.sh (or launchd when
   installed) → golden must pass again.
2. Kill cloudflared → tunnel reconnects (named) or note new URL (quick).
3. Fire 12–16 concurrent requests → expect 429/503 with `Retry-After`, NOT
   an unresponsive server (queue cap 8 now enforced, verified).

## Open — in waves, never all at once

5 users → watch 15 min → 15 users → watch 15–30 min → all seeds.

Watch: canary latency, 5xx, 503, queue-503 count, p95 latency, RSS, disk
(`/tmp` log grows ~1 KB/decision → rotate weekly).

**Pause conditions** (stop inviting): canary failure · 5xx > 1% · 503 > 5%
for 2 min · queue-503s sustained · confidence/action anomalies · restarts.

## Incidents

| symptom | action |
| --- | --- |
| abuse from one token | revoke that token only |
| sustained overload | lower per-IP rate, keep 503s, pause invites |
| worker hang (known Metal failure mode) | kill worker; restart; golden |
| Metal not recovering | reboot the Mac (last resort, ~5 min) |
| model regression | redeploy previous adapter dir; golden |
| privacy incident | set `MARVIN_LOG_PROMPTS=0`, quarantine log, record scope |

## Single biggest risk (Astra): downstream fail-open

Callers that ignore `review`/`abstain` or proceed on timeout/5xx turn a good
classifier into a security bypass. Every example client MUST implement an
explicit failure policy (default-deny), and the docs must say: `accept`
means "confident", never "permitted".
