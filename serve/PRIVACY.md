# Privacy (beta)

**What we store:** for every decision, a JSONL log line on the serving
machine: timestamp, client IP, token id, request id, prompt length (chars),
number of candidates, decision, confidence, recommended action, latency.
**The full text of your prompt is NOT logged by default** (only a short
SHA-256 prefix for correlation). Setting `MARVIN_LOG_PROMPTS=1` on the
server turns full-prompt logging on for debugging — the playground banner
states which mode is active.

**Where it lives:** only on the serving machine (local disk). If the
serving host is a rented cloud instance (e.g. Vast.ai), the host operator
can in principle access that disk — treat prompts accordingly and prefer
the self-hosted path for sensitive data.

**Retention:** logs rotate weekly; delete on request (open an issue with
your token id, or self-host where you control deletion).

**No training on your data.** Nothing you send is used to train any model.
