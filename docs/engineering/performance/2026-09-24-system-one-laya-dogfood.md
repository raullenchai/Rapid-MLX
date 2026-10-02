# System One Laya server dogfood

Date: 2026-09-24

## Result

The server-only Laya integration completed real cached-weight requests through
`/v1/systemone` and `/v1/rank`. Authentication, typed `choice`, `noul`, and
`score` answers, restart behavior, and bounded concurrent admission worked.
Rapid-MLX answers matched a direct `laya-mlx` `system_one()` call field for
field; the service intentionally replaces the internal `laya-rl-agent` model
name with its public model ID and adds billing and latency metadata.

The small qualitative screen behaved sensibly:

- duplicate-charge routing selected `billing` with probability `0.9412`;
- a credential-phishing message returned `P(true)=0.9183`;
- an attachment-crash report selected `technical` with probability `0.7226`;
- a successful backup alert returned urgent `P(true)=0.0448`;
- ranking placed the duplicate-charge verification/refund response first with
  probability `0.7877`.

This is a product-path dogfood screen, not a benchmark or broad quality eval.
It supports an experimental server claim. It does not qualify accuracy across
domains or hardware.

## Environment

- Machine: Apple M3 Ultra, 256 GB unified memory
- macOS: 26.5.2
- Python: 3.12.13
- MLX: 0.32.2
- `laya-mlx`: 0.2.0
- Rapid-MLX revision: `55c399e40`
- Model: `convaiinnovations/laya`
- Model revision: `1c5edc17a7acd8701df6fc341c0d179f1c62c982`
- Model payload already present in the standard Hugging Face cache: 807 MiB
- Device: CPU, to avoid contending with an unrelated active GPU workload
- Dtype: float16
- Batch size: 16

## Timing

All times are HTTP client wall times on loopback unless stated otherwise.

| Operation | Observed time |
| --- | ---: |
| First request before file pages were warm, three questions | 2,321 ms |
| Warm three-question request, median of 12 | 170.3 ms |
| Warm three-question range, 12 requests | 167.8–172.5 ms |
| Warm single-question requests | 51.6–54.0 ms |
| Warm three-candidate rank | 55.6 ms |
| Restart to `/health` ready, two runs | 984.5 ms, 1,008.5 ms |
| First request after page-cached restart | 200.6 ms, 200.1 ms |
| Eight concurrent single-question requests | 394.7 ms wall |

The backend serializes the MLX model, so the eight concurrent requests
completed at roughly 54, 105, 153, 203, 250, 299, 347, and 394 ms. All
returned HTTP 200 and the same `technical` choice. This gives predictable
queueing but no parallel inference throughput at the default eight-request
admission limit.

## Reproduction

No model download was performed. The run forced offline Hub behavior so a
cache miss would fail rather than mutate storage.

```bash
uv pip install --python .venv/bin/python -e '.[system-one]'

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/rapid-mlx system-one convaiinnovations/laya \
  --backend laya --device cpu --port 18700 \
  --api-key dogfood-secret
```

Representative request:

```bash
curl --fail-with-body http://127.0.0.1:18700/v1/systemone \
  -H 'Authorization: Bearer dogfood-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"ticket": "I was charged twice for my subscription."},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
          "billing": "Charges and refunds",
          "technical": "Bugs and troubleshooting",
          "sales": "Purchases and upgrades"
        }
      }
    }
  }'
```

## Limits found

- The checkpoint emits a `laya-mlx` warning that its `choice:11+`
  temperature is outside the runtime's safe calibration interval and is
  clamped. Confidence for choices with eleven or more options must therefore
  be treated as uncalibrated. Smaller choice sets used in this screen are not
  in the affected bucket.
- CPU latency is suitable for interactive routing and policy decisions, but
  requests are serialized. Tail latency grows linearly at concurrency eight.
- CLM was qualified separately on CPU after this run; see
  [System One CLM-8B server dogfood](2026-09-24-system-one-clm-dogfood.md).
