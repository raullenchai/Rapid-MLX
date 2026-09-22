# Trio-Flash pricing card & unit economics (astra session 5, 2026-09-22)

## Cost basis

- Production host: Vast.ai RTX 3090 24GB dedicated, ~$200/month (24/7) —
  replaces the Mac Studio for serving (Studio returns to development).
- Bandwidth ≈ $0; failover capacity NOT included at beta scale.

## Market anchors (measured 2026-09)

- TypeSafe Jev: ~$0.04–$0.042 per **million** input tokens, free output →
  at 300–400-token prompts that is **$0.012–$0.017 per 1k decisions**, no
  public monthly minimum. We CANNOT compete on commodity inference price.
- AWS Comprehend custom classification: ~$0.30–1.50/1k (async) or
  provisioned capacity ~$1,296/month per inference unit.
- OpenAI Moderation endpoint: free but narrow, non-custom.

## Beta price card (per-decision pricing with included bundles)

| Tier | Monthly floor | Included decisions | Overage |
| --- | ---: | ---: | --- |
| Research beta | $0 | 1,000 | no automatic overage |
| Indie | $39 | 10,000 | $4 / 1,000 |
| Team | $149 | 50,000 | $3 / 1,000 |

- Break-even on the $200 host: **1 Team + 2 Indie = $227/month**.
- At 500k Team decisions/month: ~$1,499 revenue on the same $200 GPU
  (≈87% infrastructure gross margin, before support/fees/failover).
- Price **per decision**, never per seat — seats don't track load.
- The paid hypothesis (per astra): calibration, abstention policy, support,
  custom lanes — never raw inference price.

## Positioning rule

Never claim price parity with Jev. The honest ladder: **Jev-class accuracy
(McNemar p=0.39) + 3.3× calibration + disposition contract + custom lanes**.
Beta pricing is a willingness-to-pay experiment, not a margin claim.
