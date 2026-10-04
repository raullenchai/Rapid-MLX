# Clef and Clef-Flash on M3 Ultra: local dogfood

Date: 2026-10-03 UTC. Branch: `atlas/clef-support` at `3cf67373` before the
single-frame video validation follow-up. Host: Mac Studio, M3 Ultra (60 GPU
cores), 256 GB unified memory, macOS 26.5.2. Python 3.12.14, Torch 2.14.1,
Transformers 5.15.1, Accelerate 1.15.0. Models: pinned
`Cloudflare/clef-flash` 9B revision
`17f0b0ad64efb65d273590632833508766b2aae6` and pinned
`Cloudflare/clef` 27B revision
`2f3de3dd85f379784083b0814d997ab627200f0c`, both BF16 and both using
the official joint-schema head. The default Hugging Face cache was used;
`torch.backends.mps.is_available()` was true.

## Reproduce

Install the optional runtime in a fresh environment on a Metal-capable Mac:

```bash
pip install -e '.[clef]'
rapid-mlx system-one clef-flash --host 127.0.0.1 --port 8701
# For the 27B checkpoint, stop the Flash process and run:
rapid-mlx system-one clef --host 127.0.0.1 --port 8701
```

Send this representative multi-question request to `/v1/systemone`:

```json
{
  "state": {
    "message": "I was charged twice for the same invoice. Please refund the duplicate payment.",
    "account_status": "active"
  },
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this require urgent human handling?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Charges, invoices, refunds",
        "technical": "Technical bugs and outages",
        "sales": "Plans and purchasing"
      }
    },
    "priority": {
      "type": "score",
      "instructions": "How urgent is this customer request?",
      "criteria": ["low", "medium", "high"]
    }
  }
}
```

For the image probe, make a 384×192 white PNG with Pillow and draw `RECEIPT`
and either `TOTAL $42.00` or `TOTAL $17.00` in black at `(24, 28)` and
`(24, 82)`. Send it as a PNG base64 data URL in `images`, with a `noul`
question asking whether the total equals `$42.00`. For the video probe, send
three solid red or blue 384×192 PNG frames as one nested `videos` array, with
a `noul` question asking whether the frames are mostly red. A second video
probe draws `FRAME 1: OPEN`, `FRAME 2: OPEN`, `FRAME 3: CLOSED` (and the
reverse order) in black on white frames, then asks which transition happened.
Timings below are HTTP wall time from `httpx` on loopback. Requests ran
serially. Cold downloads and model load are excluded from request timings.

## Clef-Flash (9B) observations

| Probe | Result | Wall time |
| --- | --- | ---: |
| First text request, `noul` + `choice` + `score` | 200; `team=billing` (0.9868); `priority` high (0.9205); 340 input tokens | 3.04 s |
| Seven repeated short text requests | All 200; median 0.250 s, max 0.373 s | 0.248–0.373 s |
| `/v1/rank`, three support actions | 200; billing refund case ranked first (0.9958) | 0.561 s |
| Receipt shows `$42.00`; asks if total is `$42.00` | `noul=0.9722` | 0.542 s |
| Receipt shows `$17.00`; same question | `noul=0.0073` | 0.397 s |
| Three red video frames; asks if mostly red | `noul=0.9790` | 0.889 s |
| Three blue video frames; same question | `noul=0.0061` | 0.509 s |
| Three-frame `OPEN→CLOSED` order probe | `closes=0.9646` | 3.686 s immediately after restart |
| Reversed three-frame `CLOSED→OPEN` probe | `opens=0.9589` | 3.358 s immediately after restart |

The initial snapshot download transferred 19.1 GB in about 2 minutes 50
seconds. A second start from the existing cache reached `/health` in about
4.9 seconds. The process RSS after the probes was about 19.0 GiB (19,924,240
KiB); system memory pressure remained low. The local cache entry used about
18 GiB on disk.

An earlier **two-frame** `OPEN`/`CLOSED` order reversal was ambiguous: both
directions selected `closes` with low confidence. The three-frame version
above selected both directions correctly. The first three-frame probes ran
just after a process restart and include per-shape Metal warmup.

## Clef (27B) observations

| Probe | Result | Wall time |
| --- | --- | ---: |
| First text request, same three questions | 200; `team=billing` (0.9944); `priority` high (0.9099); 340 input tokens | 7.764 s |
| Seven early short text requests | All 200; median 5.189 s while runtime warmed | 5.003–5.369 s |
| Seven later identical short text requests | All 200; median 0.775 s | 0.766–0.837 s |
| Three-question request after warmup | `team=billing` on all three repeats | 1.477–1.506 s |
| `/v1/rank`, same three support actions | Billing refund case first (0.9964) | 0.836 s |
| Receipt shows `$42.00`; asks if total is `$42.00` | `noul=0.9943` | 1.244 s |
| Receipt shows `$17.00`; same question | `noul=0.0051` | 1.102 s |
| Three red / blue video frames; asks if mostly red | `noul=0.9939` / `0.0056` | 1.595 / 1.444 s |
| Three-frame `OPEN→CLOSED` / reversed order | `closes=0.9810` / `opens=0.9864` | 1.717 / 1.488 s |

The 27B snapshot is about 55.0 GB. A restart from the existing cache reached
`/health` in 9.3 seconds. Process RSS after the probes was about 52.4 GiB
(54,936,496 KiB). The cache entry used about 51 GiB on disk. The difference
between the early 5.2-second short calls and later 0.78-second calls shows a
substantial runtime warmup effect; report both when setting user expectations.

Both models identified the synthetic receipt counterfactual and frame colors.
Both selected the correct order on the three-frame temporal control. These
small probes do not establish accuracy on real receipts or video. Transformers
logged that frame arrays have no FPS metadata and defaulted to 24 FPS; this
was non-fatal. A one-frame video originally returned the processor's
`t:1 ... temporal_factor:2` error; the API now gives a direct
minimum-two-frames error.

This run verifies full-weight local inference and API behavior for both model
sizes on one M3 Ultra. It does not qualify other Mac memory sizes, accuracy on
real customer material, or production concurrency.
