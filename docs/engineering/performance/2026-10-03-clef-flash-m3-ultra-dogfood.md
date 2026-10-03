# Clef-Flash on M3 Ultra: local dogfood

Date: 2026-10-03 UTC. Branch: `atlas/clef-support` at `3cf67373` before the
single-frame video validation follow-up. Host: Mac Studio, M3 Ultra (60 GPU
cores), 256 GB unified memory, macOS 26.5.2. Python 3.12.14, Torch 2.14.1,
Transformers 5.15.1, Accelerate 1.15.0. Model: pinned
`Cloudflare/clef-flash` revision `17f0b0ad64efb65d273590632833508766b2aae6`
in the default Hugging Face cache. This is the BF16 9B checkpoint, with the
official joint-schema head; `torch.backends.mps.is_available()` was true.

## Reproduce

Install the optional runtime in a fresh environment on a Metal-capable Mac:

```bash
pip install -e '.[clef]'
rapid-mlx system-one clef-flash --host 127.0.0.1 --port 8701
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
a `noul` question asking whether the frames are mostly red. Timings below are
HTTP wall time from `httpx` on loopback; the service was loaded once and
requests ran serially. Cold download and model load are excluded from the
request timings.

## Observations

| Probe | Result | Wall time |
| --- | --- | ---: |
| First text request, `noul` + `choice` + `score` | 200; `team=billing` (0.9868); `priority` high (0.9205); 340 input tokens | 3.04 s |
| Seven repeated short text requests | All 200; median 0.250 s, max 0.373 s | 0.248–0.373 s |
| `/v1/rank`, three support actions | 200; billing refund case ranked first (0.9958) | 0.561 s |
| Receipt shows `$42.00`; asks if total is `$42.00` | `noul=0.9722` | 0.542 s |
| Receipt shows `$17.00`; same question | `noul=0.0073` | 0.397 s |
| Three red video frames; asks if mostly red | `noul=0.9790` | 0.889 s |
| Three blue video frames; same question | `noul=0.0061` | 0.509 s |

The initial snapshot download transferred 19.1 GB in about 2 minutes 50
seconds. A second start from the existing cache reached `/health` in about
4.9 seconds. The process RSS after the probes was about 19.0 GiB (19,924,240
KiB); system memory pressure remained low. The local cache entry used about
18 GiB on disk.

The model did not show clear temporal understanding in a small synthetic
two-frame `OPEN`/`CLOSED` order-reversal probe: both directions selected
`closes` with low confidence. Solid-color frame controls above show that the
video path does deliver visual content to the model. These toy probes do not
establish video decision accuracy. Transformers logged that video frame arrays
have no FPS metadata and defaulted to 24 FPS; this was non-fatal. A one-frame
video originally returned the processor's `t:1 ... temporal_factor:2` error;
the API now gives a direct minimum-two-frames error.

This run verifies Clef-Flash local inference and API behavior on one M3 Ultra.
It does not qualify the 27B Clef checkpoint, other Mac memory sizes, accuracy
on real customer material, or production concurrency.
