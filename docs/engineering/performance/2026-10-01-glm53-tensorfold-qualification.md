# GLM-5.3-Flash TensorFold qualification

Date: 2026-10-01. Host: Mac Studio M3 Ultra, 256 GB. The test reused complete
snapshots from the default Hugging Face cache; it downloaded no model data.

## Immutable inputs

- Target: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` at
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`.
- Runtime: TensorFold 0.6.0 at
  `c4646171139ee8a3c38103eaa1699dad226ec12b`.
- MLX 0.32.3; 4-bit affine weights in groups of 64; embedded MTP head.
- Context 8,192; one request; temperature 0; seed 1234.

The server's load check admitted a 16-row exact window. Resident weights were
168.5 GiB. The benchmark ran in a low-pressure window and the service was
stopped immediately afterward.

## Reproduction

The tracked request fixture is
[`fixtures/glm53-tensorfold-ttlcache.json`](fixtures/glm53-tensorfold-ttlcache.json)
(SHA-256 `1e688842b5ff93702f83f0a7b5dac4eeca9c6cdc1734371ed74c19f6602f1eca`).
It asks for a standard-library, thread-safe TTL/LRU cache with single-flight
loading. Run the exact pinned runtime from a clean shell:

```bash
python -m pip install "tensorfold @ git+https://github.com/ashhart/TensorFold.git@c4646171139ee8a3c38103eaa1699dad226ec12b"
TARGET=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Vontra/GLM-5.3-Flash-MLX-4bit-MTP", revision="76add2a341a1cd90ad0e86bb69839ea9c35827c6"))')
tensorfold serve "$TARGET" \
  --name glm53-tf-v06 --context 8192 --max-tokens 4096 \
  --parallel 1 --mtp-drafts 3 --prefill-pass 8 --pass-cache-gib 16 \
  --port 18161 --no-update-check
curl -sS http://127.0.0.1:18161/v1/chat/completions \
  -H 'content-type: application/json' \
  --data-binary @docs/engineering/performance/fixtures/glm53-tensorfold-ttlcache.json \
  > /tmp/glm53-drafted.json
jq '. + {draft:false}' \
  docs/engineering/performance/fixtures/glm53-tensorfold-ttlcache.json \
  | curl -sS http://127.0.0.1:18161/v1/chat/completions \
      -H 'content-type: application/json' --data-binary @- \
  > /tmp/glm53-serial.json
```

The values below are the response `tensorfold.tokens_per_second` fields, which
measure decode after prefill. Compare `.tensorfold.token_sha`,
`.choices[0].message`, and `.choices[0].finish_reason` between the two files for
the exact-output check. The coding run had 460 prompt tokens and 768 completion
tokens; the smoke run had 22 and 110. Record memory pressure before each run and
discard runs with swap or another resident model, since that condition caused a
known invalid slowdown during qualification.

## Bounded result

The coding fixture used a 460-token prompt, `thinking_budget=256`, and a
768-token reply cap. Drafted and serial requests returned byte-identical
reasoning and content and the same length finish. Drafted decode measured 57.5
token/s; the same loaded engine with `draft:false` measured 47.5 token/s, a
1.21x ratio for this one fixture. A 110-token smoke measured 53.9 versus 48.0
token/s, a 1.12x ratio, with the same server token fingerprint.

These samples qualify an experimental opt-in path. They do not establish a
fixed speed multiplier or a quality improvement. An earlier run under heavy
swap made wide verification slower than serial; memory admission and the 256 GB
product floor are therefore part of the profile contract.
