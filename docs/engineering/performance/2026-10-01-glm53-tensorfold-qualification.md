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

Raw request and response artifacts remain in
`/private/tmp/harbor-desk-glm53-study/` on the qualification host.
