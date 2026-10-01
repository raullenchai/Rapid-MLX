# Vector handoff: GLM-5.3 TensorFold profile

Owner: Vector. Branch: `vector/glm53-tensorfold-product`. Host: Studio.

The change adds a target-only, catalog-qualified TensorFold MTP lane for the
new `glm5.3-flash-tensorfold` alias. It keeps the ordinary GLM alias unchanged,
pins target/runtime identities, requires 256 GB, maps Rapid's
`reasoning_max_tokens` onto TensorFold's `thinking_budget`, and reuses the
existing TensorFold HTTP provider. Unsupported tools, media, grammar, and
general batching remain fail-closed. The ordinary GLM alias is the documented
restart fallback.

Reference check: vLLM and SGLang use explicit speculative backend selection and
separate target/draft capability validation. MLX-native precedent keeps model
family loading and cache ownership inside the owning runtime. TensorFold 0.6.0
provides the GLM family lane and embedded-MTP verifier. The adopted shape keeps
TensorFold's scheduler/cache domain intact and exposes only Rapid's existing
HTTP/lifecycle boundary. A DFlash profile was rejected because the Mac path
uses the embedded MTP head and the available GLM DFlash artifact has unsuitable
license terms.

Qualification evidence is in
`docs/engineering/performance/2026-10-01-glm53-tensorfold-qualification.md`.
Next action: independent scope-locked review, then source CI and release-owner
queueing. No release or provider action belongs to this branch.
