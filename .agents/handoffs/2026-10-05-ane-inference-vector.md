# Atlas → Vector: ANE inference qualification

Branch: `atlas/ane-inference-survey-20261005`. Host: Studio.

Verified: Rapid's MLX text path supports CPU/GPU; no ANE backend is wired. Public ANE conversion examples exist, and oMLX / mlx-serve report Qwen hybrid prefill gains, but their current hybrid implementations use undocumented Apple APIs and lossy quantization. See `docs/engineering/performance/2026-10-05-ane-inference-survey.md` for sources and candidate matrix.

Unresolved: no Rapid-side ANE run, same-weight paired benchmark or full-response quality test has been performed. A private-API shipping policy decision belongs to Atlas and the human owner. The Studio is shared; coordinate a memory-safe window before large checkpoint loads.

Next action: reproduce a public Core ML small-model ANE placement/performance run, then a pinned Qwen 27B GPU-only versus hybrid prefill run with identical prompts and quality checks. Report TTFT, decode, power, memory, compilation time and ANE placement. No release commitment.
