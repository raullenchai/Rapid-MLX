# Spark 2.5 text checkpoint loader

`XHToken/Spark-X2.6-4B` and its MLX text conversion declare
`model_type=spark2_5` and `Spark2_5ForCausalLM`. Their checkpoints have no
vision config or vision weights. In the pinned runtime (`mlx-lm` 0.31.3,
`mlx-vlm` 0.7.2), only `mlx-vlm` implements this architecture. The normal
text lane fails with `Model type spark2_5 not supported`; forcing the MLLM
lane loads and serves text correctly. The catalog alias therefore does not
set the `is_text_only` routing override, while it explicitly declares
`supports_image_input=false`.

The cache-only lane classifier routes a known `spark2_5` config to the
`mlx-vlm` loader. This is scoped to the checkpoint architecture and preserves
existing `is_text_only` overrides for other aliases. The alias pins
`abenzerps/Spark-X2.6-4B-MLX-4bit@253181296bd0c643880b0f64daa0049a42148d4a`.
The model's `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value>`
wire matches `glm47`; `<think>` matches `qwen3` reasoning.

The tested entry point is the plain `rapid-mlx serve spark-x2.6-4b-4bit`
command on a fresh `[vision]` install. Removing this route would make the
alias fail at startup. A future `mlx-lm` Spark implementation can replace
this routing decision after the same HTTP qualification.
