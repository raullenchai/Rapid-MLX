# Contributing

Start with the [development setup and test commands](../../CONTRIBUTING.md).
That guide includes the Python environment, local server, pull request process,
and validation steps.

## Choosing a model for a test

- **Correctness:** use an 8-bit or higher precision model so quantization does
  not obscure behavior. Check the current model selection in
  [`golden_models.yaml`](../../scripts/pr_validate/golden_models.yaml).
- **Performance:** use a 4-bit model and record the machine, model, command,
  and run conditions with the result.
- **Startup smoke:** use a small model when the test only checks that the
  server starts and returns a response.

Run `rapid-mlx bench --help` for the current benchmark options. Model aliases
are listed by `rapid-mlx models` and defined in
[`aliases.json`](../../rapid_mlx/aliases.json).

For code structure, see [Architecture](architecture.md). For release work,
see the [release procedure](../../RELEASE.md).
