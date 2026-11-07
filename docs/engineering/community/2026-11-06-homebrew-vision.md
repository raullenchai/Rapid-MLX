# Homebrew vision dependency audit

Owner: Harbor. Tracking: [#5642](https://github.com/raullenchai/Rapid-MLX/issues/5642).
Source report: [discussion #1205](https://github.com/raullenchai/Rapid-MLX/discussions/1205), concerning v0.11.0.

## Findings (2026-11-06)

The current [homebrew-core formula](https://github.com/Homebrew/homebrew-core/blob/a1c0a8b64a92e1e2c448a5886782876b0e9bc666/Formula/r/rapid-mlx.rb)
packages Rapid-MLX 1.0.1 with Python 3.15 and `mlx-vlm` 0.7.2. It declares
Pillow, PyTorch, torchvision and OpenCV dependencies, installs its Python
resources, and adds the brewed PyTorch/torchvision environment paths.
Rapid-MLX main at `aa4d679ec` also pins `mlx-vlm==0.7.2`.
The historical missing-dependency report therefore does not establish that the
current formula still omits the vision backend.

The formula test covers CLI metadata and imports `cv2`, `imageio_ffmpeg`,
`torch`, and `torchvision`; it does **not** import `mlx_vlm` or run image inference.
This audit inspected packaging and source, not a freshly installed bottle.
The audit host has no Homebrew Rapid-MLX installation; its separate Python
installation is not evidence about a bottle. No model was downloaded.

## Supported user follow-up

For an existing Homebrew install, use Homebrew to update the managed environment:

```sh
brew update
brew upgrade rapid-mlx
brew list --versions rapid-mlx
"$(brew --prefix rapid-mlx)/bin/rapid-mlx" --version
"$(brew --prefix rapid-mlx)/libexec/bin/python" -c 'from importlib.metadata import version; import mlx_vlm; from PIL import Image; print("mlx-vlm", version("mlx-vlm"))'
```

The import check is a dependency smoke test, not an inference compatibility test.
Record its traceback, macOS/architecture, Homebrew version, and the exact model
ID if it fails. If it passes but vision inference fails, include the launch
command and sanitized request/error for a model-specific reproduction.

Do not use the old version-specific `pip --target /opt/homebrew/Cellar/...`
workaround: it mutates a Homebrew-managed environment and disappears on upgrade.
Users choosing a pip-managed installation should follow the separate
[optional extras instructions](https://rapidmlx.com/docs/extras), using the
Python environment that owns their Rapid-MLX installation.

## Remaining verification

Harbor should run the smoke test against a freshly installed current bottle on
an isolated Apple Silicon runner, then exercise a locally available vision model
with a small image and record the exact bottle/model revisions. The issue stays
open until that runtime evidence exists; formula inspection alone is insufficient
to declare the original class of failure resolved.
