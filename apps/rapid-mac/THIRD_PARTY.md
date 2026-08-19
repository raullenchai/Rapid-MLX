# Rapid-MLX Desktop — Third-Party Acknowledgements

Rapid-MLX Desktop incorporates open-source software from the following
projects. Each is used under the terms of its original license. The
full license texts are reproduced in the linked repositories and — so
the notices travel with the binary as BSD/MIT ask — inside the shipped
app itself: the Swift packages linked into the executable under
`Contents/Resources/Licenses/`, and the bundled Python payload under
`Contents/Resources/rapid-mlx/site-packages/*.dist-info/licenses/`.
`scripts/build.sh` stages the Swift set from each package's resolved
checkout and fails the build if any linked package has no license file,
so this document and the shipped bundle cannot silently disagree (#1596).

This is the document the app's **Settings → Privacy → "Open-source
credits"** link opens.

* **MTPLX** — Youssof Altoukhi — Apache License 2.0
  Rapid's prompt-lookup speculative decoding adapts MTPLX's context-copy
  design. The required in-product attribution appears in Settings → Privacy.
  https://github.com/youssofal/mtplx

**Scope.** Components this project declares directly: the Swift packages
in `Package.swift` (plus the transitive ones actually linked into the
binary), the engine requirements in the monorepo's root `pyproject.toml`,
and what `scripts/build-sidecar.sh` installs by name. The full transitive
Python closure is **not** enumerated — a shipped bundle also contains
everything those packages pull in (`pydantic`, `starlette`, `safetensors`,
`certifi`, `regex`, …), each under its own license and each carrying its
own license file in the bundle. Optional engine extras (`[vision]`,
`[audio]`, `[chat]`, and the rest) are **not** installed into the sidecar
and are out of scope. Model weights downloaded at runtime carry their own
separate licenses from their publishers.

## Swift packages

Declared in `Package.swift`. `Package.resolved` is git-ignored (see
`.gitignore`), so the manifest range is the shipped contract; the
"resolved" column is what a clean `swift package resolve` produced at the
time of writing. Run that command for the exact revisions in your build.

### Direct

* **swift-markdown-ui** — Guillermo González Real — MIT License
  `from: "2.4.0"` (resolved 2.4.1)
  https://github.com/gonzalezreal/swift-markdown-ui
  Block-level markdown rendering for assistant messages.

* **swift-markdown** — the Swift project (Apple) — Apache-2.0 with the
  Runtime Library Exception
  `from: "0.6.0"`
  https://github.com/swiftlang/swift-markdown
  Markdown parsing (GFM tables + strikethrough) on the chat streaming path,
  which `MarkdownUI`'s string-only entry point cannot provide (#1843). Its
  parser, `swift-cmark`, is listed under Transitive below.

* **Sparkle** — Sparkle Project contributors — MIT License
  `exact: "2.9.5"`
  https://github.com/sparkle-project/Sparkle
  Signed application updates, background downloads, install-on-quit, and
  atomic application replacement.

* **SwiftMath** — Computer Inspirations — MIT License
  Vendored from 1.7.3 (`fa8244ed032f4a1ade4cb0571bf87d2f1a9fd2d7`)
  under `Vendor/SwiftMath` with a resource-resolution patch for the assembled
  macOS app. The complete upstream MIT text is kept at
  `Vendor/SwiftMath/LICENSE`. The bundled math fonts retain their upstream
  `LICENSE`, `OFL.txt`, and `GUST-FONT-LICENSE.txt` notices alongside the font
  files in `Contents/Resources/mathFonts.bundle`.
  https://github.com/mgriebling/SwiftMath
  LaTeX rendering for math/STEM model responses.

### Transitive, but linked into the shipped binary

Pulled in by the markdown packages above, so they are compiled into the app
even though the manifest names neither of these products directly.

* **NetworkImage** — Guille Gonzalez — MIT License (resolved 6.0.1)
  Pulled in by `swift-markdown-ui`.
  https://github.com/gonzalezreal/NetworkImage

* **swift-cmark** — John MacFarlane and contributors — BSD-2-Clause
  (resolved 0.8.0)
  The CommonMark parser underlying both `swift-markdown-ui` and the direct
  `swift-markdown` dependency above.
  https://github.com/swiftlang/swift-cmark
  Its `COPYING` is BSD-2-Clause for cmark itself and additionally carries
  MIT notices for code cmark vendors in turn — for example `houdini*`
  © 2012 Vicent Martí, `buffer.[ch]` / `chunk.h` © 2012 GitHub, Inc., and
  the utf8proc-derived `utf8.c` © 2009 Public Software Group e. V. — plus
  CC-BY-SA-4.0 for the CommonMark spec fixture, which is test-only and not
  shipped. That `COPYING` file, not this summary, is the complete notice.

`Sources/RapidCrashHandler` is first-party C in this repository, not a
third-party package.

The in-tree `UpdateChecker.swift` is the read-only release-status source behind
the version pill and the Settings panel. It does not download or install
anything: Sparkle owns archive verification and installation, and builds without
an injected Sparkle public key have no in-app update path at all.

## Assets

* **Cheetah mascot** — the Rapid-MLX project's own artwork, derived from
  the project's `rapidmlx.com` landing-page assets. © 2026 the Rapid-MLX
  project; all rights reserved. Embedded in the app bundle (not the source
  tree under an OSS license).

## Bundled engine (the `rapid-mlx` sidecar)

Since v0.6.6 the `rapid-mlx` engine ships **inside** the app, staged by
`scripts/build-sidecar.sh` at `Contents/Resources/rapid-mlx/`. It is the
same Apache-2.0 project this app lives in:
https://github.com/raullenchai/Rapid-MLX

### Interpreter

* **CPython 3.12.13** — Python Software Foundation License 2.0
  https://www.python.org/
* **python-build-standalone** (the redistributable build, tag `20260610`)
  — MPL-2.0
  https://github.com/astral-sh/python-build-standalone

Both pinned by `PBS_VERSION` / `PBS_TAG` in `scripts/build-sidecar.sh`.

### Engine runtime dependencies

The root `pyproject.toml` `[project].dependencies` block, installed in
full. The ranges below are the **root manifest's**; the sidecar build
narrows one of them, pinning `transformers>=5.5.0,<5.13` at install time
(step 2 of `scripts/build-sidecar.sh`), so a shipped bundle never carries
a `transformers` older than 5.5.0.

| Component | Declared range | License | Project |
| --- | --- | --- | --- |
| mlx | `>=0.31.2,<0.32` | MIT | https://github.com/ml-explore/mlx |
| mlx-lm | `>=0.31.3,<0.32` | MIT | https://github.com/ml-explore/mlx-lm |
| transformers | `>=5.0.0,<5.13` | Apache-2.0 | https://github.com/huggingface/transformers |
| tokenizers | `>=0.19.0` | Apache-2.0 | https://github.com/huggingface/tokenizers |
| huggingface-hub | `>=0.23.0` | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| numpy | `>=1.24.0` | BSD-3-Clause (also 0BSD, MIT, Zlib, CC0-1.0 components) | https://github.com/numpy/numpy |
| tqdm | `>=4.66.0` | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| pyyaml | `>=6.0` | MIT | https://github.com/yaml/pyyaml |
| tomli-w | `>=1.0.0` | MIT | https://github.com/hukkin/tomli-w |
| requests | `>=2.28.0` | Apache-2.0 | https://github.com/psf/requests |
| rich | `>=13.8.0` | MIT | https://github.com/Textualize/rich |
| tabulate | `>=0.9.0` | MIT | https://github.com/astanin/python-tabulate |
| psutil | `>=5.9.0` | BSD-3-Clause | https://github.com/giampaolo/psutil |
| fastapi | `>=0.100.0` | MIT | https://github.com/fastapi/fastapi |
| uvicorn | `>=0.23.0` | BSD-3-Clause | https://github.com/Kludex/uvicorn |
| mcp | `>=1.9.3` | MIT | https://github.com/modelcontextprotocol/python-sdk |
| jsonschema | `>=4.0.0` | MIT | https://github.com/python-jsonschema/jsonschema |
| argcomplete | `>=3.6` | Apache-2.0 | https://github.com/kislyuk/argcomplete |
| websockets | `>=12.0` | BSD-3-Clause | https://github.com/python-websockets/websockets |
| openai-harmony | `>=0.0.8` | Apache-2.0 | https://github.com/openai/harmony |
| llguidance | `>=1.7.6` | MIT | https://github.com/microsoft/llguidance |

Additionally installed by name with `--no-deps`, so the gemma-4 loader
path works in a text-only bundle:

| Component | Pin | License | Project |
| --- | --- | --- | --- |
| mlx-vlm | `==0.6.3` | MIT | https://github.com/Blaizzy/mlx-vlm |
| Pillow | `>=10.0` | MIT-CMU | https://github.com/python-pillow/Pillow |

### Third-party source vendored into the engine

The engine vendors upstream code directly into its own tree, and that tree
is installed into the sidecar — so this code travels in the app,
predominantly as compiled bytecode (the sidecar build hoists `.pyc` files
and drops the matching `.py` sources, keeping every `__init__.py` so
packages stay importable; the in-tree `LICENSE` / `NOTICE` files are
package data and are kept as-is). Each vendored file records its
provenance in its own header or in a sibling `NOTICE` — always the
upstream project, and in most cases the exact revision as well.

**This list is not exhaustive**, and is not offered as one. Vendoring
happens per model family and per subsystem, so the authoritative record
is the per-file headers, not this table. To enumerate them for a given
revision:

```bash
grep -rn "endored from\|orted from\|dapted from\|derive[sd] from" \
    vllm_mlx/ videox_fun_mlx/
```

The largest and most self-contained components:

| Component | Upstream | Upstream license | In-tree |
| --- | --- | --- | --- |
| MLX Stable Audio 3 | https://github.com/Stability-AI/stable-audio-3 | MIT | `vllm_mlx/audio/sa3/` (`LICENSE`, `NOTICE`) |
| CogVideoX-Fun MLX | https://github.com/dgrauet/VideoX-Fun-mlx | Apache-2.0 | `videox_fun_mlx/` (`LICENSE`, `NOTICE`) |
| TurboQuant Metal kernels | https://github.com/arozanov/turboquant-mlx | Apache-2.0 | `vllm_mlx/kernels/turboquant_fused.metal` |
| Gemma 4 model classes | https://github.com/Blaizzy/mlx-vlm (v0.6.3) | MIT | `vllm_mlx/models/gemma4_vendored/` |
| Hunyuan 3 model class | https://github.com/ml-explore/mlx-lm (PR #1211) | MIT | `vllm_mlx/models/hy_v3.py` |
| DeepSeek V4 model classes | https://github.com/ml-explore/mlx-lm (`_ds4` branch, © Apple Inc.) | MIT | `vllm_mlx/models/deepseek_v4.py`, `deepseek_v4_cache.py`, `deepseek_v4_hyper_connection.py`, `deepseek_v4_switch.py` |
| MTP speculative-decoding head + generator | https://github.com/ml-explore/mlx-lm (PR #990) | MIT | `vllm_mlx/spec_decode/mtp/head.py`, `generator.py` |
| Request/status model, adapted | https://github.com/vllm-project/vllm | Apache-2.0 | `vllm_mlx/request.py` |
| Several tool parsers, ported | https://github.com/vllm-project/vllm, https://github.com/sgl-project/sglang | Apache-2.0 | `vllm_mlx/tool_parsers/` |

Two notes on reading that table:

* The **licence column is the upstream project's**, which is what governs
  redistribution of the vendored code. Many vendored files additionally
  carry a rapid-mlx `SPDX-License-Identifier` header — that stamp reflects
  this repository's own default and does not override the upstream terms
  above.
* The sibling files under `vllm_mlx/models/deepseek_v4_verify*.py` and
  `deepseek_v4_rollback.py` are **first-party** rapid-mlx code (Apache-2.0)
  that happens to share the prefix; they are not vendored.

Paths are relative to the monorepo root, two levels above this file.
