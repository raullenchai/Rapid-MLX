# MiniMax-H3-Image

`minimax-h3-image` generates and edits images with load-time 4-bit quantization
of the image transformer and text decoder. `minimax-h3-image-8bit` uses 8-bit
quantization. Both use the same pinned original weights and auxiliary text
encoder; an existing pinned `minimax-h3` text-encoder cache is reused.
Direct repository or local checkpoint requests default to 4-bit.

```sh
rapid-mlx pull minimax-h3-image
rapid-mlx serve minimax-h3-image --port 8000
curl http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-h3-image","prompt":"A lighthouse on a rocky island","size":"2048x2048","steps":6,"seed":42}'
```

Only six steps and guidance 1.0 are supported. Omit these parameters to use the
model defaults. Negative prompts and LoRAs are unsupported. Explicit width and
height must be positive multiples of 32. The runtime uses a minimum generation
area of 2048 × 2048 pixels, so smaller requested canvases can be enlarged.

The editing endpoint accepts up to 16 reference images, in upload order.
Refer to them as `<Picture 1>`, `<Picture 2>`, and so on in the prompt. Desktop
currently clamps reference attachments to 10; raising that limit is follow-up
work. Generation and editing share one resident model.

```sh
curl http://localhost:8000/v1/images/edits \
  -F model=minimax-h3-image -F 'prompt=Paint <Picture 1> in watercolors' \
  -F image=@reference.png -F size=2048x2048 -F steps=6 -F seed=42
```

Measured on an M3 Ultra (256 GB): whole-process peak is about 47 GB for 4-bit
and 76 GB for 8-bit, so the catalog requires 64 GB and 96 GB of unified memory
and charges 44/71 GiB of residency. Both aliases download the same
full-precision artifacts (126.1 GiB including the text encoder); quantization
happens at load time. Without `size`, an edit follows the first reference's
aspect ratio at about 1 MP.
See the [measurement worksheet](../../engineering/performance/2026-11-14-minimax-h3-image.md).

## License and required display

Weights use the [MiniMax-H3-Image Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE).
The license excludes use in the European Union, the United Kingdom, the
Republic of Korea and the United States. Commercial use is permitted under the
license. Separate written authorization
from MiniMax is required for organizations above US$1M annual revenue or for
offering a hosted service to third parties. Products using it must prominently
display **MiniMax-H3-Image**; see the license for full terms.

Model metadata and generation/edit JSON responses include
`"attribution": "MiniMax-H3-Image"`. Desktop displays
“Powered by MiniMax-H3-Image” beside the selected model.
