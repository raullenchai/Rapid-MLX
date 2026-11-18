# Public Sans

- Family: Public Sans, upright variable weight (100–900).
- Source: [Google Fonts official repository](https://github.com/google/fonts/tree/main/ofl/publicsans).
- Original: [PublicSans[wght].ttf](https://github.com/google/fonts/blob/main/ofl/publicsans/PublicSans%5Bwght%5D.ttf), saved as `PublicSans.ttf` without modification.
- Downloaded: 2026-11-18.
- Licence: SIL Open Font License 1.1; the full upstream licence is [OFL.txt](OFL.txt).

The interface uses this single family for headings, prose, controls, and tabular figures. Code diffs use the browser's local monospace fallback.

The existing server only exposes three named web assets and must stay running during this redesign. `style.css` therefore embeds the exact font bytes in its `@font-face` data URL. It is served from this Mac with the stylesheet, without a CDN or a new server route; the original font is retained here for licence/source review. To refresh the embedded bytes after an intentional font replacement:

```python
from pathlib import Path
import base64
import re
root = Path('apps/research-loop/web')
font = base64.b64encode((root / 'fonts/PublicSans.ttf').read_bytes()).decode()
css = root / 'style.css'
css.write_text(re.sub(r'data:font/ttf;base64,[A-Za-z0-9+/=]+',
                      'data:font/ttf;base64,' + font, css.read_text()))
```
