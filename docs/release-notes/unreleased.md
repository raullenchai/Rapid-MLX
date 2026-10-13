## Highlights

**Actionable image validation** — Asset-independent parameter checks (steps, dimensions, guidance, negative prompt, character limits, and input-image rules) return HTTP 400 with the model’s validation message before any download or load. HiDream’s token limit also returns HTTP 400, but is only evaluated once the model’s assets are available; a cold request downloads and loads the model first. Operational failures remain HTTP 500, and cancellation remains successful.
