# DeepSeek V4.1 fused attention for speculative verify (2026-10-21)

## Finding

A draft round is about 21 ms of drafting, 102-105 ms for the five-token target
verify and 3-5 ms of host work (#5038), so the verify forward sets the
break-even acceptance. The target forward grew by about 10 ms per extra row
(51.5 ms for one row, 98 ms for five), far more than the weight traffic of the
added rows. One cause: the fused decode attention kernel covered only one query
row, so every verify chunk fell back to the graph attention: a chronological
window gather, concatenations, a full conversion and concatenation of every
compressed row per layer, the gather, two einsums, the masked softmax and the
inverse rope, about 25 dispatches per layer.

## Change

The fused kernel takes 1-8 query rows that fit in the window ring, with one
threadgroup row per query row. Each row reads exactly the keys its own decode
step would see: ring slots that a later row of the chunk already overwrote are
read from a snapshot of those rows taken before the write. Each row is
therefore bit-identical to the one-row kernel at its position (tested on Metal
for 2, 5 and 8 rows across ring wrap-around, with and without compressed
rows). Decode (one row) is unchanged.

Verify attention now matches decode attention per row instead of using the
graph path, which removes one source of verify-versus-decode numerics
differences.

## Measurements

Full V4.1 Flash 1017 checkpoint (revision c95bb8ad), M3 Ultra 256 GB. Each cell
is a target forward of N rows after the prompt, rolled back after every call,
median of 12 after 3 warm-up calls. Base and change were interleaved twice in
the same GPU window; ranges are the two runs.

| Rows | base, 40-token prompt (ms) | change (ms) | base, 7049-token prompt (ms) | change (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 51.5-51.6 | 51.3-52.1 | 58.6-59.6 | 58.4-60.6 |
| 2 | 67.2-67.3 | 61.0-61.9 | 71.6-72.8 | 66.0-66.3 |
| 3 | 77.6-77.9 | 71.4-72.0 | 84.1-85.9 | 75.9-80.2 |
| 5 | 98.1-100.6 | 92.2-94.9 | 103.9-104.8 | 103.1-104.7 |
| 6 | 109.8-110.8 | 103.4-107.6 | 119.0-119.7 | 111.0-112.6 |
| 8 | 131.1-133.8 | 125.6-125.7 | 140.2-140.4 | 133.7-136.0 |

At the short prompt every verify size gains about 6 ms. At 7k tokens most
sizes gain 6-8 ms; the five-row cell is within noise there and should be
re-measured before quoting a long-context five-row number.

Reproduce: `/usr/bin/lockf -k /var/tmp/rapid-mlx-large-model.lock` around a
script that loads `serving.load_product_runtime(..., mixed_target_precision=True)`,
prefills the rendered chat prompt, then times
`model(tokens[:, :n], cache, enable_rollback=True)` followed by
`cache.rollback(prompt_len)` for each n.
