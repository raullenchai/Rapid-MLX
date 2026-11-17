# M5 Max short-request TTFT

Measured A/B evidence: Apple M5 Max, 36 GB, GPU generation 17,
macOS 27.0.1, MLX 0.32.4, mlx-lm 0.31.3. Model
`rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX`, revision `aa985c29`, default serve
(MTP). Each arm used a fresh server, three warmups and two rounds, with
client-side streaming TTFT and temperature 0.

Arms: base retains `prefix_boundary_min_tokens=0, admission_window_ms=8`;
p256 sets the prefix threshold to 256; w0 sets the admission window to 0;
both combines p256 and w0. Values below are median milliseconds, round 1 / round 2.

| Workload | Base | p256 | w0 | Both |
| --- | --- | --- | --- | --- |
| Short first turn | 189.4 / 185.6 | 181.0 / 178.9 | 174.8 / 173.5 | 170.8 / 171.0 |
| Two-turn chat, turn 1 | 185.7 / 176.8 | 177.9 / 180.3 | 167.2 / 167.1 | 166.6 / 168.5 |
| Two-turn chat, turn 2 | 313.1 / 306.3 | 181.3 / 184.7 | 293.2 / 289.8 | 170.7 / 169.0 |
| Approximately 1.5K-token prompt | 849.5 / 839.6 | 821.6 / 817.2 | 805.3 / 805.6 | 832.1 / 832.5 |

Output identity used non-streaming requests, temperature 0, eight two-turn
chats, maximum 64 output tokens and a cold cache for each arm. Base versus
base was 16/16 identical; base versus both was 16/16 identical. This control
qualifies these measured chats, rather than all possible outputs.

With bare mlx-lm and the same weights, a 73-token prompt took 185 ms in one
pass versus 270 ms with the server's 58 + 14 + 1 split. The per-pass floor was
approximately 40 ms:

| Tokens per pass | 1 | 4 | 8 | 16 | 64 | 128 |
| --- | --- | --- | --- | --- | --- | --- |
| Milliseconds | 40 | 45 | 77 | 111 | 123 | 192 |

A lone request also pays one 8 ms admission tick under the base policy.
Skipping a short message-boundary split reduces forwards; removing the hold
avoids that tick. The combined arm improves the measured short requests,
especially turn 2; it is slower than either isolated change on the longer
prompt, so these results do not establish a universal prefill improvement.

The builtin cell selects both knobs only on generation 17, Max tier, exactly
36 GB, MLX 0.32.4 and macOS 27.0.1, for MTP, width 1 and batch 1. Core count
is not a key. The schema cannot key on checkpoint name or revision: the
narrowest available model key is family `qwen3_5_text`, geometry
`hidden_size=5120, bits=4, group_size=64`, and quantization mode `affine`.
Live text args omit intermediate size, so it is not a key. Installed module
formats supply quantization facts omitted by the args; mixed or unknown
formats do not qualify. Checkpoints with the same facts share this scope.
Other chips, memory sizes, model facts, software and modes retain the defaults.

`admission_window_ms=0` is confined to this batch-1 MTP cell; waiting waves
larger than one retain the 8 ms hold because coalescing matters for concurrent
bursts. A later arrival can still miss a singleton's disabled hold. Skipping a
boundary below 256 tokens can require fewer than 256 additional prefill tokens
(at most 255) on the next turn relative to reusing that boundary with the same
retained earlier prefix. Eviction or a changed earlier prefix can independently
require more work. Global defaults remain threshold 0 and window 8 ms.
