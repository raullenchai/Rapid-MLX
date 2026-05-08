# SuffixDecoding cross-validation: v2 vs v3 sweep

Total: 18 aliases · stable 14 · unstable 2 · fragile 2

| Alias | v2 tier | v3 tier | status | v2 speedup (chat/json/tool/code) | v3 speedup |
|---|---|---|---|---|---|
| `bonsai-1.7b` | avoid | avoid | ✅ stable | 0.71/1.30/0.88/1.05 | 0.78/1.40/0.94/0.86 |
| `bonsai-4b` | avoid | avoid | ✅ stable | 0.73/0.80/—/1.01 | 0.79/0.88/—/— |
| `bonsai-8b` | avoid | avoid | ✅ stable | 0.78/1.27/—/1.18 | 0.82/1.47/—/1.28 |
| `deepseek-r1-32b` | neutral | avoid | ⚠ unstable | 1.03/0.99/0.98/1.00 | 0.67/0.84/0.71/0.70 |
| `deepseek-r1-8b` | avoid | avoid | ✅ stable | 1.03/0.96/—/0.29 | 0.84/0.83/—/— |
| `devstral-v2-24b` | neutral | avoid | ⚠ unstable | 0.99/1.01/—/1.03 | 0.71/0.99/—/0.75 |
| `gemma-4-26b` | avoid | unknown | ❓ fragile | —/0.20/—/— | — |
| `gemma-4-31b` | avoid | avoid | ✅ stable | 1.47/0.67/—/1.00 | 1.61/0.72/—/1.07 |
| `gemma3-27b` | unknown | unknown | ❓ fragile | — | — |
| `hermes3-8b` | avoid | avoid | ✅ stable | 0.61/0.98/0.59/0.58 | 0.68/1.10/0.62/0.69 |
| `hermes4-70b` | avoid | avoid | ✅ stable | 0.69/1.08/0.75/0.68 | 0.70/1.12/0.79/0.70 |
| `llama3-3b` | avoid | avoid | ✅ stable | 0.59/0.88/—/— | 0.64/0.95/—/— |
| `ministral-3b` | avoid | avoid | ✅ stable | 0.84/1.08/—/— | 0.67/0.92/—/— |
| `phi4-14b` | avoid | avoid | ✅ stable | 0.66/0.94/0.67/0.87 | 0.65/0.90/0.64/— |
| `qwen3-coder-30b` | avoid | avoid | ✅ stable | 0.87/0.99/1.04/0.93 | 0.45/0.89/—/1.80 |
| `qwen3-vl-30b` | neutral | neutral | ✅ stable | 1.05/1.03/—/0.99 | 1.00/1.00/—/1.01 |
| `qwen3-vl-8b` | neutral | neutral | ✅ stable | 0.99/1.03/—/1.02 | 1.00/1.00/—/1.00 |
| `smollm3-3b` | avoid | avoid | ✅ stable | 0.51/—/0.78/— | —/2.76/0.63/— |

