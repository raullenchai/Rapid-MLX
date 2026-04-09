# Integration Tests

End-to-end tests that exercise Rapid-MLX from a real client library.

These are **not** run as part of `pytest tests/` because they need:

1. A running Rapid-MLX server on `http://localhost:8000`
2. A loaded model (see each test's `MODEL_ID`)
3. The client library installed (varies per test)
4. For Docker tests: Docker running

## Running

Start the server first:

```bash
rapid-mlx serve "/path/to/your/mlx-model" \
    --tool-call-parser hermes --enable-auto-tool-choice
```

Then run any test:

```bash
# Python integrations (need pip install of the client library)
python3 tests/integrations/test_pydantic_ai_full.py
python3 tests/integrations/test_smolagents_full.py
python3 tests/integrations/test_langchain.py
python3 tests/integrations/test_anthropic_sdk.py
python3 tests/integrations/test_openwebui.py

# CLI integrations
bash tests/integrations/test_aider.sh

# Docker E2E
python3 tests/integrations/test_librechat_docker.py
```

## Test coverage matrix

| Test | Plain | Stream | Tool | Multi-tool | Structured | Notes |
|---|---|---|---|---|---|---|
| `test_pydantic_ai_full.py` | x | x | x | x | x | + multi-turn |
| `test_smolagents_full.py` | x | — | x | x | — | CodeAgent + ToolCallingAgent |
| `test_langchain.py` | x | x | x | x | x | + system prompt |
| `test_anthropic_sdk.py` | x | x | x | — | — | `/v1/messages` endpoint |
| `test_openwebui.py` | — | x | — | — | — | Docker: register, login, models, chat |
| `test_aider.sh` | — | — | — | — | — | CLI edit-and-write workflow |
| `test_librechat_docker.py` | — | — | — | — | — | Docker: register, login, endpoints, models |

## Verified model matrix

All tests verified against three models:

| Suite | Gemma 4 26B (4bit) | Qwen3.5 122B-A10B (mxfp4) | Qwopus 3.5-27B (4bit) |
|---|---|---|---|
| PydanticAI (6 tests) | 6/6 | 6/6 | 6/6 |
| LangChain (6 tests) | 6/6 | 3/4 (*) | 6/6 |
| Anthropic SDK (5 tests) | 5/5 | 2/3 (*) | 4/5 (*) |
| smolagents (4 tests) | 4/4 | 2/2 | 3/4 |
| Claw Code (3 tests) | 3/3 | — | 3/3 |
| Aider CLI | PASS | PASS | — |
| LibreChat Docker (4 tests) | 4/4 | 4/4 | — |
| Open WebUI Docker (4 tests) | — | 3/4 (*) | — |

(*) Known Qwen3.5 thinking mode edge cases:
- LangChain `with_structured_output()` overrides max_tokens to None; thinking
  burns the token budget before producing JSON. Workaround: `--max-tokens 8192`.
- Anthropic streaming: thinking content is consumed by reasoning parser; final
  answer arrives in 1 chunk instead of many. Functionally correct.
- Open WebUI non-streaming chat with `max_tokens=50`: thinking exhausts the
  budget. Increase max_tokens.

The model path is hard-coded in each test's `MODEL_ID` constant. Edit it to
match your local model path.
