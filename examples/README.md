# examples/

Runnable client-side examples against a running `rapid-mlx serve` instance.

- **API demos**: `demo_curl_{text,image,video}.sh`, `demo_openai_{text,image,video}.py` — the same three flows via raw curl and the `openai` Python client.
- **MCP / tools**: `mcp_chat.py`, `mcp_tool_use.py`, `mcp.example.json`.
- **Audio**: `tts_example.py`, `tts_multilingual.py`, `mic_transcribe.py`, `mic_live.py`, `mic_realtime.py`, `closed_captions.py`, `audio_separation_example.py`; the `assistant_*.wav` files are TTS output samples linked from `docs/guides/audio.md`.
- **Multimodal**: `mllm_example.py`, `test_video.py`.
- **Benchmarks**: `benchmark_all_models.py`, `benchmark_audio.py`, `benchmark_detokenizer.py`, `mllm_benchmark.py`.
- **API behavior checks**: `test_batching.py`, `test_batch_sync.py`, `test_openai_compatibility.py` (standalone scripts, not pytest suites).
