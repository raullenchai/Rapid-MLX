from vllm_mlx.routes.chat import _looks_like_deferred_tool_use


def test_deferred_tool_use_detects_intent_text():
    assert _looks_like_deferred_tool_use("Let me write the files individually.")


def test_deferred_tool_use_detects_raw_write_file_tail():
    assert _looks_like_deferred_tool_use('", "path": "/tmp/tsconfig.json"}')


def test_deferred_tool_use_ignores_plain_answer():
    assert not _looks_like_deferred_tool_use("The API exposes users and products.")
