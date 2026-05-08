from vllm_mlx.routes.chat import (
    _artifact_fallback_tool_call,
    _artifact_retry_hint,
    _last_tool_result_needs_more_work,
    _last_user_requests_artifact,
    _looks_like_deferred_tool_use,
    _looks_like_incomplete_artifact_answer,
    _tool_call_validation_error,
    _tool_turn_max_tokens,
)
from vllm_mlx.service.helpers import _TOOL_USE_SYSTEM_SUFFIX


def test_deferred_tool_use_detects_intent_text():
    assert _looks_like_deferred_tool_use("Let me write the files individually.")


def test_deferred_tool_use_detects_raw_write_file_tail():
    assert _looks_like_deferred_tool_use('", "path": "/tmp/tsconfig.json"}')


def test_deferred_tool_use_ignores_plain_answer():
    assert not _looks_like_deferred_tool_use("The API exposes users and products.")


def test_incomplete_artifact_answer_detects_placeholder_prose():
    assert _looks_like_incomplete_artifact_answer("documentation links placeholder content")


def test_incomplete_artifact_answer_detects_interrupted_tool_thought():
    assert _looks_like_incomplete_artifact_answer(
        "[Response interrupted by a tool use thought.](Message)"
    )


def test_incomplete_artifact_answer_detects_default_vite_content():
    assert _looks_like_incomplete_artifact_answer("Join the Vite community")


def test_incomplete_artifact_answer_detects_install_dependency_loop():
    assert _looks_like_incomplete_artifact_answer("Install dependencies")


def test_last_user_requests_artifact_detects_create_prompt():
    messages = [{"role": "user", "content": "Create snake game using html"}]

    assert _last_user_requests_artifact(messages)


def test_last_user_requests_artifact_ignores_plain_question():
    messages = [{"role": "user", "content": "What is Express?"}]

    assert not _last_user_requests_artifact(messages)


def test_artifact_retry_hint_for_vite_landing_page():
    messages = [{"role": "user", "content": "create a landing page using vite"}]

    assert "overwrite or create src/App.tsx" in _artifact_retry_hint(messages)
    assert "visible page must be about Lightning MLX" in _artifact_retry_hint(messages)
    assert "compact enough to finish" in _artifact_retry_hint(messages)
    assert "Do not use npm create vite again" in _artifact_retry_hint(messages)


def test_artifact_retry_hint_for_express_typescript():
    messages = [{"role": "user", "content": "create a REST api using express and bun and typescript"}]

    assert "Mount routers from the app entrypoint" in _artifact_retry_hint(messages)
    assert "import type" in _artifact_retry_hint(messages)
    assert "bunx tsc --noEmit" in _artifact_retry_hint(messages)


def test_last_tool_result_needs_more_work_detects_empty_directory_listing():
    messages = [
        {"role": "user", "content": "create a REST api"},
        {"role": "tool", "content": "total 0\n-rw-r--r--  1 me wheel 0 pi.log"},
    ]

    assert _last_tool_result_needs_more_work(messages)


def test_last_tool_result_needs_more_work_detects_cancelled_scaffold():
    messages = [{"role": "tool", "content": "Operation cancelled"}]

    assert _last_tool_result_needs_more_work(messages)


def test_last_tool_result_needs_more_work_detects_successful_starter_scaffold():
    messages = [{"role": "tool", "content": "Done. Now run: npm install"}]

    assert _last_tool_result_needs_more_work(messages)


def test_last_tool_result_needs_more_work_detects_starter_source_listing():
    messages = [{"role": "tool", "content": "App.css\nApp.tsx\nassets\nindex.css\nmain.tsx"}]

    assert _last_tool_result_needs_more_work(messages)


def test_last_tool_result_needs_more_work_detects_default_vite_source():
    messages = [{"role": "tool", "content": "Explore Vite\nJoin the Vite community"}]

    assert _last_tool_result_needs_more_work(messages)


def test_last_tool_result_needs_more_work_detects_failed_edit():
    messages = [{"role": "tool", "content": "Could not find the exact text"}]

    assert _last_tool_result_needs_more_work(messages)


def test_tool_call_validation_error_detects_missing_required_arguments():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
        }
    ]
    calls = [{"function": {"name": "write", "arguments": "{}"}}]

    assert _tool_call_validation_error(calls, tools) == (
        "write: missing required argument(s): content, path"
    )


def test_artifact_fallback_tool_call_for_vite_landing_uses_bash():
    messages = [{"role": "user", "content": "create a landing page for lightning-mlx using vite"}]
    tools = [{"function": {"name": "bash"}}]

    fallback = _artifact_fallback_tool_call(messages, tools)

    assert fallback is not None
    assert fallback[0]["function"]["name"] == "bash"
    assert "lightning-mlx" in fallback[0]["function"]["arguments"]
    assert "npm run build" in fallback[0]["function"]["arguments"]


def test_tool_turn_max_tokens_allows_large_file_writes():
    assert _tool_turn_max_tokens(None) == 8192
    assert _tool_turn_max_tokens(16384) == 8192


def test_tool_use_prompt_prefers_path_before_content_for_write():
    assert "path argument before the content argument" in _TOOL_USE_SYSTEM_SUFFIX


def test_tool_use_prompt_rejects_starter_templates_as_complete_work():
    assert "Starter templates are not complete work" in _TOOL_USE_SYSTEM_SUFFIX
    assert "overwrite placeholder files" in _TOOL_USE_SYSTEM_SUFFIX
    assert "A scaffold command must never be your final tool call" in _TOOL_USE_SYSTEM_SUFFIX
    assert "Do not run install" in _TOOL_USE_SYSTEM_SUFFIX


def test_tool_use_prompt_requires_typescript_validation():
    assert "For TypeScript tasks" in _TOOL_USE_SYSTEM_SUFFIX
    assert "type errors before your final response" in _TOOL_USE_SYSTEM_SUFFIX
    assert "do not run long-lived dev" in _TOOL_USE_SYSTEM_SUFFIX
    assert "If you import a local file, create that file" in _TOOL_USE_SYSTEM_SUFFIX


def test_tool_use_prompt_requires_requested_stack():
    assert "Do not substitute a different stack" in _TOOL_USE_SYSTEM_SUFFIX
    assert "if the user asks for Vite, create a Vite app" in _TOOL_USE_SYSTEM_SUFFIX
    assert "prefer writing the needed Vite files directly" in _TOOL_USE_SYSTEM_SUFFIX
    assert "Vite is only the build tool" in _TOOL_USE_SYSTEM_SUFFIX
    assert "smallest working implementation" in _TOOL_USE_SYSTEM_SUFFIX
