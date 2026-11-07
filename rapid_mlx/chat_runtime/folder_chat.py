"""Explicit folder QA: safe transport results and frozen, tool-free evidence."""

from dataclasses import dataclass

from fastapi import HTTPException
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from rapid_mlx.api.models import Message
from rapid_mlx.local_search.grounding import build_grounding_context


@dataclass
class FolderChatError(ValueError):
    code: str
    message: str


UNAVAILABLE = FolderChatError(
    "local_search_unavailable",
    "Folder search is unavailable. Check the selected folder and local server, then retry.",
)


def text_history(rows: list[dict]) -> list[Message]:
    """Folder QA cannot inherit tool calls, media, or executable attachments."""
    messages = []
    for row in rows:
        if row["role"] not in {"system", "user", "assistant"} or any(
            part["type"] != "text" for part in row["parts"]
        ):
            raise FolderChatError(
                "unsupported_local_search_history",
                "Start a new folder chat without tools or attachments.",
            )
        try:
            for part in row["parts"]:
                part["text"].encode("utf-8")
        except UnicodeEncodeError:
            raise FolderChatError(
                "invalid_local_search_query",
                "The folder chat must contain valid Unicode text.",
            ) from None
        messages.append(
            Message(
                role=row["role"],
                content="\n".join(part["text"] for part in row["parts"]),
            )
        )
    if (
        not messages
        or messages[-1].role != "user"
        or not isinstance(messages[-1].content, str)
        or not messages[-1].content.strip()
        or len(messages[-1].content) > 8192
    ):
        raise FolderChatError(
            "invalid_local_search_query", "Ask a question containing 1–8192 characters."
        )
    return messages


def prepared_context(messages, result, folders, scope):
    try:
        prepared, metadata = build_grounding_context(
            messages, result, folders, scope.folder_ids
        )
    except HTTPException as error:
        detail = error.detail.get("error", {}) if isinstance(error.detail, dict) else {}
        # Only errors from the shared formatter are admitted, never raw HTTP text.
        raise FolderChatError(
            detail.get("code", "local_search_unavailable"),
            detail.get("message", UNAVAILABLE.message),
        ) from None
    except (KeyError, TypeError, ValueError):
        raise FolderChatError(UNAVAILABLE.code, UNAVAILABLE.message) from None
    return {
        "messages": [
            {"role": message.role, "content": message.content} for message in prepared
        ],
        "metadata": metadata.model_dump(),
    }


def model_history(context: dict, owner_instructions: str):
    # Some chat templates accept a system message only at index zero. Keep the
    # owner policy and the exact frozen grounding policy together in that slot.
    system = "\n\n".join(
        [owner_instructions]
        + [
            message["content"]
            for message in context["messages"]
            if message["role"] == "system"
        ]
    )
    history = [ModelRequest(parts=[SystemPromptPart(system)])]
    for message in context["messages"]:
        if message["role"] == "system":
            continue
        if message["role"] == "assistant":
            history.append(ModelResponse(parts=[TextPart(message["content"])]))
        else:
            history.append(ModelRequest(parts=[UserPromptPart(message["content"])]))
    return history


def source_title(source: dict) -> str:
    title: str = source["title"]
    for first, last, name in (
        ("page", "end_page", "pages"),
        ("line", "end_line", "lines"),
    ):
        if source.get(first) is not None:
            end = source.get(last)
            locator = (
                str(source[first])
                if end is None or end == source[first]
                else f"{source[first]}–{end}"
            )
            title += f" · {name} {locator}"
    return title


# Owner-generated replies obey the same human-language policy as model replies.
# The HTTP grounding adapter's English protocol/error copies stay unchanged.
CHINESE_ERRORS = {
    "local_search_unavailable": "文件夹搜索暂不可用。请检查所选文件夹和本地服务后重试。",
    "local_search_unauthorized": "文件夹搜索需要授权。请重新连接本地服务。",
    "local_search_disabled": "请启用本地搜索并选择已登记的文件夹。",
    "folder_not_found": "所选文件夹不可用。请重新连接它或选择其他文件夹。",
    "local_search_full_text_unsupported": "本地服务未返回匹配的完整片段。请更新服务后再使用文件夹对话。",
    "unsupported_local_search_history": "请新建不含工具或附件的文件夹对话。",
    "invalid_local_search_query": "请输入包含 1–8192 个字符的有效文本问题。",
    "local_search_scope_mismatch": "搜索结果不属于所选文件夹。请重新连接本地服务后重试。",
    "local_search_context_too_large": "此文件夹对话超出上下文限制。请新建文件夹对话或缩小问题范围。",
    "local_search_token_count_unavailable": "无法验证文件夹上下文的大小。请更新或重新连接本地服务后重试。",
    "local_search_tools_forbidden": "文件夹对话不允许使用工具。",
    "local_search_empty_answer": "模型未返回回答。请缩小问题范围或换一个本地模型。",
}
