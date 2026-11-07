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


def model_history(context: dict):
    history = []
    for message in context["messages"]:
        if message["role"] == "assistant":
            history.append(ModelResponse(parts=[TextPart(message["content"])]))
        else:
            cls = SystemPromptPart if message["role"] == "system" else UserPromptPart
            history.append(ModelRequest(parts=[cls(message["content"])]))
    return history


def source_title(source: dict) -> str:
    title = source["title"]
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
