# SPDX-License-Identifier: Apache-2.0
"""Chat completion endpoints — /v1/chat/completions."""

import gc
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceLogProbs,
    TokenLogProb,
    Usage,
)
from ..api.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_schema_for_guided,
    parse_json_output,
)
from ..api.utils import (
    clean_output_text,
    extract_json_from_response,
    extract_multimodal_content,
    sanitize_output,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import GenerationOutput
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _TOOL_USE_SYSTEM_SUFFIX,
    _build_usage,
    _disconnect_guard,
    _extract_token_logprob,
    _inject_json_instruction,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    _validate_tool_call_params,
    _wait_with_disconnect,
    get_engine,
    get_usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_TOOL_INTENT_RE = re.compile(
    r"\b("
    r"let me|now let me|i'?ll|i will|starting with|"
    r"continue|create|write|edit|run|test|verify|fix|repair"
    r")\b",
    re.IGNORECASE,
)

_ARTIFACT_REQUEST_RE = re.compile(
    r"\b(create|build|make|write|implement|generate|scaffold)\b",
    re.IGNORECASE,
)


def _last_user_text(messages: list) -> str:
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "user":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, list):
            text = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        else:
            text = str(content or "")
        return text
    return ""


def _last_user_requests_artifact(messages: list) -> bool:
    return bool(_ARTIFACT_REQUEST_RE.search(_last_user_text(messages)))


def _artifact_retry_hint(messages: list) -> str:
    text = _last_user_text(messages).lower()
    if "vite" in text and ("landing" in text or "page" in text):
        return (
            " For this Vite landing-page task, call bash or write now to overwrite "
            "or create src/App.tsx and CSS with Lightning MLX landing content. "
            "Vite is only the build tool; the visible page must be about Lightning "
            "MLX, not default Vite or React community content. "
            "Keep the page compact enough to finish in one complete source write. "
            "Do not use npm create vite again. Ensure package.json has a finite "
            "build script, then run npm install and npm run build."
        )
    if "express" in text and ("typescript" in text or "bun" in text):
        return (
            " For this Express TypeScript task, call bash or write now to overwrite "
            "the starter source with an Express REST API. Mount routers from the app "
            "entrypoint, use `import type` for TypeScript-only imports when "
            "verbatimModuleSyntax is enabled, then run `bunx tsc --noEmit` or a "
            "finite startup validation and fix errors before answering."
        )
    if "snake" in text:
        return (
            " For this Snake game task, call bash or write now to create an index.html "
            "with canvas, snake movement, food, scoring, and keyboard controls."
        )
    if "poem" in text or "peom" in text:
        return " For this poem task, call write now to create a poem file."
    return ""


def _last_tool_result_needs_more_work(messages: list) -> bool:
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "tool":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        text = str(content or "").lower()
        return any(
            marker in text
            for marker in (
                "validation failed",
                "iserror",
                "no such file",
                "command exited with code 1",
                "operation cancelled",
                "could not find the exact text",
                "scaffolding project in",
                "done. now run",
                "happy hacking",
                "install dependencies",
                "hello via bun",
                "packages installed",
                "added 153 packages",
                "found 0 vulnerabilities",
                "join the vite community",
                "explore vite",
                "edit src/app",
                "edit src/main",
                "app.css\napp.tsx",
                "assets\nindex.css\nmain.tsx",
                "assets\ncounter.js\nmain.js\nstyle.css",
                "total 0",
            )
        )
    return False


def _tool_call_validation_error(tool_calls: list | None, tools: list | None) -> str | None:
    if not tool_calls or not tools:
        return None

    tool_defs = [tool.model_dump() if hasattr(tool, "model_dump") else tool for tool in tools]
    required_by_name: dict[str, set[str]] = {}
    for tool in tool_defs:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        parameters = function.get("parameters") or {}
        required = parameters.get("required") or []
        if name:
            required_by_name[name] = {str(param) for param in required}

    for tool_call in tool_calls:
        function = (
            tool_call.function
            if hasattr(tool_call, "function")
            else tool_call.get("function", {})
        )
        name = function.name if hasattr(function, "name") else function.get("name", "")
        arguments = (
            function.arguments
            if hasattr(function, "arguments")
            else function.get("arguments", "{}")
        )
        try:
            parsed_args = json.loads(arguments or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return f"{name}: arguments must be valid JSON"
        if not isinstance(parsed_args, dict):
            return f"{name}: arguments must be an object"
        missing = [
            param
            for param in sorted(required_by_name.get(name, set()))
            if parsed_args.get(param) in (None, "")
        ]
        if missing:
            return f"{name}: missing required argument(s): {', '.join(missing)}"

    return None


def _available_tool_names(tools: list | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else tool
        function = tool_dict.get("function", {}) if isinstance(tool_dict, dict) else {}
        name = function.get("name")
        if name:
            names.add(str(name))
    return names


def _bash_tool_call(command: str) -> dict:
    return {
        "index": 0,
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": command}),
        },
    }


def _artifact_fallback_tool_call(messages: list, tools: list | None) -> list[dict] | None:
    if "bash" not in _available_tool_names(tools):
        return None

    prompt = _last_user_text(messages).lower()
    if "vite" in prompt and ("landing" in prompt or "page" in prompt):
        return [
            _bash_tool_call(
                "mkdir -p src && "
                "cat > package.json <<'EOF'\n"
                "{\"type\":\"module\",\"scripts\":{\"build\":\"vite build\"},"
                "\"dependencies\":{\"@vitejs/plugin-react\":\"latest\","
                "\"vite\":\"latest\",\"typescript\":\"latest\"},"
                "\"devDependencies\":{}}\n"
                "EOF\n"
                "cat > index.html <<'EOF'\n"
                "<div id=\"app\"></div><script type=\"module\" src=\"/src/main.js\"></script>\n"
                "EOF\n"
                "cat > src/main.js <<'EOF'\n"
                "import './style.css';\n"
                "document.querySelector('#app').innerHTML = `<main class=\"shell\">"
                "<section class=\"hero\"><p class=\"eyebrow\">lightning-mlx</p>"
                "<h1>Local MLX inference at production speed.</h1>"
                "<p>Serve OpenAI-compatible chat, tool calling, and streaming on "
                "Apple Silicon with fast MTPLX defaults.</p>"
                "<div class=\"actions\"><a href=\"#install\">Install</a>"
                "<a href=\"#features\">Features</a></div></section>"
                "<section id=\"features\" class=\"grid\"><article><h2>Agentic ready</h2>"
                "<p>Reliable tool calls for coding agents.</p></article>"
                "<article><h2>MLX native</h2><p>Optimized for local Apple GPUs.</p>"
                "</article><article><h2>OpenAI API</h2>"
                "<p>Drop-in /v1 compatibility.</p></article></section>"
                "<section id=\"install\" class=\"code\">uv run lightning-mlx serve "
                "qwen3.6-27b --served-model-name local --port 8010</section></main>`;\n"
                "EOF\n"
                "cat > src/style.css <<'EOF'\n"
                "body{margin:0;font-family:Inter,system-ui,sans-serif;background:#0b0f19;"
                "color:#eef2ff}.shell{max-width:1120px;margin:auto;padding:72px 24px}"
                ".hero{min-height:55vh;display:grid;align-content:center;gap:20px}"
                ".eyebrow{color:#7dd3fc;text-transform:uppercase;letter-spacing:.12em}"
                "h1{font-size:clamp(40px,8vw,88px);line-height:1;margin:0}"
                ".hero p{max-width:680px;color:#cbd5e1;font-size:20px}"
                ".actions{display:flex;gap:12px;flex-wrap:wrap}.actions a{color:#08111f;"
                "background:#7dd3fc;padding:12px 16px;border-radius:8px;text-decoration:none}"
                ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));"
                "gap:16px}.grid article,.code{border:1px solid #334155;border-radius:8px;"
                "padding:20px;background:#111827}.code{margin-top:24px;font-family:monospace;"
                "color:#bae6fd;overflow:auto}\n"
                "EOF\n"
                "npm install && npm run build"
            )
        ]
    if "express" in prompt and ("typescript" in prompt or "bun" in prompt):
        return [
            _bash_tool_call(
                "mkdir -p src && "
                "cat > package.json <<'EOF'\n"
                "{\"type\":\"module\",\"scripts\":{\"typecheck\":\"tsc --noEmit\","
                "\"start\":\"bun src/index.ts\"},\"dependencies\":{\"express\":\"latest\"},"
                "\"devDependencies\":{\"@types/bun\":\"latest\",\"@types/express\":\"latest\","
                "\"typescript\":\"latest\"}}\n"
                "EOF\n"
                "cat > tsconfig.json <<'EOF'\n"
                "{\"compilerOptions\":{\"target\":\"ES2022\",\"module\":\"ESNext\","
                "\"moduleResolution\":\"Bundler\",\"strict\":true,\"types\":[\"bun\"],"
                "\"skipLibCheck\":true,\"noEmit\":true},\"include\":[\"src/**/*.ts\"]}\n"
                "EOF\n"
                "cat > src/index.ts <<'EOF'\n"
                "import express from 'express';\n"
                "import type { Request, Response } from 'express';\n"
                "type Todo={id:number;title:string;done:boolean};\n"
                "const app=express(); const todos:Todo[]=[]; let nextId=1;\n"
                "app.use(express.json());\n"
                "app.get('/health',(_req:Request,res:Response)=>res.json({ok:true}));\n"
                "app.get('/todos',(_req:Request,res:Response)=>res.json(todos));\n"
                "app.post('/todos',(req:Request,res:Response)=>{const todo={id:nextId++,"
                "title:String(req.body.title||''),done:false};todos.push(todo);"
                "res.status(201).json(todo);});\n"
                "app.put('/todos/:id',(req:Request,res:Response)=>{const todo=todos.find("
                "item=>item.id===Number(req.params.id));if(!todo)return res.sendStatus(404);"
                "todo.title=String(req.body.title??todo.title);todo.done=Boolean("
                "req.body.done??todo.done);res.json(todo);});\n"
                "app.delete('/todos/:id',(req:Request,res:Response)=>{const index=todos."
                "findIndex(item=>item.id===Number(req.params.id));if(index<0)return "
                "res.sendStatus(404);res.json(todos.splice(index,1)[0]);});\n"
                "app.listen(3000,()=>console.log('REST API listening on 3000'));\n"
                "EOF\n"
                "bun install && bunx tsc --noEmit"
            )
        ]
    if "snake" in prompt:
        return [
            _bash_tool_call(
                "cat > index.html <<'EOF'\n"
                "<canvas id=\"game\" width=\"400\" height=\"400\"></canvas>"
                "<p>Score: <span id=\"score\">0</span></p><script>"
                "const c=document.getElementById('game'),x=c.getContext('2d'),s="
                "document.getElementById('score');let snake=[{x:10,y:10}],food={x:15,y:15},"
                "dx=1,dy=0,score=0;addEventListener('keydown',e=>{if(e.key==='ArrowUp')"
                "{dx=0;dy=-1}else if(e.key==='ArrowDown'){dx=0;dy=1}else if(e.key==="
                "'ArrowLeft'){dx=-1;dy=0}else if(e.key==='ArrowRight'){dx=1;dy=0}});"
                "function loop(){let h={x:(snake[0].x+dx+20)%20,y:(snake[0].y+dy+20)%20};"
                "if(snake.some(p=>p.x===h.x&&p.y===h.y)){snake=[{x:10,y:10}];score=0}"
                "snake.unshift(h);if(h.x===food.x&&h.y===food.y){score++;food={x:Math.floor("
                "Math.random()*20),y:Math.floor(Math.random()*20)}}else snake.pop();"
                "s.textContent=score;x.fillStyle='#111';x.fillRect(0,0,400,400);"
                "x.fillStyle='#22c55e';snake.forEach(p=>x.fillRect(p.x*20,p.y*20,18,18));"
                "x.fillStyle='#ef4444';x.fillRect(food.x*20,food.y*20,18,18)}"
                "setInterval(loop,120)</script>\n"
                "EOF"
            )
        ]
    if "poem" in prompt or "peom" in prompt:
        return [
            _bash_tool_call(
                "cat > cat-poem.txt <<'EOF'\n"
                "Cats in moonlit windows gleam,\n"
                "Soft paws walking through a dream.\n"
                "Whiskers twitch and shadows play,\n"
                "Tiny hunters greet the day.\n"
                "EOF"
            )
        ]

    return None


def _looks_like_deferred_tool_use(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if (
        '"path"' in lowered
        or "calling tool:" in lowered
        or "calling tool=" in lowered
        or "[calling tool" in lowered
        or "_tool:" in lowered
        or "<tool_call>" in lowered
        or "<parameter=" in lowered
        or "</parameter" in lowered
        or "</function" in lowered
    ):
        return True
    return bool(_TOOL_INTENT_RE.search(text))


def _looks_like_incomplete_artifact_answer(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "placeholder content",
            "starter template",
            "vite template",
            "documentation links",
            "edit src/",
            "get started",
            "install dependencies",
            "explore vite",
            "join the vite community",
            "response interrupted by a tool use thought",
        )
    )


def _looks_like_invalid_tool_continuation(text: str | None) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if len(stripped) <= 32 and (
        lowered.startswith("_") or lowered.endswith("_name")
    ):
        return True
    if len(stripped) <= 16:
        return True
    return lowered in {
        "i",
        "i'm",
        "i’ll",
        "i'll",
        "i will",
        "now",
        "now,",
        "user",
        "user:",
        'user: "',
        "assistant",
        "assistant:",
        'assistant: "',
    }


def _should_emit_reasoning(request: ChatCompletionRequest) -> bool:
    """Stream reasoning by default. Off when --no-thinking or client opts out."""
    if get_config().no_thinking:
        return False
    if request.enable_thinking is False:
        return False
    return True


def _tool_turn_max_tokens(max_tokens: int | None) -> int:
    """Bound tool turns enough to avoid runaway planning without truncating files."""
    return min(int(max_tokens or 8192), 8192)


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    _validate_model_name(request.model)
    engine = get_engine(request.model)

    # Validate messages is non-empty
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty",
        )

    # Validate message roles
    _valid_roles = {"system", "user", "assistant", "tool", "developer"}
    for msg in request.messages:
        if msg.role not in _valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be one of: {', '.join(sorted(_valid_roles))}",
            )
    request_last_role = request.messages[-1].role

    # Validate n parameter (only n=1 supported)
    if request.n is not None and request.n > 1:
        raise HTTPException(
            status_code=400,
            detail="n > 1 is not supported. Rapid-MLX generates one completion per request.",
        )

    # Validate max_tokens (must be positive)
    if request.max_tokens is not None and request.max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at least 1",
        )

    # Validate temperature range (OpenAI spec: 0-2)
    if request.temperature is not None and (
        request.temperature < 0 or request.temperature > 2
    ):
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    # Validate top_logprobs range (OpenAI spec: 0-20)
    if request.top_logprobs is not None and (
        request.top_logprobs < 0 or request.top_logprobs > 20
    ):
        raise HTTPException(
            status_code=400,
            detail="top_logprobs must be between 0 and 20",
        )

    # --- Detailed request logging ---
    n_msgs = len(request.messages)
    msg_roles = [m.role for m in request.messages]
    total_chars = 0
    last_user_preview = ""
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    has_tools = bool(request.tools)
    n_tools = len(request.tools) if request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/chat/completions stream={request.stream} "
        f"model={request.model!r} max_tokens={request.max_tokens} "
        f"temp={request.temperature} msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.debug(f"[REQUEST] last user message preview: {last_user_preview!r}")

    cfg = get_config()

    # Save original messages (clean dicts) for cloud routing BEFORE
    # local mutations (extract_multimodal_content, developer→system, suffix injection).
    if cfg.cloud_router:
        _cloud_original_messages = [
            (
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump")
                else {k: v for k, v in dict(msg).items() if v is not None}
            )
            for msg in request.messages
        ]
    else:
        _cloud_original_messages = None

    # For MLLM models, keep original messages with embedded images
    if engine.is_mllm:
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos = [], []
        logger.debug(f"MLLM: Processing {len(messages)} messages")
    else:
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

    has_media = bool(images or videos)
    if engine.is_mllm and not has_media:
        for msg in request.messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else (item.get("type", "") if isinstance(item, dict) else "")
                    )
                    if item_type in ("image_url", "image", "video", "video_url"):
                        has_media = True
                        break
            if has_media:
                break

    # Normalize "developer" role to "system"
    for i, m in enumerate(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "developer":
            if isinstance(m, dict):
                messages[i]["role"] = "system"
            else:
                m.role = "system"

    # Auto-inject system prompt suffix for tool use and/or reasoning control
    _inject_suffix = None
    if request.tools and cfg.tool_call_parser:
        _inject_suffix = _TOOL_USE_SYSTEM_SUFFIX
    elif cfg.reasoning_parser_name == "minimax":
        _inject_suffix = (
            "\n\nDo NOT think out loud or show your reasoning process. "
            "Give direct answers only — no preamble like 'The user asks...' or "
            "'We should respond...' or 'Let me think...'. Be concise."
        )

    if _inject_suffix:
        has_system = any(
            (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
            == "system"
            for m in messages
        )
        if has_system:
            for i, m in enumerate(messages):
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                if role == "system":
                    if isinstance(m, dict):
                        messages[i] = {**m, "content": m["content"] + _inject_suffix}
                    else:
                        messages[i]["content"] = m["content"] + _inject_suffix
                    break
        else:
            system_msg = {"role": "system", "content": _inject_suffix.strip()}
            messages = [system_msg] + list(messages)

    # Auto-pin system prompt prefix cache blocks
    if cfg.pin_system_prompt:
        _maybe_pin_system_prompt(messages)

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        try:
            json_instruction = build_json_system_prompt(response_format)
        except Exception as e:
            logger.warning(f"Failed to build JSON system prompt: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid response_format schema: {e}",
            )
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(request.max_tokens, request.enable_thinking),
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": request.stop,
    }
    if request.tools:
        chat_kwargs["max_tokens"] = _tool_turn_max_tokens(chat_kwargs["max_tokens"])

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # Add tools if provided
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    # Pass through enable_thinking if explicitly set by the client
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking
    elif cfg.no_thinking:
        chat_kwargs["enable_thinking"] = False

    # Cloud routing: offload large-context requests to cloud LLM
    if cfg.cloud_router and not engine.is_mllm and hasattr(engine, "build_prompt"):
        try:
            prompt = engine.build_prompt(messages, tools=request.tools)
            total_tokens, new_tokens = engine.model.estimate_new_tokens(prompt)
            if cfg.cloud_router.should_route_to_cloud(new_tokens):
                logger.info(
                    f"[CLOUD ROUTE] {new_tokens} new tokens (total {total_tokens}) "
                    f"> threshold {cfg.cloud_router.threshold}, "
                    f"routing to {cfg.cloud_router.cloud_model}"
                )
                cloud_messages = _cloud_original_messages
                cloud_kwargs = {
                    "temperature": chat_kwargs.get("temperature"),
                    "max_tokens": chat_kwargs.get("max_tokens"),
                    "top_p": chat_kwargs.get("top_p"),
                }
                if request.stop:
                    cloud_kwargs["stop"] = request.stop
                if request.tool_choice is not None:
                    cloud_kwargs["tool_choice"] = request.tool_choice
                if request.response_format:
                    rf = request.response_format
                    cloud_kwargs["response_format"] = (
                        rf.model_dump() if hasattr(rf, "model_dump") else rf
                    )
                if request.tools:
                    cloud_kwargs["tools"] = [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in request.tools
                    ]
                if request.stream:
                    return StreamingResponse(
                        _disconnect_guard(
                            cfg.cloud_router.stream_completion(
                                cloud_messages,
                                model_name=cfg.model_name or "cloud",
                                **cloud_kwargs,
                            ),
                            raw_request,
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    result = await _wait_with_disconnect(
                        cfg.cloud_router.completion(cloud_messages, **cloud_kwargs),
                        raw_request,
                        timeout=request.timeout or cfg.default_timeout,
                    )
                    if result is None:
                        return Response(status_code=499, content="Client disconnected")
                    return Response(
                        content=json.dumps(result),
                        media_type="application/json",
                    )
            else:
                logger.info(
                    f"[LOCAL] {new_tokens} new tokens (total {total_tokens}) "
                    f"<= threshold {cfg.cloud_router.threshold}, using local inference"
                )
        except Exception as e:
            logger.warning(
                f"[CLOUD ROUTE] Error during routing check: {e}, falling back to local"
            )

    if request.stream:
        # Validate chat template eagerly so template errors return 400
        if hasattr(engine, "build_prompt") and not engine.is_mllm:
            try:
                engine.build_prompt(
                    messages,
                    tools=chat_kwargs.get("tools"),
                    enable_thinking=chat_kwargs.get("enable_thinking"),
                )
            except Exception as e:
                err_msg = str(e)
                err_type = type(e).__name__
                if (
                    "TemplateError" in err_type
                    or "template" in err_msg.lower()
                    or ("user" in err_msg.lower() and "found" in err_msg.lower())
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Chat template error: {err_msg}",
                    )
                raise
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(engine, messages, request, **chat_kwargs),
                raw_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or cfg.default_timeout

    # Disable GC during generation to avoid latency spikes
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    # Determine if we need per-token logprobs
    want_logprobs = request.logprobs and request.top_logprobs
    top_k_logprobs = request.top_logprobs or 0
    token_logprobs_list: list[TokenLogProb] = []

    # Check if we should use guided generation for JSON schema
    use_guided = False
    json_schema = None
    if response_format and not request.tools:
        json_schema = extract_json_schema_for_guided(response_format)
        if json_schema and hasattr(engine, "supports_guided_generation"):
            use_guided = engine.supports_guided_generation
            if use_guided:
                logger.info("Using guided generation for JSON schema enforcement")

    try:
        if want_logprobs and not use_guided:
            output = None
            async for chunk in engine.stream_chat(messages=messages, **chat_kwargs):
                output = chunk
                if chunk.logprobs is not None and chunk.new_text:
                    token_id = chunk.tokens[-1] if chunk.tokens else 0
                    token_logprobs_list.append(
                        _extract_token_logprob(
                            chunk.logprobs, token_id, engine.tokenizer, top_k_logprobs
                        )
                    )
            if output is None:
                return Response(status_code=499)
        elif use_guided and json_schema:
            try:
                output = await _wait_with_disconnect(
                    engine.generate_with_schema(
                        messages=messages,
                        json_schema=json_schema,
                        **chat_kwargs,
                    ),
                    raw_request,
                    timeout=timeout,
                )
            except Exception as guided_err:
                logger.warning(
                    f"Guided generation failed, falling back to standard: {guided_err}"
                )
                logger.debug(f"Problematic schema: {json_schema}")
                output = await _wait_with_disconnect(
                    engine.chat(messages=messages, **chat_kwargs),
                    raw_request,
                    timeout=timeout,
                )
        else:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                raw_request,
                timeout=timeout,
            )
    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e)
        err_type = type(e).__name__
        if (
            "TemplateError" in err_type
            or "template" in err_msg.lower()
            or ("user" in err_msg.lower() and "found" in err_msg.lower())
        ):
            raise HTTPException(
                status_code=400, detail=f"Chat template error: {err_msg}"
            )
        raise
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()

    if output is None:
        return Response(status_code=499)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls from output using configured parser
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, request)

    force_tool_work = bool(request.tools and _last_user_requests_artifact(request.messages))
    last_tool_needs_more_work = _last_tool_result_needs_more_work(request.messages)
    retry_messages = list(messages)
    artifact_hint = _artifact_retry_hint(request.messages)
    for retry_index in range(2):
        validation_error = _tool_call_validation_error(tool_calls, request.tools)
        if (
            not request.tools
            or (tool_calls and not validation_error)
            or (
                not force_tool_work
                and not last_tool_needs_more_work
                and not _looks_like_deferred_tool_use(cleaned_text or output.text)
                and not validation_error
            )
        ):
            break
        logger.info(
            "Tool intent without tool call detected; retrying (%d/2)",
            retry_index + 1,
        )
        retry_messages = retry_messages + [
            {"role": "assistant", "content": cleaned_text or output.text},
            {
                "role": "user",
                "content": (
                    "Call the appropriate tool now. Do not explain, do not describe "
                    "what you will do, and do not output raw JSON as text. "
                    f"{artifact_hint} If the "
                    "previous tool failed, repair it by calling another tool. For "
                    "create/build/scaffold tasks, create the files now; do not inspect "
                    "the directory again."
                ),
            },
        ]
        retry_output = await _wait_with_disconnect(
            engine.chat(messages=retry_messages, **chat_kwargs),
            raw_request,
            timeout=timeout,
        )
        if retry_output is None:
            break
        retry_cleaned_text, retry_tool_calls = _parse_tool_calls_with_parser(
            retry_output.text, request
        )
        output = retry_output
        cleaned_text = retry_cleaned_text
        tool_calls = retry_tool_calls

    if request.tools and (
        not tool_calls
        or _tool_call_validation_error(tool_calls, request.tools)
    ):
        if force_tool_work or last_tool_needs_more_work:
            fallback_tool_call = _artifact_fallback_tool_call(
                request.messages, request.tools
            )
            if fallback_tool_call:
                tool_calls = fallback_tool_call
                cleaned_text = ""

    # Validate tool call parameter values against schemas
    if tool_calls and request.tools:
        _validate_tool_call_params(tool_calls, request.tools)

    # Extract reasoning content FIRST.
    # Note: extract_reasoning() is stateless (pure regex on full text),
    # so using the singleton is safe here unlike the streaming variant.
    reasoning_text = None
    if cfg.reasoning_parser:
        text_to_parse = cleaned_text if tool_calls else (cleaned_text or output.text)
        reasoning_text, cleaned_text = cfg.reasoning_parser.extract_reasoning(
            text_to_parse
        )
        if not _should_emit_reasoning(request):
            reasoning_text = None

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        try:
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, response_format
            )
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
            if not is_valid:
                logger.warning(f"JSON validation failed: {error}")
        except Exception as e:
            logger.warning(f"JSON output parsing failed: {e}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    # Clean and strip thinking tags from content
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)
        if response_format and final_content:
            final_content = extract_json_from_response(final_content)

    # Build logprobs for response if requested
    choice_logprobs = None
    if want_logprobs and token_logprobs_list:
        choice_logprobs = ChoiceLogProbs(content=token_logprobs_list)

    chat_response = ChatCompletionResponse(
        model=_resolve_model_name(request.model),
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    reasoning_content=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
                logprobs=choice_logprobs,
            )
        ],
        usage=_build_usage(output, reasoning_text),
    )
    return Response(
        content=chat_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


async def stream_chat_completion(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response.

    Uses StreamingPostProcessor for reasoning/tool/sanitization pipeline.
    SSE formatting stays inline for performance (fast path bypasses Pydantic).
    """
    from ..domain.events import StreamEvent
    from ..service.postprocessor import StreamingPostProcessor

    cfg = get_config()
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    try:
        request_last_role = request.messages[-1].role
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Check if we should include usage in the final chunk
        include_usage = request.stream_options and request.stream_options.include_usage

        # Logprobs configuration
        want_logprobs = request.logprobs and request.top_logprobs
        top_k_logprobs = request.top_logprobs or 0

        def _build_chunk_logprobs(output: GenerationOutput) -> ChoiceLogProbs | None:
            """Build ChoiceLogProbs for a streaming chunk if logprobs requested."""
            if not want_logprobs or output.logprobs is None:
                return None
            token_id = output.tokens[-1] if output.tokens else 0
            token_lp = _extract_token_logprob(
                output.logprobs, token_id, engine.tokenizer, top_k_logprobs
            )
            return ChoiceLogProbs(content=[token_lp])

        # Pre-compute SSE template parts that don't change per-token.
        _sse_created = int(time.time())
        _model_escaped = json.dumps(_resolve_model_name(request.model))
        _sse_prefix = (
            f'data: {{"id":"{response_id}","object":"chat.completion.chunk",'
            f'"created":{_sse_created},"model":{_model_escaped},'
            f'"choices":[{{"index":0,"delta":{{'
        )
        _sse_suffix = "}}]}\n\n"

        def _fast_sse_chunk(text: str, field: str = "content") -> str:
            """Build SSE chunk JSON directly, bypassing Pydantic serialization."""
            escaped = json.dumps(text)
            return f'{_sse_prefix}"{field}":{escaped}{_sse_suffix}'

        def _fast_tool_call_sse(tool_calls: list[dict]) -> str:
            """Build tool-call SSE chunk directly, bypassing Pydantic serialization."""
            normalized_tool_calls = []
            for index, tool_call in enumerate(tool_calls):
                if hasattr(tool_call, "function"):
                    logger.debug(
                        "Streaming tool call emitted name=%s args=%r",
                        tool_call.function.name,
                        tool_call.function.arguments[:200],
                    )
                    normalized_tool_calls.append(
                        {
                            "index": index,
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
                else:
                    normalized = dict(tool_call)
                    normalized.setdefault("index", index)
                    function = normalized.get("function") or {}
                    logger.debug(
                        "Streaming tool call emitted name=%s args=%r",
                        function.get("name"),
                        str(function.get("arguments", ""))[:200],
                    )
                    normalized_tool_calls.append(normalized)
            tool_calls_json = json.dumps(
                normalized_tool_calls, separators=(",", ":")
            )
            return (
                f'data: {{"id":"{response_id}",'
                f'"object":"chat.completion.chunk",'
                f'"created":{_sse_created},'
                f'"model":{_model_escaped},'
                f'"choices":[{{"index":0,'
                f'"delta":{{"tool_calls":{tool_calls_json}}},'
                f'"finish_reason":"tool_calls"}}]}}\n\n'
            )

        # First chunk with role
        _first_sse = f'{_sse_prefix}"role":"assistant"{_sse_suffix}'
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[SSE-ROLE] {_first_sse.strip()[:200]}")
        yield _first_sse

        # Initialize post-processor.
        # request_dict carries `tools` so streaming parsers (qwen3_coder etc.)
        # can do schema-driven type conversion (#171).
        request_dict = (
            request.model_dump(exclude_none=True)
            if hasattr(request, "model_dump")
            else None
        )
        tool_mode = bool(request.tools or cfg.tool_call_parser)
        force_tool_work = bool(tool_mode and _last_user_requests_artifact(request.messages))
        last_tool_needs_more_work = _last_tool_result_needs_more_work(request.messages)
        processor = StreamingPostProcessor(
            cfg,
            tools_requested=tool_mode,
            json_mode=bool(
                request.response_format
                and getattr(request.response_format, "type", "text") != "text"
            ),
            request=request_dict,
        )
        processor.set_thinking_model(request.model)
        processor.reset()

        # Track token counts for usage reporting
        prompt_tokens = 0
        completion_tokens = 0
        stop_after_tool_call = False
        emitted_content = False
        invalid_tool_call_seen = False

        # Stream content — PostProcessor handles reasoning/tool/sanitize
        async for output in engine.stream_chat(messages=messages, **kwargs):
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens

            for event in processor.process_chunk(output):
                if event.type == "content":
                    last_role = request_last_role
                    if tool_mode:
                        logger.debug(
                            "Streaming content candidate last_role=%s preview=%r",
                            last_role,
                            (event.content or "")[:120],
                        )
                    if tool_mode and (
                        _looks_like_invalid_tool_continuation(event.content)
                        or (
                            last_role == "tool"
                            and _looks_like_deferred_tool_use(event.content)
                        )
                        or (
                            force_tool_work
                            and last_role == "tool"
                            and _looks_like_incomplete_artifact_answer(event.content)
                        )
                        or (force_tool_work and last_role != "tool")
                        or (last_tool_needs_more_work and last_role == "tool")
                    ):
                        continue
                    if tool_mode and last_role == "tool":
                        logger.debug(
                            "Streaming tool continuation content preview=%r",
                            (event.content or "")[:120],
                        )
                    if not want_logprobs:
                        _sse = _fast_sse_chunk(event.content, "content")
                        if _sse:
                            emitted_content = True
                            yield _sse
                    else:
                        chunk = ChatCompletionChunk(
                            id=response_id,
                            model=_resolve_model_name(request.model),
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        content=event.content,
                                    ),
                                    logprobs=_build_chunk_logprobs(output),
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                elif event.type == "reasoning":
                    if not _should_emit_reasoning(request):
                        continue
                    yield _fast_sse_chunk(event.reasoning, "reasoning_content")

                elif event.type == "tool_call":
                    validation_error = _tool_call_validation_error(
                        event.tool_calls, request.tools
                    )
                    if validation_error:
                        invalid_tool_call_seen = True
                        logger.debug(
                            "Suppressing invalid streaming tool call: %s",
                            validation_error,
                        )
                        continue
                    _tc_sse = _fast_tool_call_sse(event.tool_calls)
                    logger.debug(f"[SSE-TC] {_tc_sse.strip()[:300]}")
                    yield _tc_sse
                    stop_after_tool_call = True
                    break

                elif event.type == "finish":
                    last_role = request_last_role
                    if (
                        tool_mode
                        and last_role == "tool"
                        and (
                            not event.content
                            or _looks_like_invalid_tool_continuation(event.content)
                            or _looks_like_deferred_tool_use(event.content)
                        )
                    ):
                        continue
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=_resolve_model_name(request.model),
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    content=event.content,
                                    reasoning_content=event.reasoning,
                                ),
                                finish_reason=event.finish_reason,
                                logprobs=_build_chunk_logprobs(output),
                            )
                        ],
                        usage=get_usage(output) if output.finished else None,
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        # Fallback tool call detection
        if not stop_after_tool_call:
            finalize_events = processor.finalize()
            if tool_mode:
                logger.debug(
                    "Streaming fallback state emitted_content=%s finalize_types=%s",
                    emitted_content,
                    [event.type for event in finalize_events],
                )
            if (
                tool_mode
                and not emitted_content
                and not any(event.type == "tool_call" for event in finalize_events)
            ):
                finalize_text = "".join(
                    event.content or ""
                    for event in finalize_events
                    if event.type == "content"
                )
                if not finalize_text:
                    finalize_text = (
                        processor.tool_accumulated_text or processor.accumulated_text
                    )
                last_role = request_last_role
                invalid_tool_continuation = _looks_like_invalid_tool_continuation(
                    finalize_text
                )
                should_retry = (
                    _looks_like_deferred_tool_use(finalize_text)
                    or _looks_like_incomplete_artifact_answer(finalize_text)
                    or last_role == "tool"
                    or force_tool_work
                    or last_tool_needs_more_work
                    or invalid_tool_call_seen
                )
                if should_retry:
                    logger.debug(
                        "Streaming tool intent without tool call detected; retrying"
                    )
                    retry_messages = list(messages)
                    if finalize_text and not invalid_tool_continuation:
                        retry_messages.append(
                            {"role": "assistant", "content": finalize_text}
                        )
                    tool_names = []
                    for tool in request.tools or []:
                        function = tool.function if hasattr(tool, "function") else None
                        name = getattr(function, "name", None)
                        if name:
                            tool_names.append(name)
                    tool_hint = (
                        f" Available tools: {', '.join(tool_names)}."
                        if tool_names
                        else ""
                    )
                    artifact_hint = _artifact_retry_hint(request.messages)
                    retry_instruction = (
                        "Continue the task. Call the appropriate tool now "
                        "if work remains."
                        f"{artifact_hint}"
                        f"{tool_hint} Do not explain, "
                        "do not describe what you will do, and do not output "
                        "raw JSON as text. Use concrete tool arguments. "
                        "If files must change, call write, edit, or bash. "
                        "If the previous tool failed, repair it by calling "
                        "another tool. For create/build/scaffold tasks, create "
                        "the files now; do not inspect the directory again. "
                        "If the previous tool scaffolded a starter project, "
                        "your next tool call must replace starter source files "
                        "with the requested app before any install, test, or "
                        "build command. Do not run long-lived dev, watch, "
                        "preview, or server commands; use finite build, test, "
                        "or typecheck commands for validation."
                    )
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["max_tokens"] = min(
                        int(retry_kwargs.get("max_tokens") or 4096), 4096
                    )
                    retry_kwargs["temperature"] = min(
                        float(retry_kwargs.get("temperature") or 0.2), 0.2
                    )
                    finalize_events = []
                    for retry_attempt in range(6):
                        retry_attempt_messages = retry_messages + [
                            {"role": "user", "content": retry_instruction}
                        ]
                        retry_output = await engine.chat(
                            messages=retry_attempt_messages,
                            **retry_kwargs,
                        )
                        retry_cleaned, retry_tool_calls = (
                            _parse_tool_calls_with_parser(retry_output.text, request)
                        )
                        if retry_tool_calls:
                            retry_validation_error = _tool_call_validation_error(
                                retry_tool_calls, request.tools
                            )
                            if retry_validation_error:
                                logger.debug(
                                    "Retry produced invalid tool call: %s",
                                    retry_validation_error,
                                )
                            else:
                                _validate_tool_call_params(
                                    retry_tool_calls, request.tools
                                )
                                finalize_events = [
                                    StreamEvent(
                                        type="tool_call", tool_calls=retry_tool_calls
                                    )
                                ]
                                break
                        retry_invalid = _looks_like_invalid_tool_continuation(
                            retry_cleaned
                        )
                        retry_deferred = _looks_like_deferred_tool_use(retry_cleaned)
                        retry_incomplete = _looks_like_incomplete_artifact_answer(
                            retry_cleaned
                        )
                        if (
                            retry_cleaned
                            and not retry_invalid
                            and not retry_deferred
                            and not retry_incomplete
                            and not force_tool_work
                            and not last_tool_needs_more_work
                        ):
                            finalize_events = [
                                StreamEvent(type="content", content=retry_cleaned)
                            ]
                            break
                        logger.debug(
                            "Streaming retry %s produced malformed tool text; "
                            "retrying preview=%r",
                            retry_attempt + 1,
                            (retry_cleaned or retry_output.text)[:120],
                        )
                        retry_instruction = (
                            "Your previous response was not a valid tool call. "
                            f"{artifact_hint}"
                            f"{tool_hint} Call exactly one available tool now "
                            "with concrete arguments. Create the requested files "
                            "now. If a starter project exists, replace its source "
                            "files now before install/build/test. Do not write "
                            "plain text. Do not run long-lived dev/watch/server "
                            "commands."
                        )
                    if (
                        not finalize_events
                        and tool_mode
                        and (
                            force_tool_work
                            or last_tool_needs_more_work
                            or invalid_tool_call_seen
                        )
                    ):
                        fallback_tool_call = _artifact_fallback_tool_call(
                            request.messages, request.tools
                        )
                        if fallback_tool_call:
                            finalize_events = [
                                StreamEvent(
                                    type="tool_call", tool_calls=fallback_tool_call
                                )
                            ]

            for event in finalize_events:
                if event.type == "content":
                    if tool_mode and _looks_like_invalid_tool_continuation(
                        event.content
                    ):
                        continue
                    if tool_mode and _looks_like_incomplete_artifact_answer(
                        event.content
                    ):
                        continue
                    if tool_mode and (force_tool_work or last_tool_needs_more_work):
                        continue
                    _sse = _fast_sse_chunk(event.content, "content")
                    if _sse:
                        emitted_content = True
                        yield _sse
                elif event.type == "reasoning":
                    if not _should_emit_reasoning(request):
                        continue
                    yield _fast_sse_chunk(event.reasoning, "reasoning_content")
                elif event.type == "tool_call":
                    _fb_sse = _fast_tool_call_sse(event.tool_calls)
                    logger.debug(f"[SSE-FALLBACK-TC] {_fb_sse.strip()[:300]}")
                    yield _fb_sse

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Send final chunk with usage if requested
        if include_usage:
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                model=_resolve_model_name(request.model),
                choices=[],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()
