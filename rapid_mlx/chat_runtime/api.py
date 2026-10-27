# SPDX-License-Identifier: Apache-2.0
"""Authenticated chat catalog and turn API, independent of inference serving."""

from __future__ import annotations

import asyncio
import json
import secrets

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from rapid_mlx.middleware.body_depth import _replay_buffered, _replay_with_disconnect

from .events import Envelope, PermissionResponse, json_schema
from .file_extraction import MAX_INPUT_BYTES
from .file_worker import run_worker
from .owner import CAPABILITIES, ClientLeases
from .runtime import Runtime, TurnRequest
from .store import Conflict, Store

PREFIX = "/v1/rapid/chat"
MAX_ATTACHMENT_BYTES = MAX_INPUT_BYTES
MAX_REQUEST_BYTES = 2 * MAX_ATTACHMENT_BYTES
BODY_READ_TIMEOUT = 30


def _authorized(authorization: str, token: str) -> bool:
    return secrets.compare_digest(authorization.encode(), f"Bearer {token}".encode())


class ChatBodyLimitMiddleware:
    """Bound authenticated body bytes before JSON parsing, including chunked uploads."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        authorization = Request(scope).headers.get("authorization", "")
        if not _authorized(authorization, self.token):
            return await JSONResponse(
                {"detail": "Bearer authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
        body = bytearray()

        async def read_body() -> str:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return "disconnect"
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                    return "too_large"
                body.extend(chunk)
                if not message.get("more_body", False):
                    return "complete"

        try:
            outcome = await asyncio.wait_for(read_body(), BODY_READ_TIMEOUT)
        except asyncio.TimeoutError:
            return await JSONResponse(
                {"detail": "Request body timeout"},
                status_code=408,
                headers={"Connection": "close"},
            )(scope, receive, send)
        if outcome == "disconnect":
            return await self.app(scope, _replay_with_disconnect([bytes(body)]), send)
        if outcome == "too_large":
            return await JSONResponse(
                {"detail": "Request too large"},
                status_code=413,
                headers={"Connection": "close"},
            )(scope, receive, send)
        # Share the server's replay helper so streaming response readers see real
        # disconnects after the buffered request, never a synthetic early EOF.
        return await self.app(scope, _replay_buffered(bytes(body), receive), send)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationInput(Input):
    title: str = Field(default="New chat", max_length=200)
    workspace: str | None = Field(default=None, max_length=4096)


class UndoInput(Input):
    operation_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")


class CompactInput(Input):
    request: TurnRequest
    instructions: str = Field(default="", max_length=8192)


class LeaseInput(Input):
    client_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")


class ForkInput(Input):
    branch_id: str
    operation_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")


class EditInput(ForkInput):
    message_id: str
    text: str = Field(min_length=1, max_length=262144)


def _cursor(turn_id: str, value: str | None) -> int:
    if not value:
        return 0
    prefix, separator, sequence = value.rpartition(":")
    if not separator or prefix != turn_id or not sequence.isdigit():
        raise HTTPException(400, "Invalid Last-Event-ID for this turn")
    return int(sequence)


def create_app(
    store: Store,
    runtime: Runtime,
    token: str,
    *,
    metadata: dict | None = None,
    leases: ClientLeases | None = None,
) -> FastAPI:
    if not token:
        raise ValueError("Chat API requires a bearer credential")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(ChatBodyLimitMiddleware, token=token)
    metadata = metadata or {}
    leases = leases or ClientLeases()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        authorization = request.headers.get("authorization", "")
        if not _authorized(authorization, token):
            return JSONResponse(
                {"detail": "Bearer authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Never include the request body in validation/error responses: it can
        # contain an inference credential or private conversation material.
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_REQUEST_BYTES):
            return JSONResponse({"detail": "Request too large"}, status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"detail": "Invalid chat request"}, status_code=422)

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse(
            {
                "detail": "Chat state changed or operation ID was reused; refresh and retry"
            },
            status_code=409,
        )

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({"detail": "Chat item not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid_value(request, error):
        return JSONResponse({"detail": "Invalid chat operation"}, status_code=400)

    @app.get(PREFIX + "/capabilities")
    async def capabilities():
        return {
            "wire_version": 1,
            "schema_version": store.schema_version,
            "store_uuid": store.store_uuid,
            "owner_epoch": metadata.get("owner_epoch"),
            "capabilities": CAPABILITIES,
        }

    @app.get(PREFIX + "/schema")
    async def schema():
        return json_schema()

    @app.post(PREFIX + "/clients")
    async def attach(body: LeaseInput):
        leases.attach(body.client_id)
        return {"client_id": body.client_id, "lease_seconds": leases.ttl}

    @app.delete(PREFIX + "/clients/{client_id}", status_code=204)
    async def detach(client_id: str):
        leases.detach(client_id)
        return Response(status_code=204)

    @app.get(PREFIX + "/conversations")
    async def conversations():
        return await store.list_conversations()

    @app.post(PREFIX + "/conversations", status_code=201)
    async def create_conversation(body: ConversationInput):
        return await store.create_conversation(body.title, body.workspace)

    @app.get(PREFIX + "/conversations/{conversation_id}")
    async def conversation(conversation_id: str, branch_id: str | None = None):
        return await store.get_conversation(conversation_id, branch_id)

    @app.delete(PREFIX + "/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str):
        await store.delete_conversation(conversation_id)
        return Response(status_code=204)

    async def check_branch(conversation_id, branch_id):
        await store.get_conversation(conversation_id, branch_id)

    @app.post(PREFIX + "/conversations/{conversation_id}/context")
    async def context(conversation_id: str, body: TurnRequest):
        if body.conversation_id != conversation_id:
            raise ValueError("Conversation does not match request")
        await check_branch(conversation_id, body.branch_id)
        return await runtime.context_info(body)

    @app.post(PREFIX + "/conversations/{conversation_id}/compact")
    async def compact(conversation_id: str, body: CompactInput):
        if body.request.conversation_id != conversation_id:
            raise ValueError("Conversation does not match request")
        await check_branch(conversation_id, body.request.branch_id)
        return await runtime.compact(body.request, instructions=body.instructions)

    @app.post(PREFIX + "/conversations/{conversation_id}/edit")
    async def edit(conversation_id: str, body: EditInput):
        await check_branch(conversation_id, body.branch_id)
        branch = await store.edit(
            body.branch_id, body.message_id, body.text, body.operation_id
        )
        return {**branch, "branch_id": branch["id"]}

    @app.post(PREFIX + "/conversations/{conversation_id}/regenerate")
    async def regenerate(conversation_id: str, body: ForkInput):
        await check_branch(conversation_id, body.branch_id)
        branch = await store.regenerate(body.branch_id, body.operation_id)
        return {**branch, "branch_id": branch["id"]}

    @app.post(PREFIX + "/conversations/{conversation_id}/attachments", status_code=201)
    async def upload(
        conversation_id: str, request: Request, name: str = "attachment.txt"
    ):
        await store.get_conversation(conversation_id)
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > MAX_ATTACHMENT_BYTES:
                raise HTTPException(413, "Attachments are limited to 20 MiB")
        if not name or len(name) > 255:
            raise HTTPException(400, "Invalid attachment name")
        result = await run_worker(
            "rapid_mlx.chat_runtime.file_extraction", bytes(content), arguments=(name,)
        )
        if "error" in result:
            error = result["error"]
            code = error["code"]
            status = {
                "binary_unsupported": 415,
                "image_unsupported": 415,
                "document_unsupported": 415,
                "file_too_large": 413,
                "timeout": 504,
                "worker_failed": 500,
                "document_invalid": 422,
            }.get(code, 422)
            raise HTTPException(status, error)
        text = result["text"]
        return await store.put_attachment(conversation_id, name, text.encode("utf-8"))

    async def stream(turn_id: str, after: int):
        await store.get_turn(turn_id)
        # A cursor beyond durable history cannot silently swallow a terminal.
        saved = await store.replay_events(turn_id)
        if after > (saved[-1]["seq"] if saved else 0):
            raise HTTPException(
                409, "Replay cursor is ahead of durable history; refetch conversation"
            )

        async def generate():
            async for event in runtime.events(turn_id, after):
                if event is None:
                    yield b": keepalive\n\n"
                else:
                    yield Envelope.model_validate(event).sse()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post(PREFIX + "/turns")
    async def submit(
        body: TurnRequest, last_event_id: str | None = Header(default=None)
    ):
        after = _cursor(body.turn_id, last_event_id)
        await runtime.submit(body)
        return await stream(body.turn_id, after)

    @app.get(PREFIX + "/turns/{turn_id}")
    async def get_turn(turn_id: str):
        return await store.get_turn(turn_id)

    @app.get(PREFIX + "/turns/{turn_id}/events")
    async def events(turn_id: str, last_event_id: str | None = Header(default=None)):
        return await stream(turn_id, _cursor(turn_id, last_event_id))

    @app.get(PREFIX + "/turns/{turn_id}/subruns")
    async def child_runs(turn_id: str):
        return await store.list_subruns(turn_id)

    @app.get(PREFIX + "/subruns/{subrun_id}")
    async def child_run(subrun_id: str):
        return await store.get_subrun(subrun_id)

    @app.post(PREFIX + "/turns/{turn_id}/cancel")
    async def cancel(turn_id: str):
        turn = await runtime.cancel(turn_id)
        return {"turn_id": turn_id, "outcome": turn["status"]}

    @app.post(PREFIX + "/permission.response")
    async def permission_response(body: PermissionResponse):
        return await runtime.actions.respond(
            body.op_id, body.revision, body.decision, body.scope, body.reason
        )

    @app.get(PREFIX + "/operations/{op_id}")
    async def operation(op_id: str):
        return await store.get_operation(op_id)

    @app.get(PREFIX + "/conversations/{conversation_id}/operations")
    async def operations(conversation_id: str):
        return await store.list_operations(conversation_id)

    @app.post(PREFIX + "/conversations/{conversation_id}/undo")
    async def undo(conversation_id: str, body: UndoInput):
        return await runtime.actions.undo(conversation_id, body.operation_id)

    @app.get(PREFIX + "/changes")
    async def changes(request: Request, after: int = 0):
        """Revision invalidations only: clients refetch their selected snapshot."""

        async def generate():
            cursor = after
            while not await request.is_disconnected():
                revision = await store.catalog_revision()
                if revision != cursor:
                    cursor = revision
                    yield f"id: {revision}\nevent: catalog.changed\ndata: {json.dumps({'revision': revision})}\n\n"
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app
