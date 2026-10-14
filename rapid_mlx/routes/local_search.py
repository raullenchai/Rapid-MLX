# SPDX-License-Identifier: Apache-2.0
"""Frozen local-search MVP wire contract; mounted only by the opt-in lifespan."""

import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..middleware.auth import check_rate_limit, verify_api_key
from ..middleware.exception_handlers import register_request_model


class FolderRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class ReindexRequest(BaseModel):
    folder_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8192)
    limit: int = Field(default=20, ge=1, le=100)
    group_limit: int | None = Field(default=None, ge=1, le=50)
    folder_ids: list[str] | None = Field(default=None, max_length=1000)
    kinds: list[Literal["text", "image", "video"]] | None = None


for model in (FolderRequest, ReindexRequest, SearchRequest):
    register_request_model(model)


def error(message: str, code: str, param: str, status=400):
    raise HTTPException(
        status,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
                "param": param,
            }
        },
    )


def get_service(request: Request):
    return request.app.state.local_search


async def require_store(request: Request):
    try:
        await get_service(request).store.ready()
    except Exception as exc:
        error(str(exc), "store_open_failed", "local_search_dir", 503)


router = APIRouter(
    prefix="/v1/local-search",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(require_store),
    ],
)


async def require_folder(service, folder_id):
    if await service.folder(folder_id) is None:
        error("Folder not found", "folder_not_found", "folder_id", 404)


@router.get("/status")
async def status(request: Request):
    return await get_service(request).status_async()


@router.post("/folders")
async def add_folder(body: FolderRequest, request: Request, response: Response):
    try:
        folder, created = await get_service(request).add_folder(body.path)
    except ValueError as exc:
        error(str(exc), "invalid_folder_path", "path")
    except OSError:
        error(
            "path must be an existing absolute directory without '..', outside excluded directories, other than /",
            "invalid_folder_path",
            "path",
        )
    response.status_code = 201 if created else 200
    return folder


@router.delete("/folders/{folder_id}", status_code=204)
async def delete_folder(folder_id: str, request: Request):
    service = get_service(request)
    await require_folder(service, folder_id)
    await service.delete_folder(folder_id)
    return Response(status_code=204)


@router.post("/reindex", status_code=202)
async def reindex(body: ReindexRequest, request: Request):
    service = get_service(request)
    if body.folder_id is not None:
        await require_folder(service, body.folder_id)
        await service.queue(body.folder_id)
    else:
        for folder in await service.folders():
            await service.queue(folder["id"])
    return await service.status_async()


@router.post("/search")
async def search(body: SearchRequest, request: Request):
    if not body.query.strip():
        error("query must not be blank", "invalid_query", "query")
    service = get_service(request)
    if body.folder_ids is not None:
        for folder_id in body.folder_ids:
            await require_folder(service, folder_id)
    try:
        return await service.search(
            body.query, body.limit, body.folder_ids, body.kinds, body.group_limit
        )
    except sqlite3.OperationalError:
        error("Query could not be processed", "invalid_query", "query")
