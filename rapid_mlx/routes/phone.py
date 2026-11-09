# SPDX-License-Identifier: Apache-2.0
"""Public, credential-free phone bootstrap; inference retains bearer auth."""

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/connect", response_class=HTMLResponse, include_in_schema=False)
async def phone_page():
    return HTMLResponse(
        files("rapid_mlx").joinpath("phone.html").read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; connect-src 'self'; img-src data:; "
            "manifest-src data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        },
    )
