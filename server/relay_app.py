from __future__ import annotations

import asyncio
import hmac
import json
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response

if __package__:
    from .llm_relay import (
        RelayConfig,
        RelayValidationError,
        forward_response_payload,
        validate_response_payload,
    )
else:
    from llm_relay import (
        RelayConfig,
        RelayValidationError,
        forward_response_payload,
        validate_response_payload,
    )


LOGGER = logging.getLogger("cbt_simulator_relay")
RELAY_CONFIG = RelayConfig.from_environment()

app = FastAPI(
    title="CBT Patient Simulator GPT Relay",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "cbt-simulator-relay",
        "configured": RELAY_CONFIG.configured,
        "model": RELAY_CONFIG.model,
    }


@app.post("/api/responses")
async def responses(request: Request, authorization: str | None = Header(default=None)) -> Response:
    if not RELAY_CONFIG.configured:
        raise HTTPException(status_code=503, detail="The simulator relay is not configured.")
    supplied_token = ""
    if authorization and authorization.startswith("Bearer "):
        supplied_token = authorization.removeprefix("Bearer ").strip()
    if not supplied_token or not hmac.compare_digest(supplied_token, RELAY_CONFIG.token):
        raise HTTPException(status_code=401, detail="Relay authentication failed.")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > RELAY_CONFIG.max_request_bytes:
                raise HTTPException(status_code=413, detail="The relay request is too large.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > RELAY_CONFIG.max_request_bytes:
            raise HTTPException(status_code=413, detail="The relay request is too large.")
    try:
        payload = json.loads(body)
        forwarded = validate_response_payload(payload, RELAY_CONFIG)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": {"message": "The request body must be valid JSON."}}, status_code=400)
    except RelayValidationError as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=422)

    try:
        upstream = await asyncio.to_thread(forward_response_payload, forwarded, RELAY_CONFIG)
    except Exception as exc:
        LOGGER.exception("Simulator relay upstream request failed: %s", type(exc).__name__)
        return JSONResponse(
            {"error": {"message": "The model provider could not complete this request."}},
            status_code=502,
        )
    return Response(
        content=upstream.body,
        status_code=upstream.status_code,
        media_type=upstream.content_type,
    )
