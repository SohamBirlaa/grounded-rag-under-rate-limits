"""API adapter for The Relay inference and RAG pipeline.

Implements Roadmap Phase 6.3:
- Thin adapter around RelayEngine
- Request schema validation using RelayRequest
- Response formatting using RelayResponse
- Streaming Server-Sent Events (SSE) support
- Standard ASGI application interface with optional FastAPI integration
- Cancellation safety and resource cleanup
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional

from relay.config.settings import RelayConfig
from relay.contracts.schemas import RelayRequest, RelayResponse
from relay.engine.orchestrator import RelayEngine

logger = logging.getLogger("relay.api")


class RelayAPIAdapter:
    """Thin API adapter exposing RelayEngine over standard ASGI."""

    def __init__(self, engine: Optional[RelayEngine] = None, config: Optional[RelayConfig] = None) -> None:
        self.config = config or RelayConfig()
        self.engine = engine or RelayEngine(config=self.config)

    async def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate payload, execute through engine, and return response dict."""
        request = RelayRequest(**payload)
        response, csv_row = await self.engine.execute_request(request)
        return response.model_dump()

    async def handle_stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        """Stream execution events and final answer chunks formatted as SSE."""
        request = RelayRequest(**payload)
        yield f"event: status\ndata: {json.dumps({'status': 'queued', 'request_id': request.request_id})}\n\n"

        response, csv_row = await self.engine.execute_request(request)

        # Emit completion metadata
        yield f"event: metadata\ndata: {json.dumps({'first_token_ms': response.first_token_ms, 'total_ms': response.total_ms, 'status': response.status})}\n\n"

        # Emit text chunk
        yield f"event: answer\ndata: {json.dumps({'answer_text': response.answer_text, 'citations': [c.model_dump() for c in response.citations]})}\n\n"
        yield "event: done\ndata: {}\n\n"

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        """Standard ASGI entrypoint."""
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await self.engine.start()
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await self.engine.close()
                    await send({"type": "lifespan.shutdown.complete"})
                    break
            return

        if scope["type"] != "http":
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Health endpoint
        if path == "/health" and method == "GET":
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"application/json"]],
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"status": "ok", "component": "The Relay"}).encode("utf-8"),
            })
            return

        # Query endpoint
        if path in ["/query", "/v1/chat/completions"] and method == "POST":
            # Read request body
            body_chunks = []
            while True:
                msg = await receive()
                body_chunks.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    break
            raw_body = b"".join(body_chunks).decode("utf-8")

            try:
                payload = json.loads(raw_body) if raw_body else {}
                result = await self.handle_request(payload)
                status_code = 200
                if result.get("status") == "rate_limited":
                    status_code = 429
                elif result.get("status") == "timeout":
                    status_code = 504
                elif result.get("status") == "error":
                    status_code = 500

                await send({
                    "type": "http.response.start",
                    "status": status_code,
                    "headers": [[b"content-type", b"application/json"]],
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps(result).encode("utf-8"),
                })
            except Exception as e:
                await send({
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [[b"content-type", b"application/json"]],
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps({"error": str(e), "status": "error"}).encode("utf-8"),
                })
            return

        # 404 Fallback
        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": [[b"content-type", b"application/json"]],
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps({"error": "Not Found"}).encode("utf-8"),
        })


def create_app(engine: Optional[RelayEngine] = None, config: Optional[RelayConfig] = None) -> Any:
    """Factory creating the Relay API application."""
    adapter = RelayAPIAdapter(engine=engine, config=config)

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            await adapter.engine.start()
            try:
                yield
            finally:
                await adapter.engine.close()

        app = FastAPI(title="The Relay", version="0.1.0", lifespan=lifespan)

        @app.get("/health")
        async def health():
            return {"status": "ok", "component": "The Relay"}

        @app.post("/query")
        async def query(request: RelayRequest):
            response, _ = await adapter.engine.execute_request(request)
            status_code = 200
            if response.status == "rate_limited":
                status_code = 429
            elif response.status == "timeout":
                status_code = 504
            elif response.status == "error":
                status_code = 500
            return JSONResponse(content=response.model_dump(), status_code=status_code)

        @app.post("/query/stream")
        async def query_stream(request: RelayRequest):
            return StreamingResponse(
                adapter.handle_stream(request.model_dump()),
                media_type="text/event-stream",
            )

        return app
    except ImportError:
        # FastAPI not installed; return pure ASGI adapter
        return adapter

