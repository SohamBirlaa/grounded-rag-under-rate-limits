"""HTTP streaming endpoint for the live pipeline (Server-Sent Events).

  uvicorn relay.api.live_app:app --app-dir src --port 8000
  curl -N -X POST localhost:8000/ask/stream -H 'content-type: application/json' \
       -d '{"question": "What frame rate does Mimi use?"}'

Events: `data: {"type":"text","text":"..."}` per verified sentence, then one
`data: {"type":"done", ...timing...}`.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from relay.live.groq_client import GroqClient
    from relay.live.pipeline import LivePipeline, PipelineConfig
    from relay.live.ratelimit import TokenBudget
    from relay.live.retrieval import LiveRetriever

    model = os.environ.get("RELAY_MODEL", "openai/gpt-oss-20b")
    client = GroqClient(capture_dir=ROOT / "data/calibration")
    await client.__aenter__()
    retr = LiveRetriever.from_index_cache(str(ROOT / "data/corpus/indexes"))
    retr.warmup()
    budgets = {model: TokenBudget(rpm=30, tpm=6000 if "llama-3.1" in model else 8000)}
    STATE["pipe"] = LivePipeline(client, retr, budgets, PipelineConfig(model=model))
    yield
    await client.__aexit__(None, None, None)


app = FastAPI(title="The Relay (live)", lifespan=lifespan)


class Ask(BaseModel):
    question: str
    setting: str = "R5_hybrid_route_rerank"


@app.post("/ask/stream")
async def ask_stream(body: Ask) -> StreamingResponse:
    q: asyncio.Queue = asyncio.Queue()

    async def sink(text: str) -> None:
        await q.put({"type": "text", "text": text})

    async def worker() -> None:
        rec = await STATE["pipe"].run(body.question, body.setting, f"api_{uuid.uuid4().hex[:8]}", "api", sink=sink)
        await q.put({"type": "done", "status": rec.status, "path": rec.path, "first_token_ms": rec.first_token_ms,
                     "total_ms": rec.total_ms, "tokens_in": rec.tokens_in, "tokens_out": rec.tokens_out,
                     "cited": rec.cited_ids})

    async def events():
        task = asyncio.create_task(worker())
        while True:
            ev = await q.get()
            yield f"data: {json.dumps(ev)}\n\n"
            if ev["type"] == "done":
                break
        await task

    return StreamingResponse(events(), media_type="text/event-stream")
