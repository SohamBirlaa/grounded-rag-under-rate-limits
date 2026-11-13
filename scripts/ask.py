"""Ask one question and watch the verified answer stream in (real Groq).

  python scripts/ask.py "What frame rate does Mimi use?"
  python scripts/ask.py --model openai/gpt-oss-20b --show-timing "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from _live_common import ROOT, build_pipeline, load_env


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--setting", default="R5_hybrid_route_rerank")
    ap.add_argument("--show-timing", action="store_true")
    a = ap.parse_args()
    load_env()
    from relay.live.groq_client import GroqClient

    async def sink(text: str):
        sys.stdout.write(text)
        sys.stdout.flush()

    async with GroqClient(capture_dir=ROOT / "data/calibration") as client:
        pipe = build_pipeline(client, a.model)
        rec = await pipe.run(a.question, a.setting, "ask_cli", "interactive", sink=sink)
    print("\n")
    if a.show_timing:
        keys = ["status", "path", "first_token_ms", "provider_first_token_ms", "total_ms", "queue_wait_ms",
                "retrieval_ms", "compress_ms", "generation_ms", "repair_ms", "retry_wait_ms", "groq_queue_ms",
                "groq_prompt_ms", "groq_completion_ms", "tokens_in", "tokens_out", "cached_tokens", "max_tokens",
                "context_tokens_before", "context_tokens_after", "routed_papers", "top_rerank_score", "gate_problems", "error", "attempts"]
        print(json.dumps({k: getattr(rec, k) for k in keys}, indent=1, default=str))


if __name__ == "__main__":
    asyncio.run(main())
