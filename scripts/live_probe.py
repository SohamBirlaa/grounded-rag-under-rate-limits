"""Measure the REAL Groq service before load testing. Every number here comes from real responses.

Phases (pick with --phases, default "latency,ratio,concurrency"):
  latency      N sequential requests at 3 prompt sizes, paced under TPM -> TTFT/total distribution,
               Groq server-side queue/prompt/completion time, inter-token time.
  ratio        real prompt_tokens vs our local estimate -> token_ratio for the pipeline's budget.
  concurrency  bursts of 1,2,4,8 tiny parallel requests -> does latency inflate with requests in flight?
  limit        OPT-IN: rapid tiny requests until the first real 429; saves body+headers; then measures
               how long until a request succeeds again (time to recover).
  cache        gpt-oss only: same ~1.5K-token static prefix twice -> cached_tokens and whether
               x-ratelimit-remaining-tokens drops by cached tokens (do cached tokens count?).

Writes data/calibration/live_probe_<model>.json (+ live_429_*.json on 429).
Approx cost (defaults): ~45 requests, ~45K tokens on openai/gpt-oss-20b (1K req / 200K tokens per day).
Phase `daily` (OPT-IN, run LAST): ~1-token requests at ~28/min until the real per-day 429 (up to ~1,000
requests, ~35 min). It uses up the day for that model, but gives the real daily-limit body/headers/reset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from _live_common import ROOT, load_env

FILLER = ("The Transformer uses stacked self-attention and point-wise fully connected layers for both the "
          "encoder and decoder. ")


def msgs(n_words: int, q: str = "Reply with the single word OK."):
    ctx = (FILLER * (n_words // 15 + 1)).split()[:n_words]
    return [{"role": "system", "content": "You are terse."},
            {"role": "user", "content": " ".join(ctx) + "\n\n" + q}]


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


async def one(client, model, m, max_tokens, budget=None):
    r = await client.stream_chat(model, m, max_tokens, budget=budget)
    u = r.usage or {}
    gen_ms = (r.t_done - r.t_first_content) * 1000 if (r.t_done and r.t_first_content) else None
    return {
        "ok": r.ok, "error": r.error_kind, "status": r.status_code, "retries": r.retries,
        "ttft_ms": r.ms(r.t_first_content), "headers_ms": r.ms(r.t_headers), "total_ms": r.ms(r.t_done),
        "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
        "cached_tokens": u.get("cached_tokens"), "groq_queue_ms": u.get("queue_time_ms"),
        "groq_prompt_ms": u.get("prompt_time_ms"), "groq_completion_ms": u.get("completion_time_ms"),
        "itl_ms": (gen_ms / max(1, u.get("completion_tokens", 1) - 1)) if gen_ms and u.get("completion_tokens") else None,
        "headers": r.headers, "wall": time.time(),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--phases", default="latency,ratio,concurrency")
    ap.add_argument("--n", type=int, default=24, help="latency requests (split over 3 prompt sizes)")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    a = ap.parse_args()
    load_env()
    phases = set(a.phases.split(","))
    print(f"Probe {a.model}: phases={sorted(phases)}. This makes REAL calls on your free quota.")
    if not a.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return

    from _live_common import make_budget
    from relay.live.compress import tok_count
    from relay.live.groq_client import GroqClient

    out = {"model": a.model, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phases": {}}
    budget = make_budget(a.model)
    async with GroqClient(capture_dir=ROOT / "data/calibration", retry_policy="none") as c:
        if "latency" in phases or "ratio" in phases:
            rows = []
            sizes = [60, 600, 1500]  # words of context: small / typical RAG / large RAG
            for i in range(a.n):
                n_words = sizes[i % 3]
                m = msgs(n_words, "In one sentence, what does the text say the encoder and decoder use?")
                est = int(sum(tok_count(x["content"]) for x in m) * 1.15) + 80
                ok, waited, entry = await budget.acquire(est, time.monotonic() + 120)  # pace under TPM
                r = await one(c, a.model, m, 250, budget)
                r.update(size_words=n_words, local_tokens=sum(tok_count(x["content"]) for x in m), paced_wait_s=waited)
                if entry is not None:
                    budget.settle(entry, (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0))
                rows.append(r)
                print(f"  latency {i+1}/{a.n} size={n_words:4d}w ttft={r['ttft_ms'] or 0:7.1f}ms "
                      f"total={r['total_ms'] or 0:7.1f}ms groq_queue={r['groq_queue_ms']} err={r['error']}")
            okr = [r for r in rows if r["ok"]]
            summary = {}
            for n_words in sizes:
                g = [r for r in okr if r["size_words"] == n_words]
                if g:
                    summary[f"{n_words}w"] = {
                        "n": len(g), "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in g),
                        **{f"ttft_p{q}": pct([r["ttft_ms"] for r in g], q) for q in (50, 95)},
                        **{f"total_p{q}": pct([r["total_ms"] for r in g], q) for q in (50, 95)},
                        "groq_queue_p50": pct([r["groq_queue_ms"] for r in g], 50),
                        "groq_prompt_p50": pct([r["groq_prompt_ms"] for r in g], 50),
                        "itl_p50_ms": pct([r["itl_ms"] for r in g], 50),
                        "network_overhead_p50_ms": pct([(r["ttft_ms"] or 0) - (r["groq_queue_ms"] or 0)
                                                        - (r["groq_prompt_ms"] or 0) for r in g], 50),
                    }
            ratios = [r["prompt_tokens"] / r["local_tokens"] for r in okr if r["prompt_tokens"] and r["local_tokens"]]
            out["phases"]["latency"] = {"rows": rows, "by_size": summary}
            out["phases"]["ratio"] = {"token_ratio_mean": statistics.mean(ratios) if ratios else None,
                                      "token_ratio_max": max(ratios) if ratios else None, "n": len(ratios)}
            print("  by size:", json.dumps(summary, indent=1))
            print("  token ratio (real/local):", out["phases"]["ratio"])

        if "concurrency" in phases:
            conc = {}
            for k in (1, 2, 4, 8):
                await asyncio.sleep(12)  # let the minute window drain a little between bursts
                rs = await asyncio.gather(*[one(c, a.model, msgs(40), 150) for _ in range(k)])
                okr = [r for r in rs if r["ok"]]
                conc[k] = {"sent": k, "ok": len(okr), "errors": [r["error"] for r in rs if not r["ok"]],
                           "ttft_p50": pct([r["ttft_ms"] for r in okr], 50), "ttft_max": pct([r["ttft_ms"] for r in okr], 100),
                           "groq_queue_p50": pct([r["groq_queue_ms"] for r in okr], 50)}
                print(f"  concurrency {k}: {conc[k]}")
            out["phases"]["concurrency"] = conc

        if "cache" in phases:
            prefix = "You answer from this reference document.\n" + ("Reference: " + FILLER * 60)  # ~1.2K tokens
            rs = []
            for q in ("Say OK.", "Say YES.", "Say NO.", "Say MAYBE.", "Say DONE."):
                m = [{"role": "system", "content": prefix}, {"role": "user", "content": q}]
                ok, _w, entry = await budget.acquire(1500, time.monotonic() + 120)  # pace: no 429s here
                r = await one(c, a.model, m, 150, budget)
                if entry is not None:
                    budget.settle(entry, (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0))
                rs.append(r)
            out["phases"]["cache"] = [{k: r[k] for k in ("prompt_tokens", "cached_tokens", "ttft_ms", "groq_prompt_ms")}
                                      | {"tpm_remaining": (r["headers"] or {}).get("tpm_remaining")} for r in rs]
            print("  cache:", json.dumps(out["phases"]["cache"], indent=1))
            print("  -> if cached_tokens > 0 and tpm_remaining fell by ~(prompt-cached), cached tokens are not counted.")

        if "limit" in phases:
            print("  limit: sending tiny requests as fast as allowed until the first 429 (max 60)...")
            first_429, t_start = None, time.perf_counter()
            sent = 0
            while sent < 60 and first_429 is None:
                rs = await asyncio.gather(*[one(c, a.model, msgs(1500, "Say OK."), 1) for _ in range(5)])
                sent += 5
                for r in rs:
                    if r["status"] == 429:
                        first_429 = r
                        break
            rec = {"sent_before_429": sent, "first_429": first_429, "captures": c.captured_429}
            if first_429:
                t429 = time.perf_counter()
                for _ in range(90):  # poll once per second with a 1-token request
                    await asyncio.sleep(1.0)
                    r = await one(c, a.model, msgs(5, "Say OK."), 1)
                    if r["ok"]:
                        rec["recovered_after_s"] = time.perf_counter() - t429
                        rec["header_retry_after_s"] = (first_429["headers"] or {}).get("retry_after_s")
                        break
            out["phases"]["limit"] = rec
            print("  limit:", json.dumps({k: v for k, v in rec.items() if k != "first_429"}, indent=1))

        if "daily" in phases:
            print("  daily: tiny requests at ~28/min until Groq returns a per-day 429 ...")
            n, t0, last = 0, time.perf_counter(), None
            while n < 1200:
                r = await one(c, a.model, [{"role": "user", "content": "Hi"}], 1)
                n += 1
                if r["status"] == 429:
                    from relay.live.ratelimit import classify_429
                    last = r
                    if c.captured_429:
                        body = json.loads(Path(c.captured_429[-1]).read_text()).get("body", {})
                        msg = body.get("error", {}).get("message", "") if isinstance(body, dict) else str(body)
                        if classify_429(msg)[0] == "day":
                            break
                if n % 50 == 0:
                    print(f"    {n} sent, rpd_remaining={(r['headers'] or {}).get('rpd_remaining')}")
                await asyncio.sleep(60 / 28)
            out["phases"]["daily"] = {"requests_sent": n, "minutes": (time.perf_counter() - t0) / 60,
                                      "last_429": last, "captures": c.captured_429[-3:]}
            print("  daily:", json.dumps({k: v for k, v in out["phases"]["daily"].items() if k != "last_429"}, indent=1))

    path = ROOT / "data/calibration" / f"live_probe_{a.model.replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # merge: keep phases from earlier runs, keep earlier copies of re-run phases
        old = json.loads(path.read_text(encoding="utf-8"))
        for k, v in out["phases"].items():
            if k in old.get("phases", {}):
                old["phases"][f"{k}_previous_{old.get('started_utc', 'run')}"] = old["phases"][k]
            old.setdefault("phases", {})[k] = v
        old.setdefault("runs", []).append(out["started_utc"])
        out = old
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    asyncio.run(main())
