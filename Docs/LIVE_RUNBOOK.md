# Live (real Groq) runbook

Every number this produces comes from real Groq responses. The only non-real values are in
unit tests (`tests/live/`), which check logic and never write results.

## 0. Setup (once)
1. **Revoke the key that was inside the zip** (console.groq.com -> API keys) and create a new one.
2. `uv sync` (adds `httpx`, already used by the Groq adapter).
3. `export GROQ_API_KEY=...` (or `.env`, which must NOT go in the submission).
4. Tests (no network): `uv run pytest tests/live -q`

## 1. Run order and quota plan
Groq shut down `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` for free tier on 16 Aug 2026 (404
`model_not_found`). Primary model is now `openai/gpt-oss-20b` (free: 30 RPM, 8K TPM, 1K req/day, 200K tok/day,
prompt caching supported). Judge: `openai/gpt-oss-120b` (separate pool). Confirm what your account can use:
`uv run python -c "import os,httpx;print([m['id'] for m in httpx.get('https://api.groq.com/openai/v1/models',headers={'Authorization':'Bearer '+os.environ['GROQ_API_KEY']}).json()['data']])"`

| Day | Step | Command | ~Requests | ~Tokens (20b pool) |
|---|---|---|---|---|
| 1 | a | `uv run python scripts/live_eval.py --retrieval-only` | 0 | 0 |
| 1 | b | `uv run python scripts/ask.py --show-timing "..."` | 1-2 | 2K |
| 1 | c | `uv run python scripts/live_probe.py --phases latency,ratio,concurrency,cache` | ~45 | ~45K |
| 1 | d | `uv run python scripts/live_probe.py --phases limit` | ~10-30 | ~25K |
| 1 | e | `uv run python scripts/live_eval.py --judge-model openai/gpt-oss-120b --resume --yes` | ~65 (+~40 judge on 120b) | ~90K |
| 2 | f | `uv run python scripts/live_load.py --scenario <each>` (7 scenarios) | ~90 model calls | ~130K |
| 2 | g | `cp data/live/results.csv results.csv; uv run python scripts/live_tables.py` | 0 | 0 |
| 2 | h | `uv run python scripts/calibrate_standin_from_live.py openai/gpt-oss-20b` | 0 | 0 |
| 3 (optional, LAST) | i | `uv run python scripts/live_probe.py --phases daily --yes` | up to ~1,000 tiny | ~10K |

Step i uses up the day's requests for gpt-oss-20b, so run it only after everything else. It gives the REAL
per-day 429 (body, headers, reset time) for Table 1 and sections 5 and 7.

## 2. Definitions to state in the report
* `first_token_ms` = first **verified** sentence the person sees, measured from request arrival (includes our
  queueing). `provider_first_token_ms` (extra column) = first raw Groq token. The gap is the cost of the gate.
* Latency percentiles: `status == ok` rows; n printed in each cell.
* `daily_budget` scenario uses an **app-level** daily cap of 5 requests to trigger the same code path that a real
  per-day 429 triggers (that path is also unit-tested against a real-format 429). Say so; don't call it Groq's limit.
* Rate-limited / shed / timed-out requests have an empty `answer_grade`. They are counted in Table 2b, not graded.
* Extra columns after the 12 required ones: model, path, queue_wait_ms, cached_tokens, provider_first_token_ms.

## 3. What each feature is and how to show it
| Feature | Where | Evidence it worked |
|---|---|---|
| Streaming RAG (sentence-gated) | `live/gate.py`, `pipeline._generate` | first_token_ms vs provider_first_token_ms; gate_problems in traces |
| Output length control | `prompts.length_policy` | max_tokens per row in traces; finish_reason == "length" rate |
| Prompt size reduction | `live/compress.py`, lean prompt, short `[E#]` tags | context_tokens_before -> after; R5 vs R6 in Table 3 |
| Parallel calls | per-paper retrieval fan-out (`retrieval.retrieve`), concurrent requests under `TokenBudget` | subqueries in traces; concurrency probe |
| Prompt caching | static-prefix layout; only gpt-oss models cache on Groq | `cached_tokens` column; probe phase `cache` |
| Semantic + exact cache | `live/cache.py` | `path` = semantic_cache / exact_cache in cache_repeat; base-vs-big must MISS |
| Retrieval optimisation | section repair, reference filter, paper routing, RRF pool, cross-encoder re-rank | retrieval_metrics.json, Table 3 |
| RAG evals | `live/evals.py`, `scripts/live_eval.py` | eval_summary.json, eval_review.csv (+ your human grades) |
| Limits & retries | `live/ratelimit.py`, `live/groq_client.py` | live_429_*.json, retries column, naive_retry scenario |

## 4. Before using `data/eval/questions_near_miss.json`
Each question is marked `needs_human_verification`. Open the PDF and confirm the paper really does not answer it
before adding it to the eval (`--questions`). Also rewrite any question in your own words: the brief asks for
questions you wrote.
