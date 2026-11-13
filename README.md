# The Relay

A grounded question-answering service over three arXiv papers, built and measured against the **real Groq free
tier** (`openai/gpt-oss-20b`). Every number in the report comes from real calls recorded in `results.csv`.

- Report: [`report/the_relay_report.md`](report/the_relay_report.md)
- Raw results (214 real requests): [`results.csv`](results.csv)
- Traces with every timing span, per request: `data/live/traces.jsonl` (and `data/live_v1/` for the first eval run)
- Real 429 captures and probe data: `data/calibration/live_429_*.json`, `data/calibration/live_probe_openai_gpt-oss-20b.json`
- Step-by-step runbook: [`Docs/LIVE_RUNBOOK.md`](Docs/LIVE_RUNBOOK.md)

## Setup

```bash
uv sync
cp .env.example .env        # then put your own GROQ_API_KEY in .env (never commit it)
uv run pytest tests/live -q # 33 unit tests, no network
```

PDFs are not committed: `uv run python scripts/fetch_corpus.py`. The search index used for the results is
committed in `data/corpus/indexes/` (rebuild with `uv run python scripts/build_indexes.py`).

## Reproduce

```bash
uv run python scripts/ask.py --show-timing "What frame rate does the Mimi codec in Moshi use?"
uv run python scripts/live_eval.py --retrieval-only                     # retrieval metrics, no API calls
uv run python scripts/live_probe.py --phases latency,ratio,concurrency,cache
uv run python scripts/live_eval.py --judge-model openai/gpt-oss-120b --resume
uv run python scripts/live_load.py --scenario all --scale 0.5
uv run python scripts/live_tables.py                                    # Tables 2a, 2b, 3 from results.csv
uv run python scripts/regrade.py --review data/live/eval_review_graded.csv  # judge vs my human grades
```

Free-tier limits (1,000 requests / 200K tokens per day on gpt-oss-20b) mean a full rerun takes about two days of quota.

## Code map

- `src/relay/live/`: the real-Groq pipeline: streaming client (`groq_client.py`), refilling-bucket admission control
  (`ratelimit.py`), retrieval with paper routing / filtering / re-rank (`retrieval.py`), context compression
  (`compress.py`), sentence-gated streaming verifier (`gate.py`), exact + semantic caches (`cache.py`),
  RAG evals (`evals.py`), pipeline (`pipeline.py`), table builder (`tables.py`).
- `src/relay/rag/`: PDF parsing, chunking and indexes.
- `src/relay/api/live_app.py`: SSE streaming endpoint (`uvicorn relay.api.live_app:app --app-dir src`).
- Earlier code from the first version remains in `src/relay/` (engine, providers, stand-in).
