"""Shared setup for live scripts. Reads GROQ_API_KEY from env or .env (never printed)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Free-tier limits by model (Groq docs snapshot, Sept 2026). The probe confirms them from headers.
FREE_LIMITS = {
    "llama-3.1-8b-instant": dict(rpm=30, tpm=6000, rpd=14400),
    "openai/gpt-oss-20b": dict(rpm=30, tpm=8000, rpd=1000),
    "openai/gpt-oss-120b": dict(rpm=30, tpm=8000, rpd=1000),
    "llama-3.3-70b-versatile": dict(rpm=30, tpm=12000, rpd=1000),
}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists() and not os.environ.get("GROQ_API_KEY"):
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GROQ_API_KEY="):
                os.environ["GROQ_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY (env var or .env). Do not ship .env in the submission.")


def make_budget(model: str, scale: float = 1.0, local_daily_cap=None, safety: float = 0.9):
    from relay.live.ratelimit import TokenBudget

    lim = FREE_LIMITS.get(model, dict(rpm=30, tpm=6000, rpd=1000))
    return TokenBudget(rpm=max(1, int(lim["rpm"] * scale)), tpm=max(500, int(lim["tpm"] * scale)),
                       rpd=lim["rpd"], safety=safety, local_daily_request_cap=local_daily_cap)


def load_retriever(with_legacy: bool = False):
    from relay.live.retrieval import LegacySettingAdapter, LiveRetriever

    retr = LiveRetriever.from_index_cache(str(ROOT / "data/corpus/indexes"))
    legacy = None
    if with_legacy:
        from relay.rag.retriever import Retriever

        legacy = LegacySettingAdapter(Retriever.from_cache(ROOT / "data/corpus/indexes"), retr)
    return retr, legacy


def build_pipeline(client, model: str, fallback=None, budget_scale: float = 1.0, local_daily_cap=None,
                   with_legacy: bool = False, retriever=None, legacy=None, **cfg_kw):
    from relay.live.pipeline import LivePipeline, PipelineConfig

    if retriever is None:
        retriever, legacy = load_retriever(with_legacy)
    retr = retriever
    budgets = {model: make_budget(model, budget_scale, local_daily_cap)}
    if fallback:
        budgets[fallback] = make_budget(fallback, budget_scale)
    cfg = PipelineConfig(model=model, fallback_model=fallback, **cfg_kw)
    return LivePipeline(client, retr, budgets, cfg, legacy=legacy)


def load_questions(path: Path = ROOT / "data/eval/questions.json"):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))
