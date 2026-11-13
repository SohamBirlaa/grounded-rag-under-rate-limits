"""Recompute the brief's Tables 2a, 2b and 3 from results.csv (nothing else).

Definitions (state these in the report):
  * First token = first text the person sees, measured from request arrival (includes queueing).
  * Latency percentiles use status == ok rows only; n is printed next to each.
  * Succeeded / Rate-limited / Timed out % are over all rows of the scenario.
  * Retries per request = mean of `retries` over all rows.
  * Peak queue = max number of requests simultaneously in [start, start + queue_wait_ms).
  * Tokens per answer = mean(tokens_in + tokens_out) over ok rows that called the model (tokens > 0).
  * Cost per 1,000 answers = Groq on-demand list price (free tier is billed $0), cached input at 50%.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# $ per 1M tokens (input, output). Check groq.com/pricing before quoting.
PRICES: Dict[str, Tuple[float, float]] = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


def _pct(s: pd.Series, q: float) -> float:
    s = s.dropna()
    return float(np.percentile(s, q)) if len(s) else float("nan")


def _fmt(v: float) -> str:
    return "not measured" if v != v else f"{v:.0f}"


def peak_queue(df: pd.DataFrame) -> int:
    t = pd.to_datetime(df["start_time"], utc=True, format="ISO8601")
    starts = (t - t.min()).dt.total_seconds() * 1000.0
    waits = pd.to_numeric(df.get("queue_wait_ms", 0), errors="coerce").fillna(0.0)
    events = [(s, 1) for s, w in zip(starts, waits) if w > 1.0] + [(s + w, -1) for s, w in zip(starts, waits) if w > 1.0]
    cur = peak = 0
    for _, d in sorted(events, key=lambda e: (e[0], e[1])):
        cur += d
        peak = max(peak, cur)
    return peak


def cost_per_1000(df: pd.DataFrame) -> float:
    total = 0.0
    n = 0
    for _, r in df.iterrows():
        pin, pout = PRICES.get(str(r.get("model", "")), (float("nan"), float("nan")))
        cached = float(r.get("cached_tokens", 0) or 0)
        total += ((float(r["tokens_in"]) - cached) * pin + cached * pin * 0.5 + float(r["tokens_out"]) * pout) / 1e6
        n += 1
    return 1000.0 * total / n if n else float("nan")


def table_2(df: pd.DataFrame) -> Tuple[str, str]:
    a = ["| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | "
         "First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |", "|" + "---|" * 7]
    b = ["| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | "
         "Peak queue length | Tokens per answer | Cost per 1,000 answers |", "|" + "---|" * 8]
    for (scen, src), g in df.groupby(["scenario", "source"], sort=False):
        t = pd.to_datetime(g["start_time"], utc=True, format="ISO8601")
        dur = (t.max() - t.min()).total_seconds()
        rate = (len(g) - 1) / dur * 60.0 if dur > 0 else float("nan")
        ok = g[g["status"] == "ok"]
        ft, tt = pd.to_numeric(ok["first_token_ms"], errors="coerce"), pd.to_numeric(ok["total_ms"], errors="coerce")
        a.append(f"| {scen} | {src} | {len(g)} | {dur:.1f} | {rate:.1f} | "
                 f"{_fmt(_pct(ft, 50))} / {_fmt(_pct(ft, 95))} / {_fmt(_pct(ft, 99))} (n={ft.notna().sum()}) | "
                 f"{_fmt(_pct(tt, 50))} / {_fmt(_pct(tt, 95))} / {_fmt(_pct(tt, 99))} (n={tt.notna().sum()}) |")
        called = ok[(ok["tokens_in"] + ok["tokens_out"]) > 0]
        tpa = (called["tokens_in"] + called["tokens_out"]).mean() if len(called) else float("nan")
        b.append(f"| {scen} | {100 * (g.status == 'ok').mean():.1f} | {100 * (g.status == 'rate_limited').mean():.1f} | "
                 f"{100 * (g.status == 'timeout').mean():.1f} | {g['retries'].mean():.2f} | {peak_queue(g)} | "
                 f"{'not measured' if tpa != tpa else f'{tpa:.0f}'} | ${cost_per_1000(ok):.4f} |")
    return "\n".join(a), "\n".join(b)


def table_3(df: pd.DataFrame) -> str:
    e = df[df["scenario"] == "retrieval_eval"]
    out = ["| Setting | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | "
           "First token p50 / p95 (ms) | Total time p50 / p95 (ms) |", "|" + "---|" * 8]
    for s, g in e.groupby("retrieval_setting", sort=False):
        graded = g[g["answer_grade"].fillna("") != ""]
        n = len(graded)
        pc = lambda lab: 100.0 * (graded["answer_grade"] == lab).sum() / n if n else float("nan")
        ok = g[g["status"] == "ok"]
        ft, tt = pd.to_numeric(ok["first_token_ms"], errors="coerce"), pd.to_numeric(ok["total_ms"], errors="coerce")
        called = ok[(ok["tokens_in"] + ok["tokens_out"]) > 0]
        out.append(f"| {s} | {len(g)} (graded {n}) | {pc('correct'):.1f} | {pc('wrong'):.1f} | "
                   f"{pc('refused_correctly'):.1f} | {(called.tokens_in + called.tokens_out).mean():.0f} | "
                   f"{_fmt(_pct(ft, 50))} / {_fmt(_pct(ft, 95))} | {_fmt(_pct(tt, 50))} / {_fmt(_pct(tt, 95))} |")
    return "\n".join(out)


def main(path: str = "results.csv") -> str:
    df = pd.read_csv(path)
    for col in ("tokens_in", "tokens_out", "retries"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    load = df[df["scenario"] != "retrieval_eval"]
    a, b = table_2(load) if len(load) else ("(no load rows)", "(no load rows)")
    return f"### Table 2a\n{a}\n\n### Table 2b\n{b}\n\n### Table 3\n{table_3(df)}\n"
