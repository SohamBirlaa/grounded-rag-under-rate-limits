"""RAG evaluation.

Retrieval metrics (free: no API calls), per question and setting:
  hit@k            any retrieved chunk is from the target paper within ±1 page of the gold page
  mrr              1/rank of the first such chunk
  paper_precision  share of retrieved chunks from the target paper
  fact_in_context  share of gold key facts present in the retrieved text
  fact_retention   share of those facts still present AFTER compression (measures compression loss)

Answer metrics (from real Groq answers):
  fact_recall         share of gold key facts in the answer (normalised matching)
  citation_valid      every citation points at a retrieved chunk (gate guarantees it)
  cited_target_paper  at least one citation is from the target paper
  refusal             answer is a refusal / abstention
  faithfulness        gate verdict: 1 - rejected_sentences / (emitted + rejected)
  judge_*             optional LLM judge on a DIFFERENT Groq model (e.g. gpt-oss-120b) (separate rate-limit pool)

Final grade (brief's 4 labels):
  unanswerable: refusal -> refused_correctly, otherwise -> wrong
  answerable:   refusal -> refused_wrongly; fact_recall >= 2/3 and cited_target_paper (and judge agrees
                when used) -> correct; otherwise -> wrong
  Requests with no answer because of a rate limit / timeout are left ungraded (blank).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

REFUSAL_MARKERS = ("papers don't answer", "papers do not answer", "do not appear to cover", "won't guess",
                   "not in the papers", "does not contain", "do not contain", "not_in_papers",
                   "couldn't produce an answer i can verify")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.replace("−", "-").replace("–", "-").replace("‐", "-").replace("‑", "-").replace("×", "x")
    s = re.sub(r"\s*([=:/\-])\s*", r"\1", s)
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"(\d)\s+(hz|khz|ms|m|b|k)\b", r"\1\2", s)
    return " ".join(s.split())


def fact_present(fact: str, text: str) -> bool:
    f, t = norm_text(fact), norm_text(text)
    if not f:
        return False
    if re.fullmatch(r"[\d\.]+", f):  # bare number: require token boundary so "512" doesn't match "5120"
        return re.search(rf"(?<![\d\.]){re.escape(f)}(?![\d])", t) is not None
    return f in t


def is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in REFUSAL_MARKERS)


# ---------------------------------------------------------------- retrieval metrics
def retrieval_metrics(q: Dict[str, Any], chunks: Sequence[Any], compressed_text: Optional[str] = None) -> Dict[str, Any]:
    target, page = q.get("target_paper_id"), q.get("target_page")
    facts = [] if q.get("is_unanswerable") else list(q.get("key_facts") or [])
    rank = None
    for i, c in enumerate(chunks, 1):
        if target and c.paper_id == target and (page is None or abs(int(c.page) - int(page)) <= 1):
            rank = i
            break
    ctx = " ".join(c.text or "" for c in chunks)
    in_ctx = [f for f in facts if fact_present(f, ctx)]
    out = {
        "hit_at_k": None if not target else int(rank is not None),
        "mrr": None if not target else (1.0 / rank if rank else 0.0),
        "paper_precision": None if not target else sum(c.paper_id == target for c in chunks) / max(1, len(chunks)),
        "fact_in_context": None if not facts else len(in_ctx) / len(facts),
    }
    if compressed_text is not None and facts:
        out["fact_retention"] = (sum(fact_present(f, compressed_text) for f in in_ctx) / len(in_ctx)) if in_ctx else None
    return out


# ---------------------------------------------------------------- answer metrics
@dataclass
class AnswerEval:
    fact_recall: Optional[float]
    cited_target_paper: Optional[bool]
    refusal: bool
    faithfulness: Optional[float]
    judge_correct: Optional[bool]
    judge_faithful: Optional[bool]
    grade: str


def grade_answer(q: Dict[str, Any], answer: str, cited_chunk_ids: Sequence[str], status: str,
                 gate_emitted: int = 0, gate_rejected: int = 0,
                 judge: Optional[Dict[str, Any]] = None, fact_threshold: float = 2 / 3,
                 policy: str = "judge_primary") -> AnswerEval:
    refusal = is_refusal(answer) and not cited_chunk_ids
    faith = None if (gate_emitted + gate_rejected) == 0 else gate_emitted / (gate_emitted + gate_rejected)
    jc = None if not judge else judge.get("correct")
    jf = None if not judge else judge.get("faithful")
    if status != "ok" and not answer.strip():
        return AnswerEval(None, None, False, faith, jc, jf, "")
    if status == "rate_limited":  # degraded reply (excerpts / busy) is not an answer to grade
        return AnswerEval(None, None, False, faith, jc, jf, "")
    if q.get("is_unanswerable"):
        return AnswerEval(None, None, refusal, faith, jc, jf, "refused_correctly" if refusal else "wrong")
    facts = list(q.get("key_facts") or [])
    recall = sum(fact_present(f, answer) for f in facts) / len(facts) if facts else None
    target = q.get("target_paper_id")
    cited_target = any(cid.startswith(target) for cid in cited_chunk_ids) if target else None
    if refusal:
        return AnswerEval(recall, cited_target, True, faith, jc, jf, "refused_wrongly")
    rules_ok = (recall is not None and recall >= fact_threshold) and bool(cited_target)
    if policy == "judge_primary" and jc is not None:
        # Gold key facts often include details the question did not ask for, so the judge decides;
        # a citation to the right paper is still required.
        ok = bool(jc) and bool(cited_target)
    elif policy == "rules_and_judge" and jc is not None:
        ok = rules_ok and bool(jc)
    else:
        ok = rules_ok
    return AnswerEval(recall, cited_target, False, faith, jc, jf, "correct" if ok else "wrong")


# ---------------------------------------------------------------- LLM judge (real Groq call, other model)
JUDGE_SYSTEM = """You grade answers from a research-paper assistant. Reply with JSON only:
{"correct": true|false, "faithful": true|false, "reason": "<one sentence>"}
correct = the answer states the same facts as the reference answer (wording may differ; missing a key number = false).
faithful = every claim in the answer is supported by the evidence passages."""


def judge_messages(q: Dict[str, Any], answer: str, evidence_text: str) -> List[Dict[str, str]]:
    user = (f"QUESTION: {q['question']}\nREFERENCE ANSWER: {q['gold_answer']}\n"
            f"EVIDENCE:\n{evidence_text[:3500]}\nANSWER TO GRADE: {answer}\nJSON:")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_judge(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {"correct": bool(d.get("correct")), "faithful": bool(d.get("faithful")), "reason": str(d.get("reason", ""))}
    except Exception:
        return None
