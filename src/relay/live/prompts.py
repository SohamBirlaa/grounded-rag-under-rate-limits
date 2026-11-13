"""Prompt construction, output length control and prompt-caching layout.

Layout is [static system prompt][evidence][question], question LAST, so the
static prefix is byte-identical across requests. On Groq only openai/gpt-oss-*
models support prompt caching (cached tokens do not count toward rate limits),
so the long few-shot profile is only worth it there. On non-caching models every
system-prompt token is billed against TPM, so the lean profile is used.
(llama-3.1-8b-instant was shut down for free tier on 16 Aug 2026; default is gpt-oss-20b.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from relay.live.compress import Evidence

REFUSAL_SENTINEL = "NOT_IN_PAPERS"

LEAN_SYSTEM = f"""You answer questions using ONLY the numbered evidence passages from three arXiv papers.
Rules:
1. Every sentence must end with a citation tag of the evidence it uses, e.g. "d_model is 512 [E1]." Use only tags that exist.
2. Copy numbers exactly as written in the evidence. Never use outside knowledge.
3. If the evidence answers only PART of the question, answer that part with citations, then add a last line:
   {REFUSAL_SENTINEL}: <the part the evidence does not cover>
   If it answers none of it, reply with only that line. Never refuse a part the evidence does answer.
4. Be brief. No preamble, no restating the question."""

FEW_SHOT = f"""
Example A
EVIDENCE:
[E1] (paper X p.3 §2 Method) The encoder uses 12 layers with hidden size 640 and 10 heads.
[E2] (paper X p.7 §5 Results) Our model reaches 31.2 BLEU on the test set.
QUESTION: How many layers and heads does the encoder have?
ANSWER: The encoder has 12 layers and 10 attention heads [E1].

Example B
EVIDENCE:
[E1] (paper Y p.2 §1 Introduction) We study speech tokenizers for dialogue.
QUESTION: What GPU cluster was used to train paper Y?
ANSWER: {REFUSAL_SENTINEL}: the evidence does not describe the training hardware.

Example C
EVIDENCE:
[E1] (paper Z p.4 §3 Model) The codec runs at 25 Hz.
[E2] (paper Z p.9 §6 Ablations) Lowering the rate to 12 Hz hurt quality.
QUESTION: What frame rate does the codec use, and what happened at a lower rate?
ANSWER: The codec runs at 25 Hz [E1]. Lowering it to 12 Hz hurt quality [E2].
"""

CACHE_FRIENDLY_SYSTEM = LEAN_SYSTEM + "\n" + FEW_SHOT


@dataclass
class LengthPolicy:
    label: str
    max_tokens: int
    max_sentences: int


_LONG_Q = re.compile(r"\b(how does|how do|why|explain|compare|differ|difference|describe|versus|vs\.?)\b", re.I)


def length_policy(question: str, pressure: float = 0.0, is_reasoning_model: bool = False) -> LengthPolicy:
    """Output length control: cap tokens by question type, tighter under quota pressure."""
    if _LONG_Q.search(question):
        p = LengthPolicy("explain", 260, 4)
    elif len(question.split()) > 28 or question.count("?") > 1 or " and " in question.lower():
        p = LengthPolicy("multi_fact", 180, 3)
    else:
        p = LengthPolicy("fact", 120, 2)
    if pressure >= 0.75:  # close to the per-minute ceiling -> shorter answers for everyone
        p = LengthPolicy(p.label + "+compact", max(80, int(p.max_tokens * 0.6)), max(1, p.max_sentences - 1))
    if is_reasoning_model:
        # gpt-oss spends completion tokens on hidden reasoning before the answer
        p = LengthPolicy(p.label + "+reasoning", p.max_tokens + 300, p.max_sentences)
    return p


def evidence_header(ev: Evidence) -> str:
    c = ev.chunk
    return f"[{ev.tag}] ({c.paper_id} p.{c.page} §{c.section})"


def build_messages(question: str, evidence: Sequence[Evidence], profile: str, policy: LengthPolicy) -> List[Dict[str, str]]:
    system = CACHE_FRIENDLY_SYSTEM if profile == "cache_friendly" else LEAN_SYSTEM
    ev_block = "\n".join(f"{evidence_header(e)} {e.text}" for e in evidence) or "(no passages retrieved)"
    user = (f"EVIDENCE:\n{ev_block}\n\nQUESTION: {question.strip()}\n"
            f"Answer in at most {policy.max_sentences} sentences.\nANSWER:")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_repair_messages(base: List[Dict[str, str]], bad_answer: str, problems: Sequence[str]) -> List[Dict[str, str]]:
    msg = ("Your answer broke the rules: " + "; ".join(problems) +
           ". Rewrite it. Cite every sentence with an existing [E#] tag, copy numbers exactly from the cited "
           f"passage, or reply {REFUSAL_SENTINEL}: <reason> if the evidence does not answer the question.")
    return list(base) + [{"role": "assistant", "content": bad_answer}, {"role": "user", "content": msg}]


def estimate_prompt_tokens(messages: Sequence[Dict[str, str]], ratio: float = 1.25) -> int:
    """Local estimate = regex tokens x ratio + fixed chat-template overhead.

    Measured on gpt-oss-20b (24 live requests): real/local = 1.25 mean, 1.72 max (tiny prompts,
    where the fixed template overhead dominates), hence the +50 constant.
    """
    from relay.live.compress import tok_count

    return int(sum(tok_count(m["content"]) + 4 for m in messages) * ratio) + 50
