"""Sentence-gated streaming.

Trade-off this encodes: raw token streaming gives the best time-to-first-token
but can show a hallucinated number before anything checks it. We stream, but a
sentence is only released to the person once it is complete and passes:
  1. it cites at least one [E#] tag that exists in the prompt,
  2. every number in it appears in the cited passages (or in the question),
or it is the refusal sentinel. Cost: user-visible first token = time to the end
of the first verified sentence, not the first raw token. Both are recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from relay.live.compress import Evidence
from relay.live.prompts import REFUSAL_SENTINEL

CITE_GROUP = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E?\d+)*)\s*\]")
# No letter lookbehind: the PDF text glues words to numbers ("scoreof28.4"); citation tags are
# stripped before extraction, so "E1" is not mistaken for a number.
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_TRAILING_CITE = re.compile(r"[.!?]\s*(\[[^\]]*\]\s*)+$")
_ABBREV = ("e.g.", "i.e.", "et al.", "Fig.", "Sec.", "Eq.", "Tab.", "vs.", "approx.", "cf.", "No.")
_SOFT_REFUSAL = re.compile(r"\b(evidence|passages?|papers?) (does|do) not (contain|mention|describe|state|say)", re.I)


def norm_number(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def numbers_in(text: str) -> Set[str]:
    clean = CITE_GROUP.sub(" ", text)
    clean = re.sub(r"\bE\d+\b", " ", clean)
    return {norm_number(m.group(0)) for m in _NUM.finditer(clean)}


def cited_tags(sentence: str) -> List[str]:
    tags: List[str] = []
    for grp in CITE_GROUP.findall(sentence):
        for part in re.split(r"[,;]", grp):
            p = part.strip().upper()
            tags.append(p if p.startswith("E") else f"E{p}")
    return tags


def render_citation(ev: Evidence) -> str:
    c = ev.chunk
    return f"[arXiv:{c.paper_id} p.{c.page} §{c.section} ({c.chunk_id})]"


@dataclass
class Verdict:
    ok: bool
    kind: str  # cited | refusal | uncited | invalid_citation | ungrounded_number
    text: str
    rendered: str = ""
    tags: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)


class SentenceGate:
    def __init__(self, evidence: Sequence[Evidence], question: str = ""):
        self.ev: Dict[str, Evidence] = {e.tag: e for e in evidence}
        # numbers a sentence may use when citing E#: the passage text plus its page/section header
        self.ev_numbers: Dict[str, Set[str]] = {
            e.tag: numbers_in(e.text) | numbers_in(e.chunk.text or "") | numbers_in(f"{e.chunk.page} {e.chunk.section}")
            for e in evidence
        }
        self.q_numbers = numbers_in(question)
        self.buf = ""
        self.pending_uncited: List[str] = []
        self.emitted: List[Verdict] = []
        self.rejected: List[Verdict] = []
        self.refusal: Optional[str] = None
        self.missing_note: Optional[str] = None

    # ------------------------------------------------------------ verification
    def verify(self, sentence: str) -> Verdict:
        s = sentence.strip()
        if s.upper().startswith(REFUSAL_SENTINEL) or (_SOFT_REFUSAL.search(s) and not cited_tags(s)):
            reason = s.split(":", 1)[1].strip() if ":" in s else s
            return Verdict(True, "refusal", s, rendered=reason)
        tags = cited_tags(s)
        if not tags:
            return Verdict(False, "uncited", s)
        bad = [t for t in tags if t not in self.ev]
        if bad:
            return Verdict(False, "invalid_citation", s, tags=tags)
        allowed = set(self.q_numbers)
        for t in tags:
            allowed |= self.ev_numbers[t]
        missing = sorted(n for n in numbers_in(s) if n not in allowed)
        if missing:
            return Verdict(False, "ungrounded_number", s, tags=tags, missing_numbers=missing)
        rendered = CITE_GROUP.sub(lambda m: " ".join(render_citation(self.ev[t]) for t in cited_tags(m.group(0))), s)
        return Verdict(True, "cited", s, rendered=" ".join(rendered.split()), tags=tags)

    # ------------------------------------------------------------ streaming
    def _split_ready(self) -> Tuple[List[str], str]:
        """Return complete sentences; keep a trailing partial sentence in the buffer.

        A boundary is '.', '!' or '?' (or a newline) followed by whitespace and a
        character that does not start a citation, so "... 512. [E1]" stays together.
        """
        out, start, i, b = [], 0, 0, self.buf
        while i < len(b):
            ch = b[i]
            if ch == "\n" and b[start:i].strip():
                out.append(b[start:i]); start = i + 1
            elif ch == "]" and _TRAILING_CITE.search(b[start:i + 1]):
                # "... 12.5 Hz. [E1] Next" -> boundary after the trailing citation
                j = i + 1
                while j < len(b) and b[j].isspace():
                    j += 1
                if j >= len(b):
                    break
                if j > i + 1 and b[j] != "[":
                    out.append(b[start:i + 1]); start = i + 1
            elif ch in ".!?" and i + 1 < len(b) and b[i + 1].isspace():
                j = i + 1
                while j < len(b) and b[j].isspace():
                    j += 1
                if j >= len(b):
                    break  # need lookahead to know whether a citation follows
                frag = b[start:i + 1]
                is_decimal = ch == "." and i > 0 and b[i - 1].isdigit() and j < len(b) and b[j].isdigit()
                if b[j] != "[" and not is_decimal and not frag.rstrip().endswith(_ABBREV):
                    out.append(frag); start = i + 1
            i += 1
        return [s for s in out if s.strip()], b[start:]

    def feed(self, delta: str) -> List[str]:
        """Add streamed text; return rendered sentences that are safe to show now."""
        self.buf += delta
        ready, self.buf = self._split_ready()
        return self._process(ready)

    def finish(self) -> List[str]:
        tail, self.buf = self.buf, ""
        out = self._process([tail] if tail.strip() else [])
        for s in self.pending_uncited:  # never followed by a cited sentence -> drop
            self.rejected.append(Verdict(False, "uncited", s))
        self.pending_uncited = []
        return out

    def _process(self, sentences: Sequence[str]) -> List[str]:
        shown: List[str] = []
        for s in sentences:
            v = self.verify(s)
            if v.kind == "refusal":
                if not self.emitted:
                    self.refusal = v.rendered
                else:
                    self.missing_note = v.rendered  # partial answer: say what the papers don't cover
                continue
            if v.ok:
                # An uncited, number-free lead-in sentence is allowed only if the next sentence is grounded.
                for p in self.pending_uncited:
                    if not numbers_in(p):
                        lead = " ".join(p.split())
                        self.emitted.append(Verdict(True, "lead_in", p, rendered=lead))
                        shown.append(lead)
                    else:
                        self.rejected.append(Verdict(False, "ungrounded_number", p))
                self.pending_uncited = []
                self.emitted.append(v)
                shown.append(v.rendered)
            elif v.kind == "uncited" and not numbers_in(s) and len(s.split()) <= 30:
                self.pending_uncited.append(s)
            else:
                self.rejected.append(v)
        return shown

    # ------------------------------------------------------------ summary
    @property
    def violations(self) -> int:
        return sum(1 for v in self.rejected if v.kind in ("invalid_citation", "ungrounded_number"))

    def should_abort(self) -> bool:
        """Stop paying for tokens once the model is clearly not following the evidence."""
        return self.violations >= 2 and not self.emitted

    def problems(self) -> List[str]:
        return sorted({f"{v.kind}: {v.text[:80]}" + (f" (numbers {v.missing_numbers})" if v.missing_numbers else "")
                       for v in self.rejected})

    def answer_text(self) -> str:
        text = " ".join(v.rendered for v in self.emitted)
        if self.missing_note:
            text += f" (The papers don't state: {self.missing_note})"
        return text

    def cited_chunks(self) -> List[str]:
        ids: List[str] = []
        for v in self.emitted:
            for t in v.tags:
                cid = self.ev[t].chunk.chunk_id
                if cid not in ids:
                    ids.append(cid)
        return ids
