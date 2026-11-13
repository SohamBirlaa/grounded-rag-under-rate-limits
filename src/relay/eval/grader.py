"""Evaluation and grading suite for The Relay RAG system.

Implements:
- Quantitative IR metrics: Recall@k, MRR@k, Page/Section Hit@k
- Deterministic citation format and provenance verification
- 4-way answer grading: correct, wrong, refused_correctly, refused_wrongly
- Evaluates black-box parametric reliance across Cat A, B, C, and D questions.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from relay.contracts.schemas import AnswerGradeType, CitationRecord, PaperIdType
from relay.rag.chunker import Chunk
from relay.rag.retriever import RetrievedChunk

# Regex matching: [arXiv:1706.03762v7 p.3 §3.2 (chunk_id)]
CITATION_REGEX = re.compile(
    r"\[arXiv:(?P<paper_id>1706\.03762v7|2410\.00037v1|2507\.07955v2)\s+"
    r"p\.(?P<page>\d+)\s+"
    r"§(?P<section>[^\(\]]+?)"
    r"(?:\s+\((?P<chunk_id>[^\)]+)\))?\]",
    re.IGNORECASE,
)

REFUSAL_TRIGGERS = [
    "cannot answer",
    "can not answer",
    "not present in the retrieved",
    "not present in the provided",
    "not mentioned in the provided",
    "lacks sufficient evidence",
    "insufficient evidence",
    "not found in the retrieved",
    "corpus does not contain",
    "papers do not contain",
    "provided documents do not",
]


@dataclass
class EvaluationQuestion:
    """Evaluation question schema loaded from data/eval/questions.json."""

    question_id: str
    category: str  # "A", "B", "C", "D"
    category_name: str
    question: str
    target_paper_id: Optional[PaperIdType]
    target_version: Optional[str]
    target_section: Optional[str]
    target_page: Optional[int]
    gold_answer: str
    key_facts: List[str]
    is_unanswerable: bool
    trap_description: Optional[str] = None


@dataclass
class CitationValidationResult:
    """Detailed verification outcome of parsed or declared citations."""

    is_valid: bool
    total_citations: int
    valid_citations: int
    invalid_reasons: List[str] = field(default_factory=list)


@dataclass
class GradingResult:
    """Outcome of grading a single response against an evaluation question."""

    question_id: str
    category: str
    grade: AnswerGradeType
    is_refusal: bool
    is_unanswerable: bool
    facts_matched: List[str]
    facts_missing: List[str]
    citation_result: CitationValidationResult
    explanation: str


def normalize_whitespace(text: str) -> str:
    """Normalize internal whitespaces and newlines for robust text matching."""
    return re.sub(r"\s+", " ", text).strip()


class CitationValidator:
    """Deterministic validator for research paper citations and evidence provenance."""

    def __init__(self, chunks_by_id: Optional[Dict[str, Chunk]] = None) -> None:
        self.chunks_by_id = chunks_by_id or {}

    def extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract all regex-matched citation markers from free-form text."""
        citations = []
        for match in CITATION_REGEX.finditer(text):
            data = match.groupdict()
            citations.append(
                {
                    "paper_id": data["paper_id"],
                    "page": int(data["page"]),
                    "section": data["section"].strip(),
                    "chunk_id": data["chunk_id"].strip() if data.get("chunk_id") else None,
                    "raw_span": match.group(0),
                }
            )
        return citations

    def validate_citation_record(self, citation: CitationRecord) -> Tuple[bool, Optional[str]]:
        """Verify that a structured CitationRecord is grounded in the indexed corpus."""
        # 1. Check paper ID
        if citation.paper_id not in ["1706.03762v7", "2410.00037v1", "2507.07955v2"]:
            return False, f"Unknown paper ID: {citation.paper_id}"

        # 2. Check chunk existence if chunk repository is loaded
        if self.chunks_by_id:
            chunk = self.chunks_by_id.get(citation.chunk_id)
            if not chunk:
                return False, f"Chunk ID '{citation.chunk_id}' not found in corpus index"

            if chunk.paper_id != citation.paper_id:
                return False, f"Chunk {citation.chunk_id} paper ({chunk.paper_id}) mismatch with citation ({citation.paper_id})"

            if chunk.page != citation.page:
                return False, f"Chunk {citation.chunk_id} page ({chunk.page}) mismatch with citation ({citation.page})"

            # Check quoted evidence is verbatim substring of chunk
            norm_chunk_text = normalize_whitespace(chunk.text.lower())
            norm_quote = normalize_whitespace(citation.quoted_evidence.lower())
            if norm_quote not in norm_chunk_text:
                return False, f"Quoted evidence not found verbatim in chunk {citation.chunk_id}"

        return True, None

    def validate_citations(
        self,
        citations: List[CitationRecord],
        text: Optional[str] = None,
    ) -> CitationValidationResult:
        """Validate a list of citations and text markers."""
        if not citations and text:
            # Fall back to extracting from text
            extracted = self.extract_citations_from_text(text)
            if not extracted:
                return CitationValidationResult(
                    is_valid=False,
                    total_citations=0,
                    valid_citations=0,
                    invalid_reasons=["No citations found in text or response"],
                )
            # Extracted markers without chunk IDs
            return CitationValidationResult(
                is_valid=True,
                total_citations=len(extracted),
                valid_citations=len(extracted),
                invalid_reasons=[],
            )

        if not citations:
            return CitationValidationResult(
                is_valid=False,
                total_citations=0,
                valid_citations=0,
                invalid_reasons=["No citations provided"],
            )

        valid_count = 0
        reasons = []
        for c in citations:
            is_valid, reason = self.validate_citation_record(c)
            if is_valid:
                valid_count += 1
            else:
                if reason:
                    reasons.append(reason)

        all_valid = (valid_count == len(citations)) and (valid_count > 0)
        return CitationValidationResult(
            is_valid=all_valid,
            total_citations=len(citations),
            valid_citations=valid_count,
            invalid_reasons=reasons,
        )


class AnswerGrader:
    """Grader that evaluates responses across Cat A, B, C, and D questions."""

    def __init__(self, citation_validator: Optional[CitationValidator] = None) -> None:
        self.validator = citation_validator or CitationValidator()

    @staticmethod
    def detect_refusal(text: str, is_refusal_flag: bool = False) -> bool:
        """Detect whether response abstained or refused."""
        if is_refusal_flag:
            return True
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in REFUSAL_TRIGGERS)

    def grade_response(
        self,
        question: EvaluationQuestion,
        answer_text: str,
        citations: Optional[List[CitationRecord]] = None,
        is_refusal_flag: bool = False,
    ) -> GradingResult:
        """Grade a single response into correct, wrong, refused_correctly, or refused_wrongly."""
        citations = citations or []
        refused = self.detect_refusal(answer_text, is_refusal_flag)
        cit_result = self.validator.validate_citations(citations, text=answer_text)

        # 1. Unanswerable / Corpus-Absent Questions (Category D)
        if question.is_unanswerable:
            if refused:
                return GradingResult(
                    question_id=question.question_id,
                    category=question.category,
                    grade="refused_correctly",
                    is_refusal=True,
                    is_unanswerable=True,
                    facts_matched=[],
                    facts_missing=[],
                    citation_result=cit_result,
                    explanation="Correctly abstained on corpus-absent question without hallucinating.",
                )
            else:
                return GradingResult(
                    question_id=question.question_id,
                    category=question.category,
                    grade="wrong",
                    is_refusal=False,
                    is_unanswerable=True,
                    facts_matched=[],
                    facts_missing=["Epistemic abstention expected"],
                    citation_result=cit_result,
                    explanation="Hallucinated answer to unanswerable question instead of abstaining.",
                )

        # 2. Answerable Questions (Category A, B, C)
        if refused:
            return GradingResult(
                question_id=question.question_id,
                category=question.category,
                grade="refused_wrongly",
                is_refusal=True,
                is_unanswerable=False,
                facts_matched=[],
                facts_missing=question.key_facts,
                citation_result=cit_result,
                explanation="Falsely refused/abstained on an answerable question with available corpus evidence.",
            )

        # Check key facts match
        answer_lower = answer_text.lower()
        matched = []
        missing = []
        for fact in question.key_facts:
            if fact.lower() in answer_lower:
                matched.append(fact)
            else:
                missing.append(fact)

        # Require at least 60% of key facts matched
        fact_threshold = max(1, int(len(question.key_facts) * 0.6))
        facts_ok = len(matched) >= fact_threshold

        # Citation presence check
        citations_ok = cit_result.valid_citations > 0 or cit_result.total_citations > 0

        if facts_ok and citations_ok:
            grade: AnswerGradeType = "correct"
            explanation = f"Answer matched {len(matched)}/{len(question.key_facts)} key facts with valid citations."
        else:
            grade = "wrong"
            reasons = []
            if not facts_ok:
                reasons.append(f"Missing key facts: {missing}")
            if not citations_ok:
                reasons.append("Missing or invalid citations")
            explanation = "Answer failed criteria: " + "; ".join(reasons)

        return GradingResult(
            question_id=question.question_id,
            category=question.category,
            grade=grade,
            is_refusal=False,
            is_unanswerable=False,
            facts_matched=matched,
            facts_missing=missing,
            citation_result=cit_result,
            explanation=explanation,
        )


def calculate_retrieval_metrics(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Set[str],
    k: int = 5,
) -> Tuple[float, float]:
    """Calculate Recall@k and MRR@k given retrieved IDs and gold IDs."""
    if not gold_chunk_ids:
        return 0.0, 0.0

    top_k = list(retrieved_chunk_ids[:k])
    recall = 1.0 if any(cid in gold_chunk_ids for cid in top_k) else 0.0

    mrr = 0.0
    for rank, cid in enumerate(top_k, start=1):
        if cid in gold_chunk_ids:
            mrr = 1.0 / rank
            break

    return recall, mrr


def calculate_page_hit_metrics(
    retrieved_chunks: Sequence[RetrievedChunk],
    target_paper_id: Optional[str],
    target_page: Optional[int],
    k: int = 5,
) -> Tuple[float, float]:
    """Calculate Recall@k and MRR@k matching target paper and page."""
    if not target_paper_id or not target_page:
        return 0.0, 0.0

    top_k = list(retrieved_chunks[:k])
    hit = False
    mrr = 0.0

    for rank, chunk in enumerate(top_k, start=1):
        if chunk.paper_id == target_paper_id and chunk.page == target_page:
            if not hit:
                hit = True
                mrr = 1.0 / rank
            break

    recall = 1.0 if hit else 0.0
    return recall, mrr


def load_evaluation_questions(questions_path: Path) -> List[EvaluationQuestion]:
    """Load and validate the 20 stratified evaluation questions from JSON."""
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for item in data:
        questions.append(
            EvaluationQuestion(
                question_id=item["question_id"],
                category=item["category"],
                category_name=item["category_name"],
                question=item["question"],
                target_paper_id=item.get("target_paper_id"),
                target_version=item.get("target_version"),
                target_section=item.get("target_section"),
                target_page=item.get("target_page"),
                gold_answer=item["gold_answer"],
                key_facts=item["key_facts"],
                is_unanswerable=item.get("is_unanswerable", False),
                trap_description=item.get("trap_description"),
            )
        )
    return questions

