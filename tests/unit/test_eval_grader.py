"""Unit tests for the evaluation questions and AnswerGrader suite."""

from pathlib import Path
import pytest

from relay.contracts.schemas import CitationRecord
from relay.eval.grader import (
    AnswerGrader,
    CitationValidator,
    calculate_page_hit_metrics,
    calculate_retrieval_metrics,
    load_evaluation_questions,
)
from relay.rag.chunker import Chunk

QUESTIONS_PATH = Path("data/eval/questions.json")


def test_load_evaluation_questions_schema_and_distribution():
    """Verify exactly 20 stratified evaluation questions are loaded across 4 categories."""
    assert QUESTIONS_PATH.is_file(), f"Missing questions file: {QUESTIONS_PATH}"
    questions = load_evaluation_questions(QUESTIONS_PATH)
    assert len(questions) == 20

    # Count categories
    by_cat = {}
    for q in questions:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1

    assert by_cat.get("A") == 8, f"Expected 8 Cat A questions, got {by_cat.get('A')}"
    assert by_cat.get("B") == 4, f"Expected 4 Cat B questions, got {by_cat.get('B')}"
    assert by_cat.get("C") == 4, f"Expected 4 Cat C questions, got {by_cat.get('C')}"
    assert by_cat.get("D") == 4, f"Expected 4 Cat D questions, got {by_cat.get('D')}"

    # Cat D must all be unanswerable
    cat_d_questions = [q for q in questions if q.category == "D"]
    for q in cat_d_questions:
        assert q.is_unanswerable is True
        assert q.target_paper_id is None

    # Cat A, B, C must have target papers and key facts
    for q in [q for q in questions if q.category in ("A", "B", "C")]:
        assert q.is_unanswerable is False
        assert q.target_paper_id in ["1706.03762v7", "2410.00037v1", "2507.07955v2"]
        assert len(q.key_facts) >= 2
        assert q.target_page is not None and q.target_page > 0


def test_citation_validator_text_extraction():
    """Verify CitationValidator extracts arXiv citation markers from generated text."""
    validator = CitationValidator()
    text = (
        "Scaled Dot-Product Attention is computed via softmax [arXiv:1706.03762v7 p.4 §3.2.1 (chunk_01)]. "
        "Furthermore, Mimi operates at 12.5 Hz [arXiv:2410.00037v1 p.4 §3.1 (chunk_02)]."
    )
    extracted = validator.extract_citations_from_text(text)
    assert len(extracted) == 2
    assert extracted[0]["paper_id"] == "1706.03762v7"
    assert extracted[0]["page"] == 4
    assert extracted[0]["chunk_id"] == "chunk_01"

    assert extracted[1]["paper_id"] == "2410.00037v1"
    assert extracted[1]["page"] == 4
    assert extracted[1]["chunk_id"] == "chunk_02"


def test_citation_validator_record_validation():
    """Verify validation of structured CitationRecord with chunk repository lookup."""
    chunk = Chunk(
        chunk_id="chunk_attention_01",
        paper_id="1706.03762v7",
        version="v7",
        page=4,
        section="3.2.1 Scaled Dot-Product Attention",
        content_type="prose",
        text="We compute the attention function on a set of queries simultaneously.",
        token_count=12,
        structural_breadcrumb="[Doc: Attention | Page 4]",
    )
    validator = CitationValidator(chunks_by_id={chunk.chunk_id: chunk})

    # Valid citation
    valid_record = CitationRecord(
        paper_id="1706.03762v7",
        version="v7",
        section="3.2.1 Scaled Dot-Product Attention",
        page=4,
        chunk_id="chunk_attention_01",
        quoted_evidence="We compute the attention function",
    )
    is_valid, reason = validator.validate_citation_record(valid_record)
    assert is_valid is True
    assert reason is None

    # Invalid page
    invalid_page_record = CitationRecord(
        paper_id="1706.03762v7",
        version="v7",
        section="3.2.1 Scaled Dot-Product Attention",
        page=12,  # wrong page
        chunk_id="chunk_attention_01",
        quoted_evidence="We compute the attention function",
    )
    is_valid, reason = validator.validate_citation_record(invalid_page_record)
    assert is_valid is False
    assert "page" in reason

    # Hallucinated evidence
    invalid_evidence_record = CitationRecord(
        paper_id="1706.03762v7",
        version="v7",
        section="3.2.1 Scaled Dot-Product Attention",
        page=4,
        chunk_id="chunk_attention_01",
        quoted_evidence="Completely hallucinated non-existent sentence text",
    )
    is_valid, reason = validator.validate_citation_record(invalid_evidence_record)
    assert is_valid is False
    assert "verbatim" in reason


def test_answer_grader_outcomes():
    """Verify 4-way grading across Cat A and Cat D questions."""
    questions = load_evaluation_questions(QUESTIONS_PATH)
    q_catA = next(q for q in questions if q.question_id == "q_catA_01")
    q_catD = next(q for q in questions if q.question_id == "q_catD_01")

    grader = AnswerGrader()

    # 1. Correct answer on Cat A
    good_answer = "In the base model, d_model is 512, h = 8 heads, and d_k is 64 [arXiv:1706.03762v7 p.5 §3.2.2]."
    res_correct = grader.grade_response(q_catA, answer_text=good_answer)
    assert res_correct.grade == "correct"

    # 2. Refused wrongly on Cat A
    refusal_answer = "I cannot answer this question based on the provided corpus."
    res_refused_wrongly = grader.grade_response(q_catA, answer_text=refusal_answer)
    assert res_refused_wrongly.grade == "refused_wrongly"

    # 3. Refused correctly on Cat D (epistemic abstention)
    res_refused_correctly = grader.grade_response(q_catD, answer_text=refusal_answer)
    assert res_refused_correctly.grade == "refused_correctly"

    # 4. Hallucinated answer on Cat D -> wrong
    hallucination_answer = "Llama-3-70B was trained on 15 trillion tokens with 16k H100 GPUs."
    res_wrong_catD = grader.grade_response(q_catD, answer_text=hallucination_answer)
    assert res_wrong_catD.grade == "wrong"


def test_retrieval_metrics_calculation():
    """Verify Recall@k and MRR@k calculations."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    gold = {"c3", "c9"}

    recall_5, mrr_5 = calculate_retrieval_metrics(retrieved, gold, k=5)
    assert recall_5 == 1.0
    assert mrr_5 == pytest.approx(1.0 / 3.0)  # rank 3

    recall_2, mrr_2 = calculate_retrieval_metrics(retrieved, gold, k=2)
    assert recall_2 == 0.0
    assert mrr_2 == 0.0

