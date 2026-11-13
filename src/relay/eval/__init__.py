"""Evaluation and benchmark suite for The Relay.

Provides:
- EvaluationQuestion schema and loader
- Deterministic CitationValidator
- 4-way AnswerGrader: correct, wrong, refused_correctly, refused_wrongly
- Retrieval metrics: Recall@k, MRR@k, Page Hit@k
"""

from relay.eval.grader import (
    EvaluationQuestion,
    CitationValidator,
    CitationValidationResult,
    AnswerGrader,
    GradingResult,
    calculate_retrieval_metrics,
    calculate_page_hit_metrics,
    load_evaluation_questions,
    CITATION_REGEX,
)

__all__ = [
    "EvaluationQuestion",
    "CitationValidator",
    "CitationValidationResult",
    "AnswerGrader",
    "GradingResult",
    "calculate_retrieval_metrics",
    "calculate_page_hit_metrics",
    "load_evaluation_questions",
    "CITATION_REGEX",
]

