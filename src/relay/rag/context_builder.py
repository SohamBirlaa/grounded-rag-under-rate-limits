"""Context builder and prompt formatter for The Relay research paper RAG pipeline.

Formats retrieved passages into structured XML with complete provenance headers.
Generates strict grounding prompts requiring verbatim evidence and exact citation markers.
Supports Quota-Conserving Compact Retrieval Mode context truncation.
"""

from typing import List, Optional, Tuple
from relay.rag.chunker import count_tokens
from relay.rag.retriever import RetrievedChunk

GROUNDING_SYSTEM_PROMPT = """You are The Relay, an evidence-grounded research assistant specializing in deep learning papers.
Your task is to answer the user's question using ONLY the provided retrieved passages.

STRICT GROUNDING & CITATION RULES:
1. Every factual statement or quantitative claim MUST be directly supported by the retrieved passages.
2. For each claim, provide an exact citation in the format:
   [arXiv:<paper_id> p.<page> §<section> (<chunk_id>)]
   Example: [arXiv:1706.03762v7 p.3 §3.2 (1706.03762v7_sec_3_2_p03_c014)]
3. Quote relevant verbatim key phrases or numbers when discussing architectural details, hyperparameters, or benchmark results.
4. EPISTEMIC ABSTENTION RULE: If the retrieved passages do NOT contain sufficient factual evidence to answer the question, you MUST explicitly state that the provided corpus lacks sufficient evidence. Do NOT rely on prior parametric knowledge or hallucinate details.
5. If the question asks about a paper, model, or dataset not present in the passages, refuse to speculate and state:
   "I cannot answer this question based on the provided corpus. The relevant information is not present in the retrieved research documents."
"""


class ContextBuilder:
    """Assembles retrieved passages into structured context for generation."""

    def __init__(self, system_prompt: str = GROUNDING_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def build_passages_xml(self, passages: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into structured XML passages."""
        if not passages:
            return "<retrieved_passages>\n  <!-- No relevant passages retrieved -->\n</retrieved_passages>"

        lines = ["<retrieved_passages>"]
        for p in passages:
            lines.append(
                f'  <passage id="{p.chunk_id}" paper_id="{p.paper_id}" version="{p.version}" '
                f'page="{p.page}" section="{p.section}" content_type="{p.content_type}">'
            )
            # Indent passage text
            for text_line in p.text.splitlines():
                lines.append(f"    {text_line}")
            lines.append("  </passage>")
        lines.append("</retrieved_passages>")
        return "\n".join(lines)

    def build_user_prompt(self, question: str, passages: List[RetrievedChunk]) -> str:
        """Construct complete user prompt with XML passages and question."""
        passages_xml = self.build_passages_xml(passages)
        prompt = (
            f"Here are the relevant passages retrieved from the research papers:\n\n"
            f"{passages_xml}\n\n"
            f"Question: {question}\n\n"
            f"Please provide an accurate, grounded answer with exact arXiv citations and quoted evidence."
        )
        return prompt

    def build_messages(
        self, question: str, passages: List[RetrievedChunk]
    ) -> List[dict]:
        """Build standard chat messages payload for LLM providers."""
        user_prompt = self.build_user_prompt(question, passages)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def estimate_prompt_tokens(self, question: str, passages: List[RetrievedChunk]) -> int:
        """Estimate token count for system prompt + context + question."""
        user_prompt = self.build_user_prompt(question, passages)
        return count_tokens(self.system_prompt) + count_tokens(user_prompt)

