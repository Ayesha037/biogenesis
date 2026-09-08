from __future__ import annotations

from biogenesis.json_utils import parse_json_loose
from biogenesis.llm_client import LLMClient
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a biomedical research planning agent. Given a \
broad research question, break it down into 3-5 concrete, specific \
sub-questions that can each be answered by searching PubMed literature. \
Each sub-question should be narrow enough to form a good PubMed search \
query.

Respond with a single JSON object of this exact form:
{"sub_questions": ["...", "...", "..."]}

Respond with JSON only -- no prose, no markdown code fences."""


class PlannerAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def plan(self, research_question: str) -> list[str]:
        raw = self.llm.complete(
            prompt=f"Research question: {research_question}",
            system=_SYSTEM_PROMPT,
            temperature=0.3,
            json_mode=True,
        )
        parsed = parse_json_loose(raw)
        sub_questions = []
        if isinstance(parsed, dict):
            sub_questions = [str(x) for x in parsed.get("sub_questions", [])]
        logger.info("Planner produced %d sub-questions", len(sub_questions))
        return sub_questions
