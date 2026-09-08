
from __future__ import annotations

from dataclasses import dataclass, field

from biogenesis.evidence.models import Evidence
from biogenesis.json_utils import parse_json_loose
from biogenesis.llm_client import LLMClient
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a biomedical hypothesis generation agent. You \
will be given a research question and a list of evidence items, each with \
an evidence_id. Propose 1-3 specific, testable hypotheses that are \
directly supported by the given evidence. Do not use any knowledge outside \
the provided evidence.

Respond with a single JSON object of the form:
{"hypotheses": [ { ... }, { ... } ]}

Each element of "hypotheses" must have exactly these fields, and every \
string value must be properly double-quoted JSON:
  "hypothesis": one or two sentence testable hypothesis
  "rationale": short explanation of why the evidence supports it
  "supporting_evidence_ids": array of evidence_id strings you used
  "confidence": your confidence 0.0-1.0 based on evidence strength

If the evidence is insufficient to support any reasonable hypothesis, \
respond with {"hypotheses": []}.
Respond with JSON only -- no prose, no markdown code fences."""


@dataclass
class Hypothesis:
    hypothesis_id: str
    text: str
    rationale: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    critique: str = ""
    critique_verdict: str = ""  # filled in later by the Critic agent


class HypothesisGeneratorAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def generate(
        self,
        research_question: str,
        evidence_list: list[Evidence],
        memory_context: str = "",
    ) -> list[Hypothesis]:
        evidence_block = "\n".join(
            f"[{e.evidence_id}] ({e.study_type}, confidence={e.confidence}) "
            f"{e.subject} -- {e.relation} -- {e.obj}. Claim: {e.claim}"
            for e in evidence_list
        )
        memory_block = (
            f"\n\nRelevant past research from memory (for context only -- "
            f"do not treat as evidence, do not cite these as evidence_ids):\n"
            f"{memory_context}"
            if memory_context
            else ""
        )
        prompt = (
            f"Research question: {research_question}\n\n"
            f"Available evidence:\n{evidence_block if evidence_block else '(none)'}"
            f"{memory_block}"
        )
        raw = self.llm.complete(
            prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.4, json_mode=True
        )
        parsed = parse_json_loose(raw)
        items = parsed.get("hypotheses", []) if isinstance(parsed, dict) else []

        hypotheses = []
        for i, item in enumerate(items):
            try:
                hypotheses.append(
                    Hypothesis(
                        hypothesis_id=f"hyp_{i}",
                        text=item["hypothesis"],
                        rationale=item.get("rationale", ""),
                        supporting_evidence_ids=item.get("supporting_evidence_ids", []),
                        confidence=float(item.get("confidence", 0.0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping malformed hypothesis item: %s", e)

        logger.info("Generated %d hypotheses", len(hypotheses))
        return hypotheses
