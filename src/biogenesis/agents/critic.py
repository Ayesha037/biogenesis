from __future__ import annotations

from biogenesis.agents.hypothesis_generator import Hypothesis
from biogenesis.evidence.models import Evidence
from biogenesis.json_utils import parse_json_loose
from biogenesis.llm_client import LLMClient
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an adversarial scientific critic reviewing a \
biomedical hypothesis. Your job is to find weaknesses, not to be agreeable.

Check specifically for:
1. Whether the cited evidence actually supports the claim as stated
2. Whether the evidence is strong enough (study type, confidence) for the \
strength of claim being made
3. Overreach: correlation stated as causation, small studies generalized \
too broadly
4. Any missing caveats a careful researcher would flag

Return ONLY a JSON object (no prose, no markdown fences) with:
  "verdict": one of "well-supported", "needs-caveats", "weakly-supported", \
"not-supported"
  "critique": 2-4 sentences explaining the verdict
  "suggested_caveats": array of short strings, caveats to add if any"""


class CriticAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def critique(self, hypothesis: Hypothesis, evidence_list: list[Evidence]) -> Hypothesis:
        cited = [e for e in evidence_list if e.evidence_id in hypothesis.supporting_evidence_ids]
        evidence_block = "\n".join(
            f"[{e.evidence_id}] ({e.study_type}, confidence={e.confidence}) "
            f"{e.subject} -- {e.relation} -- {e.obj}. Claim: {e.claim}"
            for e in cited
        )
        prompt = (
            f"Hypothesis: {hypothesis.text}\n"
            f"Rationale given: {hypothesis.rationale}\n\n"
            f"Cited evidence:\n{evidence_block if evidence_block else '(none cited)'}"
        )
        raw = self.llm.complete(
            prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.2, json_mode=True
        )
        parsed = parse_json_loose(raw)
        result = parsed if isinstance(parsed, dict) else {}

        hypothesis.critique_verdict = result.get("verdict", "unknown")
        caveats = result.get("suggested_caveats", [])
        caveat_text = ("\nCaveats: " + "; ".join(caveats)) if caveats else ""
        hypothesis.critique = result.get("critique", "") + caveat_text

        logger.info(
            "Critiqued %s -> verdict=%s", hypothesis.hypothesis_id, hypothesis.critique_verdict
        )
        return hypothesis

    def critique_batch(
        self, hypotheses: list[Hypothesis], evidence_list: list[Evidence]
    ) -> list[Hypothesis]:
        return [self.critique(h, evidence_list) for h in hypotheses]
