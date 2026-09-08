from __future__ import annotations

from dataclasses import dataclass

from biogenesis.evidence.models import Evidence
from biogenesis.json_utils import parse_json_loose
from biogenesis.llm_client import LLMClient
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a strict scientific fact-checker. You will be \
given a CLAIM and a piece of EVIDENCE TEXT. Decide whether the evidence \
text actually supports the claim -- not whether the claim is plausible in \
general, only whether THIS SPECIFIC TEXT entails it.

Respond with a single JSON object of the form:
{"supported": true or false, "reason": "one sentence explanation"}

Be strict: if the evidence text does not explicitly support the claim, or \
only partially supports it, or is about a related but different claim, \
answer false.
Respond with JSON only -- no prose, no markdown code fences."""


@dataclass
class SupportResult:
    evidence_id: str
    supported: bool
    reason: str


class SupportChecker:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def check(self, claim: str, evidence: Evidence) -> SupportResult:
        prompt = f"CLAIM: {claim}\n\nEVIDENCE TEXT: {evidence.source_text}"
        raw = self.llm.complete(
            prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.0, json_mode=True
        )
        parsed = parse_json_loose(raw)
        if not isinstance(parsed, dict):
            logger.warning(
                "Support check for evidence %s returned unparseable output; "
                "defaulting to unsupported", evidence.evidence_id,
            )
            return SupportResult(evidence.evidence_id, supported=False, reason="unparseable LLM output")

        return SupportResult(
            evidence_id=evidence.evidence_id,
            supported=bool(parsed.get("supported", False)),
            reason=str(parsed.get("reason", "")),
        )

    def check_hypothesis(
        self, claim: str, cited_evidence: list[Evidence]
    ) -> list[SupportResult]:
        """Check every piece of evidence a hypothesis cited against its claim text."""
        return [self.check(claim, e) for e in cited_evidence]
