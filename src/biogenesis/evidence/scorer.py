from __future__ import annotations

import re
from datetime import datetime

from biogenesis.evidence.models import Evidence
from biogenesis.logging_utils import get_logger
from biogenesis.retrieval.models import Paper

logger = get_logger(__name__)

_STUDY_TYPE_WEIGHTS = {
    "meta-analysis": 1.0,
    "rct": 0.9,
    "cohort": 0.7,
    "case-control": 0.6,
    "case-study": 0.4,
    "animal-model": 0.3,
    "in-vitro": 0.25,
    "review": 0.35,
    "unknown": 0.3,
}

_CURRENT_YEAR = datetime.now().year


class EvidenceScorer:
    def score(self, evidence: Evidence, paper: Paper) -> Evidence:
        base = _STUDY_TYPE_WEIGHTS.get(evidence.study_type, 0.3)
        recency_multiplier = self._recency_multiplier(paper.pub_date)
        confidence = round(min(base * recency_multiplier, 1.0), 3)
        evidence.confidence = confidence
        return evidence

    def score_batch(self, evidence_list: list[Evidence], paper: Paper) -> list[Evidence]:
        return [self.score(e, paper) for e in evidence_list]

    @staticmethod
    def _recency_multiplier(pub_date: str) -> float:
        """
        Evidence doesn't become false with age, but for fast-moving fields a
        very old single study is often superseded. We apply a gentle decay
        rather than a hard cutoff, and never let it fall below 0.75 so a
        strong old meta-analysis is still weighted meaningfully.
        """
        match = re.search(r"(19|20)\d{2}", pub_date or "")
        if not match:
            return 0.9 
        year = int(match.group(0))
        age = max(_CURRENT_YEAR - year, 0)
        decay = max(1.0 - 0.015 * age, 0.75)
        return decay
