from __future__ import annotations

from dataclasses import dataclass

from biogenesis.knowledge_graph.builder import KnowledgeGraphBuilder
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Contradiction:
    subject: str
    obj: str
    conflicting_claims: list[dict] 
    dominant_relation: str  


class ContradictionResolver:
    def __init__(self, kg: KnowledgeGraphBuilder) -> None:
        self.kg = kg

    def find_all_contradictions(self) -> list[Contradiction]:
        contradictions = []
        seen_pairs = set()
        for subject, obj in list(self.kg.graph.edges()):
            if (subject, obj) in seen_pairs:
                continue
            seen_pairs.add((subject, obj))
            conflicting = self.kg.find_conflicting_edges(subject, obj)
            if not conflicting:
                continue
            ranked = sorted(conflicting, key=lambda e: e["confidence"], reverse=True)
            contradictions.append(
                Contradiction(
                    subject=subject,
                    obj=obj,
                    conflicting_claims=ranked,
                    dominant_relation=ranked[0]["relation"],
                )
            )
        logger.info("Found %d contradictions in the knowledge graph", len(contradictions))
        return contradictions

    def contradiction_rate(self) -> float | None:
        """
        Contradictory subject/object pairs as a fraction of all distinct
        subject/object pairs in the graph. Normalizing by total pairs (not
        just reporting a raw count) is what makes this comparable across
        runs with different amounts of retrieved evidence -- a raw count of
        "3 contradictions" means something very different in a graph with
        10 edges vs. 500 edges.

        Returns None (not 0.0) when the graph has no edges at all, since a
        rate is undefined, not zero, in that case -- reporting 0.0 would
        misleadingly imply "checked, found none" rather than "nothing to
        check".
        """
        distinct_pairs = {(u, v) for u, v in self.kg.graph.edges()}
        if not distinct_pairs:
            return None
        contradictory_pairs = sum(
            1 for u, v in distinct_pairs if self.kg.find_conflicting_edges(u, v)
        )
        return round(contradictory_pairs / len(distinct_pairs), 4)
