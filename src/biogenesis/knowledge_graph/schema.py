from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphEdge:
    subject: str
    relation: str
    obj: str
    evidence_id: str
    pmid: str
    confidence: float
