from __future__ import annotations

import json

import networkx as nx

from biogenesis.evidence.models import Evidence
from biogenesis.knowledge_graph.schema import GraphEdge
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_evidence(self, evidence: Evidence) -> None:
        """Add one evidence item as a directed edge subject -> object."""
        self.graph.add_node(evidence.subject, type="entity")
        self.graph.add_node(evidence.obj, type="entity")
        edge = GraphEdge(
            subject=evidence.subject,
            relation=evidence.relation,
            obj=evidence.obj,
            evidence_id=evidence.evidence_id,
            pmid=evidence.pmid,
            confidence=evidence.confidence,
        )
        self.graph.add_edge(
            evidence.subject,
            evidence.obj,
            key=evidence.evidence_id,
            relation=edge.relation,
            evidence_id=edge.evidence_id,
            pmid=edge.pmid,
            confidence=edge.confidence,
        )

    def add_evidence_batch(self, evidence_list: list[Evidence]) -> None:
        for e in evidence_list:
            self.add_evidence(e)
        logger.info(
            "Graph now has %d nodes / %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def neighbors(self, entity: str) -> list[dict]:
        """All outgoing relations from an entity, with evidence provenance."""
        if entity not in self.graph:
            return []
        results = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            results.append({"target": target, **data})
        return results

    def find_conflicting_edges(self, subject: str, obj: str) -> list[dict]:
        """
        Relations between the same subject/object pair that differ -- the
        raw candidates for contradiction resolution (Milestone 15).
        """
        if not self.graph.has_edge(subject, obj):
            return []
        edges = self.graph.get_edge_data(subject, obj)
        relations = {e["relation"] for e in edges.values()}
        if len(relations) <= 1:
            return []
        return list(edges.values())

    def save(self, path: str) -> None:
        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved knowledge graph to %s", path)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, multigraph=True, directed=True)
        logger.info("Loaded knowledge graph from %s", path)
