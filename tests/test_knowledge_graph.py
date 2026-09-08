from biogenesis.evidence.contradiction_resolver import ContradictionResolver
from biogenesis.evidence.models import Evidence
from biogenesis.knowledge_graph.builder import KnowledgeGraphBuilder


def _evidence(eid, subject, relation, obj, confidence):
    return Evidence(
        evidence_id=eid,
        pmid=eid,
        claim=f"{subject} {relation} {obj}",
        subject=subject,
        relation=relation,
        obj=obj,
        source_text="...",
        confidence=confidence,
    )


def test_add_evidence_creates_nodes_and_edges():
    kg = KnowledgeGraphBuilder()
    kg.add_evidence(_evidence("e1", "DrugA", "inhibits", "ProteinB", 0.8))
    assert kg.graph.number_of_nodes() == 2
    assert kg.graph.number_of_edges() == 1


def test_detects_conflicting_relations():
    kg = KnowledgeGraphBuilder()
    kg.add_evidence(_evidence("e1", "DrugA", "increases risk of", "ConditionC", 0.6))
    kg.add_evidence(_evidence("e2", "DrugA", "decreases risk of", "ConditionC", 0.9))

    conflicts = kg.find_conflicting_edges("DrugA", "ConditionC")
    assert len(conflicts) == 2


def test_no_conflict_when_relations_agree():
    kg = KnowledgeGraphBuilder()
    kg.add_evidence(_evidence("e1", "DrugA", "inhibits", "ProteinB", 0.6))
    kg.add_evidence(_evidence("e2", "DrugA", "inhibits", "ProteinB", 0.7))

    conflicts = kg.find_conflicting_edges("DrugA", "ProteinB")
    assert conflicts == []


def test_contradiction_resolver_finds_and_ranks_by_confidence():
    kg = KnowledgeGraphBuilder()
    kg.add_evidence(_evidence("e1", "DrugA", "increases risk of", "ConditionC", 0.4))
    kg.add_evidence(_evidence("e2", "DrugA", "decreases risk of", "ConditionC", 0.9))

    resolver = ContradictionResolver(kg)
    contradictions = resolver.find_all_contradictions()

    assert len(contradictions) == 1
    assert contradictions[0].dominant_relation == "decreases risk of"


def test_contradiction_rate_none_for_empty_graph():
    kg = KnowledgeGraphBuilder()
    resolver = ContradictionResolver(kg)
    assert resolver.contradiction_rate() is None


def test_contradiction_rate_zero_when_no_contradictions_exist():
    kg = KnowledgeGraphBuilder()
    kg.add_evidence(_evidence("e1", "DrugA", "inhibits", "ProteinB", 0.6))
    kg.add_evidence(_evidence("e2", "DrugC", "activates", "ProteinD", 0.7))
    resolver = ContradictionResolver(kg)
    assert resolver.contradiction_rate() == 0.0


def test_contradiction_rate_normalizes_by_total_pairs_not_raw_count():
    kg = KnowledgeGraphBuilder()
    kg.add_evidence(_evidence("e1", "DrugA", "increases risk of", "ConditionC", 0.4))
    kg.add_evidence(_evidence("e2", "DrugA", "decreases risk of", "ConditionC", 0.9))
    kg.add_evidence(_evidence("e3", "DrugX", "inhibits", "ProteinY", 0.5))
    kg.add_evidence(_evidence("e4", "DrugZ", "activates", "ProteinW", 0.5))

    resolver = ContradictionResolver(kg)
    assert resolver.contradiction_rate() == round(1 / 3, 4)
