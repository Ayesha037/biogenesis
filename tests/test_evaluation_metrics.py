from __future__ import annotations

from biogenesis.agents.hypothesis_generator import Hypothesis
from biogenesis.evidence.models import Evidence
from biogenesis.evidence.support_checker import SupportResult
from biogenesis.knowledge_graph.builder import KnowledgeGraphBuilder
from biogenesis.evaluation.metrics import (
    automated_metrics,
    human_eval_schema,
    llm_judged_metrics,
    retrieval_metrics,
)


def _evidence(eid, pmid, subject="X", relation="affects", obj="Y", confidence=0.5):
    return Evidence(
        evidence_id=eid, pmid=pmid, claim=f"{subject} {relation} {obj}",
        subject=subject, relation=relation, obj=obj, source_text="...", confidence=confidence,
    )


def _hypothesis(hid, evidence_ids, verdict="well-supported"):
    return Hypothesis(
        hypothesis_id=hid, text="some hypothesis", rationale="r",
        supporting_evidence_ids=evidence_ids, confidence=0.7, critique_verdict=verdict,
    )

def test_automated_metrics_empty_hypotheses_returns_none_not_zero():
    result = automated_metrics([], [], KnowledgeGraphBuilder())
    assert result.citation_validity_rate is None
    assert result.evidence_coverage is None
    assert result.contradiction_rate is None
    assert result.notes


def test_automated_metrics_citation_validity():
    evidence = [_evidence("e1", "p1"), _evidence("e2", "p2")]
    hyps = [_hypothesis("h1", ["e1", "e2", "does_not_exist"])]
    result = automated_metrics(hyps, evidence, KnowledgeGraphBuilder())
    assert result.citation_validity_rate == round(2 / 3, 3)


def test_automated_metrics_evidence_coverage():
    evidence = [_evidence("e1", "p1"), _evidence("e2", "p2"), _evidence("e3", "p3")]
    hyps = [_hypothesis("h1", ["e1"])]
    result = automated_metrics(hyps, evidence, KnowledgeGraphBuilder())
    assert result.evidence_coverage == round(1 / 3, 3)


def test_automated_metrics_support_rate_unavailable_when_not_supplied():
    evidence = [_evidence("e1", "p1")]
    hyps = [_hypothesis("h1", ["e1"])]
    result = automated_metrics(hyps, evidence, KnowledgeGraphBuilder())
    assert result.citation_support_rate is None
    assert result.unsupported_claim_rate is None
    assert any("not computed" in n for n in result.notes)


def test_automated_metrics_support_rate_computed_when_supplied():
    evidence = [_evidence("e1", "p1")]
    hyps = [_hypothesis("h1", ["e1"])]
    support_results = {"h1": [SupportResult("e1", supported=True, reason="matches")]}
    result = automated_metrics(hyps, evidence, KnowledgeGraphBuilder(), support_results=support_results)
    assert result.citation_support_rate == 1.0
    assert result.unsupported_claim_rate == 0.0


def test_automated_metrics_contradiction_rate_none_for_empty_graph():
    evidence = [_evidence("e1", "p1")]
    hyps = [_hypothesis("h1", ["e1"])]
    result = automated_metrics(hyps, evidence, KnowledgeGraphBuilder())
    assert result.contradiction_rate is None


def test_automated_metrics_contradiction_rate_computed_for_populated_graph():
    kg = KnowledgeGraphBuilder()
    kg.add_evidence(_evidence("e1", "p1", "DrugA", "increases risk of", "CondB", 0.5))
    kg.add_evidence(_evidence("e2", "p2", "DrugA", "decreases risk of", "CondB", 0.9))
    evidence = [_evidence("e1", "p1"), _evidence("e2", "p2")]
    hyps = [_hypothesis("h1", ["e1"])]
    result = automated_metrics(hyps, evidence, kg)
    assert result.contradiction_rate == 1.0  


def test_llm_judged_metrics_empty():
    result = llm_judged_metrics([])
    assert result.critic_approval_rate is None


def test_llm_judged_metrics_approval_rate():
    hyps = [
        _hypothesis("h1", ["e1"], verdict="well-supported"),
        _hypothesis("h2", ["e1"], verdict="needs-caveats"),
        _hypothesis("h3", ["e1"], verdict="not-supported"),
    ]
    result = llm_judged_metrics(hyps)
    assert result.critic_approval_rate == round(2 / 3, 3)
    assert result.critic_verdict_distribution == {
        "well-supported": 1, "needs-caveats": 1, "not-supported": 1
    }


def test_llm_judged_metrics_excludes_not_evaluated_from_approval_rate():
    hyps = [
        _hypothesis("h1", ["e1"], verdict="well-supported"),
        _hypothesis("h2", ["e1"], verdict="not-evaluated"),
    ]
    result = llm_judged_metrics(hyps)
    assert result.critic_approval_rate == 1.0
    assert any("not critiqued" in n for n in result.notes)


def test_human_eval_schema_never_fabricates_ratings():
    hyps = [_hypothesis("h1", ["e1"])]
    records = human_eval_schema(hyps)
    assert len(records) == 1
    assert all(v is None for v in records[0].ratings.values())
    assert set(records[0].ratings.keys()) == {
        "factual_correctness", "evidence_support", "completeness",
        "scientific_usefulness", "appropriate_uncertainty", "traceability",
    }


def test_retrieval_metrics_unavailable_without_gold_labels():
    result = retrieval_metrics(["p1", "p2", "p3"], gold_relevant_ids=None, k=3)
    assert result.precision_at_k is None
    assert result.recall_at_k is None
    assert result.ndcg_at_k is None
    assert result.notes


def test_retrieval_metrics_computed_with_gold_labels():
    result = retrieval_metrics(
        ["p1", "p2", "p3", "p4"], gold_relevant_ids=["p1", "p3", "p5"], k=4
    )
    assert result.precision_at_k == 0.5 
    assert result.recall_at_k == round(2 / 3, 3) 
    assert result.ndcg_at_k is not None
