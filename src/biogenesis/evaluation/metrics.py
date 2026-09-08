from __future__ import annotations

import math
from dataclasses import dataclass, field

from biogenesis.agents.hypothesis_generator import Hypothesis
from biogenesis.evidence.contradiction_resolver import ContradictionResolver
from biogenesis.evidence.models import Evidence
from biogenesis.evidence.support_checker import SupportResult
from biogenesis.knowledge_graph.builder import KnowledgeGraphBuilder

_APPROVED_VERDICTS = {"well-supported", "needs-caveats"}
_RATING_CRITERIA = (
    "factual_correctness",
    "evidence_support",
    "completeness",
    "scientific_usefulness",
    "appropriate_uncertainty",
    "traceability",
)


@dataclass
class AutomatedMetrics:
    num_hypotheses: int
    citation_validity_rate: float | None  
    evidence_coverage: float | None            
    avg_evidence_diversity: float | None      
    contradiction_rate: float | None           
    citation_support_rate: float | None = None  
    unsupported_claim_rate: float | None = None  
    notes: list[str] = field(default_factory=list)


def automated_metrics(
    hypotheses: list[Hypothesis],
    evidence: list[Evidence],
    kg: KnowledgeGraphBuilder,
    support_results: dict[str, list[SupportResult]] | None = None,
) -> AutomatedMetrics:
    """
    support_results, if provided, maps hypothesis_id -> list[SupportResult]
    from evidence.support_checker.SupportChecker. Passing this in (rather
    than computing it here) keeps this function's core metrics genuinely
    automated -- the LLM-judged support-checking step is the caller's
    explicit choice to run, and its results are clearly attributed to that
    separate LLM call, not silently baked into "automated_metrics".
    """
    notes: list[str] = []
    n = len(hypotheses)

    if n == 0:
        return AutomatedMetrics(
            num_hypotheses=0,
            citation_validity_rate=None,
            evidence_coverage=None,
            avg_evidence_diversity=None,
            contradiction_rate=None,
            notes=["No hypotheses were generated; all rates are undefined (None), not zero."],
        )

    evidence_by_id = {e.evidence_id: e for e in evidence}

    total_citations = 0
    valid_citations = 0
    cited_evidence_ids: set[str] = set()
    diversity_scores = []
    for h in hypotheses:
        pmids = set()
        for eid in h.supporting_evidence_ids:
            total_citations += 1
            if eid in evidence_by_id:
                valid_citations += 1
                cited_evidence_ids.add(eid)
                pmids.add(evidence_by_id[eid].pmid)
        diversity_scores.append(len(pmids))

    citation_validity_rate = (
        round(valid_citations / total_citations, 3) if total_citations > 0 else None
    )
    if total_citations == 0:
        notes.append("No hypothesis cited any evidence_id; citation_validity_rate is undefined.")

    evidence_coverage = round(len(cited_evidence_ids) / len(evidence), 3) if evidence else None
    if not evidence:
        notes.append("No evidence was extracted; evidence_coverage is undefined.")

    avg_evidence_diversity = round(sum(diversity_scores) / n, 2)

    resolver = ContradictionResolver(kg)
    contradiction_rate = resolver.contradiction_rate()
    if contradiction_rate is None:
        notes.append(
            "Knowledge graph has no edges (either disabled for this run, or no "
            "evidence was extracted); contradiction_rate is undefined, not zero."
        )

    citation_support_rate = None
    unsupported_claim_rate = None
    if support_results is not None:
        all_results = [r for results in support_results.values() for r in results]
        if all_results:
            supported = sum(1 for r in all_results if r.supported)
            citation_support_rate = round(supported / len(all_results), 3)
            unsupported_claim_rate = round(1 - citation_support_rate, 3)
        else:
            notes.append("support_results was provided but empty; support rates are undefined.")
    else:
        notes.append(
            "citation_support_rate/unsupported_claim_rate not computed for this run "
            "(no SupportChecker results supplied) -- reported as unavailable, not zero."
        )

    return AutomatedMetrics(
        num_hypotheses=n,
        citation_validity_rate=citation_validity_rate,
        evidence_coverage=evidence_coverage,
        avg_evidence_diversity=avg_evidence_diversity,
        contradiction_rate=contradiction_rate,
        citation_support_rate=citation_support_rate,
        unsupported_claim_rate=unsupported_claim_rate,
        notes=notes,
    )


@dataclass
class LLMJudgedMetrics:
    critic_approval_rate: float | None
    critic_verdict_distribution: dict[str, int]
    notes: list[str] = field(default_factory=list)


def llm_judged_metrics(hypotheses: list[Hypothesis]) -> LLMJudgedMetrics:
    n = len(hypotheses)
    if n == 0:
        return LLMJudgedMetrics(
            critic_approval_rate=None,
            critic_verdict_distribution={},
            notes=["No hypotheses were generated."],
        )

    distribution: dict[str, int] = {}
    for h in hypotheses:
        distribution[h.critique_verdict] = distribution.get(h.critique_verdict, 0) + 1

    notes = []
    if "not-evaluated" in distribution:
        notes.append(
            f"{distribution['not-evaluated']}/{n} hypotheses were not critiqued "
            f"(Critic disabled for this run) -- excluded from approval rate."
        )

    evaluated = [h for h in hypotheses if h.critique_verdict != "not-evaluated"]
    approval_rate = (
        round(sum(1 for h in evaluated if h.critique_verdict in _APPROVED_VERDICTS) / len(evaluated), 3)
        if evaluated
        else None
    )

    notes.append(
        "critic_approval_rate reflects the Critic LLM's own judgment, not a "
        "verified ground truth -- treat as a model opinion, not fact."
    )

    return LLMJudgedMetrics(
        critic_approval_rate=approval_rate,
        critic_verdict_distribution=distribution,
        notes=notes,
    )



@dataclass
class HumanEvalRecord:
    hypothesis_id: str
    ratings: dict[str, int | None]  # each criterion -> 1-5, or None if not yet rated
    rater_notes: str = ""


def human_eval_schema(hypotheses: list[Hypothesis]) -> list[HumanEvalRecord]:
    """
    Returns one unrated record per hypothesis, with every criterion set to
    None. This is a template for a human rater to fill in -- it is NOT a
    computed result and must never be reported as one. Criteria use a
    consistent 1-5 scale per the research plan: factual_correctness,
    evidence_support, completeness, scientific_usefulness,
    appropriate_uncertainty, traceability.
    """
    return [
        HumanEvalRecord(
            hypothesis_id=h.hypothesis_id,
            ratings={criterion: None for criterion in _RATING_CRITERIA},
        )
        for h in hypotheses
    ]



@dataclass
class RetrievalMetrics:
    precision_at_k: float | None
    recall_at_k: float | None
    ndcg_at_k: float | None
    k: int
    notes: list[str] = field(default_factory=list)


def retrieval_metrics(
    retrieved_ids: list[str], gold_relevant_ids: list[str] | None, k: int
) -> RetrievalMetrics:
    """
    gold_relevant_ids must come from actual human relevance judgments for a
    benchmark question. If none are supplied, every metric is explicitly
    reported as unavailable (None) rather than fabricated -- see
    experiments/benchmark_schema.py for the mechanism to attach human
    relevance labels to a benchmark item.
    """
    if gold_relevant_ids is None:
        return RetrievalMetrics(
            precision_at_k=None,
            recall_at_k=None,
            ndcg_at_k=None,
            k=k,
            notes=[
                "No gold relevance labels available for this question -- "
                "precision@k/recall@k/ndcg@k are unavailable, not zero. "
                "Provide gold_relevant_ids via human annotation to compute these."
            ],
        )

    top_k_retrieved = retrieved_ids[:k]
    gold_set = set(gold_relevant_ids)

    if not top_k_retrieved:
        return RetrievalMetrics(0.0, 0.0, 0.0, k, notes=["No items were retrieved."])

    hits = [1 if rid in gold_set else 0 for rid in top_k_retrieved]
    precision = sum(hits) / len(top_k_retrieved)
    recall = sum(hits) / len(gold_set) if gold_set else None

    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = sorted(hits, reverse=True)
    idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal_hits))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return RetrievalMetrics(
        precision_at_k=round(precision, 3),
        recall_at_k=round(recall, 3) if recall is not None else None,
        ndcg_at_k=round(ndcg, 3),
        k=k,
    )
