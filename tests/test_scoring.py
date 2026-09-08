from biogenesis.evidence.models import Evidence
from biogenesis.evidence.scorer import EvidenceScorer
from biogenesis.retrieval.models import Paper


def _make_evidence(study_type: str) -> Evidence:
    return Evidence(
        evidence_id="e1",
        pmid="1",
        claim="X affects Y",
        subject="X",
        relation="affects",
        obj="Y",
        source_text="...",
        study_type=study_type,
    )


def test_meta_analysis_scores_higher_than_case_study():
    paper = Paper(pmid="1", title="t", abstract="a", pub_date="2023")
    scorer = EvidenceScorer()

    meta = scorer.score(_make_evidence("meta-analysis"), paper)
    case = scorer.score(_make_evidence("case-study"), paper)

    assert meta.confidence > case.confidence


def test_recency_reduces_confidence_for_old_papers():
    scorer = EvidenceScorer()
    old_paper = Paper(pmid="1", title="t", abstract="a", pub_date="1990")
    new_paper = Paper(pmid="2", title="t", abstract="a", pub_date="2024")

    old_score = scorer.score(_make_evidence("cohort"), old_paper).confidence
    new_score = scorer.score(_make_evidence("cohort"), new_paper).confidence

    assert new_score >= old_score


def test_confidence_never_exceeds_one():
    scorer = EvidenceScorer()
    paper = Paper(pmid="1", title="t", abstract="a", pub_date="2024")
    e = scorer.score(_make_evidence("meta-analysis"), paper)
    assert 0.0 <= e.confidence <= 1.0
