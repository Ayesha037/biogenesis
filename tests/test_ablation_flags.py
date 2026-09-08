
from __future__ import annotations

from biogenesis.memory.store import MemoryStore
from biogenesis.orchestration.orchestrator import BioGenesisOrchestrator
from fakes import (
    FakeChromaStore,
    FakeEmbedder,
    FakeLLMClient,
    FakePubMedClient,
    evidence_json_response,
    hypotheses_json_response,
)


def _make_orchestrator(tmp_path, **flags) -> BioGenesisOrchestrator:
    from biogenesis.agents.critic import CriticAgent
    from biogenesis.agents.hypothesis_generator import HypothesisGeneratorAgent
    from biogenesis.agents.planner import PlannerAgent
    from biogenesis.evidence.extractor import EvidenceExtractor
    from biogenesis.evidence.scorer import EvidenceScorer
    from biogenesis.knowledge_graph.builder import KnowledgeGraphBuilder

    planner_llm = FakeLLMClient(default='{"sub_questions": ["fake sub question"]}')
    extractor_llm = FakeLLMClient(
        default=evidence_json_response(
            [
                {
                    "claim": "Drug A reduces Condition B",
                    "subject": "Drug A",
                    "relation": "reduces",
                    "object": "Condition B",
                    "study_type": "cohort",
                }
            ]
        )
    )
    generator_llm = FakeLLMClient(
        default=hypotheses_json_response(
            [
                {
                    "hypothesis": "Drug A may reduce Condition B",
                    "rationale": "supported by cohort evidence",
                    "supporting_evidence_ids": ["1001_ev_0"],
                    "confidence": 0.6,
                }
            ]
        )
    )
    critic_llm = FakeLLMClient(
        default='{"verdict": "needs-caveats", "critique": "reasonable", "suggested_caveats": []}'
    )

    memory = MemoryStore(db_path=tmp_path / "test_memory.db") if flags.get("use_memory", True) else None

    return BioGenesisOrchestrator(
        pubmed=FakePubMedClient(),
        embedder=FakeEmbedder(),
        vectorstore=FakeChromaStore(),
        extractor=EvidenceExtractor(llm=extractor_llm),
        scorer=EvidenceScorer(),
        kg=KnowledgeGraphBuilder(),
        planner=PlannerAgent(llm=planner_llm),
        generator=HypothesisGeneratorAgent(llm=generator_llm),
        critic=CriticAgent(llm=critic_llm),
        memory=memory,
        **flags,
    )


def test_full_system_runs_all_stages(tmp_path):
    orch = _make_orchestrator(tmp_path)
    session = orch.run("does drug a reduce condition b")

    assert session.sub_questions == ["fake sub question"]
    assert len(session.evidence) > 0
    assert orch.kg.graph.number_of_edges() > 0
    assert session.hypotheses[0].critique_verdict == "needs-caveats"
    orch.close()


def test_use_planner_false_skips_decomposition(tmp_path):
    orch = _make_orchestrator(tmp_path, use_planner=False)
    session = orch.run("does drug a reduce condition b")

    assert session.sub_questions == ["does drug a reduce condition b"]
    assert orch.planner.llm.calls == []
    orch.close()


def test_use_critic_false_skips_critique_and_marks_not_evaluated(tmp_path):
    orch = _make_orchestrator(tmp_path, use_critic=False)
    session = orch.run("does drug a reduce condition b")

    assert len(orch.critic.llm.calls) == 0
    assert all(h.critique_verdict == "not-evaluated" for h in session.hypotheses)
    orch.close()


def test_use_knowledge_graph_false_leaves_graph_empty(tmp_path):
    orch = _make_orchestrator(tmp_path, use_knowledge_graph=False)
    session = orch.run("does drug a reduce condition b")

    assert orch.kg.graph.number_of_nodes() == 0
    assert orch.kg.graph.number_of_edges() == 0
    # evidence extraction should still have run
    assert len(session.evidence) > 0
    orch.close()


def test_use_evidence_scoring_false_leaves_default_confidence(tmp_path):
    orch = _make_orchestrator(tmp_path, use_evidence_scoring=False)
    session = orch.run("does drug a reduce condition b")

    assert len(session.evidence) > 0
    assert all(e.confidence == 0.0 for e in session.evidence)
    orch.close()


def test_use_memory_false_skips_retrieval_injection(tmp_path):
    orch = _make_orchestrator(tmp_path, use_memory=False)
    assert orch.memory is None

    session = orch.run("does drug a reduce condition b")
    assert session.memory_context_used == ""
    orch.close()


def test_use_semantic_retrieval_false_uses_all_fetched_papers(tmp_path):
    orch = _make_orchestrator(tmp_path, use_semantic_retrieval=False)
    session = orch.run("does drug a reduce condition b")

    assert session.retrieved_pmids == ["1001"]
    orch.close()


def test_support_checking_off_by_default_produces_no_support_results(tmp_path):
    orch = _make_orchestrator(tmp_path)
    session = orch.run("does drug a reduce condition b")

    assert orch.use_support_checking is False
    assert session.support_results == {}
    orch.close()


def test_support_checking_enabled_checks_every_cited_evidence(tmp_path):
    from biogenesis.evidence.support_checker import SupportChecker

    support_llm = FakeLLMClient(default='{"supported": true, "reason": "entailed"}')
    orch = _make_orchestrator(
        tmp_path,
        use_support_checking=True,
        support_checker=SupportChecker(llm=support_llm),
    )
    session = orch.run("does drug a reduce condition b")

    assert len(session.hypotheses) == 1
    hyp = session.hypotheses[0]
    assert hyp.hypothesis_id in session.support_results
    results = session.support_results[hyp.hypothesis_id]
    assert len(results) == len(hyp.supporting_evidence_ids)
    assert all(r.supported for r in results)

    assert len(support_llm.calls) == len(hyp.supporting_evidence_ids)
    orch.close()


def test_support_checking_skips_hypotheses_with_no_valid_citations(tmp_path):
    from biogenesis.evidence.support_checker import SupportChecker

    support_llm = FakeLLMClient(default='{"supported": false, "reason": "n/a"}')
    generator_llm = FakeLLMClient(
        default=hypotheses_json_response(
            [
                {
                    "hypothesis": "Uncited hypothesis",
                    "rationale": "no evidence cited",
                    "supporting_evidence_ids": [],
                    "confidence": 0.1,
                }
            ]
        )
    )
    from biogenesis.agents.hypothesis_generator import HypothesisGeneratorAgent

    orch = _make_orchestrator(
        tmp_path,
        use_support_checking=True,
        support_checker=SupportChecker(llm=support_llm),
    )
    orch.generator = HypothesisGeneratorAgent(llm=generator_llm)
    session = orch.run("does drug a reduce condition b")

    assert session.support_results == {}
    assert support_llm.calls == []
    orch.close()
