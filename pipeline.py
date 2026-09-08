from __future__ import annotations

import sys

from biogenesis.evaluation.metrics import automated_metrics, llm_judged_metrics
from biogenesis.logging_utils import get_logger
from biogenesis.orchestration.orchestrator import BioGenesisOrchestrator

logger = get_logger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python pipeline.py "<your research question>"')
        sys.exit(1)

    research_question = " ".join(sys.argv[1:])
    orchestrator = BioGenesisOrchestrator(use_support_checking=True)
    session = orchestrator.run(research_question)

    print("\n" + "=" * 70)
    print(f"RESEARCH QUESTION: {session.research_question}")
    print("=" * 70)
    print(f"\nSub-questions investigated ({len(session.sub_questions)}):")
    for sq in session.sub_questions:
        print(f"  - {sq}")

    print(f"\nPapers retrieved as relevant: {len(session.retrieved_pmids)}")
    print(f"Evidence collected: {len(session.evidence)} items")
    if session.memory_context_used:
        print(f"\nMemory context injected:\n{session.memory_context_used}")

    print(f"\nHYPOTHESES ({len(session.hypotheses)}):\n")
    for h in session.hypotheses:
        print(f"[{h.hypothesis_id}] {h.text}")
        print(f"  Rationale: {h.rationale}")
        print(f"  Supporting evidence: {h.supporting_evidence_ids}")
        print(f"  Generator confidence: {h.confidence}")
        print(f"  Critic verdict: {h.critique_verdict}")
        print(f"  Critique: {h.critique}")
        print()

    auto = automated_metrics(
        session.hypotheses,
        session.evidence,
        orchestrator.kg,
        support_results=session.support_results or None,
    )
    judged = llm_judged_metrics(session.hypotheses)

    print("=" * 70)
    print("AUTOMATED METRICS (deterministic, no LLM judgment)")
    print("=" * 70)
    print(f"  Hypotheses generated:      {auto.num_hypotheses}")
    print(f"  Citation validity rate:    {auto.citation_validity_rate}")
    print(f"  Evidence coverage:         {auto.evidence_coverage}")
    print(f"  Avg. evidence diversity:   {auto.avg_evidence_diversity}")
    print(f"  Contradiction rate:        {auto.contradiction_rate}")
    print(f"  Citation support rate:     {auto.citation_support_rate} (needs SupportChecker)")
    if auto.notes:
        print("  Notes:")
        for note in auto.notes:
            print(f"    - {note}")

    print("\n" + "=" * 70)
    print("LLM-JUDGED METRICS (Critic's opinion, not verified ground truth)")
    print("=" * 70)
    print(f"  Critic approval rate:      {judged.critic_approval_rate}")
    print(f"  Verdict distribution:      {judged.critic_verdict_distribution}")

    if orchestrator.memory is not None:
        orchestrator.memory.save_session(session.research_question, session.hypotheses)

    kg_path = "data/vectorstore/knowledge_graph.json"
    orchestrator.kg.save(kg_path)
    print(f"\nKnowledge graph saved to {kg_path}")
    if orchestrator.memory is not None:
        print("Session saved to persistent memory (data/memory/biogenesis_memory.db)")

    orchestrator.close()


if __name__ == "__main__":
    main()
