from __future__ import annotations

from dataclasses import dataclass, field

from biogenesis.agents.critic import CriticAgent
from biogenesis.agents.hypothesis_generator import Hypothesis, HypothesisGeneratorAgent
from biogenesis.agents.planner import PlannerAgent
from biogenesis.embeddings.embedder import Embedder
from biogenesis.evidence.extractor import EvidenceExtractor
from biogenesis.evidence.models import Evidence
from biogenesis.evidence.scorer import EvidenceScorer
from biogenesis.evidence.support_checker import SupportChecker, SupportResult
from biogenesis.knowledge_graph.builder import KnowledgeGraphBuilder
from biogenesis.logging_utils import get_logger
from biogenesis.memory.store import MemoryStore
from biogenesis.preprocessing.chunker import chunk_paper
from biogenesis.preprocessing.cleaner import clean_text
from biogenesis.retrieval.pubmed_client import PubMedClient
from biogenesis.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)


@dataclass
class ResearchSession:
    research_question: str
    sub_questions: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    retrieved_pmids: list[str] = field(default_factory=list)
    memory_context_used: str = ""
    support_results: dict[str, list[SupportResult]] = field(default_factory=dict)

class BioGenesisOrchestrator:
    def __init__(
        self,
        use_planner: bool = True,
        use_semantic_retrieval: bool = True,
        use_evidence_scoring: bool = True,
        use_knowledge_graph: bool = True,
        use_critic: bool = True,
        use_memory: bool = True,
        use_support_checking: bool = False,
        top_k: int = 8,
        papers_per_subquestion: int = 5,
        pubmed: PubMedClient | None = None,
        embedder: Embedder | None = None,
        vectorstore: ChromaStore | None = None,
        extractor: EvidenceExtractor | None = None,
        scorer: EvidenceScorer | None = None,
        kg: KnowledgeGraphBuilder | None = None,
        planner: PlannerAgent | None = None,
        generator: HypothesisGeneratorAgent | None = None,
        critic: CriticAgent | None = None,
        memory: MemoryStore | None = None,
        support_checker: SupportChecker | None = None,
    ) -> None:

        self.use_planner = use_planner
        self.use_semantic_retrieval = use_semantic_retrieval
        self.use_evidence_scoring = use_evidence_scoring
        self.use_knowledge_graph = use_knowledge_graph
        self.use_critic = use_critic
        self.use_memory = use_memory
        self.use_support_checking = use_support_checking
        self.top_k = top_k
        self.papers_per_subquestion = papers_per_subquestion

        self.pubmed = pubmed or PubMedClient()
        self.embedder = embedder or Embedder()
        self.vectorstore = vectorstore or ChromaStore()
        self.extractor = extractor or EvidenceExtractor()
        self.scorer = scorer or EvidenceScorer()
        self.kg = kg or KnowledgeGraphBuilder()
        self.planner = planner or PlannerAgent()
        self.generator = generator or HypothesisGeneratorAgent()
        self.critic = critic or CriticAgent()
        self.support_checker = support_checker or SupportChecker()
        self.memory = memory if memory is not None else (MemoryStore() if use_memory else None)

    def run(self, research_question: str) -> ResearchSession:
        session = ResearchSession(research_question=research_question)

        if self.use_planner:
            logger.info("=== Planning ===")
            session.sub_questions = self.planner.plan(research_question) or [research_question]
        else:
            session.sub_questions = [research_question]

        logger.info("=== Retrieval & Indexing ===")
        all_papers = {}
        for sq in session.sub_questions:
            papers = self.pubmed.search_and_fetch(sq, max_results=self.papers_per_subquestion)
            for paper in papers:
                paper.abstract = clean_text(paper.abstract)
                all_papers[paper.pmid] = paper
                chunks = chunk_paper(paper.pmid, paper.title, paper.abstract)
                if chunks:
                    embeddings = self.embedder.embed([c.text for c in chunks])
                    self.vectorstore.add_chunks(chunks, embeddings)

        logger.info("Retrieved %d unique papers", len(all_papers))

        if self.use_semantic_retrieval:
            logger.info("=== Semantic Retrieval (top_k=%d) ===", self.top_k)
            relevant_pmids: set[str] = set()
            for sq in session.sub_questions:
                query_embedding = self.embedder.embed_one(sq)
                hits = self.vectorstore.query(query_embedding, top_k=self.top_k)
                for hit in hits:
                    relevant_pmids.add(hit["metadata"]["pmid"])
            unique_papers = [p for pmid, p in all_papers.items() if pmid in relevant_pmids]
            logger.info(
                "Semantic retrieval selected %d/%d papers as relevant",
                len(unique_papers), len(all_papers),
            )
        else:
            unique_papers = list(all_papers.values())

        session.retrieved_pmids = [p.pmid for p in unique_papers]

        logger.info("=== Evidence Extraction & Scoring ===")
        for paper in unique_papers:
            if not paper.abstract:
                continue
            raw_evidence = self.extractor.extract(paper)
            if self.use_evidence_scoring:
                raw_evidence = self.scorer.score_batch(raw_evidence, paper)
            session.evidence.extend(raw_evidence)

        if self.use_knowledge_graph:
            logger.info("=== Knowledge Graph Construction ===")
            self.kg.add_evidence_batch(session.evidence)

        memory_context = ""
        if self.use_memory and self.memory is not None:
            logger.info("=== Memory Retrieval ===")
            related = self.memory.find_related_by_question(research_question)
            if related:
                memory_context = "\n".join(
                    f"- ({r['created_at']}) {r['hypothesis']} [prior verdict: {r['verdict']}]"
                    for r in related
                )
                logger.info("Retrieved %d related past hypotheses from memory", len(related))
        session.memory_context_used = memory_context

        logger.info("=== Hypothesis Generation ===")
        session.hypotheses = self.generator.generate(
            research_question, session.evidence, memory_context=memory_context
        )

        if self.use_critic:
            logger.info("=== Critique ===")
            session.hypotheses = self.critic.critique_batch(session.hypotheses, session.evidence)
        else:
            for h in session.hypotheses:
                h.critique_verdict = "not-evaluated"
                h.critique = "Critic disabled for this run (ablation)."

        if self.use_support_checking:
            logger.info("=== Citation Support Checking ===")
            evidence_by_id = {e.evidence_id: e for e in session.evidence}
            for h in session.hypotheses:
                cited = [
                    evidence_by_id[eid]
                    for eid in h.supporting_evidence_ids
                    if eid in evidence_by_id
                ]
                if cited:
                    session.support_results[h.hypothesis_id] = self.support_checker.check_hypothesis(
                        h.text, cited
                    )
            total_checked = sum(len(v) for v in session.support_results.values())
            logger.info(
                "Checked %d citations across %d hypotheses for support",
                total_checked, len(session.support_results),
            )

        return session

    def close(self) -> None:
        if self.memory is not None:
            self.memory.close()
