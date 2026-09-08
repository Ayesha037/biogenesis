from __future__ import annotations

from dataclasses import dataclass, field

from biogenesis.embeddings.embedder import Embedder
from biogenesis.evidence.extractor import EvidenceExtractor
from biogenesis.evidence.models import Evidence
from biogenesis.evidence.scorer import EvidenceScorer
from biogenesis.llm_client import LLMClient
from biogenesis.logging_utils import get_logger
from biogenesis.preprocessing.chunker import chunk_paper
from biogenesis.preprocessing.cleaner import clean_text
from biogenesis.retrieval.pubmed_client import PubMedClient
from biogenesis.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)


@dataclass
class BaselineResult:
    config_type: str
    question: str
    answer_text: str
    citation_type: str  # "none" | "chunk" | "evidence"
    cited_ids: list[str] = field(default_factory=list)
    retrieved_pmids: list[str] = field(default_factory=list)
    evidence_pool: list[Evidence] = field(default_factory=list)


_LLM_ONLY_SYSTEM = """You are a biomedical research assistant. Answer the \
research question directly using your own knowledge. Be concise (2-4 \
sentences) and state your confidence, but note you have not consulted any \
external literature for this answer."""


def run_llm_only(question: str, llm: LLMClient | None = None) -> BaselineResult:
    """BASELINE 1: Question -> LLM -> Answer. No retrieval at all."""
    llm = llm or LLMClient()
    answer = llm.complete(prompt=question, system=_LLM_ONLY_SYSTEM, temperature=0.3)
    return BaselineResult(
        config_type="llm_only",
        question=question,
        answer_text=answer,
        citation_type="none",
    )


_STANDARD_RAG_SYSTEM = """You are a biomedical research assistant. Answer \
the research question using ONLY the numbered excerpts provided below. \
Cite excerpt numbers in square brackets, e.g. [1], [3], inline in your \
answer. If the excerpts don't contain enough information, say so \
explicitly rather than filling gaps with outside knowledge."""


def run_standard_rag(
    question: str,
    top_k: int = 8,
    max_papers: int = 10,
    pubmed: PubMedClient | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> BaselineResult:
    """
    BASELINE 2: Question -> PubMed retrieval -> top-k relevant chunks -> LLM -> Answer.

    Uses a fresh, isolated Chroma collection per call (not the shared
    persistent one BioGenesis uses) so baseline runs never see evidence
    indexed by other experiment configurations -- each configuration must
    only receive the information its own design specifies.
    """
    pubmed = pubmed or PubMedClient()
    embedder = embedder or Embedder()
    llm = llm or LLMClient()
    vectorstore = ChromaStore(collection_name=f"baseline_rag_{abs(hash(question))}")

    papers = pubmed.search_and_fetch(question, max_results=max_papers)
    for paper in papers:
        paper.abstract = clean_text(paper.abstract)
        chunks = chunk_paper(paper.pmid, paper.title, paper.abstract)
        if chunks:
            embeddings = embedder.embed([c.text for c in chunks])
            vectorstore.add_chunks(chunks, embeddings)

    query_embedding = embedder.embed_one(question)
    hits = vectorstore.query(query_embedding, top_k=top_k)

    excerpt_block = "\n".join(
        f"[{i+1}] (PMID {h['metadata']['pmid']}) {h['text']}" for i, h in enumerate(hits)
    )
    prompt = f"Research question: {question}\n\nExcerpts:\n{excerpt_block if excerpt_block else '(none retrieved)'}"
    answer = llm.complete(prompt=prompt, system=_STANDARD_RAG_SYSTEM, temperature=0.3)

    return BaselineResult(
        config_type="standard_rag",
        question=question,
        answer_text=answer,
        citation_type="chunk",
        cited_ids=[h["chunk_id"] for h in hits],
        retrieved_pmids=list({h["metadata"]["pmid"] for h in hits}),
    )


def run_evidence_aware_rag(
    question: str,
    top_k: int = 8,
    max_papers: int = 10,
    pubmed: PubMedClient | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> BaselineResult:
    """
    BASELINE 3: Question -> PubMed retrieval -> Evidence extraction ->
    Evidence scoring -> LLM -> Answer.

    Reuses EvidenceExtractor, EvidenceScorer, and HypothesisGeneratorAgent
    directly (the same modules the full system uses), but with no Planner
    (single question, no decomposition), no Critic, and no Knowledge Graph
    -- isolating exactly what the evidence extraction + scoring layer adds
    over raw-chunk retrieval (Baseline 2), and what the Planner/Critic/KG
    layers add on top of that (SYSTEM 4/5/6).
    """
    from biogenesis.agents.hypothesis_generator import HypothesisGeneratorAgent

    pubmed = pubmed or PubMedClient()
    embedder = embedder or Embedder()
    llm = llm or LLMClient()
    extractor = EvidenceExtractor(llm=llm)
    scorer = EvidenceScorer()
    generator = HypothesisGeneratorAgent(llm=llm)
    vectorstore = ChromaStore(collection_name=f"baseline_evrag_{abs(hash(question))}")

    papers = pubmed.search_and_fetch(question, max_results=max_papers)
    for paper in papers:
        paper.abstract = clean_text(paper.abstract)
        chunks = chunk_paper(paper.pmid, paper.title, paper.abstract)
        if chunks:
            embeddings = embedder.embed([c.text for c in chunks])
            vectorstore.add_chunks(chunks, embeddings)

    query_embedding = embedder.embed_one(question)
    hits = vectorstore.query(query_embedding, top_k=top_k)
    relevant_pmids = {h["metadata"]["pmid"] for h in hits}
    relevant_papers = [p for p in papers if p.pmid in relevant_pmids]

    evidence: list[Evidence] = []
    for paper in relevant_papers:
        if not paper.abstract:
            continue
        raw = extractor.extract(paper)
        scored = scorer.score_batch(raw, paper)
        evidence.extend(scored)

    hypotheses = generator.generate(question, evidence)
    if hypotheses:
        answer_text = "\n".join(f"{h.text} (confidence={h.confidence})" for h in hypotheses)
        cited_ids = [eid for h in hypotheses for eid in h.supporting_evidence_ids]
    else:
        answer_text = "(no hypotheses generated from available evidence)"
        cited_ids = []

    return BaselineResult(
        config_type="evidence_aware_rag",
        question=question,
        answer_text=answer_text,
        citation_type="evidence",
        cited_ids=cited_ids,
        retrieved_pmids=list(relevant_pmids),
        evidence_pool=evidence,
    )
