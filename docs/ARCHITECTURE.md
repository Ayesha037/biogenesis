# BioGenesis Architecture

## Why not a standard RAG chatbot?

A standard RAG system retrieves text similar to a query and asks an LLM to
summarize it. BioGenesis differs in three structural ways:

1. **Evidence, not text, is the unit of reasoning.** Every claim extracted
   from a paper is a structured object (`Evidence`) with a subject,
   relation, object, study type, and confidence score — not a blob of
   retrieved text. This lets downstream components (the knowledge graph,
   the agents) reason about *claims*, not paragraphs.

2. **Multiple specialized agents, not one prompt.** A Planner decomposes
   the question, a Generator proposes hypotheses strictly from retrieved
   evidence, and a Critic adversarially checks those hypotheses. Splitting
   these roles across separate LLM calls with separate prompts produces
   more reliable behavior than asking one model to plan, generate, and
   self-critique in a single pass — models are poor at grading their own
   uncorrected output.

3. **Persistent memory across sessions.** Past hypotheses and their critic
   verdicts are saved to a local SQLite database. `MemoryStore.find_related_by_question`
   is called before hypothesis generation and its results are injected
   into the generator's prompt (`memory_context`, gated by `use_memory` —
   see `orchestration/orchestrator.py`), so this is genuine
   memory-augmented reasoning, not just persistence. The retrieval itself
   is naive keyword overlap, not semantic search — see "Known limitations"
   in the README.

## Data flow

```
research question
      │
      ▼
 PlannerAgent  ──────►  sub-questions                        [ablation: use_planner]
      │
      ▼
 PubMedClient  ──────►  Paper objects (title, abstract, metadata)
      │
      ▼
 cleaner + chunker  ──►  Chunk objects
      │
      ▼
 Embedder  ──────────►  vectors  ──────►  ChromaStore (persisted)
      │
      ▼
 Semantic retrieval ──►  top-k relevant chunks per sub-question,       [ablation: use_semantic_retrieval]
                          narrowed down to their source papers
      │
      ▼
 EvidenceExtractor (LLM) ──►  Evidence objects (claim, subject, relation, object)
      │
      ▼
 EvidenceScorer  ──────►  Evidence.confidence filled in                [ablation: use_evidence_scoring]
      │
      ▼
 KnowledgeGraphBuilder ──►  NetworkX MultiDiGraph (persisted as JSON)   [ablation: use_knowledge_graph]
      │
      ▼
 MemoryStore.find_related_by_question ──► past hypotheses injected     [ablation: use_memory]
                                            into the generator's prompt
      │
      ▼
 HypothesisGeneratorAgent (LLM) ──►  Hypothesis objects, citing evidence_ids
      │
      ▼
 CriticAgent (LLM) ──────►  Hypothesis.critique_verdict, Hypothesis.critique  [ablation: use_critic]
      │
      ▼
 SupportChecker (LLM, opt-in) ──►  per-citation entailment check,      [opt-in: use_support_checking]
                                    independent of the Critic's verdict
      │
      ▼
 evaluation/metrics.py ──►  automated_metrics() / llm_judged_metrics() /
                             human_eval_schema() / retrieval_metrics(),
                             each explicitly labeled by evidence category
      │
      ▼
 MemoryStore.save_session ──►  this session persisted to SQLite
                                (only happens if use_memory=True)
```

## Why these specific free tools

| Concern | Tool | Why |
|---|---|---|
| LLM reasoning | Groq API (free tier) | No local GPU needed; fast; open models; no credit card |
| Literature | NCBI E-utilities | The standard, sanctioned, free API for PubMed |
| Embeddings | sentence-transformers (local, CPU) | Free, small, no API cost, no network dependency once downloaded |
| Vector search | Chroma | Embedded, no server, persists to disk, free |
| Graph | NetworkX | Pure Python, in-memory, zero setup, free |
| Memory | SQLite | Built into Python, zero setup, free |

Every one of these can be swapped later (e.g. Groq → a local Ollama model,
Chroma → Qdrant, NetworkX → Neo4j) without touching any other module,
because each is wrapped behind a small class (`LLMClient`, `ChromaStore`,
`KnowledgeGraphBuilder`) that the rest of the codebase depends on instead
of the underlying library directly.

## Where hallucination risk is architecturally reduced (not eliminated)

- The Generator only ever sees evidence actually retrieved and extracted —
  it cannot cite a paper it wasn't shown.
- Every hypothesis must list which `evidence_id`s it used.
- Two independent checks then look at those citations, and are kept
  deliberately separate rather than blended into one score:
  - The **Critic** (`agents/critic.py`) judges the hypothesis holistically
    — evidence strength, overreach, missing caveats — via a single
    adversarial LLM call per hypothesis.
  - The **SupportChecker** (`evidence/support_checker.py`, opt-in via
    `use_support_checking`) asks a narrower, mechanical question per
    citation: does this specific evidence's source text actually entail
    this specific claim? This is what `citation_support_rate` /
    `unsupported_claim_rate` in `evaluation/metrics.py` are built from.
- The evaluation framework (`evaluation/metrics.py`) tracks these as
  distinct, labeled categories (automated / LLM-judged / human) over time,
  so a change that makes the system cite evidence less reliably is
  measurable, not just anecdotal.

This does not make the system immune to hallucination — an LLM can still
misread a paper's abstract, and both the Critic and SupportChecker are
themselves LLM judgments, not verified ground truth — but it makes errors
checkable rather than invisible.
