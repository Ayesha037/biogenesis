# 🧬 BioGenesis

**A persistent, evidence-aware multi-agent biomedical research assistant.**

BioGenesis takes a research question, retrieves relevant PubMed literature, extracts and scores structured evidence, builds a knowledge graph, and orchestrates three collaborating LLM agents — **Planner → Hypothesis Generator → Scientific Critic** — to produce hypotheses that are grounded in cited evidence, adversarially reviewed, and persisted across sessions in a local memory store.

It is built as a **research artifact**: every architectural choice is there to be ablated and measured, not just to work.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-65%20passing-brightgreen)](#-testing)
[![Cost](https://img.shields.io/badge/cost-%240%20(free%20tiers%20only)-success)](#-why-0-cost)
[![Status](https://img.shields.io/badge/status-research%20in%20progress-yellow)](#-experimental-status)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#-license)

---

## Table of Contents

- [Why BioGenesis](#-why-biogenesis)
- [Architecture](#-architecture)
- [Repository Layout](#-repository-layout)
- [Getting Started](#-getting-started)
- [Running the Full System](#-running-the-full-system)
- [Testing](#-testing)
- [Research Framework](#-research-framework)
- [Experimental Status](#-experimental-status)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)
- [Why $0 Cost](#-why-0-cost)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Why BioGenesis

Most "chat with your papers" tools stop at retrieval-augmented generation: fetch some chunks, stuff them in a prompt, hope the model doesn't hallucinate. BioGenesis asks a sharper question:

> **Can an evidence-aware multi-agent architecture measurably improve the grounding, traceability, contradiction-handling, and reliability of biomedical research synthesis compared to a plain LLM or a standard RAG pipeline — and can that improvement actually be proven, not just claimed?**

To answer that honestly, BioGenesis is built so that:

- Every component (Planner, semantic retrieval, evidence scoring, knowledge graph, memory, critic) is behind an **ablation flag** on the *same* orchestrator — there's no forked "demo version" vs. "real version" of the pipeline.
- Every metric is labeled by *what kind of claim it is*: **automated** (deterministic, mechanical), **LLM-judged** (a model's opinion, reported as such), and **human** (a template waiting for a person to fill in). They are never blended into one "accuracy" number.
- Nothing is fabricated to make a slide look better: if gold relevance labels don't exist yet, `precision@k` reports `None` with a note — it does not silently become `0.0` or get skipped.

## 🏗️ Architecture

```
research question
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  Planner Agent            decomposes into sub-questions        │  [ablation: use_planner]
└───────────────────────────────────────────────────────────────┘
   │
   ▼
 PubMed retrieval (per sub-question)
   │
   ▼
 Preprocessing → chunking → embeddings → Chroma vector index
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  Semantic retrieval    top-k relevant chunks per sub-question  │  [ablation: use_semantic_retrieval]
└───────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  Evidence extraction   LLM pulls structured claims             │
│  Evidence scoring      study-type hierarchy + recency decay    │  [ablation: use_evidence_scoring]
└───────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  Knowledge graph       NetworkX entity/claim graph              │  [ablation: use_knowledge_graph]
│  Memory retrieval      related past hypotheses injected        │  [ablation: use_memory]
└───────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  Hypothesis Generator  proposes hypotheses, must cite evidence  │
└───────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  Scientific Critic     adversarial review of each hypothesis    │  [ablation: use_critic]
└───────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  Citation support check   entailment: does the citation        │  [opt-in: use_support_checking]
│                            actually support the claim?          │
└───────────────────────────────────────────────────────────────┘
   │
   ▼
 Evaluation (automated + LLM-judged, kept separate)
   │
   ▼
 Persistence → SQLite memory + knowledge-graph JSON
```

Every `[ablation: …]` flag is a constructor argument on `BioGenesisOrchestrator`. The full system and every ablation run through the **exact same code path** — this is what makes the comparisons in the results section meaningful rather than anecdotal.

## 📁 Repository Layout

```
biogenesis/
├── src/biogenesis/
│   ├── config.py               # all settings, loaded from .env
│   ├── llm_client.py           # Groq API wrapper shared by every agent
│   ├── json_utils.py           # defensive JSON parsing/repair for LLM output
│   ├── retrieval/              # PubMed client
│   ├── preprocessing/          # cleaning + chunking
│   ├── embeddings/             # sentence-transformers (lazy-loaded)
│   ├── vectorstore/            # Chroma wrapper
│   ├── evidence/               # extraction, scoring, contradiction resolution,
│   │                           #   citation-support checking
│   ├── knowledge_graph/        # NetworkX graph builder
│   ├── agents/                 # Planner, Hypothesis Generator, Critic
│   ├── orchestration/          # wires all agents together + ablation flags
│   ├── evaluation/             # automated / LLM-judged / human metrics
│   └── memory/                 # persistent SQLite memory + retrieval
├── experiments/
│   ├── benchmark_schema.py     # benchmark item schema + loader
│   ├── benchmark.json          # example benchmark questions (no fabricated gold data)
│   ├── baselines.py            # LLM-only / standard RAG / evidence-aware RAG
│   ├── runner.py               # CLI: run any config against any benchmark
│   ├── configs/*.yaml          # 9 experiment configs (3 baselines + 6 ablations)
│   └── results/                # JSONL output (gitignored)
├── pipeline.py                 # quick manual full-system run
├── tests/                      # 65 offline, fakes-based unit tests
├── data/                       # local vector DB + memory DB
└── .env.example
```

## 🚀 Getting Started

### 1. Create a virtual environment

```bash
git clone https://github.com/Ayesha037/biogenesis.git
cd biogenesis
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

> `sentence-transformers` downloads a small (~80 MB) embedding model on first run. This needs internet access once; every run after that is offline.

### 3. Add your free API keys

```bash
cp .env.example .env
```

| Variable | Where to get it | Required? |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) — free, no card | **Yes** |
| `NCBI_EMAIL` | Your email address | Recommended (NCBI usage policy) |
| `NCBI_API_KEY` | [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) → API Key Management — free | Optional (raises PubMed rate limit 3 → 10 req/sec) |

Everything else — vector DB, knowledge graph, memory — is local and needs no account.

## ▶️ Running the Full System

```bash
python pipeline.py "does metformin reduce cancer risk in diabetic patients"
```

This runs every component end to end: plan → retrieve → embed → semantically filter → extract evidence → score → build the knowledge graph → pull in relevant memory → generate hypotheses → critique → check citation support → evaluate → persist to `data/`.

First run is slower while the embedding model downloads and the vector store populates; subsequent runs reuse the local index.

## 🧪 Testing

```bash
pytest tests/
```

All 65 tests run **offline and deterministically** — every external dependency (LLM, PubMed, embeddings, vector store) is a dependency-injected fake (`tests/fakes.py`), so no API key is needed to verify correctness. Coverage includes PubMed XML parsing, evidence scoring heuristics, knowledge-graph contradiction detection, JSON repair, the citation support checker, the three-way evaluation-metric split, benchmark schema validation, result serialization (with an explicit check that API keys never leak into output), and — critically — a test that every ablation flag **actually disables** its component rather than existing as dead weight.

## 🔬 Research Framework

BioGenesis ships with a full experimental harness so its central claim can be tested, not just asserted.

**Nine configurations, one benchmark, one comparable output schema:**

| Config | Type | Tests |
|---|---|---|
| `llm_only` | Baseline 1 | Floor — no retrieval at all |
| `standard_rag` | Baseline 2 | Raw chunk retrieval, no evidence structure |
| `evidence_aware_rag` | Baseline 3 | + evidence extraction/scoring, no Planner/Critic/KG |
| `full_biogenesis` | Full system | Everything on |
| `no_knowledge_graph` | Ablation | Full minus knowledge graph |
| `no_critic` | Ablation | Full minus critic |
| `no_evidence_scoring` | Ablation | Full minus evidence scoring |
| `no_planner` | Ablation | Full minus sub-question decomposition |
| `no_memory` | Ablation | Full minus memory-augmented reasoning |

Each ablation config differs from `full_biogenesis.yaml` by **exactly one flag** — enforced by a dedicated test so comparisons can't silently become confounded.

```bash
# Single config, full benchmark
python experiments/runner.py --config full_biogenesis --benchmark experiments/benchmark.json

# Compare against a baseline
python experiments/runner.py --config standard_rag --benchmark experiments/benchmark.json

# Full system, also computing citation_support_rate (adds one LLM call per citation)
python experiments/runner.py --config full_biogenesis --benchmark experiments/benchmark.json --check-support
```

Metrics are split into three categories that are **never** blended:

- **Automated** — citation validity, evidence coverage/diversity, contradiction rate, citation support rate. Deterministic.
- **LLM-judged** — Critic approval rate and verdict distribution. Explicitly labeled as model opinion.
- **Human** — an unrated template (`human_eval_schema()`) for a person to score. No rating is ever invented.
- **Retrieval** — precision/recall/nDCG@k, computed only when gold-relevant PMIDs exist for a benchmark item; otherwise reported as `None` with a note, never a fabricated zero.

## 📊 Experimental Status

Live progress on the benchmark run (Groq free-tier rate limits mean this is happening in batches — not a limitation of the architecture, just of a $0 budget):

| Configuration | Progress | Status |
|---|---:|:---:|
| LLM-only (baseline) | 40 / 40 | ✅ Complete |
| Standard RAG (baseline) | 40 / 40 | ✅ Complete |
| Evidence-aware RAG (baseline) | 22 / 40 | 🔄 In progress (18 remaining) |
| Full BioGenesis | 0 / 40 | ⏳ Queued |

> **No comparative performance claim is made anywhere in this repository until the full benchmark run completes.** Once all four configurations finish, results will be published here alongside the automated + LLM-judged metrics, ahead of submission to arXiv. Structural correctness of all 9 configs (including the 6 ablations not shown above) has already been verified offline with fakes — see [Verification Performed](#verification-performed).

### Verification Performed

- **65/65 unit and integration tests passing** — including a check that every ablation flag genuinely disables its component.
- **A real end-to-end attempt** against live PubMed/Groq confirmed the runner fails *safely*: network errors are captured as `status="error"` records with the real traceback, not silently swallowed or faked.
- **An offline structural run** (fake LLM/PubMed responses) executed all 9 configs × multiple questions to confirm the runner, dispatch, ablation flags, and JSONL serialization work end-to-end — this proves the wiring is correct, not the real-world answer quality, which is exactly what the benchmark above is measuring.

## ⚠️ Known Limitations

Being upfront about these, since the research framework's whole point is not overselling results:

- Evidence extraction and hypothesis quality depend on the Groq-hosted model's reasoning — a free 70B-class model is good but not infallible. Treat outputs as a research **aid**, not ground truth.
- Evidence scoring is a transparent heuristic (study-type hierarchy + recency decay), not a learned or validated model.
- Entity resolution is naive: `"Metformin"` and `"metformin hydrochloride"` are currently distinct graph nodes. Real biomedical entity normalization (e.g. UMLS linking) is future work.
- Contradiction resolution *surfaces* conflicts; it does not adjudicate them.
- Memory retrieval is keyword-overlap based, not semantic.
- This is a **research prototype**, not a clinical decision-support tool.

## 🗺️ Roadmap

- [ ] Finish the `full_biogenesis` benchmark run (in progress, Groq free-tier throttled)
- [ ] Expand `experiments/benchmark.json` from 8 to 30–50 questions
- [ ] Source real gold relevance labels for retrieval metrics (precision/recall/nDCG@k)
- [ ] Run `--check-support` at scale for real citation-support numbers
- [ ] Conduct human evaluation using the `human_eval_schema()` template
- [ ] Add biomedical entity normalization (UMLS/MeSH linking) before graph insertion
- [ ] Swap the embedding model for a biomedical-domain one (e.g. PubMedBERT-based)
- [ ] Add full-text retrieval via the PMC Open Access subset, not just abstracts
- [ ] Submit results paper to arXiv

## 💸 Why $0 Cost

Every component runs on a free tier or free open-source software — no paid plan, no credit card, anywhere in the stack:

| Component | Free tier used |
|---|---|
| LLM inference | Groq API (free) |
| Literature retrieval | NCBI PubMed E-utilities (free) |
| Embeddings | `sentence-transformers`, local, open-source |
| Vector store | Chroma, local, open-source |
| Knowledge graph | NetworkX, local, open-source |
| Memory | SQLite, local |

## 📄 Citation

A results paper is in preparation for arXiv once the full benchmark run completes. In the meantime, if you reference this work:

```bibtex
@software{biogenesis2026,
  author = {Ayesha},
  title  = {BioGenesis: An Evidence-Aware Multi-Agent Biomedical Research Assistant},
  year   = {2026},
  url    = {https://github.com/Ayesha037/biogenesis}
}
```

## 📜 License

MIT — see [`LICENSE`](LICENSE).

---

<p align="center">Built as a research artifact, one ablation flag at a time.</p>
