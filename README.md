# BioGenesis

A persistent, evidence-aware multi-agent biomedical AI research assistant.

BioGenesis takes a research question, retrieves relevant PubMed literature,
extracts and scores structured evidence, builds a knowledge graph, and uses
three collaborating agents (Planner -> Hypothesis Generator -> Scientific
Critic) to produce evidence-grounded, critiqued hypotheses -- with results
persisted across sessions in a local memory database.

**Cost: $0.** Every component runs on free tiers or free open-source
software. Nothing requires a paid plan or a credit card.

---

## 1. What's inside

```
biogenesis/
├── src/biogenesis/
│   ├── config.py                  # all settings, reads from .env
│   ├── logging_utils.py           # shared logger
│   ├── llm_client.py              # Groq API wrapper (shared by all agents)
│   ├── json_utils.py              # defensive JSON parsing/repair for LLM output
│   ├── retrieval/                 # PubMed client
│   ├── preprocessing/             # cleaning + chunking
│   ├── embeddings/                # sentence-transformers (lazy-loaded)
│   ├── vectorstore/                # Chroma wrapper
│   ├── evidence/                  # extraction, scoring, contradiction
│   │                                 resolution, citation support checking
│   ├── knowledge_graph/           # NetworkX graph builder
│   ├── agents/                    # Planner, Hypothesis Generator, Critic
│   ├── orchestration/             # wires all agents together, ablation flags
│   ├── evaluation/                # automated / LLM-judged / human metrics
│   └── memory/                    # persistent SQLite memory + retrieval
├── experiments/                   # reproducible research framework
│   ├── benchmark_schema.py        # benchmark item schema + loader
│   ├── benchmark.json             # example benchmark questions (no fabricated gold data)
│   ├── baselines.py               # LLM-only / standard RAG / evidence-aware RAG
│   ├── runner.py                  # CLI: run any config against any benchmark
│   ├── configs/*.yaml             # 9 experiment configs (3 baselines + 6 ablations)
│   └── results/                   # JSONL output (gitignored)
├── pipeline.py                    # quick manual full-system run
├── tests/                         # unit tests (offline, fakes-based, no API key needed)
├── data/                          # local storage: vector DB, memory DB
├── .env.example                   # <-- copy this to .env and fill in keys
└── requirements.txt
```

---

## 2. Setup (one-time)

### Step 1 — Create a virtual environment

```bash
cd biogenesis
python3 -m venv .venv
source .venv/bin/activate     
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

> Note: `sentence-transformers` will download a small (~80MB) model the
> first time it runs. This requires internet access once, then works
> offline.

### Step 3 — Add your free API keys  ⚠️ THIS IS WHERE YOU NEED TO ACT

```bash
cp .env.example .env
```

Then open `.env` in any text editor and fill in:

| Variable | Where to get it | Required? |
|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys — free sign-up, no credit card | **Yes**, agents won't run without it |
| `NCBI_EMAIL` | Just your email address | Recommended (NCBI usage policy) |
| `NCBI_API_KEY` | https://www.ncbi.nlm.nih.gov/account/ → Settings → API Key Management — free | Optional, raises PubMed rate limit from 3→10 req/sec |

That's the **only** manual setup step. Everything else (vector DB,
knowledge graph, memory) is local and needs no account.

---

## 3. Run it (quick manual sanity check)

```bash
python pipeline.py "does metformin reduce cancer risk in diabetic patients"
```

This runs the **full system** (every component enabled) end-to-end:
1. **Plan** — breaks your question into sub-questions
2. **Retrieve** — searches PubMed for each sub-question
3. **Preprocess & embed** — cleans text, chunks it, embeds it, stores it in Chroma
4. **Semantic retrieval** — embeds each sub-question and queries Chroma for its top-k most relevant chunks; only papers those chunks came from proceed
5. **Extract evidence** — pulls structured claims out of each relevant paper
6. **Score evidence** — weights claims by study type + recency
7. **Build knowledge graph** — links entities via NetworkX
8. **Memory retrieval** — pulls related past hypotheses into context for the generator
9. **Generate hypotheses** — proposes evidence-grounded hypotheses
10. **Critique** — an adversarial agent checks each hypothesis
11. **Check citation support** — a separate, narrower LLM pass verifies each cited evidence item's own text actually entails the claim (enabled for this manual run; opt-in elsewhere — see Section 7.3)
12. **Evaluate** — prints automated metrics (deterministic) and LLM-judged metrics (Critic's opinion), kept clearly separate
13. **Persist** — saves everything to `data/memory/biogenesis_memory.db` and `data/vectorstore/knowledge_graph.json`

First run will be slower (downloading the embedding model, populating the
vector store). Subsequent runs reuse the local Chroma index.

For running specific ablations, baselines, or a benchmark of questions
instead of one-off manual runs, see **Section 8, Running experiments**.

---

## 4. Run the tests

Tests that check pure logic (no API key needed) live separately from
anything requiring Groq -- all external dependencies (LLM, PubMed,
embeddings, vector store) are injected fakes in these tests (see
`tests/fakes.py`), so the full suite runs instantly, offline, and
deterministically:

```bash
pytest tests/
```

65 tests cover: PubMed XML parsing, text cleaning/chunking, evidence
scoring heuristics, knowledge graph contradiction detection (including the
normalized contradiction rate), JSON repair, the citation support checker
(both standalone and wired into the orchestrator/runner), the three-way evaluation metrics split (automated / LLM-judged / human
schema), benchmark schema validation, experiment config loading and
dispatch switching, result serialization (including a check that API keys
never leak into output), and -- critically -- that every ablation flag
**actually disables** its component rather than just existing as an
unused parameter.

---

## 5. Milestone map (original build)

| # | Milestone | Module |
|---|---|---|
| 1 | Architecture & setup | `config.py`, `logging_utils.py` |
| 2 | PubMed retrieval | `retrieval/` |
| 3 | Preprocessing | `preprocessing/` |
| 4 | Embeddings | `embeddings/` |
| 5 | Vector database | `vectorstore/` |
| 6 | Evidence extraction | `evidence/extractor.py` |
| 7 | Evidence scoring | `evidence/scorer.py` |
| 8 | Knowledge graph | `knowledge_graph/` |
| 9 | Research Planner Agent | `agents/planner.py` |
| 10 | Hypothesis Generation Agent | `agents/hypothesis_generator.py` |
| 11 | Scientific Critic Agent | `agents/critic.py` |
| 12 | Multi-agent orchestration | `orchestration/orchestrator.py` |
| 13 | Evaluation framework | `evaluation/metrics.py` |
| 14 | Scientific memory | `memory/store.py` |
| 15 | Contradiction resolution | `evidence/contradiction_resolver.py` |
| 16 | Complete pipeline | `pipeline.py` |

See `docs/ARCHITECTURE.md` for the reasoning behind each original design decision.

---

## 6. Known limitations (be honest with yourself about these)

- Evidence extraction and hypothesis generation quality depend entirely on
  the Groq-hosted model's reasoning — a free 70B-class model is good but
  not infallible; always treat outputs as a research **aid**, not ground truth.
- Evidence scoring is a transparent heuristic (study-type hierarchy +
  recency decay), not a learned/validated model.
- Entity resolution is naive: "Metformin" and "metformin hydrochloride"
  are currently treated as different graph nodes. A real research-grade
  system would need biomedical entity normalization (e.g. UMLS linking).
- Contradiction resolution surfaces conflicts; it does not adjudicate them.
- Memory retrieval is naive keyword overlap, not semantic search.
- This is a research prototype, not a clinical decision-support tool.

---

## 7. Research framework

### 7.1 Research question

*Can an evidence-aware multi-agent architecture improve the grounding,
traceability, contradiction handling, and reliability of biomedical
research synthesis compared with simpler LLM/RAG approaches?*

### 7.2 Current architecture

```
research question
  → Planner (decomposes into sub-questions)                [ablation: use_planner]
  → PubMed retrieval (per sub-question)
  → Preprocessing (clean + chunk)
  → Embeddings → Chroma indexing
  → Semantic retrieval (top-k relevant chunks per sub-q)    [ablation: use_semantic_retrieval]
  → Evidence extraction (LLM, structured claims)
  → Evidence scoring (study-type + recency heuristic)       [ablation: use_evidence_scoring]
  → Knowledge graph construction (NetworkX)                 [ablation: use_knowledge_graph]
  → Memory retrieval (past related hypotheses injected)     [ablation: use_memory]
  → Hypothesis generation (LLM, must cite evidence_ids)
  → Scientific Critic (LLM, adversarial review)              [ablation: use_critic]
  → Citation support checking (LLM, per-citation entailment) [opt-in: use_support_checking]
  → Evaluation (automated + LLM-judged, kept separate)
  → Memory persistence (only if use_memory=True — see the table below)
```

Every `[ablation: ...]` flag is a constructor argument on
`BioGenesisOrchestrator` (`src/biogenesis/orchestration/orchestrator.py`),
so the *same code path* runs for the full system and every ablation —
there is no separate forked copy of the pipeline per configuration.
`use_support_checking` is a separate opt-in flag, not one of the six
True-by-default ablation flags, because it adds real LLM cost proportional
to citation count (see Section 7.3).

**What changed from the original build, and why:**

| Problem found | Fix |
|---|---|
| Chroma was populated but never queried — evidence extraction ran over every retrieved paper regardless of relevance | Real per-sub-question semantic retrieval now narrows to top-k relevant papers before extraction |
| "Grounding" meant only "has an evidence_id" | `evidence/support_checker.py` adds a separate, explicit citation→claim entailment check (LLM-judged, reported as such — not folded into automated metrics) |
| `SupportChecker` existed and was unit-tested but was never called from any real run, so `citation_support_rate` was always unavailable | Wired into `BioGenesisOrchestrator` (`use_support_checking`, opt-in), `pipeline.py` (on by default for the manual run), and `experiments/runner.py` (`--check-support` / per-config `use_support_checking:`) |
| Contradictions reported as a raw, non-comparable count | `ContradictionResolver.contradiction_rate()` normalizes by total subject/object pairs; returns `None` (not `0.0`) when there's nothing to check |
| Critic's LLM opinion was blended into the same metric namespace as mechanical checks | `evaluation/metrics.py` split into `automated_metrics()` / `llm_judged_metrics()` / `human_eval_schema()`, each explicitly labeled |
| Memory was persisted but never read back into reasoning | `MemoryStore.find_related_by_question()` is now called before hypothesis generation and injected into the generator's prompt, gated by `use_memory` |
| `use_memory=False` disables reasoning-injection, but docs implied persistence still ran in the background | Corrected: with `use_memory=False` the orchestrator never constructs a `MemoryStore`, so persistence is skipped too — a genuine "memory absent" ablation, not just "memory not consulted" |

### 7.3 Experimental configurations

Nine configurations, all consuming the **same benchmark**, all producing a
comparable output record (see `experiments/runner.py::run_one_item`):

| Config | Type | What it tests |
|---|---|---|
| `llm_only` | Baseline 1 | Floor: no retrieval at all |
| `standard_rag` | Baseline 2 | Raw chunk retrieval, no evidence structure |
| `evidence_aware_rag` | Baseline 3 | + evidence extraction/scoring, no Planner/Critic/KG |
| `full_biogenesis` | System 6 | Everything on |
| `no_knowledge_graph` | System 4 | Full minus knowledge graph |
| `no_critic` | System 5 | Full minus critic |
| `no_evidence_scoring` | Ablation | Full minus evidence scoring |
| `no_planner` | Ablation | Full minus sub-question decomposition |
| `no_memory` | Ablation | Full minus memory-augmented reasoning |

Each ablation config differs from `full_biogenesis.yaml` by **exactly one
flag** — enforced by a test (`test_ablation_configs_differ_from_full_by_exactly_one_flag`)
so ablations can't silently drift into confounded comparisons.

Configs live in `experiments/configs/*.yaml`. Baselines 1–3 are separate
functions in `experiments/baselines.py` (not orchestrator flags), because
they have no Planner/Critic/KG concept at all — see the module docstring
for why they're wired this way rather than duplicating the codebase.

### 7.4 Benchmark format

`experiments/benchmark.json` — currently **8 example questions**, not the
full 30–50 target. Schema in `experiments/benchmark_schema.py`. Fields:
`id`, `question`, `category` (one of 8 categories: drug→disease/outcome,
treatment→outcome, gene→disease, biomarker→disease, drug→adverse event,
mechanism, association/risk, conflicting evidence), optional `sub_questions`,
`expected_evidence_characteristics`, `gold_relevant_pmids`,
`gold_reference`. **No gold answers or relevance labels are fabricated** —
those fields are simply absent until genuinely sourced (e.g. human
annotation), and the evaluation code explicitly reports the corresponding
metrics as unavailable rather than substituting invented data.

### 7.5 Evaluation metrics

`src/biogenesis/evaluation/metrics.py`, split into three categories that
are never blended:

- **Automated** (`automated_metrics`) — citation validity rate, evidence
  coverage, evidence diversity, contradiction rate, and (if a
  `SupportChecker` pass was run) citation support rate / unsupported claim
  rate. Deterministic given the same inputs.
- **LLM-judged** (`llm_judged_metrics`) — Critic approval rate and verdict
  distribution. Explicitly labeled as a model opinion, not ground truth.
- **Human** (`human_eval_schema`) — returns an *unrated* template
  (factual_correctness, evidence_support, completeness,
  scientific_usefulness, appropriate_uncertainty, traceability, each 1–5)
  for a person to fill in. No ratings are ever fabricated.
- **Retrieval** (`retrieval_metrics`) — precision@k / recall@k / ndcg@k,
  computed only when `gold_relevant_pmids` are supplied for a benchmark
  item; otherwise every field is `None` with an explanatory note, not a
  fabricated zero.

### 7.6 Ablation studies

See 7.3 — six configurations (full + five single-component removals) plus
the semantic-retrieval flag, all sharing one code path.

### 7.7 Running experiments

```bash
# Single config, full benchmark
python experiments/runner.py --config full_biogenesis --benchmark experiments/benchmark.json

# Compare against a baseline
python experiments/runner.py --config standard_rag --benchmark experiments/benchmark.json

# An ablation, limited to the first 3 questions
python experiments/runner.py --config no_knowledge_graph --benchmark experiments/benchmark.json --limit 3

# Full system, also computing citation_support_rate (adds one LLM call
# per hypothesis citation -- see Section 7.3)
python experiments/runner.py --config full_biogenesis --benchmark experiments/benchmark.json --check-support
```

Results are written as JSONL to `experiments/results/<config>__<timestamp>.jsonl`
(gitignored — not committed). Each line contains: config name/type,
benchmark question id, timestamp, seed, retrieved paper/evidence IDs,
generated hypotheses with citations, critic output, contradiction/automated/
LLM-judged metrics, and the model name — **never an API key**. If an item
fails (network error, unparseable output, etc.), its record has
`status="error"` with the exception and traceback; the run continues to
the next item rather than crashing or fabricating a result.

### 7.8 What results are — and are not — currently available

**Available:** the framework itself, runnable today against real PubMed/Groq
with your own API key. Automated metrics compute correctly. The offline
structural correctness of all 9 configs has been verified (see Section 9).

**Not yet available:**
- No experiment has been run against real literature at benchmark scale —
  the 8-question example benchmark needs expanding to 30–50 before any
  comparative claim about BioGenesis vs. baselines can be made.
- No human relevance labels exist yet, so `precision@k` / `recall@k` /
  `ndcg@k` report as unavailable for every current benchmark item.
- No human evaluation has been performed — `human_eval_schema()` produces
  only unrated templates.
- No citation-support-rate numbers exist yet from a *real* run — `SupportChecker`
  is now wired into the orchestrator/pipeline/runner (`use_support_checking`
  / `--check-support`), and is exercised end-to-end in the offline
  structural run (Section 9), but no one has run it against real
  PubMed/Groq output yet. It remains an explicit opt-in, not automatic, to
  keep its added LLM cost visible.
- **No comparative claim about BioGenesis's performance vs. the baselines
  is made anywhere in this repository.** That requires actually running
  the benchmark, which requires your Groq/NCBI credentials and real
  literature access.

### 7.9 Reproducibility

- `--seed` on the runner sets Python's `random` seed per item (LLM sampling
  itself has its own temperature-driven variance not fully eliminable via
  seeding — this is noted, not hidden).
- Every result record embeds its exact config and model name.
- Config files are plain YAML, diffable, and version-controllable.
- `.gitignore` excludes `.env`, `.venv/`, `__pycache__/`, `*.pyc`,
  `experiments/results/`, and generated vector-store/memory data —
  nothing environment-specific or generated is meant to be committed.

---

## 8. Natural next steps

- Expand `experiments/benchmark.json` to 30–50 questions
- Source real gold relevance labels (human annotation) for retrieval metrics
- Run `--check-support` against real literature to get real citation-support numbers (the wiring is done — see 7.8)
- Run actual human evaluation using the `human_eval_schema()` template
- Add biomedical entity normalization (UMLS/MeSH ID linking) before graph insertion
- Swap the embedding model for a biomedical-domain one (e.g. PubMedBERT-based)
- Add full-text retrieval via the PMC Open Access subset, not just abstracts

## 9. Verification performed so far

- **65 unit/integration tests pass**, covering: PubMed XML parsing,
  preprocessing, evidence scoring, knowledge graph + contradiction rate,
  JSON repair, the support checker (standalone, and wired into the
  orchestrator's opt-in `use_support_checking` step and the runner's
  `--check-support` flag), all three evaluation-metric categories,
  benchmark schema validation, config loading, dispatch switching across
  all 9 configuration types, result serialization (incl. no-API-key-leak
  check), and — specifically — that every ablation flag actually disables
  its component (not just exists as an unused parameter).
- **A real end-to-end attempt** was made against live PubMed/Groq from the
  development sandbox; it correctly failed due to no network access to
  those services from that environment, and the runner correctly recorded
  this as a `status="error"` record with the real exception and traceback,
  rather than crashing or silently succeeding. **You should run the real
  thing yourself** with your own `.env` credentials (Section 2, Step 3) —
  the command is in Section 7.7.
- **An offline structural run** (dependency-injected fake LLM/PubMed
  responses, clearly not real literature) was executed across all 9
  configs × 2 questions to confirm the full runner pipeline — config
  loading, dispatch, ablation flags, metric computation, JSONL
  serialization — works correctly end-to-end. This proves the *wiring* is
  correct; it says nothing about real-world answer quality, which can only
  be assessed by running against real PubMed/Groq.
