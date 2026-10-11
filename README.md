# BioGenesis 🧬

**A biomedical AI that has to show its evidence, built so that every design choice can be measured.**

![Status](https://img.shields.io/badge/status-research%20prototype-orange) ![License](https://img.shields.io/badge/license-MIT-blue)

> **Research prototype. Not a clinical tool.** Outputs are suggestions from the literature for researchers to check. They are not medical advice.

---

## The short version

Ask BioGenesis a research question and it searches PubMed, pulls out the evidence, builds hypotheses, and has to **cite real papers** for each one. Then a critic argues against it, and a checker looks at whether each cited paper really supports the claim.

I didn't just build it. I built a test harness around it, so I can switch each part on and off and see what actually helps. Some parts might turn out to help less than I hoped, and I'll publish that too.

---

## Why this exists

Most "chat with papers" tools do the same thing: grab a few chunks of text, put them in a prompt, and hope the model doesn't make things up. BioGenesis asks a more specific question:

> **Does adding structure to the evidence (scored claims, a knowledge graph, a critic, citation checking) make biomedical hypotheses more reliable than a plain LLM or standard RAG?**

To find out, the project has a pipeline and an evaluation setup: **9 configurations** (3 baselines, the full system, 5 ablations) that share one code path, a **40-question benchmark**, and metrics that are reported separately and never mixed into one score.

*(An "ablation" just means removing one part to see what changes.)*

---

## What it does

Say you ask: *"What evidence exists that metformin reduces colorectal cancer risk in people with type 2 diabetes?"*

```
Question → Planner → PubMed retrieval → Evidence scoring → Knowledge graph
        → Hypotheses that must cite papers → Critic → Citation-support check → Memory
```

1. **Plan.** Break the question into smaller sub-questions.
2. **Retrieve.** Search PubMed for each one.
3. **Ground.** Split and embed the abstracts, pull out structured evidence claims, and score them (study design plus how recent).
4. **Link.** Build a knowledge graph of entities and claims, and flag contradictions.
5. **Generate.** Propose hypotheses that must cite specific papers (by PMID).
6. **Critique.** A second agent argues against each hypothesis.
7. **Verify (optional).** Check whether each cited abstract supports the claim it's attached to.
8. **Remember.** Save hypotheses in a local SQLite memory for later questions.

Steps 1, 3, 4, 6 and 8 each have a single on/off flag, so I can isolate what each one does. The full system and every ablation run through the same orchestrator. There's no special demo path, and a test checks that each ablation differs from `full_biogenesis` by exactly one flag.

Design notes are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## The experiment

| Config | Role | What it tests |
|---|---|---|
| `llm_only` | Baseline | No retrieval at all |
| `standard_rag` | Baseline | Raw text chunks, no evidence structure |
| `evidence_aware_rag` | Baseline | Adds evidence extraction and scoring |
| `full_biogenesis` | Full system | Everything on |
| `no_planner` | Ablation | No sub-question breakdown |
| `no_evidence_scoring` | Ablation | No evidence scoring |
| `no_knowledge_graph` | Ablation | No knowledge graph |
| `no_memory` | Ablation | No memory |
| `no_critic` | Ablation | No critic |

**Benchmark.** 40 questions in 8 categories of 5: drug-disease outcome, treatment outcome, gene-disease, biomarker-disease, drug adverse event, mechanism, association/risk, and conflicting evidence (`experiments/benchmark.json`). It contains questions only. I didn't invent gold answers.

**Fixed settings.** `openai/gpt-oss-120b` via Groq, seed 42, PubMed via NCBI E-utilities, local sentence-transformers embeddings.

**Metrics, kept separate:**
- **Automatic checks:** citation validity (are the cited PMIDs real and retrieved?), evidence coverage and diversity, contradiction rate, citation support rate.
- **LLM-judged:** the critic's verdicts. This is a model's opinion, not ground truth.
- **Human:** a rating template is ready (`human_eval_schema()`). I never generate fake ratings.
- **Retrieval quality** (precision, recall, nDCG@k): only when gold PMIDs exist. Otherwise it's `None`, never a made-up zero.

---

## Where things stand

*Last updated: 11 October 2026. I update this after every run.*

| Config | Runs finished | Runs with at least one hypothesis | Notes |
|---|---|---|---|
| `llm_only` | 40 / 40 | n/a | Answer-level judging not done yet |
| `standard_rag` | 40 / 40 | n/a | Answer-level judging not done yet |
| `evidence_aware_rag` | 40 / 40 | n/a | Free-tier rate limits caused many retries |
| `full_biogenesis` | 40 / 40 | 27 / 40 | See early numbers below |
| `no_planner` | 31 / 40 | 16 / 40 | Partial, 9 questions left |
| `no_evidence_scoring`, `no_knowledge_graph`, `no_memory`, `no_critic` | not run yet | | |

"Finished" means the pipeline ran without crashing. It does **not** mean the output was good. Some finished runs found no papers or produced no hypotheses, which is why the second column exists.

### Early numbers (`full_biogenesis`, the 27 questions that produced hypotheses)

| Metric | Value | In plain words |
|---|---|---|
| Citation validity | 1.00 | Every cited paper was real and had been retrieved |
| Citation support rate | 0.60 | About 6 in 10 cited papers backed up the claim they were attached to |
| Contradiction rate | 0.04 | Few contradictions were flagged between evidence items |
| Evidence items per question | 14.3 (mean) | |

These are **early** numbers. They come from runs across several code versions, and I'll recompute them from one clean final run. I'm not highlighting critic approval, because most verdicts are "needs caveats", which the metric counts as approval, so it tells us very little.

**I'm not claiming BioGenesis beats the baselines.** I'll only compare once every configuration has been re-run on the same code version and the same questions.

---

## Quickstart

```bash
git clone https://github.com/Ayesha037/biogenesis.git
cd biogenesis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env     # then add your keys
```

| Variable | Where to get it | Needed? |
|---|---|---|
| `GROQ_API_KEY` | console.groq.com/keys (free tier) | Yes |
| `NCBI_EMAIL` | Your email (NCBI usage policy) | Recommended |
| `NCBI_API_KEY` | NCBI account settings | Optional (higher rate limit) |

Never commit `.env`. It's gitignored, but leave it out when you zip or share the project too.

**Run one question:**

```bash
python pipeline.py "does metformin reduce cancer risk in diabetic patients"
```

The first run downloads the embedding model (about 80 MB, once).

**Run an experiment:**

```bash
python experiments/runner.py --config full_biogenesis --benchmark experiments/benchmark.json
python experiments/runner.py --config standard_rag --benchmark experiments/benchmark.json --limit 5
python experiments/runner.py --config full_biogenesis --benchmark experiments/benchmark.json --check-support
```

Results go to `experiments/results/` as JSONL. Failures are saved as `status="error"` records with the traceback. They're never hidden or replaced with fake output.

**Run the tests** (no API key needed, all external calls are faked):

```bash
pytest tests/
```

---

## Repo layout

```
biogenesis/
├── src/biogenesis/
│   ├── retrieval/        PubMed client
│   ├── preprocessing/    cleaning and chunking
│   ├── embeddings/       sentence-transformers wrapper
│   ├── vectorstore/      Chroma wrapper
│   ├── evidence/         extraction, scoring, contradictions, support checking
│   ├── knowledge_graph/  NetworkX graph builder
│   ├── agents/           planner, hypothesis generator, critic
│   ├── orchestration/    pipeline wiring and ablation flags
│   ├── evaluation/       automatic, LLM-judged and human metrics
│   └── memory/           SQLite memory and retrieval
├── experiments/          benchmark, 9 configs, baselines, runner
├── tests/                offline unit tests
├── docs/ARCHITECTURE.md  design notes
└── pipeline.py           single-question entry point
```

---

## What doesn't work well yet

- **Free-tier limits.** Rate limits caused many retries and some failed runs. The results will only be as consistent as my final re-run.
- **Retrieval failures.** Some questions return no PubMed results. Without the planner, the raw question is often a poor search query.
- **Rough evidence scoring.** It uses study type plus recency decay. It isn't a validated model.
- **Simple entity matching.** "Metformin" and "metformin hydrochloride" become separate graph nodes. Proper normalisation (UMLS/MeSH) is future work.
- **Contradictions are flagged, not resolved.**
- **Memory can leak.** It's keyword-based and shared across questions in one run, so earlier hypotheses can bleed into later ones. The final comparison will isolate memory per question.
- **LLM judging isn't ground truth**, and human evaluation isn't done yet.
- **Abstracts only.** No full-text retrieval.

---

## Roadmap

- [ ] Fix the empty-hypothesis failures (generator JSON retry, search fallback when the planner is off)
- [ ] Re-run every configuration on one commit, with memory isolated per question
- [ ] Add an answer-level judge for the baselines and a human-rated sample
- [ ] Paired comparison with bootstrap confidence intervals, plus per-category results
- [ ] Biomedical entity normalisation and a biomedical embedding model
- [ ] Short technical report (Zenodo / arXiv)

---

## Cite this

```bibtex
@software{summaiyya2026biogenesis,
  author = {Mohammad Ayesha Summaiyya},
  title  = {BioGenesis: An Evidence-Aware Multi-Agent System for Biomedical Hypothesis Generation},
  year   = {2026},
  url    = {https://github.com/Ayesha037/biogenesis}
}
```

## License

MIT. See [LICENSE](LICENSE).
