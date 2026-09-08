from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_schema import BenchmarkItem, load_benchmark  # noqa: E402

from biogenesis.config import settings  # noqa: E402
from biogenesis.evaluation.metrics import (  # noqa: E402
    automated_metrics,
    llm_judged_metrics,
    retrieval_metrics,
)
from biogenesis.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_config(config_arg: str) -> dict:
    path = Path(config_arg)
    if not path.exists():
        path = CONFIGS_DIR / f"{config_arg}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config '{config_arg}' not found as a path or in {CONFIGS_DIR}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def run_one_item(config: dict, item: BenchmarkItem, seed: int, check_support: bool = False) -> dict:
    """Dispatch to the correct pipeline based on config['type'], and build
    a comparable result record regardless of which pipeline ran."""
    random.seed(seed)
    config_type = config["type"]

    record: dict = {
        "experiment_config": config["name"],
        "config_type": config_type,
        "benchmark_id": item.id,
        "question": item.question,
        "category": item.category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model": settings.groq_model,
        "status": "ok",
    }

    try:
        if config_type == "llm_only":
            from baselines import run_llm_only

            result = run_llm_only(item.question)
            record.update(
                answer_text=result.answer_text,
                citation_type=result.citation_type,
                cited_ids=result.cited_ids,
                retrieved_pmids=result.retrieved_pmids,
                num_evidence=0,
                automated_metrics=None,
                llm_judged_metrics=None,
                automated_metrics_note="llm_only has no evidence pool; grounding metrics are not applicable.",
            )

        elif config_type == "standard_rag":
            from baselines import run_standard_rag

            result = run_standard_rag(
                item.question,
                top_k=config.get("top_k", 8),
                max_papers=config.get("max_papers", 10),
            )
            record.update(
                answer_text=result.answer_text,
                citation_type=result.citation_type,
                cited_ids=result.cited_ids,
                retrieved_pmids=result.retrieved_pmids,
                num_evidence=0,
                automated_metrics=None,
                llm_judged_metrics=None,
                automated_metrics_note=(
                    "standard_rag cites raw chunk IDs, not verified evidence items; "
                    "citation_validity/support metrics (built for Evidence objects) "
                    "do not apply to this configuration."
                ),
            )
            rm = retrieval_metrics(
                result.retrieved_pmids, item.gold_relevant_pmids, k=config.get("top_k", 8)
            )
            record["retrieval_metrics"] = _dataclass_to_dict(rm)

        elif config_type == "evidence_aware_rag":
            from baselines import run_evidence_aware_rag

            result = run_evidence_aware_rag(
                item.question,
                top_k=config.get("top_k", 8),
                max_papers=config.get("max_papers", 10),
            )
            record.update(
                answer_text=result.answer_text,
                citation_type=result.citation_type,
                cited_ids=result.cited_ids,
                retrieved_pmids=result.retrieved_pmids,
                num_evidence=len(result.evidence_pool),
                evidence_ids=[e.evidence_id for e in result.evidence_pool],
            )
            rm = retrieval_metrics(
                result.retrieved_pmids, item.gold_relevant_pmids, k=config.get("top_k", 8)
            )
            record["retrieval_metrics"] = _dataclass_to_dict(rm)

        elif config_type == "biogenesis":
            from biogenesis.orchestration.orchestrator import BioGenesisOrchestrator

            orchestrator = BioGenesisOrchestrator(
                use_planner=config.get("use_planner", True),
                use_semantic_retrieval=config.get("use_semantic_retrieval", True),
                use_evidence_scoring=config.get("use_evidence_scoring", True),
                use_knowledge_graph=config.get("use_knowledge_graph", True),
                use_critic=config.get("use_critic", True),
                use_memory=config.get("use_memory", True),
                # config-file opt-in OR the --check-support CLI flag, either
                # turns this on; both default to False (see orchestrator
                # module docstring "Citation support checking" for why this
                # is opt-in rather than always-on).
                use_support_checking=config.get("use_support_checking", check_support),
                top_k=config.get("top_k", 8),
                papers_per_subquestion=config.get("papers_per_subquestion", 5),
            )
            try:
                session = orchestrator.run(item.question)

                auto = automated_metrics(
                    session.hypotheses,
                    session.evidence,
                    orchestrator.kg,
                    support_results=session.support_results or None,
                )
                judged = llm_judged_metrics(session.hypotheses)

                record.update(
                    sub_questions=session.sub_questions,
                    retrieved_pmids=session.retrieved_pmids,
                    num_evidence=len(session.evidence),
                    evidence_ids=[e.evidence_id for e in session.evidence],
                    hypotheses=[
                        {
                            "hypothesis_id": h.hypothesis_id,
                            "text": h.text,
                            "rationale": h.rationale,
                            "supporting_evidence_ids": h.supporting_evidence_ids,
                            "confidence": h.confidence,
                            "critique_verdict": h.critique_verdict,
                            "critique": h.critique,
                        }
                        for h in session.hypotheses
                    ],
                    memory_context_used=bool(session.memory_context_used),
                    automated_metrics=_dataclass_to_dict(auto),
                    llm_judged_metrics=_dataclass_to_dict(judged),
                )
                rm = retrieval_metrics(
                    session.retrieved_pmids, item.gold_relevant_pmids, k=config.get("top_k", 8)
                )
                record["retrieval_metrics"] = _dataclass_to_dict(rm)

                if orchestrator.memory is not None:
                    orchestrator.memory.save_session(item.question, session.hypotheses)
            finally:
                orchestrator.close()

        else:
            raise ValueError(f"Unknown config type: {config_type}")

    except Exception as e:  # noqa: BLE001 -- intentionally broad: we must
        # record ANY failure, not just anticipated ones, per Part 8/9
        # (never fabricate success, never silently drop a failed item).
        logger.error("Item %s failed: %s", item.id, e)
        record["status"] = "error"
        record["error"] = str(e)
        record["traceback"] = traceback.format_exc()

    return record


def _dataclass_to_dict(obj) -> dict:
    if obj is None:
        return None
    return {k: v for k, v in vars(obj).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BioGenesis experiment configuration.")
    parser.add_argument("--config", required=True, help="Config name or path to YAML file")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON file")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N items")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-support",
        action="store_true",
        help=(
            "Also run SupportChecker (one extra LLM call per hypothesis "
            "citation) for 'biogenesis' config types, to compute "
            "citation_support_rate/unsupported_claim_rate. Off by default "
            "since it adds real LLM cost proportional to citation count -- "
            "see orchestrator.py's 'Citation support checking' docstring."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    items = load_benchmark(args.benchmark)
    if args.limit:
        items = items[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{config['name']}__{timestamp}.jsonl"

    logger.info("Running config '%s' on %d benchmark items", config["name"], len(items))

    n_ok, n_error = 0, 0
    with open(output_path, "w") as f:
        for item in items:
            logger.info("--- %s: %s ---", item.id, item.question)
            record = run_one_item(config, item, seed=args.seed, check_support=args.check_support)
            f.write(json.dumps(record) + "\n")
            f.flush()
            if record["status"] == "ok":
                n_ok += 1
            else:
                n_error += 1

    logger.info(
        "Run complete: %d ok, %d errors. Results written to %s", n_ok, n_error, output_path
    )
    print(f"\n{n_ok} succeeded, {n_error} failed. Results: {output_path}")


if __name__ == "__main__":
    main()
