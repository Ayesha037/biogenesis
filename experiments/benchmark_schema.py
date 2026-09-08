from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_CATEGORIES = {
    "drug_disease_outcome",
    "treatment_outcome",
    "gene_disease",
    "biomarker_disease",
    "drug_adverse_event",
    "mechanism",
    "association_risk",
    "conflicting_evidence",
}


@dataclass
class BenchmarkItem:
    id: str
    question: str
    category: str
    sub_questions: list[str] = field(default_factory=list)
    expected_evidence_characteristics: str | None = None
    gold_relevant_pmids: list[str] | None = None
    gold_reference: str | None = None

    def validate(self) -> list[str]:
        """Returns a list of validation problems; empty list = valid."""
        problems = []
        if not self.id:
            problems.append("id is required")
        if not self.question:
            problems.append(f"[{self.id}] question is required")
        if self.category not in VALID_CATEGORIES:
            problems.append(
                f"[{self.id}] category '{self.category}' not in {sorted(VALID_CATEGORIES)}"
            )
        return problems


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    items_data = data.get("items", data if isinstance(data, list) else [])
    items = []
    all_problems = []
    for item_dict in items_data:
        item = BenchmarkItem(
            id=item_dict["id"],
            question=item_dict["question"],
            category=item_dict["category"],
            sub_questions=item_dict.get("sub_questions", []),
            expected_evidence_characteristics=item_dict.get("expected_evidence_characteristics"),
            gold_relevant_pmids=item_dict.get("gold_relevant_pmids"),
            gold_reference=item_dict.get("gold_reference"),
        )
        problems = item.validate()
        if problems:
            all_problems.extend(problems)
        else:
            items.append(item)

    if all_problems:
        raise ValueError(
            "Benchmark validation failed:\n" + "\n".join(f"  - {p}" for p in all_problems)
        )

    return items
