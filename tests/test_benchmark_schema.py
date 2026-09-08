from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark_schema import VALID_CATEGORIES, BenchmarkItem, load_benchmark

_BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "experiments" / "benchmark.json"


def test_loads_real_benchmark_file():
    items = load_benchmark(_BENCHMARK_PATH)
    assert len(items) >= 5
    for item in items:
        assert item.id
        assert item.question
        assert item.category in VALID_CATEGORIES


def test_rejects_invalid_category(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({
        "items": [{"id": "x1", "question": "q?", "category": "not_a_real_category"}]
    }))
    with pytest.raises(ValueError, match="not in"):
        load_benchmark(bad_file)


def test_rejects_missing_question(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({
        "items": [{"id": "x1", "question": "", "category": "mechanism"}]
    }))
    with pytest.raises(ValueError, match="question is required"):
        load_benchmark(bad_file)


def test_optional_fields_default_sensibly(tmp_path):
    f = tmp_path / "minimal.json"
    f.write_text(json.dumps({
        "items": [{"id": "x1", "question": "does X affect Y?", "category": "mechanism"}]
    }))
    items = load_benchmark(f)
    assert items[0].sub_questions == []
    assert items[0].gold_relevant_pmids is None
    assert items[0].gold_reference is None


def test_benchmark_json_does_not_fabricate_gold_data():
    items = load_benchmark(_BENCHMARK_PATH)
    for item in items:
        assert item.gold_relevant_pmids is None
        assert item.gold_reference is None
