from __future__ import annotations

import json

import pytest
import runner as runner_module
from baselines import BaselineResult
from benchmark_schema import BenchmarkItem



def test_load_config_by_name_resolves_all_shipped_configs():
    for name in [
        "llm_only", "standard_rag", "evidence_aware_rag",
        "full_biogenesis", "no_knowledge_graph", "no_critic",
        "no_evidence_scoring", "no_planner", "no_memory",
    ]:
        config = runner_module.load_config(name)
        assert config["name"] == name
        assert "type" in config


def test_load_config_missing_raises():
    with pytest.raises(FileNotFoundError):
        runner_module.load_config("this_config_does_not_exist")


def test_ablation_configs_differ_from_full_by_exactly_one_flag():
    full = runner_module.load_config("full_biogenesis")
    flag_keys = [
        "use_planner", "use_semantic_retrieval", "use_evidence_scoring",
        "use_knowledge_graph", "use_critic", "use_memory",
    ]
    ablations = ["no_knowledge_graph", "no_critic", "no_evidence_scoring", "no_planner", "no_memory"]
    for name in ablations:
        config = runner_module.load_config(name)
        differences = [k for k in flag_keys if config.get(k) != full.get(k)]
        assert len(differences) == 1, f"{name} differs from full by {differences}, expected exactly 1"


def _item(config_type_id="bg-test") -> BenchmarkItem:
    return BenchmarkItem(id=config_type_id, question="does X affect Y?", category="mechanism")


def test_dispatch_calls_llm_only_baseline(monkeypatch):
    called = {}

    def fake_run_llm_only(question, llm=None):
        called["question"] = question
        return BaselineResult(config_type="llm_only", question=question, answer_text="answer", citation_type="none")

    import baselines
    monkeypatch.setattr(baselines, "run_llm_only", fake_run_llm_only)

    config = {"name": "llm_only", "type": "llm_only"}
    record = runner_module.run_one_item(config, _item(), seed=1)

    assert called["question"] == "does X affect Y?"
    assert record["config_type"] == "llm_only"
    assert record["status"] == "ok"
    assert record["automated_metrics"] is None  # llm_only has no evidence pool


def test_dispatch_calls_standard_rag_baseline(monkeypatch):
    def fake_run_standard_rag(question, top_k=8, max_papers=10, pubmed=None, embedder=None, llm=None):
        return BaselineResult(
            config_type="standard_rag", question=question, answer_text="a",
            citation_type="chunk", cited_ids=["c1"], retrieved_pmids=["p1"],
        )

    import baselines
    monkeypatch.setattr(baselines, "run_standard_rag", fake_run_standard_rag)

    config = {"name": "standard_rag", "type": "standard_rag", "top_k": 5}
    record = runner_module.run_one_item(config, _item(), seed=1)

    assert record["config_type"] == "standard_rag"
    assert record["citation_type"] == "chunk"
    assert record["retrieved_pmids"] == ["p1"]


def test_dispatch_calls_evidence_aware_rag_baseline(monkeypatch):
    def fake_run_evidence_aware_rag(question, top_k=8, max_papers=10, pubmed=None, embedder=None, llm=None):
        return BaselineResult(
            config_type="evidence_aware_rag", question=question, answer_text="a",
            citation_type="evidence", cited_ids=["e1"], retrieved_pmids=["p1"], evidence_pool=[],
        )

    import baselines
    monkeypatch.setattr(baselines, "run_evidence_aware_rag", fake_run_evidence_aware_rag)

    config = {"name": "evidence_aware_rag", "type": "evidence_aware_rag"}
    record = runner_module.run_one_item(config, _item(), seed=1)

    assert record["config_type"] == "evidence_aware_rag"
    assert record["citation_type"] == "evidence"


def test_dispatch_calls_biogenesis_orchestrator_with_config_flags(monkeypatch):
    from dataclasses import dataclass, field

    captured_kwargs = {}

    @dataclass
    class FakeSession:
        sub_questions: list = field(default_factory=lambda: ["sq1"])
        retrieved_pmids: list = field(default_factory=lambda: ["p1"])
        evidence: list = field(default_factory=list)
        hypotheses: list = field(default_factory=list)
        memory_context_used: str = ""
        support_results: dict = field(default_factory=dict)

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.kg = __import__(
                "biogenesis.knowledge_graph.builder", fromlist=["KnowledgeGraphBuilder"]
            ).KnowledgeGraphBuilder()
            self.memory = None

        def run(self, question):
            return FakeSession()

        def close(self):
            pass

    import biogenesis.orchestration.orchestrator as orch_module
    monkeypatch.setattr(orch_module, "BioGenesisOrchestrator", FakeOrchestrator)

    config = {
        "name": "no_critic", "type": "biogenesis",
        "use_critic": False, "use_knowledge_graph": True, "top_k": 3,
    }
    record = runner_module.run_one_item(config, _item(), seed=1)

    assert captured_kwargs["use_critic"] is False
    assert captured_kwargs["top_k"] == 3
    assert record["status"] == "ok"
    assert record["sub_questions"] == ["sq1"]


def test_check_support_flag_reaches_orchestrator(monkeypatch):
    from dataclasses import dataclass, field

    captured_kwargs = {}

    @dataclass
    class FakeSession:
        sub_questions: list = field(default_factory=lambda: ["sq1"])
        retrieved_pmids: list = field(default_factory=lambda: ["p1"])
        evidence: list = field(default_factory=list)
        hypotheses: list = field(default_factory=list)
        memory_context_used: str = ""
        support_results: dict = field(default_factory=dict)

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.kg = __import__(
                "biogenesis.knowledge_graph.builder", fromlist=["KnowledgeGraphBuilder"]
            ).KnowledgeGraphBuilder()
            self.memory = None

        def run(self, question):
            return FakeSession()

        def close(self):
            pass

    import biogenesis.orchestration.orchestrator as orch_module
    monkeypatch.setattr(orch_module, "BioGenesisOrchestrator", FakeOrchestrator)

    config = {"name": "full_biogenesis", "type": "biogenesis"}

    record_off = runner_module.run_one_item(config, _item(), seed=1, check_support=False)
    assert captured_kwargs["use_support_checking"] is False
    assert record_off["status"] == "ok"

    record_on = runner_module.run_one_item(config, _item(), seed=1, check_support=True)
    assert captured_kwargs["use_support_checking"] is True
    assert record_on["status"] == "ok"


def test_check_support_config_key_overrides_cli_flag(monkeypatch):
    from dataclasses import dataclass, field

    captured_kwargs = {}

    @dataclass
    class FakeSession:
        sub_questions: list = field(default_factory=lambda: ["sq1"])
        retrieved_pmids: list = field(default_factory=lambda: ["p1"])
        evidence: list = field(default_factory=list)
        hypotheses: list = field(default_factory=list)
        memory_context_used: str = ""
        support_results: dict = field(default_factory=dict)

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.kg = __import__(
                "biogenesis.knowledge_graph.builder", fromlist=["KnowledgeGraphBuilder"]
            ).KnowledgeGraphBuilder()
            self.memory = None

        def run(self, question):
            return FakeSession()

        def close(self):
            pass

    import biogenesis.orchestration.orchestrator as orch_module
    monkeypatch.setattr(orch_module, "BioGenesisOrchestrator", FakeOrchestrator)

    # explicit config key wins over the CLI flag, even when the CLI flag is off
    config = {"name": "full_biogenesis", "type": "biogenesis", "use_support_checking": True}
    runner_module.run_one_item(config, _item(), seed=1, check_support=False)
    assert captured_kwargs["use_support_checking"] is True


def test_dispatch_unknown_type_reports_error_not_crash():
    config = {"name": "bogus", "type": "not_a_real_pipeline_type"}
    record = runner_module.run_one_item(config, _item(), seed=1)
    assert record["status"] == "error"
    assert "Unknown config type" in record["error"]


def test_dispatch_failure_is_captured_not_silently_swallowed(monkeypatch):
    def failing_run_llm_only(question, llm=None):
        raise RuntimeError("simulated network failure")

    import baselines
    monkeypatch.setattr(baselines, "run_llm_only", failing_run_llm_only)

    config = {"name": "llm_only", "type": "llm_only"}
    record = runner_module.run_one_item(config, _item(), seed=1)

    assert record["status"] == "error"
    assert "simulated network failure" in record["error"]
    assert "traceback" in record



def test_record_is_json_serializable(monkeypatch):
    def fake_run_llm_only(question, llm=None):
        return BaselineResult(config_type="llm_only", question=question, answer_text="a", citation_type="none")

    import baselines
    monkeypatch.setattr(baselines, "run_llm_only", fake_run_llm_only)

    config = {"name": "llm_only", "type": "llm_only"}
    record = runner_module.run_one_item(config, _item(), seed=1)

    serialized = json.dumps(record) 
    round_tripped = json.loads(serialized)
    assert round_tripped["benchmark_id"] == "bg-test"


def test_record_never_contains_api_key(monkeypatch):
    def fake_run_llm_only(question, llm=None):
        return BaselineResult(config_type="llm_only", question=question, answer_text="a", citation_type="none")

    import baselines
    monkeypatch.setattr(baselines, "run_llm_only", fake_run_llm_only)
    monkeypatch.setenv("GROQ_API_KEY", "sk-should-never-appear-in-output")

    config = {"name": "llm_only", "type": "llm_only"}
    record = runner_module.run_one_item(config, _item(), seed=1)

    serialized = json.dumps(record)
    assert "sk-should-never-appear-in-output" not in serialized
