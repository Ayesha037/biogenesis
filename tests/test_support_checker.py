from __future__ import annotations

from biogenesis.evidence.models import Evidence
from biogenesis.evidence.support_checker import SupportChecker
from fakes import FakeLLMClient


def _evidence(source_text: str) -> Evidence:
    return Evidence(
        evidence_id="e1", pmid="p1", claim="c", subject="X", relation="affects",
        obj="Y", source_text=source_text,
    )


def test_supported_claim_parsed_correctly():
    llm = FakeLLMClient(default='{"supported": true, "reason": "text explicitly states this"}')
    checker = SupportChecker(llm=llm)
    result = checker.check("Drug A reduces Condition B", _evidence("Drug A reduced Condition B in trial."))
    assert result.supported is True
    assert result.evidence_id == "e1"


def test_unsupported_claim_parsed_correctly():
    llm = FakeLLMClient(default='{"supported": false, "reason": "text is unrelated"}')
    checker = SupportChecker(llm=llm)
    result = checker.check("Drug A reduces Condition B", _evidence("Drug C was studied for Condition D."))
    assert result.supported is False


def test_unparseable_output_defaults_to_unsupported_not_supported():
    llm = FakeLLMClient(default="I cannot determine this.")
    checker = SupportChecker(llm=llm)
    result = checker.check("Drug A reduces Condition B", _evidence("some text"))
    assert result.supported is False


def test_check_hypothesis_checks_every_cited_evidence():
    llm = FakeLLMClient(default='{"supported": true, "reason": "ok"}')
    checker = SupportChecker(llm=llm)
    evidence_list = [_evidence("text1"), _evidence("text2")]
    results = checker.check_hypothesis("some claim", evidence_list)
    assert len(results) == 2
    assert len(llm.calls) == 2
