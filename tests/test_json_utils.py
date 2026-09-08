from biogenesis.json_utils import parse_json_loose


def test_parses_well_formed_json_untouched():
    raw = '{"evidence": [{"claim": "X causes Y", "confidence": 0.8, "flag": true}]}'
    result = parse_json_loose(raw)
    assert result["evidence"][0]["confidence"] == 0.8
    assert result["evidence"][0]["flag"] is True


def test_repairs_unquoted_string_values():

    raw = """{"evidence": [
  {
    "claim": Metformin use is associated with reduced overall cancer risk,
    "subject": Metformin,
    "relation": reduces risk of,
    "object": overall cancer,
    "study_type": meta-analysis
  }
]}"""
    result = parse_json_loose(raw)
    assert result is not None
    item = result["evidence"][0]
    assert item["claim"] == "Metformin use is associated with reduced overall cancer risk"
    assert item["subject"] == "Metformin"
    assert item["study_type"] == "meta-analysis"


def test_does_not_corrupt_arrays_or_numbers_while_repairing():
    raw = """{"hypotheses": [
  {
    "hypothesis": Some bare text here,
    "confidence": 0.5,
    "supporting_evidence_ids": ["e1", "e2"]
  }
]}"""
    result = parse_json_loose(raw)
    assert result["hypotheses"][0]["hypothesis"] == "Some bare text here"
    assert result["hypotheses"][0]["confidence"] == 0.5
    assert result["hypotheses"][0]["supporting_evidence_ids"] == ["e1", "e2"]


def test_strips_markdown_code_fences():
    raw = '```json\n{"evidence": []}\n```'
    result = parse_json_loose(raw)
    assert result == {"evidence": []}


def test_returns_none_for_unrecoverable_garbage():
    raw = "I'm sorry, I cannot help with that."
    result = parse_json_loose(raw)
    assert result is None
