from __future__ import annotations

from groq import BadRequestError

from biogenesis.evidence.models import Evidence
from biogenesis.json_utils import parse_json_loose
from biogenesis.llm_client import LLMClient
from biogenesis.logging_utils import get_logger
from biogenesis.retrieval.models import Paper

logger = get_logger(__name__)


_SYSTEM_PROMPT = """You are a biomedical evidence extraction system. \
Given a paper's title and abstract, extract every distinct factual claim \
about a relationship between two biomedical entities (e.g. gene-disease, \
drug-condition, protein-pathway).

Respond with a single JSON object of the form:
{"evidence": [ { ... }, { ... } ]}

Each element of "evidence" must have exactly these fields, and every value \
must be a properly double-quoted JSON string:
  "claim": one-sentence plain-English statement of the claim
  "subject": the first entity (e.g. a drug, gene, or exposure)
  "relation": short relation phrase (e.g. "increases risk of", "inhibits", \
"is associated with", "has no effect on")
  "object": the second entity (e.g. a disease, protein, or outcome)
  "study_type": one of "meta-analysis", "rct", "cohort", "case-control", \
"case-study", "in-vitro", "animal-model", "review", "unknown"

If the abstract contains no extractable claims, respond with {"evidence": []}.
Respond with JSON only -- no prose, no markdown code fences."""


def _is_json_validation_error(exc: BaseException) -> bool:
    """Return True only for Groq's structured-output JSON validation error."""
    if not isinstance(exc, BadRequestError):
        return False

    error_code = getattr(exc, "code", None)
    if error_code == "json_validate_failed":
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") == "json_validate_failed":
            return True

    return "json_validate_failed" in str(exc)


class EvidenceExtractor:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def _build_prompt(self, paper: Paper) -> str:
        return f"Title: {paper.title}\n\nAbstract: {paper.abstract}"

    def _to_evidence(
        self,
        parsed: object,
        paper: Paper,
    ) -> list[Evidence]:
        items = parsed.get("evidence", []) if isinstance(parsed, dict) else []

        if not isinstance(items, list):
            logger.warning(
                "Evidence payload for PMID %s did not contain a valid "
                "'evidence' list. Returning empty evidence.",
                paper.pmid,
            )
            return []

        evidence_list: list[Evidence] = []

        for i, item in enumerate(items):
            try:
                evidence_list.append(
                    Evidence(
                        evidence_id=f"{paper.pmid}_ev_{i}",
                        pmid=paper.pmid,
                        claim=item["claim"],
                        subject=item["subject"],
                        relation=item["relation"],
                        obj=item["object"],
                        source_text=paper.abstract,
                        study_type=item.get("study_type", "unknown"),
                    )
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed evidence item from PMID %s: %s",
                    paper.pmid,
                    exc,
                )

        return evidence_list

    def extract(self, paper: Paper) -> list[Evidence]:
        prompt = self._build_prompt(paper)

        # Normal path: force structured JSON output.
        try:
            raw = self.llm.complete(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=0.1,
                json_mode=True,
            )

            parsed = parse_json_loose(raw)
            evidence_list = self._to_evidence(parsed, paper)

            logger.info(
                "Extracted %d evidence items from PMID %s using JSON mode",
                len(evidence_list),
                paper.pmid,
            )
            return evidence_list

        except BadRequestError as exc:
            if not _is_json_validation_error(exc):
                raise

            logger.warning(
                "JSON-mode evidence extraction failed for PMID %s with "
                "json_validate_failed. Trying one fallback request without "
                "forced JSON response format.",
                paper.pmid,
            )

            # One and only one fallback request.
            try:
                fallback_raw = self.llm.complete(
                    prompt=prompt,
                    system=_SYSTEM_PROMPT,
                    temperature=0.1,
                    json_mode=False,
                )
            except Exception as fallback_exc:
                logger.error(
                    "Fallback evidence extraction failed for PMID %s: %s",
                    paper.pmid,
                    fallback_exc,
                )
                return []

            try:
                fallback_parsed = parse_json_loose(fallback_raw)
            except Exception as parse_exc:
                logger.error(
                    "Fallback evidence parsing failed for PMID %s: %s",
                    paper.pmid,
                    parse_exc,
                )
                return []

            evidence_list = self._to_evidence(fallback_parsed, paper)

            logger.info(
                "Extracted %d evidence items from PMID %s using fallback "
                "non-forced-JSON mode",
                len(evidence_list),
                paper.pmid,
            )
            return evidence_list