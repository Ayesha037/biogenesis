
from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from biogenesis.config import settings
from biogenesis.logging_utils import get_logger
from biogenesis.retrieval.models import Paper

logger = get_logger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    def __init__(self) -> None:
        self.api_key = settings.ncbi_api_key
        self.email = settings.ncbi_email
        # With a key we can safely do ~10 req/sec; without, stay under 3/sec.
        self._min_interval = 0.11 if self.api_key else 0.35
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _common_params(self) -> dict:
        params = {"tool": "biogenesis", "email": self.email or "biogenesis@example.com"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search(self, query: str, max_results: int = 20) -> list[str]:
        """Return a list of PMIDs matching the query."""
        self._throttle()
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=15)
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        logger.info("PubMed search '%s' -> %d PMIDs", query, len(pmids))
        return pmids

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def fetch(self, pmids: list[str]) -> list[Paper]:
        """Fetch full citation records for a list of PMIDs."""
        if not pmids:
            return []
        self._throttle()
        params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        return self._parse_efetch_xml(resp.text)

    def search_and_fetch(self, query: str, max_results: int = 20) -> list[Paper]:
        pmids = self.search(query, max_results=max_results)
        return self.fetch(pmids)

    @staticmethod
    def _parse_efetch_xml(xml_text: str) -> list[Paper]:
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []

        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            abstract_parts = [
                "".join(node.itertext()).strip()
                for node in article.findall(".//AbstractText")
            ]
            abstract = " ".join(p for p in abstract_parts if p)

            authors = []
            for author in article.findall(".//Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                if last is not None and fore is not None:
                    authors.append(f"{fore.text} {last.text}")
                elif last is not None:
                    authors.append(last.text)

            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else ""

            year_el = article.find(".//PubDate/Year")
            medline_date_el = article.find(".//PubDate/MedlineDate")
            pub_date = (
                year_el.text if year_el is not None
                else (medline_date_el.text if medline_date_el is not None else "")
            )

            doi = ""
            for aid in article.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text or ""

            mesh_terms = [
                m.text for m in article.findall(".//MeshHeading/DescriptorName")
                if m.text
            ]

            if pmid and (title or abstract):
                papers.append(
                    Paper(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal or "",
                        pub_date=pub_date or "",
                        doi=doi,
                        mesh_terms=mesh_terms,
                    )
                )

        logger.info("Parsed %d papers from EFetch response", len(papers))
        return papers
