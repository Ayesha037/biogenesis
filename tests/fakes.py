from __future__ import annotations

import json

import numpy as np

from biogenesis.retrieval.models import Paper


class FakeLLMClient:

    def __init__(self, responses: list[str] | None = None, default: str | None = None):
        self._responses = list(responses or [])
        self._default = default if default is not None else '{"result": "ok"}'
        self.calls: list[dict] = []

    def complete(self, prompt, system="", temperature=0.3, max_tokens=1024, json_mode=False):
        self.calls.append(
            {"prompt": prompt, "system": system, "temperature": temperature, "json_mode": json_mode}
        )
        if self._responses:
            return self._responses.pop(0)
        return self._default


class FakePubMedClient:
    def __init__(self, papers: list[Paper] | None = None):
        self._papers = papers or [
            Paper(
                pmid="1001",
                title="Fake study on Drug A and Condition B",
                abstract="Drug A reduced incidence of Condition B in a cohort study.",
                pub_date="2023",
            )
        ]
        self.search_calls: list[str] = []

    def search_and_fetch(self, query: str, max_results: int = 20) -> list[Paper]:
        self.search_calls.append(query)
        return list(self._papers)


class FakeEmbedder:

    _DIM = 8

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vec(t) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    def _vec(self, text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(self._DIM)


class FakeChromaStore:

    def __init__(self, collection_name: str | None = None):
        self._chunks = []

    def add_chunks(self, chunks, embeddings):
        for c in chunks:
            self._chunks.append(c)

    def query(self, query_embedding, top_k: int = 5):
        hits = []
        for c in self._chunks[:top_k]:
            hits.append(
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "metadata": {"pmid": c.pmid, "source_field": c.source_field},
                    "distance": 0.0,
                }
            )
        return hits

    def count(self):
        return len(self._chunks)


def evidence_json_response(claims: list[dict]) -> str:
    return json.dumps({"evidence": claims})


def hypotheses_json_response(hyps: list[dict]) -> str:
    return json.dumps({"hypotheses": hyps})
