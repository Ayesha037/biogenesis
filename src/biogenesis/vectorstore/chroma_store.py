"""
Vector store wrapper around Chroma.

What a vector database is for
--------------------------------
Once text is embedded into vectors, "search" becomes "find the K nearest
vectors to my query vector" -- an Approximate Nearest Neighbor (ANN) search
problem. Doing this with a plain Python list of vectors is fine for a few
hundred items but becomes slow at scale. A vector database (Chroma here)
builds an index that makes this fast even as the collection grows, and
handles persistence to disk so embeddings survive between runs.

Why Chroma specifically
-------------------------
It's free, runs embedded (no separate server process to manage), persists
to a local folder, and has a simple Python API -- a good fit for a research
prototype that needs to "just work" locally without infra overhead.
"""

from __future__ import annotations

import chromadb

from biogenesis.config import settings
from biogenesis.logging_utils import get_logger
from biogenesis.preprocessing.chunker import Chunk

logger = get_logger(__name__)


class ChromaStore:
    def __init__(self, collection_name: str | None = None) -> None:
        self.client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[{"pmid": c.pmid, "source_field": c.source_field} for c in chunks],
        )
        logger.info("Upserted %d chunks into Chroma", len(chunks))

    def query(self, query_embedding, top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
