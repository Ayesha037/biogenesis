from __future__ import annotations

from functools import lru_cache

import numpy as np

from biogenesis.config import settings
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", settings.embedding_model_name)
    return SentenceTransformer(settings.embedding_model_name)


class Embedder:
    def __init__(self) -> None:
        self.model = _get_model()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (len(texts), dim)."""
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
