
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore"
    memory_dir: Path = PROJECT_ROOT / "data" / "memory"

    ncbi_api_key: str = field(default_factory=lambda: os.getenv("NCBI_API_KEY", ""))
    ncbi_email: str = field(default_factory=lambda: os.getenv("NCBI_EMAIL", ""))

    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_collection_name: str = "biogenesis_evidence"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.vectorstore_dir, self.memory_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
