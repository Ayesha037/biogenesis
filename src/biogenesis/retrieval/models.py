
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    pub_date: str = ""
    doi: str = ""
    mesh_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "doi": self.doi,
            "mesh_terms": self.mesh_terms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(**d)
