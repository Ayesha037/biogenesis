from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from biogenesis.agents.hypothesis_generator import Hypothesis
from biogenesis.config import settings
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)

_DB_PATH = settings.memory_dir / "biogenesis_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    hypothesis_id TEXT,
    text TEXT NOT NULL,
    rationale TEXT,
    confidence REAL,
    critique_verdict TEXT,
    critique TEXT,
    supporting_evidence_ids TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
"""


class MemoryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def save_session(self, research_question: str, hypotheses: list[Hypothesis]) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (research_question) VALUES (?)", (research_question,)
        )
        session_id = cur.lastrowid
        for h in hypotheses:
            self.conn.execute(
                """INSERT INTO hypotheses
                   (session_id, hypothesis_id, text, rationale, confidence,
                    critique_verdict, critique, supporting_evidence_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    h.hypothesis_id,
                    h.text,
                    h.rationale,
                    h.confidence,
                    h.critique_verdict,
                    h.critique,
                    json.dumps(h.supporting_evidence_ids),
                ),
            )
        self.conn.commit()
        logger.info("Saved session %d with %d hypotheses to memory", session_id, len(hypotheses))
        return session_id

    def find_related_hypotheses(self, keyword: str) -> list[dict]:
        """Simple keyword search over past hypotheses -- a starting point;
        could be upgraded to embedding-based search over memory later."""
        cur = self.conn.execute(
            """SELECT h.text, h.critique_verdict, s.research_question, s.created_at
               FROM hypotheses h JOIN sessions s ON h.session_id = s.id
               WHERE h.text LIKE ?""",
            (f"%{keyword}%",),
        )
        rows = cur.fetchall()
        return [
            {
                "hypothesis": r[0],
                "verdict": r[1],
                "research_question": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]

    def find_related_by_question(self, research_question: str, limit: int = 5) -> list[dict]:
        """
        Retrieve past hypotheses relevant to a new research question, for
        injection into the current reasoning pass (memory-augmented
        reasoning, as distinct from mere persistence -- see module
        docstring). This is a naive keyword-overlap search, not semantic
        search; it is a deliberately simple starting point flagged in the
        README limitations, not a claim of sophisticated retrieval.
        """
        stopwords = {
            "the", "a", "an", "is", "are", "does", "do", "in", "of", "to",
            "and", "or", "for", "with", "on", "at", "by", "from", "risk",
        }
        keywords = [
            w.strip(".,?!").lower()
            for w in research_question.split()
            if len(w) > 4 and w.lower() not in stopwords
        ]
        seen_texts: set[str] = set()
        results: list[dict] = []
        for kw in keywords:
            for row in self.find_related_hypotheses(kw):
                if row["hypothesis"] not in seen_texts:
                    seen_texts.add(row["hypothesis"])
                    results.append(row)
                if len(results) >= limit:
                    return results
        return results

    def close(self) -> None:
        self.conn.close()
