from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evidence:
    evidence_id: str
    pmid: str
    claim: str            
    subject: str  
    relation: str   
    obj: str   
    source_text: str    
    study_type: str = "unknown"  
    confidence: float = 0.0   

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "pmid": self.pmid,
            "claim": self.claim,
            "subject": self.subject,
            "relation": self.relation,
            "obj": self.obj,
            "source_text": self.source_text,
            "study_type": self.study_type,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return cls(**d)
