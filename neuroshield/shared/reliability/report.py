# Shared Reliability Report schema emitted by both tracks.
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

@dataclass
class ReliabilityReport:
    case_id: str
    track: str
    model: str
    prediction: dict
    confidence: float
    uncertainty: dict
    calibration: dict
    ood: dict
    counterfactual_stability: dict
    model_agreement: dict
    reliability: str
    decision: str
    decision_reasons: list[str] = field(default_factory=list)
    explanation: str | None = None
    explanation_evidence: list[dict] = field(default_factory=list)
    disclaimer: str = 'Research prototype. Not a clinically validated diagnostic system. Not medical advice.'
    generated_at: str = field(default_factory=lambda : datetime.now(timezone.utc).isoformat())

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
