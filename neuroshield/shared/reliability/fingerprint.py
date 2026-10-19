# Failure Fingerprint: structured JSON record of a systematic failure.
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

@dataclass
class FailureCase:
    case_id: str
    condition: str
    error_description: str
    metrics: dict = field(default_factory=dict)

@dataclass
class FailureFingerprint:
    track: str
    model: str
    failure_category: str
    description: str
    evidence_cases: list[FailureCase] = field(default_factory=list)
    aggregate_signals: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda : f'fp-{uuid.uuid4().hex[:8]}')
    created_at: str = field(default_factory=lambda : datetime.now(timezone.utc).isoformat())
    repair_attempted: bool = False
    repair_notes: str = ''

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> 'FailureFingerprint':
        with open(path) as f:
            d = json.load(f)
        d['evidence_cases'] = [FailureCase(**c) for c in d.get('evidence_cases', [])]
        return FailureFingerprint(**d)
