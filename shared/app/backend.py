# FastAPI backend: one interface for both tracks.
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
app = FastAPI(title='NeuroShield', version='0.1.0')
TRACK_RUNNERS: dict[str, callable] = {}

def register_track(track_name: str, runner) -> None:
    TRACK_RUNNERS[track_name] = runner

class RunRequest(BaseModel):
    track: str
    case_id: str

@app.get('/health')
def health():
    return {'status': 'ok', 'tracks': sorted(TRACK_RUNNERS)}

@app.post('/run')
def run(req: RunRequest):
    if req.track not in TRACK_RUNNERS:
        raise HTTPException(404, f"Unknown track '{req.track}'. Available: {sorted(TRACK_RUNNERS)}")
    try:
        return TRACK_RUNNERS[req.track](req.case_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
