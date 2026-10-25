"""NeuroShield demo: FastAPI backend serving the dashboard and real model inference.

One interface, two separate models. No combined diagnostic claim is made.
"""
from __future__ import annotations

import os
import json
import shutil
import time
import uuid
import threading
import tempfile
import traceback
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ARTIFACTS_DIR = os.environ.get("NEUROSHIELD_ARTIFACTS", "/artifacts")
RESULTS_DIR = os.environ.get("NEUROSHIELD_RESULTS", "/tmp/neuroshield_results")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
JOB_TTL_SECONDS = 3600

os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="NeuroShield Demo", version="1.0.0")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_rate: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_PER_MIN = 10

_state = {"track1_loaded": False, "track2_loaded": False, "load_error": {}}


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


def _rate_check(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    _rate[ip] = [t for t in _rate[ip] if now - t < 60]
    if len(_rate[ip]) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    _rate[ip].append(now)


# Public-safe mode (NEUROSHIELD_PUBLIC_SAFE=1): drop the BraTS 2020 sample cases.
# Their license requires citation and does not grant public redistribution, so a
# public deployment must not serve them. This removes them from /api/samples AND
# makes the sample-run endpoints answer 404 for those ids (metadata and routes,
# not just UI). Deploy bundles must also exclude artifacts/samples/track2/ -
# scripts/build_bundle.sh does this when PUBLIC_SAFE=1.
PUBLIC_SAFE = os.environ.get("NEUROSHIELD_PUBLIC_SAFE", "").lower() in ("1", "true", "yes")


def load_samples() -> dict:
    path = os.path.join(ARTIFACTS_DIR, "samples.json")
    if not os.path.exists(path):
        return {"track1": [], "track2": []}
    with open(path) as f:
        data = json.load(f)
    if PUBLIC_SAFE:
        data["track2"] = []
    # Only list samples whose files are actually present, so a package without
    # sample data never shows buttons that would fail at run time.
    def _present(s):
        return os.path.exists(os.path.join(ARTIFACTS_DIR, s["path"]))
    data["track1"] = [s for s in data.get("track1", []) if _present(s)]
    data["track2"] = [s for s in data.get("track2", []) if _present(s)]
    return data


@app.on_event("startup")
def startup() -> None:
    def _load():
        try:
            from runners import track1
            track1.load_models(os.path.join(ARTIFACTS_DIR, "track1"))
            _state["track1_loaded"] = True
        except Exception as e:
            _state["load_error"]["track1"] = str(e)
        try:
            from runners.track2 import Track2Runner
            model_dir = os.path.join(
                ARTIFACTS_DIR, "track2", "model", "Dataset001_BraTS2020",
                "nnUNetTrainer_100epochs__nnUNetPlans__2d")
            app.state.track2_runner = Track2Runner(model_dir)
            _state["track2_loaded"] = True
        except Exception as e:
            _state["load_error"]["track2"] = str(e)
    threading.Thread(target=_load, daemon=True).start()
    threading.Thread(target=_janitor, daemon=True).start()


def _janitor() -> None:
    while True:
        time.sleep(600)
        cutoff = time.time() - JOB_TTL_SECONDS
        with _jobs_lock:
            stale = [j for j, v in _jobs.items() if v.get("created", 0) < cutoff]
        for j in stale:
            shutil.rmtree(os.path.join(RESULTS_DIR, j), ignore_errors=True)
            with _jobs_lock:
                _jobs.pop(j, None)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "track1_ready": _state["track1_loaded"],
        "track2_ready": _state["track2_loaded"],
        "errors": _state["load_error"],
    }


@app.get("/api/samples")
def samples():
    return load_samples()


# ---------------- Track 1 ----------------

def _run_track1_job(job_id: str, scan_path: str, true_label: int | None) -> None:
    try:
        _set_job(job_id, status="running", stage="Reading scan and preparing 5 slices")
        from runners import track1
        _set_job(job_id, stage="Running the 3 trained models")
        result = track1.run_inference(scan_path)
        if true_label is not None:
            result["sample_true_label"] = true_label
            result["sample_true_label_text"] = track1.CLASS_LABELS.get(true_label, str(true_label))
        _set_job(job_id, status="done", result=result)
    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, status="error", error=_friendly_error(e))


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    if "Expected a 3D scan" in msg:
        return "That file does not look like a single 3D MRI volume. Please upload an OASIS-style structural scan (.nii, .nii.gz or .hdr/.img pair)."
    if "cannot" in msg.lower() or "load" in msg.lower():
        return "We could not read that scan. Please use an uncompressed NIfTI (.nii / .nii.gz) or an Analyze .hdr + .img pair."
    return "Something went wrong while running the model. Please try a different scan."


@app.post("/api/track1/run")
async def track1_run(request: Request, sample_id: str | None = Form(default=None),
                     scan: UploadFile | None = File(default=None),
                     scan_pair: UploadFile | None = File(default=None)):
    _rate_check(request)
    if not _state["track1_loaded"]:
        raise HTTPException(503, "Alzheimer's model is still loading. Try again in a moment.")
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    true_label = None

    if sample_id:
        samples_meta = {s["id"]: s for s in load_samples()["track1"]}
        if sample_id not in samples_meta:
            raise HTTPException(404, "Unknown sample.")
        meta = samples_meta[sample_id]
        scan_path = os.path.join(ARTIFACTS_DIR, meta["path"])
        true_label = meta.get("true_label")
    elif scan is not None:
        data = await scan.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (60 MB limit).")
        fname = (scan.filename or "").lower()
        ext = ".nii.gz" if fname.endswith(".nii.gz") else os.path.splitext(fname)[1]
        if ext not in (".nii", ".nii.gz", ".img", ".hdr"):
            raise HTTPException(400, "Please upload .nii, .nii.gz, or an .img/.hdr pair.")
        scan_path = os.path.join(job_dir, "upload" + ext)
        with open(scan_path, "wb") as f:
            f.write(data)
        if ext in (".img", ".hdr"):
            if scan_pair is None:
                raise HTTPException(400, "Analyze format needs both files: upload the matching .hdr and .img together.")
            pdata = await scan_pair.read()
            pext = ".img" if ext == ".hdr" else ".hdr"
            with open(os.path.join(job_dir, "upload" + pext), "wb") as f:
                f.write(pdata)
            scan_path = os.path.join(job_dir, "upload.hdr" if ext == ".hdr" else "upload.img")
        # convert uploaded analyze/nii to plain .nii for nibabel safety
        try:
            import nibabel as nib
            img = nib.load(scan_path)
            conv = os.path.join(job_dir, "scan.nii")
            nib.save(img, conv)
            scan_path = conv
        except Exception:
            raise HTTPException(400, "We could not read that scan file.")
    else:
        raise HTTPException(400, "Choose a sample scan or upload your own.")

    _set_job(job_id, status="queued", created=time.time(), track="track1")
    threading.Thread(target=_run_track1_job, args=(job_id, scan_path, true_label), daemon=True).start()
    return {"job_id": job_id}


# ---------------- Track 2 ----------------

def _run_track2_job(job_id: str, modality_paths: dict, case_id: str, remove_flair: bool, gt_path: str | None = None) -> None:
    try:
        runner: "Track2Runner" = app.state.track2_runner
        job_dir = os.path.join(RESULTS_DIR, job_id)

        def status(msg):
            _set_job(job_id, status="running", stage=msg)

        result = runner.run(modality_paths, case_id, job_dir, remove_flair, gt_path=gt_path, status_cb=status)
        _set_job(job_id, status="done", result=result)
    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, status="error", error="The segmentation could not be completed on this input. If you uploaded your own case, check it has the 4 BraTS sequences (FLAIR, T1, T1ce, T2) as .nii or .nii.gz files.")


@app.post("/api/track2/run")
async def track2_run(request: Request, sample_id: str | None = Form(default=None),
                     remove_flair: bool = Form(default=False),
                     flair: UploadFile | None = File(default=None),
                     t1: UploadFile | None = File(default=None),
                     t1ce: UploadFile | None = File(default=None),
                     t2: UploadFile | None = File(default=None)):
    _rate_check(request)
    if not _state["track2_loaded"]:
        raise HTTPException(503, "Tumor model is still loading. Try again in a moment.")
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    if sample_id:
        samples_meta = {s["id"]: s for s in load_samples()["track2"]}
        if sample_id not in samples_meta:
            raise HTTPException(404, "Unknown sample.")
        base = os.path.join(ARTIFACTS_DIR, samples_meta[sample_id]["path"])
        modality_paths = {m: os.path.join(base, f"{m}.nii.gz") for m in ("flair", "t1", "t1ce", "t2")}
        gt_path = os.path.join(base, "seg.nii.gz")
        case_id = sample_id
    else:
        uploads = {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2}
        if any(u is None for u in uploads.values()):
            raise HTTPException(400, "A brain tumor case needs all 4 sequences: FLAIR, T1, T1ce and T2.")
        modality_paths = {}
        for name, up in uploads.items():
            data = await up.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name} file too large (60 MB limit).")
            fname = up.filename or ""
            ext = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
            path = os.path.join(job_dir, f"{name}{ext}")
            with open(path, "wb") as f:
                f.write(data)
            modality_paths[name] = path
        case_id = "upload"

    _set_job(job_id, status="queued", created=time.time(), track="track2")
    threading.Thread(target=_run_track2_job,
                     args=(job_id, modality_paths, case_id, bool(remove_flair),
                           gt_path if sample_id else None), daemon=True).start()
    return {"job_id": job_id}


# ---------------- Jobs ----------------

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Unknown or expired job.")
    return {k: v for k, v in job.items() if k != "result"}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Unknown or expired job.")
    if job.get("status") != "done":
        raise HTTPException(409, "Job not finished.")
    return job["result"]


@app.get("/api/jobs/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str):
    safe = os.path.normpath(path)
    if safe.startswith("..") or os.path.isabs(safe):
        raise HTTPException(400, "Bad path")
    full = os.path.join(RESULTS_DIR, job_id, safe)
    if not os.path.exists(full):
        raise HTTPException(404, "Not found")
    return FileResponse(full)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error."})


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
