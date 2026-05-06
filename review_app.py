from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tribe_ui.destrieux_functions import region_function


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
STATIC = ROOT / "review_static"
INTERACTIVE_MESH = "fsaverage4"
_BRAIN_PLOTTER_LOCK = Lock()
_NEUROSYNTH_PATH = ROOT / "tribe_ui" / "destrieux_neurosynth.json"


_EMOTION_TERMS = {
    "emotion", "emotional",
    "fear", "fearful", "anxiety", "stress", "threat",
    "sad", "happy",
    "disgust", "arousal",
    "regulation", "appraisal",
    "empathy", "empathic",
    "pain",
    "salience", "salient", "salience network",
    "reward", "loss", "punishment", "value", "motivation", "incentive",
}


@lru_cache(maxsize=1)
def neurosynth_decoding() -> dict[str, list[dict]]:
    if not _NEUROSYNTH_PATH.exists():
        return {}
    return json.loads(_NEUROSYNTH_PATH.read_text())


def emotion_tag_for(decoded: list[dict]) -> dict | None:
    best = None
    for entry in decoded or []:
        term = str(entry.get("term", "")).lower().strip()
        if term in _EMOTION_TERMS:
            if best is None or entry.get("z", 0) > best.get("z", 0):
                best = entry
    return best

app = FastAPI(title="TRIBE Review Timeline")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

UPLOAD_DIR = OUTPUTS / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_LOCAL_TRIBE = os.environ.get("TRIBE_ALLOW_LOCAL", "").lower() in {"1", "true", "yes"}

_JOBS: dict[str, dict[str, object]] = {}
_JOBS_LOCK = Lock()


def _set_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        _JOBS.setdefault(job_id, {})
        _JOBS[job_id].update(fields)


def _get_job(job_id: str) -> dict[str, object]:
    with _JOBS_LOCK:
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return dict(_JOBS[job_id])


def _execute_hf_job(job_id: str, video_path: Path, with_text: bool, max_timesteps: int | None, flavor: str) -> None:
    from tribe_ui.hf_jobs import (
        fetch_job_outputs,
        finalize_remote_review,
        job_log_tail,
        submit_tribe_job,
        wait_for_job,
    )

    _set_job(job_id, status="running", stage="submitting", message="Uploading source and video to HF")
    submitted = submit_tribe_job(
        video_path,
        with_transcript=with_text,
        max_timesteps=max_timesteps,
        flavor=flavor,
    )
    _set_job(
        job_id,
        stage="queued",
        message=f"HF Job {submitted['job_id']} submitted",
        hf_job_id=submitted["job_id"],
        hf_job_url=submitted["job_url"],
    )

    def on_status(stage: str) -> None:
        _set_job(job_id, stage=stage, message=f"HF job: {stage}")

    final_stage = wait_for_job(
        submitted["job_id"],
        namespace=submitted["namespace"],
        poll_seconds=15,
        on_status=on_status,
    )
    if final_stage not in {"COMPLETED", "SUCCEEDED"}:
        logs = job_log_tail(submitted["job_id"], namespace=submitted["namespace"], lines=30)
        raise RuntimeError(f"HF job ended with stage {final_stage}.\n\nLogs:\n{logs[-3000:]}")

    _set_job(job_id, stage="downloading", message="Downloading prediction artifacts")
    run_dir = fetch_job_outputs(artifact_repo=submitted["artifact_repo"], run_id=submitted["run_id"])
    _set_job(job_id, stage="rendering", message="Rendering brain activation video locally")
    finalize_remote_review(run_dir)
    _set_job(job_id, run_dir=str(run_dir))


def _execute_local_job(job_id: str, video_path: Path, with_text: bool, max_timesteps: int | None) -> None:
    if not ALLOW_LOCAL_TRIBE:
        raise RuntimeError(
            "Local TRIBE inference is disabled. Set TRIBE_ALLOW_LOCAL=1 to enable, or use HF Jobs."
        )
    from tribe_ui.pipeline import run_tribe_review

    _set_job(job_id, status="running", stage="local", message="Running TRIBE locally (this can take a while)")
    result = run_tribe_review(
        video_path,
        with_transcript=with_text,
        device="auto",
        feature_device="auto",
        max_timesteps=max_timesteps,
    )
    _set_job(job_id, run_dir=str(result["run_dir"]))


def _run_job(job_id: str, video_path: Path, backend: str, with_text: bool, max_timesteps: int | None, flavor: str) -> None:
    try:
        if backend == "hf":
            _execute_hf_job(job_id, video_path, with_text, max_timesteps, flavor)
        else:
            _execute_local_job(job_id, video_path, with_text, max_timesteps)
        run_dir = Path(str(_get_job(job_id).get("run_dir", "")))
        run_id = run_id_from_path(run_dir) if run_dir else None
        _set_job(job_id, status="done", stage="done", message="Run complete", run_id=run_id)
    except Exception as exc:
        _set_job(
            job_id,
            status="error",
            stage="error",
            message=str(exc),
            traceback=traceback.format_exc(),
        )


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    backend: str = Form("hf"),
    with_text: bool = Form(False),
    max_timesteps: int = Form(0),
    flavor: str = Form("t4-small"),
) -> dict[str, object]:
    if backend not in {"hf", "local"}:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    target = UPLOAD_DIR / f"{job_id}{suffix}"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    limit = None if max_timesteps <= 0 else int(max_timesteps)
    _set_job(
        job_id,
        status="queued",
        stage="queued",
        message="Job queued",
        backend=backend,
        filename=file.filename,
    )
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, target, backend, with_text, limit, flavor),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job = _get_job(job_id)
    job.pop("traceback", None)
    return job


def discover_runs() -> list[Path]:
    runs = []
    for preds in OUTPUTS.glob("**/preds.npy"):
        run_dir = preds.parent
        if (run_dir / "input_normalized.mp4").exists() and (
            run_dir / "brain_left_right.mp4"
        ).exists():
            runs.append(run_dir)
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def run_by_id(run_id: str) -> Path:
    for run in discover_runs():
        if run_id == run_id_from_path(run):
            return run
    raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


def run_id_from_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix().replace("/", "__")


def label_text(label) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8", errors="replace")
    return str(label)


CHART_COLORS = [
    "#f05a5a",
    "#f59f00",
    "#facc15",
    "#26a269",
    "#22c7a9",
    "#4f7cff",
    "#8b5cf6",
    "#ec4899",
    "#a3e635",
    "#38bdf8",
]


def clean_region_name(label: str) -> str:
    return (
        label.replace("_", " ")
        .replace("G and S", "G+S")
        .replace("Lat Fis", "Lat. fissure")
        .replace("Pole occipital", "Occipital pole")
        .replace("Pole temporal", "Temporal pole")
    )


def grouped_activation(preds_path: Path) -> dict[str, object]:
    from nilearn import datasets

    preds = np.load(preds_path)
    atlas = datasets.fetch_atlas_surf_destrieux(verbose=0)
    labels = [label_text(label) for label in atlas["labels"]]
    hemi_maps = {"L": atlas["map_left"], "R": atlas["map_right"]}
    hemi_offset = {"L": 0, "R": preds.shape[1] // 2}

    series = []
    for hemi, hemi_map in hemi_maps.items():
        offset = hemi_offset[hemi]
        for index, label in enumerate(labels):
            if index == 0:
                continue
            hemi_mask = hemi_map == index
            if not hemi_mask.any():
                continue
            vertex_indices = np.flatnonzero(hemi_mask) + offset
            values = np.abs(preds[:, vertex_indices]).mean(axis=1)
            signed = preds[:, vertex_indices].mean(axis=1)
            peak = float(values.max(initial=0.0))
            series.append(
                {
                    "key": f"{hemi}_{label}",
                    "name": f"{hemi} {clean_region_name(label)}",
                    "region": clean_region_name(label),
                    "atlasLabel": label,
                    "hemi": hemi,
                    "color": CHART_COLORS[len(series) % len(CHART_COLORS)],
                    "values": [float(v) for v in values],
                    "signed": [float(v) for v in signed],
                    "peak": peak,
                    "vertices": int(len(vertex_indices)),
                    "function": region_function(label, hemi),
                    "functionNeurosynth": neurosynth_decoding().get(f"{hemi}_{label}", []),
                    "emotionTag": emotion_tag_for(neurosynth_decoding().get(f"{hemi}_{label}", [])),
                }
            )

    series.sort(key=lambda item: item["peak"], reverse=True)
    for index, item in enumerate(series):
        item["rank"] = index + 1
        item["selected"] = index < 12

    overall = np.abs(preds).mean(axis=1)
    return {
        "seconds": list(range(len(preds))),
        "overall": [float(v) for v in overall],
        "series": series,
        "shape": list(preds.shape),
    }


@lru_cache(maxsize=1)
def _brain_plotter_cached():
    from tribev2.plotting import PlotBrain

    return PlotBrain(mesh=INTERACTIVE_MESH)


def brain_plotter():
    with _BRAIN_PLOTTER_LOCK:
        return _brain_plotter_cached()


@lru_cache(maxsize=1)
def brain_mesh() -> dict[str, object]:
    plotter = brain_plotter()
    mesh = plotter._mesh["both"]
    coords = mesh["coords"].astype(float)
    faces = mesh["faces"].astype(int)
    bg_map = mesh["bg_map"].astype(float)
    center = coords.mean(axis=0)
    scale = float(np.max(np.linalg.norm(coords - center, axis=1)))
    return {
        "mesh": INTERACTIVE_MESH,
        "coords": coords.round(4).tolist(),
        "faces": faces.tolist(),
        "bg": bg_map.round(4).tolist(),
        "center": center.round(4).tolist(),
        "scale": scale,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/runs")
def runs() -> dict[str, object]:
    items = []
    for run in discover_runs():
        preds = np.load(run / "preds.npy", mmap_mode="r")
        items.append(
            {
                "id": run_id_from_path(run),
                "name": run.name,
                "path": run.relative_to(ROOT).as_posix(),
                "seconds": int(preds.shape[0]),
                "vertices": int(preds.shape[1]),
                "inputVideo": f"/api/runs/{run_id_from_path(run)}/input",
                "brainVideo": f"/api/runs/{run_id_from_path(run)}/brain",
            }
        )
    return {"runs": items}


@app.get("/api/runs/{run_id}/data")
def run_data(run_id: str) -> dict[str, object]:
    run = run_by_id(run_id)
    data = grouped_activation(run / "preds.npy")
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {
        "run": {
            "id": run_id,
            "name": run.name,
            "path": run.relative_to(ROOT).as_posix(),
        },
        "summary": summary,
        **data,
    }


@app.get("/api/runs/{run_id}/brain-mesh")
def run_brain_mesh(run_id: str) -> dict[str, object]:
    run_by_id(run_id)
    return brain_mesh()


@app.get("/api/runs/{run_id}/brain-frame/{second}")
def run_brain_frame(run_id: str, second: int) -> dict[str, object]:
    run = run_by_id(run_id)
    preds = np.load(run / "preds.npy", mmap_mode="r")
    if second < 0 or second >= preds.shape[0]:
        raise HTTPException(status_code=404, detail=f"Frame not found: {second}")
    lim = float(np.percentile(np.abs(preds), 99))
    if lim == 0:
        lim = 1.0
    values = brain_plotter().get_stat_map(np.asarray(preds[second]))["both"]
    return {
        "mesh": INTERACTIVE_MESH,
        "second": second,
        "limit": lim,
        "values": values.astype(float).round(6).tolist(),
    }


@app.get("/api/runs/{run_id}/input")
def input_video(run_id: str) -> FileResponse:
    return FileResponse(run_by_id(run_id) / "input_normalized.mp4", media_type="video/mp4")


@app.get("/api/runs/{run_id}/brain")
def brain_video(run_id: str) -> FileResponse:
    return FileResponse(run_by_id(run_id) / "brain_left_right.mp4", media_type="video/mp4")
