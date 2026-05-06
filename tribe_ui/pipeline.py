from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis import activation_plot, activation_timeseries, plain_language_summary, roi_summary


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def auto_feature_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def local_config_update(feature_device: str) -> dict[str, object]:
    return {
        "data.text_feature.device": feature_device,
        "data.audio_feature.device": feature_device,
        "data.image_feature.image.device": feature_device,
        "data.video_feature.image.device": feature_device,
        "data.num_workers": 0,
    }


def normalize_video(video_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "scale='min(720,iw)':-2",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def build_events(model: TribeModel, video_path: Path, with_transcript: bool) -> pd.DataFrame:
    from tribev2.demo_utils import get_audio_and_text_events

    if with_transcript:
        return model.get_events_dataframe(video_path=video_path)

    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    return get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)


def run_tribe_review(
    video_path: str | Path,
    *,
    with_transcript: bool = True,
    device: str = "auto",
    feature_device: str = "auto",
    output_root: Path = Path("outputs/ui_runs"),
    max_timesteps: int | None = None,
) -> dict[str, object]:
    """Run TRIBE inference and produce review artifacts for the UI."""

    started = time.strftime("%Y%m%d_%H%M%S")
    video_path = Path(video_path).expanduser().resolve()
    run_dir = output_root / f"{video_path.stem}_{started}"
    run_dir.mkdir(parents=True, exist_ok=True)

    uploaded_path = run_dir / video_path.name
    if video_path != uploaded_path:
        shutil.copy2(video_path, uploaded_path)
    normalized_path = normalize_video(uploaded_path, run_dir / "input_normalized.mp4")

    resolved_device = auto_device() if device == "auto" else device
    resolved_feature_device = (
        auto_feature_device() if feature_device == "auto" else feature_device
    )

    from tribev2.demo_utils import TribeModel

    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=Path("cache"),
        device=resolved_device,
        config_update=local_config_update(resolved_feature_device),
    )

    events = build_events(model, normalized_path, with_transcript=with_transcript)
    events_path = run_dir / "events.tsv"
    events.to_csv(events_path, sep="\t", index=False)

    preds, segments = model.predict(events=events)
    if max_timesteps is not None:
        preds = preds[:max_timesteps]
        segments = segments[:max_timesteps]

    preds_path = run_dir / "preds.npy"
    segments_path = run_dir / "segments.pkl"
    np.save(preds_path, preds)
    with segments_path.open("wb") as f:
        pickle.dump(segments, f)

    from .render import render_left_right_brain_video

    brain_path = render_left_right_brain_video(
        preds_path,
        run_dir / "brain_left_right.mp4",
        segments_path=segments_path,
    )
    chart_path = run_dir / "activation_timeseries.csv"
    activation_timeseries(preds_path).to_csv(chart_path, index=False)

    roi_path = run_dir / "roi_summary.csv"
    try:
        roi_summary(preds_path).to_csv(roi_path, index=False)
    except Exception as exc:
        roi_path.write_text(f"ROI summary failed: {exc}\n")

    summary = {
        "run_dir": str(run_dir),
        "input_video": str(normalized_path),
        "brain_video": str(brain_path),
        "events_path": str(events_path),
        "preds_path": str(preds_path),
        "segments_path": str(segments_path),
        "chart_path": str(chart_path),
        "roi_path": str(roi_path),
        "with_transcript": with_transcript,
        "device": resolved_device,
        "feature_device": resolved_feature_device,
        "preds_shape": list(preds.shape),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_existing_review(run_dir: str | Path) -> tuple[str, str, object, pd.DataFrame, str]:
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        input_video = summary.get("input_video") or summary.get("video", "")
        brain_video = summary.get("brain_video") or str(run_dir / "brain_left_right.mp4")
        preds_path = Path(summary.get("preds_path", run_dir / "preds.npy"))
    else:
        input_video = str(run_dir / "input_normalized.mp4")
        brain_video = str(run_dir / "brain_left_right.mp4")
        preds_path = run_dir / "preds.npy"
    chart = activation_plot(preds_path)
    table = roi_summary(preds_path).head(40)
    return input_video, brain_video, chart, table, plain_language_summary(preds_path)


def environment_note() -> str:
    hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return f"GPU: {gpu}. HF token configured: {hf_token}."
