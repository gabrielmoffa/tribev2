# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub",
# ]
# ///
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def install_source(source_tgz: Path) -> None:
    run(["uv", "pip", "install", "--python", sys.executable, f"{source_tgz}[plotting]"])


def ensure_ffmpeg(work: Path) -> None:
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    bin_dir = work / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    link = bin_dir / "ffmpeg"
    if not link.exists():
        link.symlink_to(ffmpeg_exe)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    print("Using bundled ffmpeg:", shutil.which("ffmpeg"), flush=True)


def normalize_video(video_path: Path, out_path: Path) -> Path:
    run(
        [
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
    )
    return out_path


def local_config_update(feature_device: str) -> dict[str, object]:
    return {
        "data.text_feature.device": feature_device,
        "data.audio_feature.device": feature_device,
        "data.image_feature.image.device": feature_device,
        "data.video_feature.image.device": feature_device,
        "data.num_workers": 0,
    }


def build_events(model, video_path: Path, with_transcript: bool):
    import pandas as pd
    from neuralset.events.utils import standardize_events
    from tribev2.demo_utils import get_audio_and_text_events

    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    if not with_transcript:
        return events

    from tribev2.eventstransforms import ExtractWordsFromAudio

    events = ExtractWordsFromAudio()(events)
    words = events.type == "Word"
    if words.any():
        for sequence_id, idx in events[words].groupby("sequence_id").groups.items():
            sentence = " ".join(events.loc[list(idx), "text"].astype(str).tolist())
            running = []
            for row_idx in idx:
                running.append(str(events.at[row_idx, "text"]))
                context = " ".join(running)
                events.at[row_idx, "sentence"] = sentence
                events.at[row_idx, "context"] = context
                events.at[row_idx, "sentence_char"] = sentence.find(
                    str(events.at[row_idx, "text"])
                )
                events.at[row_idx, "text_char"] = len(context) - len(
                    str(events.at[row_idx, "text"])
                )
    return standardize_events(events)


def render_brain(preds_path: Path, segments_path: Path, out_path: Path) -> Path:
    from tribe_ui.render import render_left_right_brain_video

    return render_left_right_brain_video(
        preds_path,
        out_path,
        segments_path=segments_path,
        fps=6,
        cmap="coolwarm",
    )


def summarize(preds_path: Path, out_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    preds = np.load(preds_path)
    pd.DataFrame(
        {
            "second": np.arange(len(preds), dtype=int),
            "absolute_mean": np.abs(preds).mean(axis=1),
            "std": preds.std(axis=1),
            "positive_peak": preds.max(axis=1),
            "negative_peak": preds.min(axis=1),
        }
    ).to_csv(out_dir / "activation_timeseries.csv", index=False)
    try:
        from tribe_ui.analysis import roi_summary

        roi_summary(preds_path).to_csv(out_dir / "roi_summary.csv", index=False)
    except Exception as exc:
        (out_dir / "roi_summary_error.txt").write_text(str(exc))
    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "shape": list(preds.shape),
                "min": float(preds.min()),
                "max": float(preds.max()),
                "mean": float(preds.mean()),
                "std": float(preds.std()),
                "abs_mean_by_second": [float(v) for v in abs(preds).mean(axis=1)],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-repo", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--with-transcript", action="store_true")
    parser.add_argument("--feature-device", default="cuda")
    parser.add_argument("--max-timesteps", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    artifact_repo = args.artifact_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for gated model downloads and uploads")

    api = HfApi(token=token)
    work = Path(tempfile.mkdtemp(prefix="tribe_job_"))
    out_dir = work / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading source and input from artifact store", flush=True)
    source_tgz = Path(
        hf_hub_download(
            artifact_repo,
            args.source_file,
            repo_type="dataset",
            token=token,
        )
    )
    input_video = Path(
        hf_hub_download(
            artifact_repo,
            args.input_file,
            repo_type="dataset",
            token=token,
        )
    )

    install_source(source_tgz)
    ensure_ffmpeg(work)

    import torch
    from tribev2.demo_utils import TribeModel

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0), flush=True)

    normalized = normalize_video(input_video, work / "input_normalized.mp4")
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=work / "cache",
        device="cuda" if torch.cuda.is_available() else "cpu",
        config_update=local_config_update(args.feature_device),
    )
    events = build_events(model, normalized, with_transcript=args.with_transcript)
    events.to_csv(out_dir / "events.tsv", sep="\t", index=False)

    print("Running TRIBE prediction", flush=True)
    preds, segments = model.predict(events=events)
    if args.max_timesteps > 0:
        preds = preds[: args.max_timesteps]
        segments = segments[: args.max_timesteps]

    import numpy as np

    preds_path = out_dir / "preds.npy"
    segments_path = out_dir / "segments.pkl"
    np.save(preds_path, preds)
    with segments_path.open("wb") as f:
        pickle.dump(segments, f)
    shutil.copy2(normalized, out_dir / "input_normalized.mp4")
    if args.render:
        render_brain(preds_path, segments_path, out_dir / "brain_left_right.mp4")
    summarize(preds_path, out_dir)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "artifact_repo": artifact_repo,
                "run_id": args.run_id,
                "with_transcript": args.with_transcript,
                "feature_device": args.feature_device,
                "output_prefix": f"runs/{args.run_id}/outputs",
            },
            indent=2,
        )
    )

    print("Uploading outputs", flush=True)
    api.upload_folder(
        repo_id=artifact_repo,
        repo_type="dataset",
        folder_path=out_dir,
        path_in_repo=f"runs/{args.run_id}/outputs",
        commit_message=f"Upload TRIBE outputs for {args.run_id}",
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
