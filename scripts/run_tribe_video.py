#!/usr/bin/env python3
"""Run TRIBE v2 inference for one video and optionally render a brain movie."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tribev2.demo_utils import TribeModel, get_audio_and_text_events


def build_events(model: TribeModel, video_path: Path, with_transcript: bool) -> pd.DataFrame:
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


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def local_config_update(feature_device: str) -> dict[str, object]:
    return {
        "data.text_feature.device": feature_device,
        "data.audio_feature.device": feature_device,
        "data.image_feature.image.device": feature_device,
        "data.video_feature.image.device": feature_device,
        "data.num_workers": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--feature-device",
        default="cpu",
        choices=["auto", "cpu", "cuda", "accelerate"],
        help="Device for heavy audio/video/text feature extractors. CPU is safest on Apple Silicon.",
    )
    parser.add_argument(
        "--with-transcript",
        action="store_true",
        help="Run WhisperX speech transcription and text features. Slower, but fuller multimodal inference.",
    )
    parser.add_argument(
        "--render-mp4",
        action="store_true",
        help="Render a cortical-surface activity movie. Requires the plotting extra and ffmpeg.",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Limit saved/rendered timesteps for quick tests.",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    device = auto_device() if args.device == "auto" else args.device
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=args.cache_dir,
        device=device,
        config_update=local_config_update(args.feature_device),
    )

    events = build_events(model, video_path, with_transcript=args.with_transcript)
    events_path = args.out_dir / "events.tsv"
    events.to_csv(events_path, sep="\t", index=False)

    preds, segments = model.predict(events=events)
    if args.max_timesteps is not None:
        preds = preds[: args.max_timesteps]
        segments = segments[: args.max_timesteps]

    preds_path = args.out_dir / "preds.npy"
    segments_path = args.out_dir / "segments.pkl"
    summary_path = args.out_dir / "summary.json"
    np.save(preds_path, preds)
    with segments_path.open("wb") as f:
        pickle.dump(segments, f)
    summary_path.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "device": device,
                "feature_device": args.feature_device,
                "with_transcript": args.with_transcript,
                "preds_shape": list(preds.shape),
                "events_path": str(events_path),
                "preds_path": str(preds_path),
                "segments_path": str(segments_path),
            },
            indent=2,
        )
    )

    if args.render_mp4:
        from tribev2.plotting import PlotBrain

        plotter = PlotBrain(mesh="fsaverage5")
        plotter.plot_timesteps_mp4(
            preds,
            args.out_dir / "brain_activity.mp4",
            segments=segments,
            cmap="fire",
            norm_percentile=99,
            vmin=0.6,
            alpha_cmap=(0, 0.2),
            interpolated_fps=12,
        )

    print(summary_path.read_text())


if __name__ == "__main__":
    main()
