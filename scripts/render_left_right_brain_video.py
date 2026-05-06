#!/usr/bin/env python3
"""Render TRIBE predictions as a 3D cortical activation video."""

from __future__ import annotations

import argparse
from pathlib import Path

from tribe_ui.render import render_left_right_brain_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--cmap", default="coolwarm")
    args = parser.parse_args()

    render_left_right_brain_video(
        args.preds,
        args.out,
        segments_path=args.segments,
        fps=args.fps,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
