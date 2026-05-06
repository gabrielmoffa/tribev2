#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import time
import uuid
from pathlib import Path

from huggingface_hub import HfApi, get_token


DEFAULT_ARTIFACT_REPO = os.environ.get("HF_ARTIFACT_REPO", "")
DEFAULT_NAMESPACE = os.environ.get("HF_NAMESPACE", "")
DEFAULT_ARTIFACT_REPO_NAME = "tribe-job-artifacts"


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _default_namespace(api: HfApi, token: str) -> str:
    whoami = api.whoami(token=token)
    namespace = str(whoami.get("name") or "").strip()
    if not namespace:
        raise SystemExit("Could not determine Hugging Face username from token.")
    return namespace


def make_source_tar(root: Path, out_path: Path) -> Path:
    include = [
        "pyproject.toml",
        "tribev2",
        "tribe_ui",
        "scripts/render_left_right_brain_video.py",
    ]
    with tarfile.open(out_path, "w:gz") as tar:
        for item in include:
            path = root / item
            if path.is_dir():
                for file in path.rglob("*"):
                    if "__pycache__" in file.parts or file.suffix == ".pyc":
                        continue
                    tar.add(file, arcname=str(file.relative_to(root)))
            else:
                tar.add(path, arcname=item)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--artifact-repo", default=DEFAULT_ARTIFACT_REPO or None)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE or None)
    parser.add_argument("--flavor", default="t4-small")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--with-transcript", action="store_true")
    parser.add_argument("--max-timesteps", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token = token or get_token()
    if not token:
        raise SystemExit("No HF token found. Run `hf auth login` or export HF_TOKEN.")

    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    api = HfApi(token=token)
    namespace = _clean(args.namespace) or _default_namespace(api, token)
    artifact_repo = _clean(args.artifact_repo) or f"{namespace}/{DEFAULT_ARTIFACT_REPO_NAME}"
    api.create_repo(artifact_repo, repo_type="dataset", private=True, exist_ok=True)

    staging = root / "cache" / "hf_jobs" / run_id
    staging.mkdir(parents=True, exist_ok=True)
    source_tgz = make_source_tar(root, staging / "tribe_source.tgz")
    input_copy = staging / video.name
    shutil.copy2(video, input_copy)

    source_repo_path = f"runs/{run_id}/inputs/tribe_source.tgz"
    video_repo_path = f"runs/{run_id}/inputs/{video.name}"
    api.upload_file(
        repo_id=artifact_repo,
        repo_type="dataset",
        path_or_fileobj=source_tgz,
        path_in_repo=source_repo_path,
        commit_message=f"Upload source for {run_id}",
    )
    api.upload_file(
        repo_id=artifact_repo,
        repo_type="dataset",
        path_or_fileobj=input_copy,
        path_in_repo=video_repo_path,
        commit_message=f"Upload input video for {run_id}",
    )

    script_args = [
        "--artifact-repo",
        artifact_repo,
        "--run-id",
        run_id,
        "--input-file",
        video_repo_path,
        "--source-file",
        source_repo_path,
    ]
    if args.with_transcript:
        script_args.append("--with-transcript")
    if args.max_timesteps:
        script_args.extend(["--max-timesteps", str(args.max_timesteps)])

    job = api.run_uv_job(
        str(root / "scripts" / "hf_job_tribe_review.py"),
        script_args=script_args,
        python="3.11",
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
        namespace=namespace,
        token=token,
    )
    print(f"RUN_ID={run_id}")
    print(f"ARTIFACT_REPO={artifact_repo}")
    print(f"JOB_ID={job.id}")
    print(f"JOB_URL={job.url}")


if __name__ == "__main__":
    main()
