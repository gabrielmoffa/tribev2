from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
import uuid
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi, get_token, hf_hub_download

from .analysis import activation_timeseries, roi_summary
from .render import render_left_right_brain_video


DEFAULT_ARTIFACT_REPO_NAME = "tribe-job-artifacts"
DEFAULT_ARTIFACT_REPO = os.environ.get("HF_ARTIFACT_REPO", "")
DEFAULT_NAMESPACE = os.environ.get("HF_NAMESPACE", "")


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def hf_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()
    )
    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")
    return token


def _default_namespace(api: HfApi, token: str) -> str:
    whoami = api.whoami(token=token)
    namespace = str(whoami.get("name") or "").strip()
    if not namespace:
        raise RuntimeError("Could not determine Hugging Face username from token.")
    return namespace


def _resolve_hf_target(
    api: HfApi,
    token: str,
    *,
    artifact_repo: str | None = None,
    namespace: str | None = None,
) -> tuple[str, str]:
    resolved_namespace = (
        _clean(namespace)
        or _clean(DEFAULT_NAMESPACE)
        or _default_namespace(api, token)
    )
    resolved_repo = (
        _clean(artifact_repo)
        or _clean(DEFAULT_ARTIFACT_REPO)
        or f"{resolved_namespace}/{DEFAULT_ARTIFACT_REPO_NAME}"
    )
    return resolved_namespace, resolved_repo


def make_source_tar(root: Path, out_path: Path) -> Path:
    include = [
        "pyproject.toml",
        "tribev2",
        "tribe_ui",
        "scripts/render_left_right_brain_video.py",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def _job_stage(job) -> str:
    status = getattr(job, "status", None)
    if isinstance(status, dict):
        return str(status.get("stage") or status.get("status") or "UNKNOWN")
    return str(getattr(status, "stage", status) or "UNKNOWN")


def submit_tribe_job(
    video_path: str | Path,
    *,
    with_transcript: bool,
    max_timesteps: int | None = None,
    artifact_repo: str | None = None,
    namespace: str | None = None,
    flavor: str = "t4-small",
    timeout: str = "2h",
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    video = Path(video_path).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(video)

    token = hf_token()
    api = HfApi(token=token)
    namespace, artifact_repo = _resolve_hf_target(
        api,
        token,
        artifact_repo=artifact_repo,
        namespace=namespace,
    )
    api.create_repo(artifact_repo, repo_type="dataset", private=True, exist_ok=True)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
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
    if with_transcript:
        script_args.append("--with-transcript")
    if max_timesteps:
        script_args.extend(["--max-timesteps", str(max_timesteps)])

    job = api.run_uv_job(
        str(root / "scripts" / "hf_job_tribe_review.py"),
        script_args=script_args,
        python="3.11",
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": token},
        namespace=namespace,
        token=token,
    )
    return {
        "run_id": run_id,
        "job_id": job.id,
        "job_url": job.url,
        "artifact_repo": artifact_repo,
        "namespace": namespace,
    }


def wait_for_job(
    job_id: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    poll_seconds: int = 15,
    on_status: Callable[[str], None] | None = None,
) -> str:
    api = HfApi(token=hf_token())
    terminal = {"COMPLETED", "SUCCEEDED", "FAILED", "ERROR", "CANCELED", "CANCELLED"}
    while True:
        job = api.inspect_job(job_id=job_id, namespace=namespace, token=hf_token())
        stage = _job_stage(job).upper()
        if on_status:
            on_status(stage)
        if stage in terminal:
            return stage
        time.sleep(poll_seconds)


def job_log_tail(
    job_id: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    lines: int = 60,
) -> str:
    api = HfApi(token=hf_token())
    logs = list(
        api.fetch_job_logs(
            job_id=job_id,
            namespace=namespace,
            follow=False,
            token=hf_token(),
        )
    )
    return "".join(logs[-lines:])


def fetch_job_outputs(
    *,
    artifact_repo: str,
    run_id: str,
    output_root: Path = Path("outputs/ui_runs"),
) -> Path:
    token = hf_token()
    api = HfApi(token=token)
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parents[1] / output_root
    prefix = f"runs/{run_id}/outputs/"
    files = [
        path
        for path in api.list_repo_files(artifact_repo, repo_type="dataset", token=token)
        if path.startswith(prefix)
    ]
    if not files:
        raise RuntimeError(f"No output files found under {prefix}.")

    run_dir = output_root / f"hf_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for repo_path in files:
        downloaded = Path(
            hf_hub_download(
                artifact_repo,
                repo_path,
                repo_type="dataset",
                token=token,
            )
        )
        local_path = run_dir / Path(repo_path).name
        shutil.copy2(downloaded, local_path)
    return run_dir


def finalize_remote_review(run_dir: Path) -> dict[str, object]:
    preds_path = run_dir / "preds.npy"
    segments_path = run_dir / "segments.pkl"
    input_video = run_dir / "input_normalized.mp4"
    if not preds_path.exists():
        raise RuntimeError(f"Missing predictions: {preds_path}")

    brain_video = run_dir / "brain_left_right.mp4"
    if not brain_video.exists():
        render_left_right_brain_video(
            preds_path,
            brain_video,
            segments_path=segments_path if segments_path.exists() else None,
        )

    activation_timeseries(preds_path).to_csv(
        run_dir / "activation_timeseries.csv", index=False
    )
    try:
        roi_summary(preds_path).to_csv(run_dir / "roi_summary.csv", index=False)
    except Exception as exc:
        (run_dir / "roi_summary_error.txt").write_text(str(exc))

    summary = {
        "run_dir": str(run_dir),
        "input_video": str(input_video),
        "brain_video": str(brain_video),
        "preds_path": str(preds_path),
        "segments_path": str(segments_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
