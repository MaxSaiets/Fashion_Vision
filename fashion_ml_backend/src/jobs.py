from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import uuid


JOBS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "jobs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job_dir() -> tuple[str, Path]:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def manifest_path(job_id: str) -> Path:
    return job_dir(job_id) / "manifest.json"


def write_manifest(job_id: str, payload: dict) -> dict:
    path = manifest_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def read_manifest(job_id: str) -> dict | None:
    path = manifest_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(job_id: str, **fields) -> dict:
    current = read_manifest(job_id)
    if current is None:
        raise FileNotFoundError(job_id)
    current.update(fields)
    current["updated_at"] = utc_now_iso()
    return write_manifest(job_id, current)


def create_manifest(job_id: str, feed_options: dict, file_names: list[str]) -> dict:
    payload = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "file_names": file_names,
        "feed_options": feed_options,
        "summary": {"total_items": 0, "categories": [], "colors": [], "patterns": [], "top_params": []},
        "items": [],
        "artifacts": {},
        "feeds": {},
        "error": None,
    }
    return write_manifest(job_id, payload)
