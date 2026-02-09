from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tube2lora.config import AppConfig
from tube2lora.utils.io import atomic_write_json, ensure_dir, read_json


PIPELINE_STAGES = [
    "download",
    "transcribe",
    "normalize",
    "analyze",
    "filter",
    "generate",
    "train",
    "evaluate",
]


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(slots=True)
class RunContext:
    config: AppConfig | None
    run_id: str
    config_hash: str
    root: Path

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def run_meta_path(self) -> Path:
        return self.root / "run.json"

    @property
    def pipeline_log_path(self) -> Path:
        return self.logs_dir / "pipeline.log"

    def stage_dir(self, stage: str) -> Path:
        path = self.stages_dir / stage
        ensure_dir(path)
        return path

    def stage_artifact_dir(self, stage: str) -> Path:
        path = self.artifacts_dir / stage
        ensure_dir(path)
        return path

    def stage_manifest_path(self, stage: str) -> Path:
        ensure_dir(self.manifests_dir)
        return self.manifests_dir / f"{stage}.json"

    def read_meta(self) -> dict[str, Any]:
        return read_json(self.run_meta_path, default={})

    def write_meta(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.run_meta_path, payload)

    def update_stage_status(
        self,
        stage: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        meta = self.read_meta()
        stage_status = dict(meta.get("stage_status", {}))
        stage_entry = dict(stage_status.get(stage, {}))
        stage_entry["status"] = status
        stage_entry["updated_at"] = utc_now_iso()
        if details:
            stage_entry.update(details)
        stage_status[stage] = stage_entry

        meta["stage_status"] = stage_status
        meta["updated_at"] = utc_now_iso()

        if status == "failed":
            meta["status"] = "failed"
        elif all(
            stage_status.get(item, {}).get("status") == "completed" for item in PIPELINE_STAGES
        ):
            meta["status"] = "completed"
        else:
            meta["status"] = "running"

        self.write_meta(meta)


class ManifestStore:
    def __init__(self, path: Path):
        self.path = path
        self.payload = read_json(path, default={"items": {}})
        self.items: dict[str, dict[str, Any]] = {
            key: value
            for key, value in self.payload.get("items", {}).items()
            if isinstance(value, dict)
        }

    def should_skip(self, item_id: str, input_hash: str) -> bool:
        record = self.items.get(item_id)
        if not record:
            return False
        return (
            record.get("status") in {"success", "skipped"}
            and record.get("input_hash") == input_hash
        )

    def mark(
        self,
        item_id: str,
        *,
        status: str,
        input_hash: str,
        output: dict[str, Any] | None = None,
        error: str | None = None,
        skipped_reason: str | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "item_id": item_id,
            "status": status,
            "input_hash": input_hash,
            "updated_at": utc_now_iso(),
        }
        if output:
            row["output"] = output
        if error:
            row["error"] = error
        if skipped_reason:
            row["skipped_reason"] = skipped_reason
        self.items[item_id] = row
        self.save()

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "updated_at": utc_now_iso(),
                "items": self.items,
            },
        )

    def summary(self) -> dict[str, int]:
        success = 0
        failed = 0
        skipped = 0
        for row in self.items.values():
            status = row.get("status")
            if status == "success":
                success += 1
            elif status == "failed":
                failed += 1
            elif status == "skipped":
                skipped += 1
        return {
            "total": len(self.items),
            "success": success,
            "failed": failed,
            "skipped": skipped,
        }


class RunManager:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        ensure_dir(self.runs_dir)

    def resolve_run(
        self,
        config: AppConfig,
        config_hash: str,
        run_id: str | None = None,
    ) -> RunContext:
        if run_id:
            root = self.runs_dir / run_id
            if not root.exists():
                raise FileNotFoundError(f"Run ID not found: {run_id}")
            return RunContext(config=config, run_id=run_id, config_hash=config_hash, root=root)

        resumable = self._find_latest_resumable(config=config, config_hash=config_hash)
        if resumable is not None:
            return resumable

        return self._create_run(config=config, config_hash=config_hash)

    def latest_run(self) -> RunContext | None:
        runs = self.list_runs()
        if not runs:
            return None
        latest = runs[0]
        run_meta = read_json(latest / "run.json", default={})
        return RunContext(
            config_hash=str(run_meta.get("config_hash", "")),
            config=None,
            run_id=latest.name,
            root=latest,
        )

    def list_runs(self) -> list[Path]:
        roots = [
            path
            for path in self.runs_dir.iterdir()
            if path.is_dir() and (path / "run.json").exists()
        ]
        roots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return roots

    def _find_latest_resumable(self, config: AppConfig, config_hash: str) -> RunContext | None:
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        for root in self.list_runs():
            meta = read_json(root / "run.json", default={})
            if meta.get("config_hash") != config_hash:
                continue
            status = meta.get("status")
            if status in {"running", "failed"}:
                candidates.append((root.stat().st_mtime, root, meta))

        if not candidates:
            return None

        _, root, _ = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
        return RunContext(config=config, run_id=root.name, config_hash=config_hash, root=root)

    def _create_run(self, config: AppConfig, config_hash: str) -> RunContext:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{config_hash[:10]}"
        root = self.runs_dir / run_id

        ensure_dir(root)
        ensure_dir(root / "stages")
        ensure_dir(root / "artifacts")
        ensure_dir(root / "logs")
        ensure_dir(root / "manifests")

        meta = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "running",
            "config_hash": config_hash,
            "config_path": str(config.config_path),
            "stage_status": {
                stage: {"status": "pending", "updated_at": utc_now_iso()}
                for stage in PIPELINE_STAGES
            },
        }
        atomic_write_json(root / "run.json", meta)
        return RunContext(config=config, run_id=run_id, config_hash=config_hash, root=root)
