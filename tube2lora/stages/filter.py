from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from datasketch import MinHash

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import stable_dict_hash
from tube2lora.utils.io import iter_jsonl, write_jsonl


def _to_minhash(hash_values: list[int]) -> MinHash:
    minhash = MinHash(num_perm=len(hash_values))
    minhash.hashvalues = np.array(hash_values, dtype=np.uint64)
    return minhash


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "filter"
    context.update_stage_status(stage_name, "running")

    if not context.config.filter.enabled:
        report = StageReport(stage=stage_name, total=0, success=0, failed=0, skipped=0, output_path=None)
        context.update_stage_status(stage_name, "completed", details={"summary": report.__dict__})
        return report

    source_manifest = context.stage_dir("analyze") / "metrics.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError("Analyze stage output not found.")

    rows = [row for row in iter_jsonl(source_manifest)]
    ready_rows = [row for row in rows if row.get("status") == "ready"]

    stage_dir = context.stage_dir(stage_name)
    keep_path = stage_dir / "kept.jsonl"
    drop_path = stage_dir / "dropped.jsonl"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    kept_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []

    cfg = context.config.filter

    kept_signatures: list[tuple[str, MinHash]] = []

    for row in ready_rows:
        video_id = str(row["video_id"])
        input_hash = stable_dict_hash(
            {
                "video_id": video_id,
                "row": row,
                "filter_cfg": cfg.model_dump(mode="json"),
            }
        )

        language = str(row.get("language", "unknown"))
        word_count = int(row.get("word_count", 0))
        quality_score = float(row.get("quality_score", 0.0))

        reason: str | None = None

        if language not in cfg.language_allowlist:
            reason = "language"
        elif word_count < cfg.min_words:
            reason = "min_words"
        elif quality_score < cfg.min_quality_score:
            reason = "quality"

        if reason is None:
            signature_raw = row.get("minhash")
            if not isinstance(signature_raw, list):
                reason = "missing_minhash"
            else:
                current_sig = _to_minhash([int(v) for v in signature_raw])
                duplicate_of = None
                for existing_id, existing_sig in kept_signatures:
                    sim = current_sig.jaccard(existing_sig)
                    if sim >= cfg.dedup_threshold:
                        duplicate_of = existing_id
                        break

                if duplicate_of is not None:
                    reason = f"dedup:{duplicate_of}"
                else:
                    kept_signatures.append((video_id, current_sig))

        if reason is None:
            out_row = {
                "video_id": video_id,
                "title": row.get("title"),
                "status": "kept",
                "text_path": row.get("text_path"),
                "language": language,
                "word_count": word_count,
                "quality_score": quality_score,
            }
            kept_rows.append(out_row)
            manifest.mark(video_id, status="success", input_hash=input_hash, output=out_row)
        else:
            out_row = {
                "video_id": video_id,
                "title": row.get("title"),
                "status": "dropped",
                "reason": reason,
                "text_path": row.get("text_path"),
                "language": language,
                "word_count": word_count,
                "quality_score": quality_score,
            }
            dropped_rows.append(out_row)
            manifest.mark(
                video_id,
                status="skipped",
                input_hash=input_hash,
                output=out_row,
                skipped_reason=reason,
            )

    kept_rows.sort(key=lambda item: str(item.get("video_id", "")))
    dropped_rows.sort(key=lambda item: str(item.get("video_id", "")))
    write_jsonl(keep_path, kept_rows)
    write_jsonl(drop_path, dropped_rows)

    report = StageReport(
        stage=stage_name,
        total=len(ready_rows),
        success=len(kept_rows),
        failed=0,
        skipped=len(dropped_rows),
        output_path=str(keep_path),
    )
    context.update_stage_status(
        stage_name,
        "completed",
        details={
            "summary": {
                "total": report.total,
                "success": report.success,
                "failed": report.failed,
                "skipped": report.skipped,
            },
            "output_path": str(keep_path),
            "dropped_path": str(drop_path),
        },
    )
    return report
