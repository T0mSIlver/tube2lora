from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from datasketch import MinHash

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import stable_dict_hash
from tube2lora.utils.io import iter_jsonl, write_jsonl


FILTER_SCHEMA_VERSION = "v3"


def _to_minhash(hash_values: list[int]) -> MinHash:
    minhash = MinHash(num_perm=len(hash_values))
    minhash.hashvalues = np.array(hash_values, dtype=np.uint64)
    return minhash


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_stage_metrics(row: dict[str, Any], stage: str) -> dict[str, Any]:
    profile = row.get("profile")
    if not isinstance(profile, dict):
        raise RuntimeError(f"Filter row missing profile object for video_id={row.get('video_id')}")
    stage_payload = profile.get(stage)
    if not isinstance(stage_payload, dict):
        raise RuntimeError(f"Filter row missing profile.{stage} for video_id={row.get('video_id')}")
    metrics = stage_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(
            f"Filter row missing profile.{stage}.metrics for video_id={row.get('video_id')}"
        )
    return metrics


def _load_analyze_rows(source_manifest: Path) -> list[dict[str, Any]]:
    return [row for row in iter_jsonl(source_manifest) if isinstance(row, dict)]


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "filter"
    context.update_stage_status(stage_name, "running")

    if not context.config.filter.enabled:
        report = StageReport(stage=stage_name, total=0, success=0, failed=0, skipped=0, output_path=None)
        context.update_stage_status(
            stage_name,
            "completed",
            details={
                "summary": {
                    "total": report.total,
                    "success": report.success,
                    "failed": report.failed,
                    "skipped": report.skipped,
                }
            },
        )
        return report

    source_manifest = context.stage_dir("analyze") / "profiles.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError("Analyze stage output not found.")

    rows = _load_analyze_rows(source_manifest)
    ready_rows = [row for row in rows if row.get("status") == "ready"]

    stage_dir = context.stage_dir(stage_name)
    keep_path = stage_dir / "kept.jsonl"
    drop_path = stage_dir / "dropped.jsonl"
    metrics_path = stage_dir / "metrics.jsonl"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    kept_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []

    cfg = context.config.filter
    kept_signatures: list[tuple[str, MinHash]] = []

    for row in ready_rows:
        video_id = str(row["video_id"])
        title = str(row.get("title", ""))

        input_hash = stable_dict_hash(
            {
                "schema": FILTER_SCHEMA_VERSION,
                "video_id": video_id,
                "row": row,
                "filter_cfg": cfg.model_dump(mode="json"),
            }
        )

        analyze_metrics = _require_stage_metrics(row, "analyze")
        normalize_metrics = _require_stage_metrics(row, "normalize")

        if "language" not in analyze_metrics:
            raise RuntimeError(f"Analyze profile missing language for video_id={video_id}")
        if "word_count" not in analyze_metrics:
            raise RuntimeError(f"Analyze profile missing word_count for video_id={video_id}")
        if "token_count" not in analyze_metrics:
            raise RuntimeError(f"Analyze profile missing token_count for video_id={video_id}")
        if "quality_score" not in analyze_metrics:
            raise RuntimeError(f"Analyze profile missing quality_score for video_id={video_id}")
        if "minhash" not in analyze_metrics:
            raise RuntimeError(f"Analyze profile missing minhash for video_id={video_id}")

        language = str(analyze_metrics["language"])
        word_count = int(analyze_metrics["word_count"])
        token_count = int(analyze_metrics["token_count"])
        quality_score = float(analyze_metrics["quality_score"])
        signature_raw = analyze_metrics["minhash"]

        normalize_token_ratio = _as_float(normalize_metrics.get("token_ratio"))
        normalize_length_ratio = _as_float(normalize_metrics.get("length_ratio"))
        normalize_similarity = _as_float(normalize_metrics.get("char_similarity"))
        if normalize_token_ratio is None:
            raise RuntimeError(f"Normalize profile missing token_ratio for video_id={video_id}")
        if normalize_length_ratio is None:
            raise RuntimeError(f"Normalize profile missing length_ratio for video_id={video_id}")
        if normalize_similarity is None:
            raise RuntimeError(f"Normalize profile missing char_similarity for video_id={video_id}")

        reason: str | None = None

        if language not in cfg.language_allowlist:
            reason = "language"
        elif token_count < cfg.min_tokens:
            reason = "min_tokens"
        elif word_count < cfg.min_words:
            reason = "min_words"
        elif quality_score < cfg.min_quality_score:
            reason = "quality"
        elif (
            cfg.normalize_min_token_ratio is not None
            and normalize_token_ratio < cfg.normalize_min_token_ratio
        ):
            reason = "normalize_token_ratio_low"
        elif (
            cfg.normalize_max_token_ratio is not None
            and normalize_token_ratio > cfg.normalize_max_token_ratio
        ):
            reason = "normalize_token_ratio_high"
        elif (
            cfg.normalize_min_length_ratio is not None
            and normalize_length_ratio < cfg.normalize_min_length_ratio
        ):
            reason = "normalize_length_ratio_low"
        elif (
            cfg.normalize_max_length_ratio is not None
            and normalize_length_ratio > cfg.normalize_max_length_ratio
        ):
            reason = "normalize_length_ratio_high"
        elif (
            cfg.normalize_min_similarity is not None
            and normalize_similarity < cfg.normalize_min_similarity
        ):
            reason = "normalize_similarity"

        duplicate_of: str | None = None
        if reason is None:
            if not isinstance(signature_raw, list):
                reason = "missing_minhash"
            else:
                current_sig = _to_minhash([int(v) for v in signature_raw])
                for existing_id, existing_sig in kept_signatures:
                    sim = current_sig.jaccard(existing_sig)
                    if sim >= cfg.dedup_threshold:
                        duplicate_of = existing_id
                        break

                if duplicate_of is not None:
                    reason = f"dedup:{duplicate_of}"
                else:
                    kept_signatures.append((video_id, current_sig))

        decision_metrics = {
            "language": language,
            "word_count": word_count,
            "token_count": token_count,
            "quality_score": quality_score,
            "normalize_token_ratio": normalize_token_ratio,
            "normalize_length_ratio": normalize_length_ratio,
            "normalize_similarity": normalize_similarity,
            "duplicate_of": duplicate_of,
        }

        if reason is None:
            out_row = {
                "video_id": video_id,
                "title": title,
                "status": "kept",
                "text_path": row.get("text_path"),
                "profile": row.get("profile", {}),
                "metrics": decision_metrics,
            }
            kept_rows.append(out_row)
            decision_rows.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "status": "kept",
                    "reason": None,
                    "metrics": decision_metrics,
                }
            )
            manifest.mark(video_id, status="success", input_hash=input_hash, output=out_row)
        else:
            out_row = {
                "video_id": video_id,
                "title": title,
                "status": "dropped",
                "reason": reason,
                "text_path": row.get("text_path"),
                "profile": row.get("profile", {}),
                "metrics": decision_metrics,
            }
            dropped_rows.append(out_row)
            decision_rows.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "status": "dropped",
                    "reason": reason,
                    "metrics": decision_metrics,
                }
            )
            manifest.mark(
                video_id,
                status="skipped",
                input_hash=input_hash,
                output=out_row,
                skipped_reason=reason,
            )

    kept_rows.sort(key=lambda item: str(item.get("video_id", "")))
    dropped_rows.sort(key=lambda item: str(item.get("video_id", "")))
    decision_rows.sort(key=lambda item: str(item.get("video_id", "")))
    write_jsonl(keep_path, kept_rows)
    write_jsonl(drop_path, dropped_rows)
    write_jsonl(metrics_path, decision_rows)

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
            "metrics_path": str(metrics_path),
        },
    )
    return report
