from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasketch import MinHash
from langdetect import LangDetectException, detect
from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import sha256_file, stable_dict_hash
from tube2lora.utils.io import iter_jsonl, write_jsonl
from tube2lora.utils.text_metrics import (
    count_sentences,
    count_tokens,
    flesch_to_unit_interval,
    lexical_metrics,
    readability_metrics,
    safe_ratio,
    tokenize_words,
)


ANALYZE_PROFILE_SCHEMA = "v3"


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _quality_score(
    *,
    word_count: int,
    sentence_count: int,
    unique_ratio: float,
    hapax_ratio: float,
    long_word_ratio: float,
    flesch_reading_ease: float | None,
) -> float:
    if word_count == 0:
        return 0.0

    sentence_density = min(1.0, sentence_count / max(1, word_count / 20))
    lexical_score = 0.6 * unique_ratio + 0.4 * hapax_ratio
    readability_score = flesch_to_unit_interval(flesch_reading_ease)
    complexity_penalty = min(1.0, long_word_ratio * 2.0)

    score = (
        0.45 * lexical_score
        + 0.25 * sentence_density
        + 0.25 * readability_score
        + 0.05 * (1.0 - complexity_penalty)
    )
    return max(0.0, min(1.0, score))


def _minhash_signature(tokens: list[str], num_perm: int) -> list[int]:
    minhash = MinHash(num_perm=num_perm)
    for token in tokens:
        minhash.update(token.encode("utf-8"))
    return [int(value) for value in minhash.hashvalues]


def _metrics_rows_by_video(path: Path, *, stage: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required {stage} metrics manifest is missing: {path}")

    rows = [row for row in iter_jsonl(path)]
    by_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{stage} metrics row missing 'metrics' object for video_id={video_id}")
        by_video[video_id] = row
    return by_video


def _profile_stage_entry(stage: str, row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"{stage} metrics row missing 'metrics' object")
    return {
        "status": str(row.get("status", "unknown")),
        "metrics": metrics,
    }


def _analyze_metrics_row(row: dict[str, object]) -> dict[str, object]:
    if row.get("status") == "ready":
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            return metrics
        return {}
    return {"error": row.get("error", "unknown")}


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "analyze"
    context.update_stage_status(stage_name, "running")

    if not context.config.analyze.enabled:
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

    normalize_manifest = context.stage_dir("normalize") / "normalized.jsonl"
    if not normalize_manifest.exists():
        raise FileNotFoundError("Normalize stage output not found.")

    download_metrics_by_video = _metrics_rows_by_video(
        context.stage_dir("download") / "metrics.jsonl",
        stage="download",
    )
    transcribe_metrics_by_video = _metrics_rows_by_video(
        context.stage_dir("transcribe") / "metrics.jsonl",
        stage="transcribe",
    )
    normalize_metrics_by_video = _metrics_rows_by_video(
        context.stage_dir("normalize") / "metrics.jsonl",
        stage="normalize",
    )

    normalize_rows = [row for row in iter_jsonl(normalize_manifest)]
    ready_rows = [row for row in normalize_rows if row.get("status") == "ready"]

    stage_dir = context.stage_dir(stage_name)
    profiles_path = stage_dir / "profiles.jsonl"
    metrics_path = stage_dir / "metrics.jsonl"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    output_rows: list[dict[str, object]] = []
    success = 0
    failed = 0
    skipped = 0

    for row in tqdm(ready_rows, desc="analyze", unit="video"):
        video_id = str(row["video_id"])
        title = str(row.get("title", ""))
        text_path = Path(str(row["text_path"]))

        download_metrics_row = download_metrics_by_video.get(video_id)
        transcribe_metrics_row = transcribe_metrics_by_video.get(video_id)
        normalize_metrics_row = normalize_metrics_by_video.get(video_id)
        if download_metrics_row is None:
            raise RuntimeError(f"Missing download metrics for video_id={video_id}")
        if transcribe_metrics_row is None:
            raise RuntimeError(f"Missing transcribe metrics for video_id={video_id}")
        if normalize_metrics_row is None:
            raise RuntimeError(f"Missing normalize metrics for video_id={video_id}")

        input_hash = stable_dict_hash(
            {
                "schema": ANALYZE_PROFILE_SCHEMA,
                "video_id": video_id,
                "text_sha256": sha256_file(text_path),
                "analyze_cfg": context.config.analyze.model_dump(mode="json"),
                "download_metrics_row": download_metrics_row,
                "transcribe_metrics_row": transcribe_metrics_row,
                "normalize_metrics_row": normalize_metrics_row,
            }
        )

        if manifest.should_skip(video_id, input_hash):
            cached = manifest.items[video_id].get("output")
            if isinstance(cached, dict):
                output_rows.append(cached)
            skipped += 1
            continue

        try:
            text = text_path.read_text(encoding="utf-8").strip()
            tokens = tokenize_words(text)
            word_count = len(tokens)
            sentence_count = count_sentences(text)
            language = _detect_language(text)
            token_count = count_tokens(text, model=context.config.generate.model)
            lexical = lexical_metrics(text)
            readability = readability_metrics(text)
            unique_ratio = float(lexical["unique_ratio"])
            hapax_ratio = float(lexical["hapax_ratio"])
            long_word_ratio = float(lexical["long_word_ratio"])
            quality_score = _quality_score(
                word_count=word_count,
                sentence_count=sentence_count,
                unique_ratio=unique_ratio,
                hapax_ratio=hapax_ratio,
                long_word_ratio=long_word_ratio,
                flesch_reading_ease=readability["flesch_reading_ease"],
            )
            signature = _minhash_signature(tokens, context.config.analyze.minhash_num_perm)

            analyze_metrics = {
                "language": language,
                "token_count": token_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "unique_ratio": unique_ratio,
                "hapax_ratio": hapax_ratio,
                "long_word_ratio": long_word_ratio,
                "avg_word_length": lexical["avg_word_length"],
                "flesch_reading_ease": readability["flesch_reading_ease"],
                "automated_readability_index": readability["automated_readability_index"],
                "gunning_fog": readability["gunning_fog"],
                "quality_score": quality_score,
                "minhash": signature,
                "avg_words_per_sentence": safe_ratio(word_count, max(1, sentence_count)),
            }

            profile = {
                "download": _profile_stage_entry("download", download_metrics_row),
                "transcribe": _profile_stage_entry("transcribe", transcribe_metrics_row),
                "normalize": _profile_stage_entry("normalize", normalize_metrics_row),
                "analyze": {"status": "ready", "metrics": analyze_metrics},
            }

            out_row = {
                "video_id": video_id,
                "title": title,
                "status": "ready",
                "text_path": str(text_path),
                "profile": profile,
                "language": language,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "unique_ratio": unique_ratio,
                "token_count": token_count,
                "quality_score": quality_score,
                "minhash": signature,
                "metrics": analyze_metrics,
            }
            output_rows.append(out_row)
            manifest.mark(video_id, status="success", input_hash=input_hash, output=out_row)
            success += 1
        except Exception as exc:
            logger.exception("Analyze failed for %s (%s)", video_id, title)
            out_row = {
                "video_id": video_id,
                "title": title,
                "status": "failed",
                "error": str(exc),
            }
            output_rows.append(out_row)
            manifest.mark(video_id, status="failed", input_hash=input_hash, output=out_row, error=str(exc))
            failed += 1

    output_rows.sort(key=lambda item: str(item.get("video_id", "")))
    write_jsonl(profiles_path, output_rows)
    write_jsonl(
        metrics_path,
        [
            {
                "video_id": row.get("video_id"),
                "title": row.get("title"),
                "status": row.get("status"),
                "metrics": _analyze_metrics_row(row),
            }
            for row in output_rows
        ],
    )

    report = StageReport(
        stage=stage_name,
        total=len(ready_rows),
        success=success,
        failed=failed,
        skipped=skipped,
        output_path=str(profiles_path),
    )
    context.update_stage_status(
        stage_name,
        "completed" if failed == 0 else "failed",
        details={
            "summary": {
                "total": report.total,
                "success": report.success,
                "failed": report.failed,
                "skipped": report.skipped,
            },
            "output_path": str(profiles_path),
            "metrics_path": str(metrics_path),
        },
    )
    return report
