from __future__ import annotations

import logging
import re
from pathlib import Path

from datasketch import MinHash
from langdetect import LangDetectException, detect
from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import sha256_file, stable_dict_hash
from tube2lora.utils.io import iter_jsonl, write_jsonl


WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _quality_score(word_count: int, unique_ratio: float, sentence_count: int) -> float:
    if word_count == 0:
        return 0.0
    sentence_density = min(1.0, sentence_count / max(1, word_count / 20))
    score = 0.6 * unique_ratio + 0.4 * sentence_density
    return max(0.0, min(1.0, score))


def _minhash_signature(tokens: list[str], num_perm: int) -> list[int]:
    minhash = MinHash(num_perm=num_perm)
    for token in tokens:
        minhash.update(token.encode("utf-8"))
    return [int(value) for value in minhash.hashvalues]


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

    source_manifest = context.stage_dir("normalize") / "normalized.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError("Normalize stage output not found.")

    rows = [row for row in iter_jsonl(source_manifest)]
    ready_rows = [row for row in rows if row.get("status") == "ready"]

    stage_dir = context.stage_dir(stage_name)
    output_manifest_path = stage_dir / "metrics.jsonl"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    output_rows: list[dict[str, object]] = []
    success = 0
    failed = 0
    skipped = 0

    for row in tqdm(ready_rows, desc="analyze", unit="video"):
        video_id = str(row["video_id"])
        title = str(row.get("title", ""))
        text_path = Path(str(row["text_path"]))

        input_hash = stable_dict_hash(
            {
                "video_id": video_id,
                "text_sha256": sha256_file(text_path),
                "analyze_cfg": context.config.analyze.model_dump(mode="json"),
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
            tokens = _tokenize(text)
            word_count = len(tokens)
            unique_ratio = (len(set(tokens)) / word_count) if word_count else 0.0
            sentence_count = len(SENTENCE_RE.findall(text))
            language = _detect_language(text)
            quality_score = _quality_score(word_count, unique_ratio, sentence_count)
            signature = _minhash_signature(tokens, context.config.analyze.minhash_num_perm)

            out_row = {
                "video_id": video_id,
                "title": title,
                "status": "ready",
                "text_path": str(text_path),
                "language": language,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "unique_ratio": unique_ratio,
                "quality_score": quality_score,
                "minhash": signature,
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
    write_jsonl(output_manifest_path, output_rows)

    report = StageReport(
        stage=stage_name,
        total=len(ready_rows),
        success=success,
        failed=failed,
        skipped=skipped,
        output_path=str(output_manifest_path),
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
            "output_path": str(output_manifest_path),
        },
    )
    return report
