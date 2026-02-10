from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.config import load_normalize_prompt_template
from tube2lora.llm.client import ChatRequest, OpenAIChatClient
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import sha256_bytes, sha256_file, stable_dict_hash
from tube2lora.utils.io import atomic_write_json, atomic_write_text, iter_jsonl, write_jsonl
from tube2lora.utils.text_metrics import (
    char_similarity,
    count_sentences,
    count_tokens,
    count_words,
    lexical_metrics,
    readability_metrics,
    safe_ratio,
)

NORMALIZE_OUTPUT_START = "<<<NORMALIZED_TRANSCRIPT_START>>>"
NORMALIZE_OUTPUT_END = "<<<NORMALIZED_TRANSCRIPT_END>>>"
NORMALIZE_BANNED_PHRASES = (
    "here is the cleaned transcript",
    "here’s the cleaned transcript",
    "cleaned transcript:",
    "---",
    "note:",
)
NORMALIZE_SCHEMA_VERSION = "v3"


@dataclass(slots=True)
class SourceSegment:
    start: float | None
    end: float | None
    text: str


@dataclass(slots=True)
class NormalizeChunk:
    chunk_index: int
    start_segment_index: int
    end_segment_index: int
    start_time: float | None
    end_time: float | None
    text: str


def _parse_segment_timestamp(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_segments_from_text(text: str) -> list[SourceSegment]:
    cleaned = text.strip()
    if not cleaned:
        return []

    parts = [part.strip() for part in re.split(r"\n\s*\n+", cleaned) if part.strip()]
    if not parts:
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]

    return [SourceSegment(start=None, end=None, text=part) for part in parts]


def _read_transcript_segments(transcript_path: Path) -> list[SourceSegment]:
    if transcript_path.suffix == ".txt":
        return _fallback_segments_from_text(transcript_path.read_text(encoding="utf-8"))

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []

    segments_payload = payload.get("segments")
    parsed_segments: list[SourceSegment] = []
    if isinstance(segments_payload, list):
        for item in segments_payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            parsed_segments.append(
                SourceSegment(
                    start=_parse_segment_timestamp(item.get("start")),
                    end=_parse_segment_timestamp(item.get("end")),
                    text=text,
                )
            )

    if parsed_segments:
        return parsed_segments

    text = str(payload.get("text", "")).strip()
    return _fallback_segments_from_text(text)


def _first_non_null_start(segments: list[SourceSegment]) -> float | None:
    for segment in segments:
        if segment.start is not None:
            return segment.start
    return None


def _last_non_null_end(segments: list[SourceSegment]) -> float | None:
    for segment in reversed(segments):
        if segment.end is not None:
            return segment.end
    return None


def _build_chunks(
    segments: list[SourceSegment],
    *,
    target_chars: int,
    hard_max_chars: int,
    overlap_chars: int,
) -> list[NormalizeChunk]:
    if not segments:
        return []

    chunks: list[NormalizeChunk] = []
    start_idx = 0
    chunk_index = 0
    total_segments = len(segments)

    while start_idx < total_segments:
        end_idx = start_idx
        chunk_chars = 0

        while end_idx < total_segments:
            segment_text = segments[end_idx].text.strip()
            if segment_text:
                added_chars = len(segment_text) + (1 if chunk_chars > 0 else 0)
                projected_chars = chunk_chars + added_chars
                if end_idx > start_idx and chunk_chars >= target_chars:
                    break
                if end_idx > start_idx and projected_chars > hard_max_chars:
                    break
                chunk_chars = projected_chars
            end_idx += 1

        if end_idx <= start_idx:
            end_idx = start_idx + 1

        chunk_segments = segments[start_idx:end_idx]
        chunk_text = " ".join(segment.text.strip() for segment in chunk_segments if segment.text.strip()).strip()
        if chunk_text:
            chunks.append(
                NormalizeChunk(
                    chunk_index=chunk_index,
                    start_segment_index=start_idx,
                    end_segment_index=end_idx - 1,
                    start_time=_first_non_null_start(chunk_segments),
                    end_time=_last_non_null_end(chunk_segments),
                    text=chunk_text,
                )
            )
            chunk_index += 1

        if end_idx >= total_segments:
            break

        if overlap_chars <= 0:
            start_idx = end_idx
            continue

        overlap_start = end_idx
        overlap_count = 0
        while overlap_start > start_idx:
            overlap_start -= 1
            overlap_count += len(segments[overlap_start].text.strip()) + 1
            if overlap_count >= overlap_chars:
                break

        if overlap_start <= start_idx:
            overlap_start = min(end_idx, start_idx + 1)

        start_idx = overlap_start

    return chunks


def _dedupe_boundary(previous_chunk: str, current_chunk: str) -> str:
    previous = previous_chunk.strip()
    current = current_chunk.strip()
    if not previous:
        return current
    if not current:
        return ""

    previous_lines = [line.strip() for line in previous.splitlines() if line.strip()]
    current_lines = [line.strip() for line in current.splitlines() if line.strip()]
    max_line_overlap = min(len(previous_lines), len(current_lines), 3)

    for overlap in range(max_line_overlap, 0, -1):
        if previous_lines[-overlap:] == current_lines[:overlap]:
            trimmed_lines = current_lines[overlap:]
            return "\n".join(trimmed_lines).strip()

    previous_words = previous.split()
    current_words = current.split()
    max_word_overlap = min(len(previous_words), len(current_words), 80)
    for overlap in range(max_word_overlap, 10, -1):
        if previous_words[-overlap:] == current_words[:overlap]:
            return " ".join(current_words[overlap:]).strip()

    return current


def _merge_normalized_chunks(chunks: list[str]) -> str:
    merged: list[str] = []
    for chunk in chunks:
        candidate = chunk.strip()
        if not candidate:
            continue
        if not merged:
            merged.append(candidate)
            continue
        deduped = _dedupe_boundary(merged[-1], candidate)
        if deduped:
            merged.append(deduped)
    return "\n\n".join(merged).strip()


def _extract_delimited_normalized_text(raw: str) -> str:
    start_idx = raw.find(NORMALIZE_OUTPUT_START)
    if start_idx < 0:
        raise RuntimeError("Normalized response missing start separator")

    content_start = start_idx + len(NORMALIZE_OUTPUT_START)
    end_idx = raw.find(NORMALIZE_OUTPUT_END, content_start)
    if end_idx < 0:
        raise RuntimeError("Normalized response missing end separator")

    extracted = raw[content_start:end_idx].strip()
    if not extracted:
        raise RuntimeError("Normalized response between separators is empty")
    return extracted


def _validate_normalized_chunk_text(chunk_text: str) -> None:
    lowered = chunk_text.lower()
    for phrase in NORMALIZE_BANNED_PHRASES:
        if phrase in lowered:
            raise RuntimeError(f"Normalized chunk contains forbidden meta text: '{phrase}'")


def _extract_metrics_from_row(row: dict[str, object]) -> dict[str, object]:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        return metrics

    status = str(row.get("status", "unknown"))
    if status == "ready":
        return {
            "num_chunks": int(row.get("num_chunks", 0)),
            "normalized_num_chars": int(row.get("num_chars", 0)),
        }
    if status == "skipped":
        return {"skip_reason": row.get("skip_reason", "unknown")}
    if status == "failed":
        return {"error": row.get("error", "unknown")}
    return {}


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "normalize"
    context.update_stage_status(stage_name, "running")

    if not context.config.normalize.enabled:
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

    source_manifest = context.stage_dir("transcribe") / "transcripts.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError("Transcribe stage output not found.")

    rows = [row for row in iter_jsonl(source_manifest)]
    ready_rows = [row for row in rows if row.get("status") == "ready"]

    prompt_template = load_normalize_prompt_template(context.config.normalize.prompt_template)
    prompt_hash = stable_dict_hash(prompt_template.model_dump(mode="json"))

    client = OpenAIChatClient(context.config.llm)

    stage_dir = context.stage_dir(stage_name)
    manifest = ManifestStore(context.stage_manifest_path(stage_name))
    output_manifest_path = stage_dir / "normalized.jsonl"
    metrics_manifest_path = stage_dir / "metrics.jsonl"

    output_rows: list[dict[str, object]] = []
    success = 0
    failed = 0
    skipped = 0

    for row in tqdm(ready_rows, desc="normalize", unit="video"):
        video_id = str(row["video_id"])
        title = str(row.get("title", ""))
        transcript_path = Path(str(row["transcript_path"]))

        input_hash = stable_dict_hash(
            {
                "schema": NORMALIZE_SCHEMA_VERSION,
                "video_id": video_id,
                "transcript_sha256": sha256_file(transcript_path),
                "prompt_hash": prompt_hash,
                "normalize_cfg": context.config.normalize.model_dump(mode="json"),
            }
        )

        if manifest.should_skip(video_id, input_hash):
            cached = manifest.items[video_id].get("output")
            if isinstance(cached, dict):
                output_rows.append(cached)
            skipped += 1
            continue

        try:
            transcript_segments = _read_transcript_segments(transcript_path)
            if not transcript_segments:
                out_row = {
                    "video_id": video_id,
                    "title": title,
                    "status": "skipped",
                    "skip_reason": "empty_transcript",
                }
                output_rows.append(out_row)
                manifest.mark(
                    video_id,
                    status="skipped",
                    input_hash=input_hash,
                    output=out_row,
                    skipped_reason="empty_transcript",
                )
                skipped += 1
                continue

            chunks = _build_chunks(
                transcript_segments,
                target_chars=context.config.normalize.chunk_target_chars,
                hard_max_chars=context.config.normalize.chunk_hard_max_chars,
                overlap_chars=context.config.normalize.chunk_overlap_chars,
            )
            if not chunks:
                raise RuntimeError("No transcript chunks were generated for normalization")

            logger.info("Normalizing %s in %d chunks", video_id, len(chunks))

            normalized_chunks: list[str] = []
            chunk_rows: list[dict[str, object]] = []
            chunk_dir = stage_dir / "chunks" / video_id
            chunk_dir.mkdir(parents=True, exist_ok=True)
            for chunk in chunks:
                user_prompt = prompt_template.user_prompt_template.format(
                    transcript=chunk.text,
                    title=title,
                    video_id=video_id,
                    chunk_index=chunk.chunk_index,
                    num_chunks=len(chunks),
                    chunk_start_time=chunk.start_time,
                    chunk_end_time=chunk.end_time,
                )
                completion = client.complete(
                    ChatRequest(
                        model=context.config.normalize.model,
                        system_prompt=prompt_template.system_prompt,
                        user_prompt=user_prompt,
                        temperature=context.config.normalize.temperature,
                        max_tokens=context.config.normalize.max_tokens,
                    )
                )
                normalized_chunk = _extract_delimited_normalized_text(completion)
                _validate_normalized_chunk_text(normalized_chunk)

                normalized_chunks.append(normalized_chunk)
                source_excerpt_path = chunk_dir / f"chunk_{chunk.chunk_index:04d}_source.txt"
                normalized_excerpt_path = chunk_dir / f"chunk_{chunk.chunk_index:04d}_normalized.txt"
                atomic_write_text(source_excerpt_path, chunk.text + "\n")
                atomic_write_text(normalized_excerpt_path, normalized_chunk + "\n")
                chunk_rows.append(
                    {
                        "chunk_index": chunk.chunk_index,
                        "start_segment_index": chunk.start_segment_index,
                        "end_segment_index": chunk.end_segment_index,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "source_num_chars": len(chunk.text),
                        "normalized_num_chars": len(normalized_chunk),
                        "source_excerpt_path": str(source_excerpt_path),
                        "normalized_excerpt_path": str(normalized_excerpt_path),
                        "source_excerpt_sha256": sha256_bytes(chunk.text.encode("utf-8")),
                        "normalized_excerpt_sha256": sha256_bytes(
                            normalized_chunk.encode("utf-8")
                        ),
                        "source_token_count": count_tokens(
                            chunk.text,
                            model=context.config.normalize.model,
                        ),
                        "normalized_token_count": count_tokens(
                            normalized_chunk,
                            model=context.config.normalize.model,
                        ),
                    }
                )

            normalized_text = _merge_normalized_chunks(normalized_chunks)
            if not normalized_text:
                raise RuntimeError("Normalized transcript is empty after chunk merge")

            source_text = " ".join(
                segment.text.strip()
                for segment in transcript_segments
                if segment.text.strip()
            ).strip()
            source_num_chars = len(source_text)
            normalized_num_chars = len(normalized_text)
            source_token_count = count_tokens(source_text, model=context.config.normalize.model)
            normalized_token_count = count_tokens(normalized_text, model=context.config.normalize.model)
            source_word_count = count_words(source_text)
            normalized_word_count = count_words(normalized_text)
            source_sentence_count = count_sentences(source_text)
            normalized_sentence_count = count_sentences(normalized_text)
            source_lexical = lexical_metrics(source_text)
            normalized_lexical = lexical_metrics(normalized_text)
            source_readability = readability_metrics(source_text)
            normalized_readability = readability_metrics(normalized_text)

            metrics = {
                "num_chunks": len(chunks),
                "source_num_chars": source_num_chars,
                "normalized_num_chars": normalized_num_chars,
                "length_delta_chars": normalized_num_chars - source_num_chars,
                "length_ratio": safe_ratio(normalized_num_chars, source_num_chars),
                "source_token_count": source_token_count,
                "normalized_token_count": normalized_token_count,
                "token_delta": normalized_token_count - source_token_count,
                "token_ratio": safe_ratio(normalized_token_count, source_token_count),
                "source_word_count": source_word_count,
                "normalized_word_count": normalized_word_count,
                "word_delta": normalized_word_count - source_word_count,
                "word_ratio": safe_ratio(normalized_word_count, source_word_count),
                "source_sentence_count": source_sentence_count,
                "normalized_sentence_count": normalized_sentence_count,
                "sentence_delta": normalized_sentence_count - source_sentence_count,
                "char_similarity": char_similarity(source_text, normalized_text),
                "source_unique_ratio": source_lexical["unique_ratio"],
                "normalized_unique_ratio": normalized_lexical["unique_ratio"],
                "source_hapax_ratio": source_lexical["hapax_ratio"],
                "normalized_hapax_ratio": normalized_lexical["hapax_ratio"],
                "source_avg_word_length": source_lexical["avg_word_length"],
                "normalized_avg_word_length": normalized_lexical["avg_word_length"],
                "source_long_word_ratio": source_lexical["long_word_ratio"],
                "normalized_long_word_ratio": normalized_lexical["long_word_ratio"],
                "source_flesch_reading_ease": source_readability["flesch_reading_ease"],
                "normalized_flesch_reading_ease": normalized_readability["flesch_reading_ease"],
                "source_automated_readability_index": source_readability[
                    "automated_readability_index"
                ],
                "normalized_automated_readability_index": normalized_readability[
                    "automated_readability_index"
                ],
                "source_gunning_fog": source_readability["gunning_fog"],
                "normalized_gunning_fog": normalized_readability["gunning_fog"],
            }

            normalized_json = {
                "video_id": video_id,
                "title": title,
                "source_transcript_path": str(transcript_path),
                "num_chunks": len(chunks),
                "chunks": chunk_rows,
                "normalized_text": normalized_text,
                "num_chars": len(normalized_text),
                "metrics": metrics,
            }

            json_path = stage_dir / f"{video_id}.json"
            txt_path = stage_dir / f"{video_id}.txt"
            atomic_write_json(json_path, normalized_json)
            atomic_write_text(txt_path, normalized_text + "\n")

            out_row = {
                "video_id": video_id,
                "title": title,
                "status": "ready",
                "normalized_path": str(json_path),
                "text_path": str(txt_path),
                "num_chunks": len(chunks),
                "num_chars": len(normalized_text),
                "metrics": metrics,
            }
            output_rows.append(out_row)
            manifest.mark(video_id, status="success", input_hash=input_hash, output=out_row)
            success += 1
        except Exception as exc:
            logger.exception("Normalize failed for %s (%s)", video_id, title)
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
    write_jsonl(
        metrics_manifest_path,
        [
            {
                "video_id": row.get("video_id"),
                "title": row.get("title"),
                "status": row.get("status"),
                "metrics": _extract_metrics_from_row(row),
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
            "metrics_path": str(metrics_manifest_path),
        },
    )
    return report
