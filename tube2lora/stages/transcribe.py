from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yt_dlp
from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.stages.common import StageReport
from tube2lora.transcribe.backends import (
    FasterWhisperTranscriber,
    TranscriptionResult,
    Transcriber,
    fetch_youtube_transcript,
)
from tube2lora.utils.hashing import sha256_file, stable_dict_hash
from tube2lora.utils.io import atomic_write_json, atomic_write_text, iter_jsonl, write_jsonl


def _select_backend(context: RunContext, logger: logging.Logger) -> Transcriber | None:
    backend = context.config.transcribe.backend
    if backend == "faster_whisper":
        return FasterWhisperTranscriber(context.config.transcribe.faster_whisper, logger)
    if backend == "youtube_native":
        return None
    raise ValueError(f"Unsupported transcribe backend: {backend}")


def _download_audio(video_url: str, video_id: str, target_dir: Path) -> Path | None:
    target_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(target_dir / f"{video_id}.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(video_url, download=True)

    matches = list(target_dir.glob(f"{video_id}.*"))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_size, reverse=True)
    return matches[0]


def _ensure_audio_file(row: dict[str, Any], audio_cache_dir: Path, logger: logging.Logger) -> Path | None:
    media_path = row.get("media_path")
    if media_path:
        existing = Path(str(media_path))
        if existing.exists() and existing.stat().st_size > 0:
            return existing

    video_id = str(row["video_id"])
    url = str(row["url"])
    logger.info("No media_path for %s, downloading audio on demand", video_id)
    return _download_audio(url, video_id, audio_cache_dir)


def _serialize_transcript(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "source": result.source,
        "language": result.language,
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in result.segments
        ],
        "text": result.text,
    }


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "transcribe"
    context.update_stage_status(stage_name, "running")

    download_manifest_path = context.stage_dir("download") / "videos.jsonl"
    if not download_manifest_path.exists():
        raise FileNotFoundError(
            "Download stage output not found. Run 'tube2lora download' first or run full pipeline."
        )

    rows = [row for row in iter_jsonl(download_manifest_path)]
    ready_rows = [row for row in rows if row.get("status") == "ready"]

    stage_dir = context.stage_dir(stage_name)
    artifact_dir = context.stage_artifact_dir(stage_name)
    audio_cache_dir = artifact_dir / "audio"
    output_manifest_path = stage_dir / "transcripts.jsonl"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    transcriber = _select_backend(context, logger)

    output_rows: list[dict[str, Any]] = []
    success = 0
    failed = 0
    skipped = 0

    for row in tqdm(ready_rows, desc="transcribe", unit="video"):
        video_id = str(row["video_id"])
        url = str(row["url"])
        title = str(row.get("title", ""))

        audio_hash = None
        media_path = row.get("media_path")
        if media_path:
            media_file = Path(str(media_path))
            if media_file.exists() and media_file.is_file():
                audio_hash = sha256_file(media_file)

        input_hash = stable_dict_hash(
            {
                "video_id": video_id,
                "url": url,
                "title": title,
                "transcribe_cfg": context.config.transcribe.model_dump(mode="json"),
                "audio_hash": audio_hash,
            }
        )

        if manifest.should_skip(video_id, input_hash):
            cached_output = manifest.items[video_id].get("output")
            if isinstance(cached_output, dict):
                output_rows.append(cached_output)
            skipped += 1
            continue

        try:
            transcript_result: TranscriptionResult | None = None

            if context.config.transcribe.prefer_manual_transcript:
                yt_result = fetch_youtube_transcript(
                    video_id=video_id,
                    language_allowlist=context.config.transcribe.language_allowlist,
                    manual_only=True,
                    allow_auto_generated=context.config.transcribe.allow_auto_generated,
                    logger=logger,
                )

                if yt_result is not None:
                    if yt_result.source == "youtube_manual_filtered" and not yt_result.segments:
                        out_row = {
                            "video_id": video_id,
                            "title": title,
                            "url": url,
                            "status": "skipped",
                            "skip_reason": "language_filtered_manual_transcript",
                            "language": yt_result.language,
                        }
                        output_rows.append(out_row)
                        manifest.mark(
                            video_id,
                            status="skipped",
                            input_hash=input_hash,
                            output=out_row,
                            skipped_reason="language_filtered_manual_transcript",
                        )
                        skipped += 1
                        continue

                    transcript_result = yt_result

            if transcript_result is None:
                if context.config.transcribe.backend == "youtube_native":
                    out_row = {
                        "video_id": video_id,
                        "title": title,
                        "url": url,
                        "status": "skipped",
                        "skip_reason": "no_manual_transcript_and_backend_is_youtube_native",
                    }
                    output_rows.append(out_row)
                    manifest.mark(
                        video_id,
                        status="skipped",
                        input_hash=input_hash,
                        output=out_row,
                        skipped_reason="no_manual_transcript",
                    )
                    skipped += 1
                    continue

                audio_file = _ensure_audio_file(row, audio_cache_dir, logger)
                if audio_file is None:
                    raise RuntimeError("Unable to locate or download audio for transcription")

                if transcriber is None:
                    raise RuntimeError("No backend transcriber available")

                transcript_result = transcriber.transcribe(audio_file)

            if transcript_result.language and (
                transcript_result.language not in context.config.transcribe.language_allowlist
            ):
                out_row = {
                    "video_id": video_id,
                    "title": title,
                    "url": url,
                    "status": "skipped",
                    "skip_reason": "language_filtered",
                    "language": transcript_result.language,
                }
                output_rows.append(out_row)
                manifest.mark(
                    video_id,
                    status="skipped",
                    input_hash=input_hash,
                    output=out_row,
                    skipped_reason="language_filtered",
                )
                skipped += 1
                continue

            serialized = _serialize_transcript(transcript_result)
            json_path = stage_dir / f"{video_id}.json"
            txt_path = stage_dir / f"{video_id}.txt"
            atomic_write_json(json_path, serialized)
            atomic_write_text(txt_path, serialized["text"] + "\n")

            out_row = {
                "video_id": video_id,
                "title": title,
                "url": url,
                "status": "ready",
                "transcript_path": str(json_path),
                "text_path": str(txt_path),
                "source": serialized["source"],
                "language": serialized["language"],
                "num_segments": len(serialized["segments"]),
                "num_chars": len(serialized["text"]),
            }
            output_rows.append(out_row)
            manifest.mark(video_id, status="success", input_hash=input_hash, output=out_row)
            success += 1
        except Exception as exc:
            logger.exception("Transcription failed for %s (%s)", video_id, title)
            out_row = {
                "video_id": video_id,
                "title": title,
                "url": url,
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
