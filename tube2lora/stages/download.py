from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yt_dlp
from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import stable_dict_hash
from tube2lora.utils.io import iter_jsonl, write_jsonl


def _read_source_urls(source_type: str, source: str) -> list[str]:
    if source_type == "url_file":
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"URL file does not exist: {path}")
        urls: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            urls.append(value)
        return urls
    return [source]


def _flatten_entries(info: dict[str, Any]) -> list[dict[str, Any]]:
    if "entries" in info and isinstance(info["entries"], list):
        flattened: list[dict[str, Any]] = []
        for entry in info["entries"]:
            if not entry:
                continue
            if "entries" in entry and isinstance(entry["entries"], list):
                flattened.extend([x for x in entry["entries"] if x])
            else:
                flattened.append(entry)
        return flattened
    return [info]


def _collect_video_entries(source_urls: list[str], logger: logging.Logger) -> list[dict[str, Any]]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
    }

    entries: list[dict[str, Any]] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in source_urls:
            logger.info("Discovering videos from %s", url)
            info = ydl.extract_info(url, download=False)
            entries.extend(_flatten_entries(info))
    return entries


def _is_short(entry: dict[str, Any]) -> bool:
    url = str(entry.get("webpage_url", ""))
    duration = entry.get("duration")
    if "/shorts/" in url:
        return True
    if isinstance(duration, (int, float)) and duration > 0 and duration < 61:
        return True
    return False


def _is_live(entry: dict[str, Any]) -> bool:
    if entry.get("is_live"):
        return True
    live_status = str(entry.get("live_status", "")).lower()
    return live_status in {"is_live", "was_live", "post_live"}


def _download_media(
    *,
    video_url: str,
    video_id: str,
    output_dir: Path,
    media: str,
    audio_format: str,
    video_format: str,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / f"{video_id}.%(ext)s")

    if media == "audio":
        opts = {
            "format": audio_format,
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
    elif media == "video":
        opts = {
            "format": video_format,
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
        }
    else:
        return None

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        vid = info.get("id", video_id)

    candidates = list(output_dir.glob(f"{vid}.*"))
    if not candidates:
        candidates = list(output_dir.glob(f"{video_id}.*"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "download"
    context.update_stage_status(stage_name, "running")

    stage_dir = context.stage_dir(stage_name)
    artifact_dir = context.stage_artifact_dir(stage_name)
    media_dir = artifact_dir / "media"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))
    output_manifest_path = stage_dir / "videos.jsonl"

    source_urls = _read_source_urls(context.config.input.source_type, context.config.input.source)
    raw_entries = _collect_video_entries(source_urls, logger)

    deduped: dict[str, dict[str, Any]] = {}
    for entry in raw_entries:
        video_id = str(entry.get("id", "")).strip()
        if not video_id:
            continue
        deduped[video_id] = entry

    rows: list[dict[str, Any]] = []
    success = 0
    failed = 0
    skipped = 0

    for video_id, entry in tqdm(deduped.items(), desc="download", unit="video"):
        title = str(entry.get("title", ""))
        url = str(entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}")
        duration = entry.get("duration")
        live = _is_live(entry)
        short = _is_short(entry)

        input_hash = stable_dict_hash(
            {
                "video_id": video_id,
                "url": url,
                "download_cfg": context.config.download.model_dump(mode="json"),
                "duration": duration,
                "title": title,
                "live": live,
                "short": short,
            }
        )

        if manifest.should_skip(video_id, input_hash):
            cached = manifest.items[video_id].get("output", {})
            if isinstance(cached, dict):
                rows.append(cached)
            skipped += 1
            continue

        if short and not context.config.download.include_shorts:
            row = {
                "video_id": video_id,
                "title": title,
                "url": url,
                "duration": duration,
                "live_status": entry.get("live_status"),
                "status": "skipped",
                "skip_reason": "short_excluded",
            }
            rows.append(row)
            manifest.mark(video_id, status="skipped", input_hash=input_hash, output=row, skipped_reason="short_excluded")
            skipped += 1
            continue

        if live and not context.config.download.include_live:
            row = {
                "video_id": video_id,
                "title": title,
                "url": url,
                "duration": duration,
                "live_status": entry.get("live_status"),
                "status": "skipped",
                "skip_reason": "live_excluded",
            }
            rows.append(row)
            manifest.mark(video_id, status="skipped", input_hash=input_hash, output=row, skipped_reason="live_excluded")
            skipped += 1
            continue

        try:
            media_path = None
            if context.config.download.media != "none":
                downloaded = _download_media(
                    video_url=url,
                    video_id=video_id,
                    output_dir=media_dir,
                    media=context.config.download.media,
                    audio_format=context.config.download.yt_dlp_audio_format,
                    video_format=context.config.download.yt_dlp_video_format,
                )
                if downloaded is not None:
                    media_path = str(downloaded)

            row = {
                "video_id": video_id,
                "title": title,
                "description": str(entry.get("description", "")),
                "url": url,
                "duration": duration,
                "channel": entry.get("channel"),
                "channel_id": entry.get("channel_id"),
                "upload_date": entry.get("upload_date"),
                "status": "ready",
                "media_path": media_path,
            }
            rows.append(row)
            manifest.mark(video_id, status="success", input_hash=input_hash, output=row)
            success += 1
        except Exception as exc:
            logger.exception("Download failed for %s (%s)", video_id, title)
            row = {
                "video_id": video_id,
                "title": title,
                "url": url,
                "status": "failed",
                "error": str(exc),
            }
            rows.append(row)
            manifest.mark(video_id, status="failed", input_hash=input_hash, output=row, error=str(exc))
            failed += 1

    # Preserve already-written rows when resuming if needed.
    if output_manifest_path.exists() and not rows:
        rows = [row for row in iter_jsonl(output_manifest_path)]
    else:
        rows.sort(key=lambda r: (str(r.get("status", "")), str(r.get("video_id", ""))))
        write_jsonl(output_manifest_path, rows)

    report = StageReport(
        stage=stage_name,
        total=len(deduped),
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
