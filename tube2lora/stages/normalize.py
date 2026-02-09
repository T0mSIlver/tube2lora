from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.config import load_normalize_prompt_template
from tube2lora.llm.client import ChatRequest, OpenAIChatClient
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import sha256_file, stable_dict_hash
from tube2lora.utils.io import atomic_write_json, atomic_write_text, iter_jsonl, write_jsonl


def _read_transcript_text(transcript_path: Path) -> str:
    if transcript_path.suffix == ".txt":
        return transcript_path.read_text(encoding="utf-8").strip()

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ""
    text = payload.get("text")
    return str(text).strip() if text is not None else ""


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "normalize"
    context.update_stage_status(stage_name, "running")

    if not context.config.normalize.enabled:
        report = StageReport(stage=stage_name, total=0, success=0, failed=0, skipped=0, output_path=None)
        context.update_stage_status(stage_name, "completed", details={"summary": report.__dict__})
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
                "video_id": video_id,
                "transcript_sha256": sha256_file(transcript_path),
                "prompt_hash": prompt_hash,
                "model": context.config.normalize.model,
                "temperature": context.config.normalize.temperature,
                "max_tokens": context.config.normalize.max_tokens,
            }
        )

        if manifest.should_skip(video_id, input_hash):
            cached = manifest.items[video_id].get("output")
            if isinstance(cached, dict):
                output_rows.append(cached)
            skipped += 1
            continue

        try:
            transcript_text = _read_transcript_text(transcript_path)
            if not transcript_text:
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

            user_prompt = prompt_template.user_prompt_template.format(
                transcript=transcript_text,
                title=title,
                video_id=video_id,
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

            normalized_text = completion.strip()
            normalized_json = {
                "video_id": video_id,
                "title": title,
                "source_transcript_path": str(transcript_path),
                "normalized_text": normalized_text,
                "num_chars": len(normalized_text),
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
                "num_chars": len(normalized_text),
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
