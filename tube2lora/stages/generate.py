from __future__ import annotations

import logging
import re
from pathlib import Path

from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.config import load_prompt_template
from tube2lora.llm.client import ChatRequest, OpenAIChatClient
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import sha256_file, stable_dict_hash
from tube2lora.utils.io import iter_jsonl, write_jsonl


def _clean_closing_sentence(raw: str) -> str:
    candidate = raw.strip()
    if "\n" in candidate:
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if lines:
            candidate = lines[-1]

    pattern = r"^(\*\*|\*|\"|'|_|`)|(\*\*|\*|\"|'|_|`)$"
    while True:
        updated = re.sub(pattern, "", candidate).strip()
        if updated == candidate:
            break
        candidate = updated

    return " ".join(candidate.split())


def _find_verbatim_index(buffer: str, closing_sentence: str) -> int:
    closing_sentence = " ".join(closing_sentence.split())
    words = closing_sentence.split()

    def extend_to_punctuation(start_idx: int) -> int:
        window = buffer[start_idx : start_idx + 500]
        match = re.search(r"[.?!]", window)
        if match:
            return start_idx + match.end()
        return start_idx

    exact_idx = buffer.find(closing_sentence)
    if exact_idx != -1:
        end_idx = exact_idx + len(closing_sentence)
        if not closing_sentence.endswith((".", "?", "!")):
            end_idx = extend_to_punctuation(end_idx)
        return end_idx

    if words:
        pattern = r"\s+".join(re.escape(part) for part in words)
        match = re.search(pattern, buffer)
        if match:
            end_idx = match.end()
            if end_idx > 0 and buffer[end_idx - 1] not in {".", "?", "!"}:
                end_idx = extend_to_punctuation(end_idx)
            return end_idx

    if len(words) > 3:
        for pct in (0.75, 0.5, 0.25):
            limit = max(3, int(len(words) * pct))
            if limit >= len(words):
                continue
            prefix = r"\s+".join(re.escape(part) for part in words[:limit])
            match = re.search(prefix, buffer)
            if not match:
                continue

            start = match.end()
            suffix_words = min(3, len(words) - limit)
            if suffix_words > 0:
                suffix = r"\s+".join(re.escape(part) for part in words[-suffix_words:])
                window = buffer[start : start + 500]
                suffix_match = re.search(suffix, window)
                if suffix_match:
                    end_idx = start + suffix_match.end()
                    if end_idx > 0 and buffer[end_idx - 1] not in {".", "?", "!"}:
                        end_idx = extend_to_punctuation(end_idx)
                    return end_idx

            fallback = extend_to_punctuation(start)
            if fallback > start:
                return fallback
            return start

        mini_prefix = r"\s+".join(re.escape(part) for part in words[:3])
        match = re.search(mini_prefix, buffer)
        if match:
            return extend_to_punctuation(match.end())

    return -1


def _semantic_chunks(
    text: str,
    *,
    title: str,
    video_id: str,
    client: OpenAIChatClient,
    model: str,
    system_message: str,
    semantic_boundary_prompt: str,
    buffer_chars: int,
    fallback_min_chars: int,
    fallback_max_chars: int,
    min_tail_chars: int,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    chunks: list[str] = []
    pointer = 0

    while pointer < len(text):
        buffer = text[pointer : pointer + buffer_chars]
        if not buffer.strip():
            break

        try:
            prompt = semantic_boundary_prompt.format(
                buffer=buffer[:5000],
                title=title,
                video_id=video_id,
            )
            boundary = client.complete(
                ChatRequest(
                    model=model,
                    system_prompt=system_message,
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            closing_sentence = _clean_closing_sentence(boundary)
            cut_point = _find_verbatim_index(buffer, closing_sentence) if closing_sentence else -1
        except Exception:
            cut_point = -1

        if cut_point == -1:
            search_start = fallback_min_chars
            search_end = fallback_max_chars
            if len(buffer) < search_start:
                cut_point = len(buffer)
            else:
                limit = min(len(buffer), search_end)
                window = buffer[search_start:limit]
                last_punc = max(window.rfind("."), window.rfind("?"), window.rfind("!"))
                if last_punc != -1:
                    cut_point = search_start + last_punc + 1
                else:
                    last_space = window.rfind(" ")
                    if last_space != -1:
                        cut_point = search_start + last_space + 1
                    else:
                        cut_point = min(len(buffer), search_start + 500)

        if cut_point < len(buffer):
            while cut_point < len(buffer) and not buffer[cut_point].isspace():
                cut_point += 1

        if cut_point <= 0:
            break

        chunk = re.sub(r"\s+", " ", buffer[:cut_point]).strip()
        if chunk:
            chunks.append(chunk)

        pointer += cut_point
        if len(text) - pointer < min_tail_chars:
            tail = re.sub(r"\s+", " ", text[pointer:]).strip()
            if tail:
                chunks.append(tail)
            break

    return chunks


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "generate"
    context.update_stage_status(stage_name, "running")

    if not context.config.generate.enabled:
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

    source_manifest = context.stage_dir("filter") / "kept.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError("Filter stage output not found.")

    rows = [row for row in iter_jsonl(source_manifest)]
    kept_rows = [row for row in rows if row.get("status") == "kept"]

    prompt_template = load_prompt_template(context.config.generate.prompt_template)
    prompt_hash = stable_dict_hash(prompt_template.model_dump(mode="json"))

    stage_dir = context.stage_dir(stage_name)
    per_video_dir = stage_dir / "videos"
    per_video_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = stage_dir / "dataset_messages.jsonl"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    client = OpenAIChatClient(context.config.llm)

    all_entries: list[dict[str, object]] = []
    success = 0
    failed = 0
    skipped = 0

    for row in tqdm(kept_rows, desc="generate", unit="video"):
        video_id = str(row["video_id"])
        title = str(row.get("title", ""))
        text_path = Path(str(row["text_path"]))

        input_hash = stable_dict_hash(
            {
                "video_id": video_id,
                "text_sha256": sha256_file(text_path),
                "generate_cfg": context.config.generate.model_dump(mode="json"),
                "prompt_hash": prompt_hash,
            }
        )

        if manifest.should_skip(video_id, input_hash):
            cached_output = manifest.items[video_id].get("output")
            if isinstance(cached_output, dict) and "entries_path" in cached_output:
                entries_path = Path(str(cached_output["entries_path"]))
                all_entries.extend(iter_jsonl(entries_path))
            skipped += 1
            continue

        try:
            transcript_text = text_path.read_text(encoding="utf-8").strip()
            if not transcript_text:
                out_row = {
                    "video_id": video_id,
                    "status": "skipped",
                    "skip_reason": "empty_normalized_text",
                }
                manifest.mark(
                    video_id,
                    status="skipped",
                    input_hash=input_hash,
                    output=out_row,
                    skipped_reason="empty_text",
                )
                skipped += 1
                continue

            chunks = _semantic_chunks(
                transcript_text,
                title=title,
                video_id=video_id,
                client=client,
                model=context.config.generate.model,
                system_message=prompt_template.system_message,
                semantic_boundary_prompt=prompt_template.semantic_boundary_prompt,
                buffer_chars=context.config.generate.buffer_chars,
                fallback_min_chars=context.config.generate.fallback_min_chars,
                fallback_max_chars=context.config.generate.fallback_max_chars,
                min_tail_chars=context.config.generate.min_tail_chars,
                temperature=context.config.generate.temperature,
                max_tokens=context.config.generate.max_tokens,
            )

            video_entries: list[dict[str, object]] = []
            for chunk_idx, chunk in enumerate(chunks):
                facts_prompt = prompt_template.fact_extraction_prompt.format(
                    chunk=chunk,
                    title=title,
                    video_id=video_id,
                )
                facts = client.complete(
                    ChatRequest(
                        model=context.config.generate.model,
                        system_prompt=prompt_template.system_message,
                        user_prompt=facts_prompt,
                        temperature=context.config.generate.temperature,
                        max_tokens=context.config.generate.max_tokens,
                    )
                )

                user_message = prompt_template.user_message_template.format(
                    facts=facts,
                    title=title,
                    video_id=video_id,
                    chunk_index=chunk_idx,
                )

                messages: list[dict[str, str]] = []
                if prompt_template.system_message.strip():
                    messages.append(
                        {
                            "role": "system",
                            "content": prompt_template.system_message.strip(),
                        }
                    )
                messages.append({"role": "user", "content": user_message})
                messages.append({"role": "assistant", "content": chunk})

                entry = {
                    "messages": messages,
                    "metadata": {
                        "video_id": video_id,
                        "title": title,
                        "chunk_index": chunk_idx,
                        "source_len": len(chunk),
                        "facts": facts,
                    },
                }
                video_entries.append(entry)

            video_entries_path = per_video_dir / f"{video_id}.jsonl"
            write_jsonl(video_entries_path, video_entries)
            all_entries.extend(video_entries)

            out_row = {
                "video_id": video_id,
                "status": "ready",
                "entries_path": str(video_entries_path),
                "num_entries": len(video_entries),
            }
            manifest.mark(video_id, status="success", input_hash=input_hash, output=out_row)
            success += 1
        except Exception as exc:
            logger.exception("Generate failed for %s (%s)", video_id, title)
            out_row = {
                "video_id": video_id,
                "status": "failed",
                "error": str(exc),
            }
            manifest.mark(video_id, status="failed", input_hash=input_hash, output=out_row, error=str(exc))
            failed += 1

    write_jsonl(output_manifest_path, all_entries)

    report = StageReport(
        stage=stage_name,
        total=len(kept_rows),
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
