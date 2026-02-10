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
from tube2lora.utils.text_metrics import (
    count_sentences,
    count_tokens,
    count_words,
    lexical_metrics,
    readability_metrics,
    safe_ratio,
)

GENERATE_SCHEMA_VERSION = "v3"
SEMANTIC_BOUNDARY_START = "<<<SCENE_END_SENTENCE_START>>>"
SEMANTIC_BOUNDARY_END = "<<<SCENE_END_SENTENCE_END>>>"
FACTS_BLOCK_START = "<<<FACTS_START>>>"
FACTS_BLOCK_END = "<<<FACTS_END>>>"


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


def _extract_delimited_block(raw: str, *, start: str, end: str, label: str) -> str:
    start_idx = raw.find(start)
    if start_idx < 0:
        raise RuntimeError(f"{label} response missing start separator")
    body_start = start_idx + len(start)
    end_idx = raw.find(end, body_start)
    if end_idx < 0:
        raise RuntimeError(f"{label} response missing end separator")
    extracted = raw[body_start:end_idx].strip()
    if not extracted:
        raise RuntimeError(f"{label} response between separators is empty")
    return extracted


def _extract_boundary_sentence(raw: str) -> str:
    boundary = _extract_delimited_block(
        raw,
        start=SEMANTIC_BOUNDARY_START,
        end=SEMANTIC_BOUNDARY_END,
        label="semantic boundary",
    )
    sentence = _clean_closing_sentence(boundary)
    if not sentence:
        raise RuntimeError("semantic boundary sentence is empty after cleanup")
    return sentence


def _extract_facts_block(raw: str) -> str:
    facts = _extract_delimited_block(
        raw,
        start=FACTS_BLOCK_START,
        end=FACTS_BLOCK_END,
        label="facts",
    )
    lines = [line.strip() for line in facts.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("facts response is empty after cleanup")
    if not any(line.startswith(("-", "*")) for line in lines):
        raise RuntimeError("facts response must contain bullet lines")
    return "\n".join(lines)


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
            boundary_raw = client.complete(
                ChatRequest(
                    model=model,
                    system_prompt=system_message,
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            closing_sentence = _extract_boundary_sentence(boundary_raw)
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


def _extract_metrics_from_row(row: dict[str, object]) -> dict[str, object]:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        return metrics

    status = str(row.get("status", "unknown"))
    if status == "ready":
        return {
            "num_entries": int(row.get("num_entries", 0)),
            "num_source_chunks": int(row.get("num_source_chunks", 0)),
            "num_dropped_chunks": int(row.get("num_dropped_chunks", 0)),
        }
    if status == "skipped":
        return {
            "skip_reason": row.get("skip_reason", "unknown"),
            "num_source_chunks": int(row.get("num_source_chunks", 0)),
            "num_dropped_chunks": int(row.get("num_dropped_chunks", 0)),
        }
    if status == "failed":
        return {"error": row.get("error", "unknown")}
    return {}


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


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
    output_rows_path = stage_dir / "videos.jsonl"
    metrics_manifest_path = stage_dir / "metrics.jsonl"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    client = OpenAIChatClient(context.config.llm)

    all_entries: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    success = 0
    failed = 0
    skipped = 0

    for row in tqdm(kept_rows, desc="generate", unit="video"):
        video_id = str(row["video_id"])
        title = str(row.get("title", ""))
        text_path = Path(str(row["text_path"]))

        input_hash = stable_dict_hash(
            {
                "schema": GENERATE_SCHEMA_VERSION,
                "video_id": video_id,
                "text_sha256": sha256_file(text_path),
                "generate_cfg": context.config.generate.model_dump(mode="json"),
                "prompt_hash": prompt_hash,
            }
        )

        if manifest.should_skip(video_id, input_hash):
            cached_output = manifest.items[video_id].get("output")
            if isinstance(cached_output, dict):
                if "entries_path" in cached_output:
                    entries_path = Path(str(cached_output["entries_path"]))
                    all_entries.extend(iter_jsonl(entries_path))
                output_rows.append(cached_output)
            skipped += 1
            continue

        try:
            transcript_text = text_path.read_text(encoding="utf-8").strip()
            if not transcript_text:
                out_row = {
                    "video_id": video_id,
                    "title": title,
                    "status": "skipped",
                    "skip_reason": "empty_normalized_text",
                    "metrics": {
                        "skip_reason": "empty_normalized_text",
                    },
                }
                output_rows.append(out_row)
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
            assistant_token_counts: list[int] = []
            user_token_counts: list[int] = []
            facts_token_counts: list[int] = []
            chunk_token_counts: list[int] = []
            dropped_chunks = 0
            system_token_count = count_tokens(
                prompt_template.system_message.strip(),
                model=context.config.generate.model,
            )
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    facts_prompt = prompt_template.fact_extraction_prompt.format(
                        chunk=chunk,
                        title=title,
                        video_id=video_id,
                    )
                    facts_raw = client.complete(
                        ChatRequest(
                            model=context.config.generate.model,
                            system_prompt=prompt_template.system_message,
                            user_prompt=facts_prompt,
                            temperature=context.config.generate.temperature,
                            max_tokens=context.config.generate.max_tokens,
                        )
                    )
                    facts = _extract_facts_block(facts_raw)

                    user_message = prompt_template.user_message_template.format(
                        facts=facts,
                        title=title,
                        video_id=video_id,
                        chunk_index=chunk_idx,
                    )

                    chunk_tokens = count_tokens(chunk, model=context.config.generate.model)
                    assistant_tokens = chunk_tokens
                    user_tokens = count_tokens(user_message, model=context.config.generate.model)
                    facts_tokens = count_tokens(facts, model=context.config.generate.model)

                    assistant_token_counts.append(assistant_tokens)
                    user_token_counts.append(user_tokens)
                    facts_token_counts.append(facts_tokens)
                    chunk_token_counts.append(chunk_tokens)

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
                            "source_token_count": chunk_tokens,
                            "facts": facts,
                            "facts_token_count": facts_tokens,
                            "user_token_count": user_tokens,
                            "assistant_token_count": assistant_tokens,
                            "system_token_count": system_token_count,
                        },
                    }
                    video_entries.append(entry)
                except Exception as chunk_exc:
                    dropped_chunks += 1
                    logger.warning(
                        "Generate chunk skipped for %s (%s), chunk=%d: %s",
                        video_id,
                        title,
                        chunk_idx,
                        chunk_exc,
                    )
                    continue

            if not video_entries:
                out_row = {
                    "video_id": video_id,
                    "title": title,
                    "status": "skipped",
                    "skip_reason": "no_valid_chunks",
                    "num_source_chunks": len(chunks),
                    "num_dropped_chunks": dropped_chunks,
                    "metrics": {
                        "skip_reason": "no_valid_chunks",
                        "num_source_chunks": len(chunks),
                        "num_dropped_chunks": dropped_chunks,
                    },
                }
                output_rows.append(out_row)
                manifest.mark(
                    video_id,
                    status="skipped",
                    input_hash=input_hash,
                    output=out_row,
                    skipped_reason="no_valid_chunks",
                )
                skipped += 1
                continue

            video_entries_path = per_video_dir / f"{video_id}.jsonl"
            write_jsonl(video_entries_path, video_entries)
            all_entries.extend(video_entries)

            source_token_count = count_tokens(transcript_text, model=context.config.generate.model)
            source_lexical = lexical_metrics(transcript_text)
            source_readability = readability_metrics(transcript_text)
            source_word_count = count_words(transcript_text)
            source_sentence_count = count_sentences(transcript_text)
            total_assistant_tokens = sum(assistant_token_counts)
            total_user_tokens = sum(user_token_counts)
            total_facts_tokens = sum(facts_token_counts)

            out_row = {
                "video_id": video_id,
                "title": title,
                "status": "ready",
                "entries_path": str(video_entries_path),
                "num_entries": len(video_entries),
                "num_source_chunks": len(chunks),
                "num_dropped_chunks": dropped_chunks,
                "metrics": {
                    "num_entries": len(video_entries),
                    "num_source_chunks": len(chunks),
                    "num_dropped_chunks": dropped_chunks,
                    "source_num_chars": len(transcript_text),
                    "source_word_count": source_word_count,
                    "source_token_count": source_token_count,
                    "source_sentence_count": source_sentence_count,
                    "source_unique_ratio": source_lexical["unique_ratio"],
                    "source_hapax_ratio": source_lexical["hapax_ratio"],
                    "source_long_word_ratio": source_lexical["long_word_ratio"],
                    "source_flesch_reading_ease": source_readability["flesch_reading_ease"],
                    "source_automated_readability_index": source_readability[
                        "automated_readability_index"
                    ],
                    "source_gunning_fog": source_readability["gunning_fog"],
                    "assistant_token_total": total_assistant_tokens,
                    "assistant_token_avg": _mean(assistant_token_counts),
                    "assistant_token_max": max(assistant_token_counts) if assistant_token_counts else 0,
                    "user_token_total": total_user_tokens,
                    "user_token_avg": _mean(user_token_counts),
                    "facts_token_total": total_facts_tokens,
                    "facts_token_avg": _mean(facts_token_counts),
                    "source_chunk_token_avg": _mean(chunk_token_counts),
                    "assistant_to_source_token_ratio": safe_ratio(
                        total_assistant_tokens,
                        source_token_count,
                    ),
                },
            }
            output_rows.append(out_row)
            manifest.mark(video_id, status="success", input_hash=input_hash, output=out_row)
            success += 1
        except Exception as exc:
            logger.exception("Generate failed for %s (%s)", video_id, title)
            out_row = {
                "video_id": video_id,
                "title": title,
                "status": "failed",
                "error": str(exc),
                "metrics": {
                    "error": str(exc),
                },
            }
            output_rows.append(out_row)
            manifest.mark(video_id, status="failed", input_hash=input_hash, output=out_row, error=str(exc))
            failed += 1

    write_jsonl(output_manifest_path, all_entries)
    output_rows.sort(key=lambda item: str(item.get("video_id", "")))
    write_jsonl(output_rows_path, output_rows)
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
            "videos_path": str(output_rows_path),
            "metrics_path": str(metrics_manifest_path),
        },
    )
    return report
