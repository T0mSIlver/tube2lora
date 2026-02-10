from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.config import (
    EditorialRewriteConfig,
    EditorialRewritePromptTemplate,
    TalkAsCreatorConfig,
    TalkAsCreatorPromptTemplate,
    load_editorial_rewrite_prompt_template,
    load_talk_as_creator_prompt_template,
)
from tube2lora.llm.client import ChatRequest, OpenAIChatClient
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import sha256_file, stable_dict_hash
from tube2lora.utils.io import atomic_write_json, iter_jsonl, write_jsonl
from tube2lora.utils.text_metrics import (
    char_similarity,
    count_tokens,
    safe_ratio,
    tokenize_words,
)

GENERATE_SCHEMA_VERSION = "v5"
TALK_GENERATOR_VERSION = "v1"
EDITORIAL_GENERATOR_VERSION = "v1"

TALK_CHAT_START = "<<<TALK_CHAT_START>>>"
TALK_CHAT_END = "<<<TALK_CHAT_END>>>"
EDIT_OUTPUT_START = "<<<EDIT_OUTPUT_START>>>"
EDIT_OUTPUT_END = "<<<EDIT_OUTPUT_END>>>"

EDITORIAL_BRIEF_INSTRUCTIONS: dict[str, str] = {
    "cleanup_filler": "Remove filler words and verbal clutter while preserving meaning and voice.",
    "tighten_60_percent": "Tighten the text to about 60% of its original length without dropping key facts.",
    "expand_with_examples": "Expand with concrete examples grounded only in the input text.",
    "convert_to_script": "Convert to a concise documentary script style with clear pacing.",
}

ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,}(?:\s+[A-Z]{2,})*)\b")


@dataclass(slots=True)
class ChunkRecord:
    video_id: str
    title: str
    chunk_index: int
    segment_start_idx: int
    segment_end_idx: int
    start_time: float | None
    end_time: float | None
    raw_excerpt: str
    raw_excerpt_path: Path
    raw_excerpt_sha256: str
    normalized_excerpt: str
    normalized_excerpt_path: Path
    normalized_excerpt_sha256: str


def _as_int(value: object, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid integer for {field}: {value!r}") from exc


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_text_file(path: Path, *, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"{label} file is empty: {path}")
    return text


def _load_chunk_records(video_id: str, title: str, normalize_json_path: Path) -> list[ChunkRecord]:
    if not normalize_json_path.exists():
        raise FileNotFoundError(f"Normalize JSON not found for video_id={video_id}: {normalize_json_path}")

    payload = json.loads(normalize_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Normalize JSON payload is invalid for video_id={video_id}")

    chunks_payload = payload.get("chunks")
    if not isinstance(chunks_payload, list) or not chunks_payload:
        raise RuntimeError(
            f"Normalize JSON chunks are missing for video_id={video_id}. Re-run normalize stage."
        )

    records: list[ChunkRecord] = []
    for raw_chunk in chunks_payload:
        if not isinstance(raw_chunk, dict):
            continue

        chunk_index = _as_int(raw_chunk.get("chunk_index"), field="chunk_index")
        segment_start_idx = _as_int(
            raw_chunk.get("start_segment_index"),
            field="start_segment_index",
        )
        segment_end_idx = _as_int(raw_chunk.get("end_segment_index"), field="end_segment_index")

        source_excerpt_path = Path(str(raw_chunk.get("source_excerpt_path", ""))).resolve()
        normalized_excerpt_path = Path(str(raw_chunk.get("normalized_excerpt_path", ""))).resolve()
        if not str(source_excerpt_path) or not str(normalized_excerpt_path):
            raise RuntimeError(
                f"Chunk excerpt paths missing for video_id={video_id}, chunk_index={chunk_index}. "
                "Re-run normalize stage."
            )

        raw_excerpt = _read_text_file(source_excerpt_path, label="source excerpt")
        normalized_excerpt = _read_text_file(normalized_excerpt_path, label="normalized excerpt")

        raw_excerpt_sha256 = str(raw_chunk.get("source_excerpt_sha256", "")).strip()
        normalized_excerpt_sha256 = str(raw_chunk.get("normalized_excerpt_sha256", "")).strip()
        if not raw_excerpt_sha256 or not normalized_excerpt_sha256:
            raise RuntimeError(
                f"Chunk excerpt hashes missing for video_id={video_id}, chunk_index={chunk_index}. "
                "Re-run normalize stage."
            )

        records.append(
            ChunkRecord(
                video_id=video_id,
                title=title,
                chunk_index=chunk_index,
                segment_start_idx=segment_start_idx,
                segment_end_idx=segment_end_idx,
                start_time=_as_float(raw_chunk.get("start_time")),
                end_time=_as_float(raw_chunk.get("end_time")),
                raw_excerpt=raw_excerpt,
                raw_excerpt_path=source_excerpt_path,
                raw_excerpt_sha256=raw_excerpt_sha256,
                normalized_excerpt=normalized_excerpt,
                normalized_excerpt_path=normalized_excerpt_path,
                normalized_excerpt_sha256=normalized_excerpt_sha256,
            )
        )

    if not records:
        raise RuntimeError(f"No valid chunks found for video_id={video_id}")

    records.sort(key=lambda item: item.chunk_index)
    return records


def _extract_delimited_block(raw: str, *, start: str, end: str, label: str) -> str:
    start_idx = raw.find(start)
    if start_idx < 0:
        raise RuntimeError(f"{label} response missing start separator")
    body_start = start_idx + len(start)
    end_idx = raw.find(end, body_start)
    if end_idx < 0:
        raise RuntimeError(f"{label} response missing end separator")
    content = raw[body_start:end_idx].strip()
    if not content:
        raise RuntimeError(f"{label} response content is empty")
    return content


def _deterministic_probability_hit(
    *,
    video_id: str,
    chunk_index: int,
    probability: float,
    salt: str,
) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    h = stable_dict_hash(
        {
            "video_id": video_id,
            "chunk_index": chunk_index,
            "salt": salt,
        }
    )
    sample = int(h[:8], 16) / float(0xFFFFFFFF)
    return sample < probability


def _topic_hint(text: str) -> str:
    words = text.strip().split()
    if not words:
        return "general discussion"
    return " ".join(words[:16])


def _novel_token_ratio(candidate_text: str, source_text: str) -> float:
    candidate_tokens = tokenize_words(candidate_text)
    if not candidate_tokens:
        return 0.0
    source_vocab = set(tokenize_words(source_text))
    novel = sum(1 for token in candidate_tokens if token not in source_vocab)
    return safe_ratio(novel, len(candidate_tokens))


def _extract_named_entities(text: str) -> set[str]:
    entities = {match.group(0).strip() for match in ENTITY_RE.finditer(text)}
    return {value for value in entities if len(value) > 1}


def _new_named_entities_count(source_text: str, output_text: str) -> int:
    source_entities = _extract_named_entities(source_text)
    output_entities = _extract_named_entities(output_text)
    return len(output_entities - source_entities)


def _parse_talk_messages(raw_response: str, *, turn_pairs: int) -> list[dict[str, str]]:
    payload_text = _extract_delimited_block(
        raw_response,
        start=TALK_CHAT_START,
        end=TALK_CHAT_END,
        label="talk_as_creator",
    )
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise RuntimeError("talk_as_creator payload must be a JSON object")

    messages_payload = payload.get("messages")
    if not isinstance(messages_payload, list):
        raise RuntimeError("talk_as_creator payload missing messages array")

    expected_messages = turn_pairs * 2
    if len(messages_payload) != expected_messages:
        raise RuntimeError(
            f"talk_as_creator expected {expected_messages} messages, got {len(messages_payload)}"
        )

    messages: list[dict[str, str]] = []
    for index, item in enumerate(messages_payload):
        if not isinstance(item, dict):
            raise RuntimeError("talk_as_creator message must be an object")

        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        expected_role = "user" if index % 2 == 0 else "assistant"
        if role != expected_role:
            raise RuntimeError(
                f"talk_as_creator role mismatch at index {index}: expected {expected_role}, got {role}"
            )
        if not content:
            raise RuntimeError(f"talk_as_creator message content is empty at index {index}")
        messages.append({"role": role, "content": content})

    return messages


def _parse_editorial_output(raw_response: str) -> str:
    output = _extract_delimited_block(
        raw_response,
        start=EDIT_OUTPUT_START,
        end=EDIT_OUTPUT_END,
        label="editorial_rewrite",
    )
    cleaned = output.strip()
    if not cleaned:
        raise RuntimeError("editorial_rewrite output is empty")
    return cleaned


def _select_briefs_for_chunk(
    *,
    briefs: list[str],
    examples_per_chunk: int,
    chunk_index: int,
) -> list[str]:
    selected: list[str] = []
    for offset in range(examples_per_chunk):
        selected.append(briefs[(chunk_index + offset) % len(briefs)])
    return selected


def _editorial_quality_failure_reason(
    *,
    brief: str,
    similarity: float,
    length_ratio: float,
    new_named_entities: int,
    cfg: EditorialRewriteConfig,
) -> str | None:
    if brief == "cleanup_filler":
        if similarity < cfg.cleanup_min_similarity:
            return "cleanup_similarity_too_low"
        if similarity > cfg.cleanup_max_similarity:
            return "cleanup_similarity_too_high"
        return None

    if brief == "tighten_60_percent":
        min_ratio = cfg.tighten_target_ratio - cfg.tighten_ratio_tolerance
        max_ratio = cfg.tighten_target_ratio + cfg.tighten_ratio_tolerance
        if length_ratio < min_ratio or length_ratio > max_ratio:
            return "tighten_ratio_out_of_range"
        return None

    if brief == "expand_with_examples":
        if length_ratio < cfg.expand_min_ratio or length_ratio > cfg.expand_max_ratio:
            return "expand_ratio_out_of_range"
        if new_named_entities > cfg.max_new_named_entities:
            return "expand_new_named_entities"
        return None

    if brief == "convert_to_script":
        if length_ratio < 0.6 or length_ratio > 1.6:
            return "script_ratio_out_of_range"
        if similarity < 0.5:
            return "script_similarity_too_low"
        return None

    return None


def _base_metadata(
    *,
    chunk: ChunkRecord,
    generator_name: Literal["talk_as_creator", "editorial_rewrite"],
    generator_version: str,
    prompt_template_id: str,
    source_excerpt_path: Path,
    source_excerpt_sha256: str,
) -> dict[str, object]:
    return {
        "generator_name": generator_name,
        "generator_version": generator_version,
        "prompt_template_id": prompt_template_id,
        "video_id": chunk.video_id,
        "title": chunk.title,
        "chunk_index": chunk.chunk_index,
        "segment_start_idx": chunk.segment_start_idx,
        "segment_end_idx": chunk.segment_end_idx,
        "start_time": chunk.start_time,
        "end_time": chunk.end_time,
        "source_excerpt_path": str(source_excerpt_path),
        "source_excerpt_sha256": source_excerpt_sha256,
        "raw_excerpt_path": str(chunk.raw_excerpt_path),
        "raw_excerpt_sha256": chunk.raw_excerpt_sha256,
        "normalized_excerpt_path": str(chunk.normalized_excerpt_path),
        "normalized_excerpt_sha256": chunk.normalized_excerpt_sha256,
    }


def _run_talk_as_creator_for_video(
    *,
    chunk_records: list[ChunkRecord],
    cfg: TalkAsCreatorConfig,
    prompt_template: TalkAsCreatorPromptTemplate,
    client: OpenAIChatClient,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    entries: list[dict[str, object]] = []
    dropped_chunks = 0
    dropped_reasons: dict[str, int] = {}
    copy_similarity_values: list[float] = []
    novel_token_ratio_values: list[float] = []

    for chunk in chunk_records:
        source_text = chunk.normalized_excerpt
        source_tokens = count_tokens(source_text, model=model)
        if source_tokens < cfg.min_source_tokens:
            dropped_chunks += 1
            dropped_reasons["source_too_short"] = dropped_reasons.get("source_too_short", 0) + 1
            continue

        requires_clarifying_question = _deterministic_probability_hit(
            video_id=chunk.video_id,
            chunk_index=chunk.chunk_index,
            probability=cfg.clarifying_question_probability,
            salt="talk_as_creator_clarify",
        )

        try:
            user_prompt = prompt_template.user_prompt_template.format(
                title=chunk.title,
                video_id=chunk.video_id,
                chunk_index=chunk.chunk_index,
                source_excerpt=source_text,
                topic_hint=_topic_hint(source_text),
                turn_pairs=cfg.turn_pairs,
                target_tokens_per_assistant_turn=cfg.target_tokens_per_assistant_turn,
                requires_clarifying_question=(
                    "yes" if requires_clarifying_question else "no"
                ),
            )
            completion = client.complete(
                ChatRequest(
                    model=model,
                    system_prompt=prompt_template.system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            messages = _parse_talk_messages(completion, turn_pairs=cfg.turn_pairs)

            assistant_messages = [
                msg["content"]
                for msg in messages
                if msg.get("role") == "assistant"
            ]
            assistant_text = "\n\n".join(assistant_messages)
            copy_similarity = char_similarity(assistant_text, source_text)
            novel_token_ratio = _novel_token_ratio(assistant_text, source_text)

            if copy_similarity > cfg.max_copy_similarity:
                dropped_chunks += 1
                dropped_reasons["copy_similarity"] = dropped_reasons.get("copy_similarity", 0) + 1
                continue

            if novel_token_ratio < cfg.min_novel_token_ratio:
                dropped_chunks += 1
                dropped_reasons["novel_token_ratio"] = dropped_reasons.get(
                    "novel_token_ratio",
                    0,
                ) + 1
                continue

            if requires_clarifying_question and not any(
                "?" in text for text in assistant_messages
            ):
                dropped_chunks += 1
                dropped_reasons["missing_clarifying_question"] = dropped_reasons.get(
                    "missing_clarifying_question",
                    0,
                ) + 1
                continue

            metadata = _base_metadata(
                chunk=chunk,
                generator_name="talk_as_creator",
                generator_version=TALK_GENERATOR_VERSION,
                prompt_template_id=prompt_template.name,
                source_excerpt_path=chunk.normalized_excerpt_path,
                source_excerpt_sha256=chunk.normalized_excerpt_sha256,
            )
            metadata.update(
                {
                    "requires_clarifying_question": requires_clarifying_question,
                    "turn_pairs": cfg.turn_pairs,
                    "target_tokens_per_assistant_turn": cfg.target_tokens_per_assistant_turn,
                    "copy_similarity_to_source": copy_similarity,
                    "novel_token_ratio": novel_token_ratio,
                }
            )

            entries.append({"messages": messages, "metadata": metadata})
            copy_similarity_values.append(copy_similarity)
            novel_token_ratio_values.append(novel_token_ratio)
        except Exception as exc:
            dropped_chunks += 1
            dropped_reasons["llm_or_parse_error"] = dropped_reasons.get("llm_or_parse_error", 0) + 1
            dropped_reasons[f"llm_or_parse_error::{type(exc).__name__}"] = dropped_reasons.get(
                f"llm_or_parse_error::{type(exc).__name__}",
                0,
            ) + 1

    status = "ready" if entries else "skipped"
    video_row = {
        "status": status,
        "num_entries": len(entries),
        "num_source_chunks": len(chunk_records),
        "num_dropped_chunks": dropped_chunks,
        "metrics": {
            "num_entries": len(entries),
            "num_source_chunks": len(chunk_records),
            "num_dropped_chunks": dropped_chunks,
            "copy_similarity_avg": safe_ratio(
                sum(copy_similarity_values),
                len(copy_similarity_values),
            ),
            "novel_token_ratio_avg": safe_ratio(
                sum(novel_token_ratio_values),
                len(novel_token_ratio_values),
            ),
            "dropped_reasons": dropped_reasons,
        },
    }
    if status == "skipped":
        video_row["skip_reason"] = "no_valid_chunks"
        video_row["metrics"]["skip_reason"] = "no_valid_chunks"
    return video_row, entries


def _run_editorial_rewrite_for_video(
    *,
    chunk_records: list[ChunkRecord],
    cfg: EditorialRewriteConfig,
    prompt_template: EditorialRewritePromptTemplate,
    client: OpenAIChatClient,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidate_rows: list[dict[str, object]] = []
    dropped_examples = 0
    attempted_examples = 0
    dropped_reasons: dict[str, int] = {}

    for chunk in chunk_records:
        input_text = chunk.normalized_excerpt if cfg.use_normalized_input else chunk.raw_excerpt
        input_tokens = count_tokens(input_text, model=model)
        if input_tokens < cfg.min_source_tokens:
            dropped_examples += 1
            dropped_reasons["source_too_short"] = dropped_reasons.get("source_too_short", 0) + 1
            continue

        selected_briefs = _select_briefs_for_chunk(
            briefs=list(cfg.briefs),
            examples_per_chunk=cfg.examples_per_chunk,
            chunk_index=chunk.chunk_index,
        )

        for brief in selected_briefs:
            attempted_examples += 1
            brief_instructions = EDITORIAL_BRIEF_INSTRUCTIONS[brief]

            try:
                user_prompt = prompt_template.user_prompt_template.format(
                    edit_brief=brief,
                    brief_instructions=brief_instructions,
                    title=chunk.title,
                    video_id=chunk.video_id,
                    chunk_index=chunk.chunk_index,
                    input_text=input_text,
                )
                completion = client.complete(
                    ChatRequest(
                        model=model,
                        system_prompt=prompt_template.system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
                output_text = _parse_editorial_output(completion)

                similarity = char_similarity(output_text, input_text)
                output_tokens = count_tokens(output_text, model=model)
                length_ratio = safe_ratio(output_tokens, input_tokens)
                new_named_entities = _new_named_entities_count(input_text, output_text)

                source_excerpt_path = (
                    chunk.normalized_excerpt_path
                    if cfg.use_normalized_input
                    else chunk.raw_excerpt_path
                )
                source_excerpt_sha256 = (
                    chunk.normalized_excerpt_sha256
                    if cfg.use_normalized_input
                    else chunk.raw_excerpt_sha256
                )

                metadata = _base_metadata(
                    chunk=chunk,
                    generator_name="editorial_rewrite",
                    generator_version=EDITORIAL_GENERATOR_VERSION,
                    prompt_template_id=prompt_template.name,
                    source_excerpt_path=source_excerpt_path,
                    source_excerpt_sha256=source_excerpt_sha256,
                )

                candidate_rows.append(
                    {
                        "edit_brief": brief,
                        "instruction": brief_instructions,
                        "input_text": input_text,
                        "output_text": output_text,
                        "metadata": metadata,
                        "quality_metrics": {
                            "copy_similarity_to_input": similarity,
                            "output_to_input_token_ratio": length_ratio,
                            "new_named_entities": new_named_entities,
                            "input_token_count": input_tokens,
                            "output_token_count": output_tokens,
                        },
                    }
                )
            except Exception as exc:
                dropped_examples += 1
                dropped_reasons["llm_or_parse_error"] = dropped_reasons.get("llm_or_parse_error", 0) + 1
                dropped_reasons[f"llm_or_parse_error::{type(exc).__name__}"] = dropped_reasons.get(
                    f"llm_or_parse_error::{type(exc).__name__}",
                    0,
                ) + 1

    candidate_stats = {
        "num_source_chunks": len(chunk_records),
        "num_attempted_examples": attempted_examples,
        "num_generated_candidates": len(candidate_rows),
        "num_generation_dropped_examples": dropped_examples,
        "generation_dropped_reasons": dropped_reasons,
    }
    return candidate_stats, candidate_rows


def _apply_editorial_quality_gates(
    *,
    candidate_rows: list[dict[str, object]],
    candidate_stats: dict[str, object],
    cfg: EditorialRewriteConfig,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    entries: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    gate_dropped = 0
    gate_dropped_reasons: dict[str, int] = {}
    accepted_similarity_values: list[float] = []
    accepted_length_ratios: list[float] = []

    for candidate in candidate_rows:
        brief = str(candidate.get("edit_brief", ""))
        quality_metrics = candidate.get("quality_metrics")
        if not isinstance(quality_metrics, dict):
            gate_dropped += 1
            gate_dropped_reasons["missing_quality_metrics"] = gate_dropped_reasons.get(
                "missing_quality_metrics",
                0,
            ) + 1
            continue

        similarity = float(quality_metrics.get("copy_similarity_to_input", 0.0))
        length_ratio = float(quality_metrics.get("output_to_input_token_ratio", 0.0))
        new_named_entities = int(quality_metrics.get("new_named_entities", 0))
        failure_reason = _editorial_quality_failure_reason(
            brief=brief,
            similarity=similarity,
            length_ratio=length_ratio,
            new_named_entities=new_named_entities,
            cfg=cfg,
        )

        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            gate_dropped += 1
            gate_dropped_reasons["missing_metadata"] = gate_dropped_reasons.get("missing_metadata", 0) + 1
            continue

        decision_row = {
            "status": "accepted" if failure_reason is None else "rejected",
            "reason": failure_reason,
            "edit_brief": brief,
            "metadata": {
                "video_id": metadata.get("video_id"),
                "chunk_index": metadata.get("chunk_index"),
                "segment_start_idx": metadata.get("segment_start_idx"),
                "segment_end_idx": metadata.get("segment_end_idx"),
            },
            "quality_metrics": quality_metrics,
        }
        decision_rows.append(decision_row)

        if failure_reason is not None:
            gate_dropped += 1
            gate_dropped_reasons[failure_reason] = gate_dropped_reasons.get(failure_reason, 0) + 1
            continue

        entry_metadata = dict(metadata)
        entry_metadata.update(
            {
                "edit_brief": brief,
                "copy_similarity_to_input": similarity,
                "output_to_input_token_ratio": length_ratio,
                "new_named_entities": new_named_entities,
            }
        )

        entries.append(
            {
                "edit_brief": brief,
                "instruction": candidate.get("instruction"),
                "input_text": candidate.get("input_text"),
                "output_text": candidate.get("output_text"),
                "metadata": entry_metadata,
            }
        )
        accepted_similarity_values.append(similarity)
        accepted_length_ratios.append(length_ratio)

    generation_dropped = int(candidate_stats.get("num_generation_dropped_examples", 0))
    total_dropped = generation_dropped + gate_dropped
    combined_dropped_reasons: dict[str, int] = {}

    generation_reasons = candidate_stats.get("generation_dropped_reasons")
    if isinstance(generation_reasons, dict):
        for key, value in generation_reasons.items():
            combined_dropped_reasons[str(key)] = int(value)

    for key, value in gate_dropped_reasons.items():
        combined_dropped_reasons[key] = combined_dropped_reasons.get(key, 0) + value

    status = "ready" if entries else "skipped"
    video_row = {
        "status": status,
        "num_entries": len(entries),
        "num_source_chunks": int(candidate_stats.get("num_source_chunks", 0)),
        "num_attempted_examples": int(candidate_stats.get("num_attempted_examples", 0)),
        "num_generated_candidates": len(candidate_rows),
        "num_generation_dropped_examples": generation_dropped,
        "num_gate_rejected_examples": gate_dropped,
        "num_dropped_examples": total_dropped,
        "metrics": {
            "num_entries": len(entries),
            "num_source_chunks": int(candidate_stats.get("num_source_chunks", 0)),
            "num_attempted_examples": int(candidate_stats.get("num_attempted_examples", 0)),
            "num_generated_candidates": len(candidate_rows),
            "num_generation_dropped_examples": generation_dropped,
            "num_gate_rejected_examples": gate_dropped,
            "num_dropped_examples": total_dropped,
            "copy_similarity_avg": safe_ratio(sum(accepted_similarity_values), len(accepted_similarity_values)),
            "output_to_input_token_ratio_avg": safe_ratio(sum(accepted_length_ratios), len(accepted_length_ratios)),
            "dropped_reasons": combined_dropped_reasons,
        },
    }
    if status == "skipped":
        video_row["skip_reason"] = "no_valid_examples"
        video_row["metrics"]["skip_reason"] = "no_valid_examples"

    return video_row, entries, decision_rows


def _editorial_generation_config(cfg: EditorialRewriteConfig) -> dict[str, object]:
    return {
        "briefs": list(cfg.briefs),
        "use_normalized_input": cfg.use_normalized_input,
        "examples_per_chunk": cfg.examples_per_chunk,
        "min_source_tokens": cfg.min_source_tokens,
    }


def _editorial_gate_config(cfg: EditorialRewriteConfig) -> dict[str, object]:
    return {
        "cleanup_min_similarity": cfg.cleanup_min_similarity,
        "cleanup_max_similarity": cfg.cleanup_max_similarity,
        "tighten_target_ratio": cfg.tighten_target_ratio,
        "tighten_ratio_tolerance": cfg.tighten_ratio_tolerance,
        "expand_min_ratio": cfg.expand_min_ratio,
        "expand_max_ratio": cfg.expand_max_ratio,
        "max_new_named_entities": cfg.max_new_named_entities,
    }


def _generator_dataset_filename(generator_name: str) -> str:
    if generator_name == "talk_as_creator":
        return "dataset_messages.jsonl"
    if generator_name == "editorial_rewrite":
        return "dataset_edits.jsonl"
    raise ValueError(f"Unknown generator: {generator_name}")


def _extract_metrics_from_row(row: dict[str, object]) -> dict[str, object]:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        return metrics

    status = str(row.get("status", "unknown"))
    if status == "ready":
        return {
            "num_entries": int(row.get("num_entries", 0)),
        }
    if status == "skipped":
        return {
            "skip_reason": row.get("skip_reason", "unknown"),
        }
    if status == "failed":
        return {
            "error": row.get("error", "unknown"),
        }
    return {}


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

    enabled_generators = list(context.config.generate.generators)
    talk_prompt = (
        load_talk_as_creator_prompt_template(context.config.generate.talk_as_creator.prompt_template)
        if "talk_as_creator" in enabled_generators
        else None
    )
    editorial_prompt = (
        load_editorial_rewrite_prompt_template(context.config.generate.editorial_rewrite.prompt_template)
        if "editorial_rewrite" in enabled_generators
        else None
    )

    prompt_hashes: dict[str, str] = {}
    if talk_prompt is not None:
        prompt_hashes["talk_as_creator"] = stable_dict_hash(talk_prompt.model_dump(mode="json"))
    if editorial_prompt is not None:
        prompt_hashes["editorial_rewrite"] = stable_dict_hash(
            editorial_prompt.model_dump(mode="json")
        )

    stage_manifest = ManifestStore(context.stage_manifest_path(stage_name))
    client = OpenAIChatClient(context.config.llm)

    generator_dirs: dict[str, Path] = {}
    generator_video_rows: dict[str, list[dict[str, object]]] = {}
    generator_entries: dict[str, list[dict[str, object]]] = {}

    for generator_name in enabled_generators:
        generator_dir = context.root / generator_name
        (generator_dir / "videos").mkdir(parents=True, exist_ok=True)
        generator_dirs[generator_name] = generator_dir
        generator_video_rows[generator_name] = []
        generator_entries[generator_name] = []

    success_items = 0
    failed_items = 0
    skipped_items = 0

    for row in tqdm(kept_rows, desc="generate", unit="video"):
        video_id = str(row["video_id"])
        title = str(row.get("title", ""))
        normalize_json_path = context.stage_dir("normalize") / f"{video_id}.json"

        try:
            chunk_records = _load_chunk_records(video_id, title, normalize_json_path)
        except Exception as exc:
            logger.exception("Failed to load normalized chunks for %s (%s)", video_id, title)
            for generator_name in enabled_generators:
                failed_items += 1
                failed_row = {
                    "video_id": video_id,
                    "title": title,
                    "status": "failed",
                    "error": str(exc),
                    "metrics": {
                        "error": str(exc),
                    },
                }
                generator_video_rows[generator_name].append(failed_row)
                item_id = f"{generator_name}:{video_id}"
                stage_manifest.mark(
                    item_id,
                    status="failed",
                    input_hash=stable_dict_hash(
                        {
                            "schema": GENERATE_SCHEMA_VERSION,
                            "generator": generator_name,
                            "video_id": video_id,
                            "error": str(exc),
                        }
                    ),
                    output=failed_row,
                    error=str(exc),
                )
            continue

        for generator_name in enabled_generators:
            item_id = f"{generator_name}:{video_id}"
            cached_item = stage_manifest.items.get(item_id, {})
            cached_output = cached_item.get("output") if isinstance(cached_item, dict) else None

            try:
                if generator_name == "talk_as_creator":
                    input_hash = stable_dict_hash(
                        {
                            "schema": GENERATE_SCHEMA_VERSION,
                            "generator": generator_name,
                            "generator_version": TALK_GENERATOR_VERSION,
                            "video_id": video_id,
                            "normalize_json_sha256": sha256_file(normalize_json_path),
                            "generate_model": context.config.generate.model,
                            "temperature": context.config.generate.temperature,
                            "max_tokens": context.config.generate.max_tokens,
                            "generator_cfg": context.config.generate.talk_as_creator.model_dump(mode="json"),
                            "prompt_hash": prompt_hashes[generator_name],
                        }
                    )

                    if stage_manifest.should_skip(item_id, input_hash):
                        if isinstance(cached_output, dict):
                            generator_video_rows[generator_name].append(cached_output)
                            if cached_output.get("status") == "ready":
                                entries_path = cached_output.get("entries_path")
                                if isinstance(entries_path, str) and entries_path:
                                    generator_entries[generator_name].extend(iter_jsonl(Path(entries_path)))
                        skipped_items += 1
                        continue

                    if talk_prompt is None:
                        raise RuntimeError("talk_as_creator prompt is not loaded")
                    video_row, entries = _run_talk_as_creator_for_video(
                        chunk_records=chunk_records,
                        cfg=context.config.generate.talk_as_creator,
                        prompt_template=talk_prompt,
                        client=client,
                        model=context.config.generate.model,
                        temperature=context.config.generate.temperature,
                        max_tokens=context.config.generate.max_tokens,
                    )

                    video_row["video_id"] = video_id
                    video_row["title"] = title

                    if entries:
                        entries_path = generator_dirs[generator_name] / "videos" / f"{video_id}.jsonl"
                        write_jsonl(entries_path, entries)
                        generator_entries[generator_name].extend(entries)
                        video_row["entries_path"] = str(entries_path)
                        success_items += 1
                        stage_manifest.mark(
                            item_id,
                            status="success",
                            input_hash=input_hash,
                            output=video_row,
                        )
                    else:
                        video_row["entries_path"] = None
                        skipped_items += 1
                        stage_manifest.mark(
                            item_id,
                            status="skipped",
                            input_hash=input_hash,
                            output=video_row,
                            skipped_reason=str(video_row.get("skip_reason", "no_valid_examples")),
                        )

                    generator_video_rows[generator_name].append(video_row)
                    continue

                if editorial_prompt is None:
                    raise RuntimeError("editorial_rewrite prompt is not loaded")

                llm_input_hash = stable_dict_hash(
                    {
                        "schema": GENERATE_SCHEMA_VERSION,
                        "generator": generator_name,
                        "generator_version": EDITORIAL_GENERATOR_VERSION,
                        "video_id": video_id,
                        "normalize_json_sha256": sha256_file(normalize_json_path),
                        "generate_model": context.config.generate.model,
                        "temperature": context.config.generate.temperature,
                        "max_tokens": context.config.generate.max_tokens,
                        "editorial_generation_cfg": _editorial_generation_config(
                            context.config.generate.editorial_rewrite
                        ),
                        "prompt_hash": prompt_hashes[generator_name],
                    }
                )
                gates_input_hash = stable_dict_hash(
                    {
                        "schema": GENERATE_SCHEMA_VERSION,
                        "generator": generator_name,
                        "generator_version": EDITORIAL_GENERATOR_VERSION,
                        "editorial_gate_cfg": _editorial_gate_config(
                            context.config.generate.editorial_rewrite
                        ),
                    }
                )
                combined_input_hash = stable_dict_hash(
                    {
                        "llm_input_hash": llm_input_hash,
                        "gates_input_hash": gates_input_hash,
                    }
                )

                if (
                    isinstance(cached_output, dict)
                    and cached_output.get("llm_input_hash") == llm_input_hash
                    and cached_output.get("gates_input_hash") == gates_input_hash
                    and stage_manifest.should_skip(item_id, combined_input_hash)
                ):
                    generator_video_rows[generator_name].append(cached_output)
                    if cached_output.get("status") == "ready":
                        entries_path = cached_output.get("entries_path")
                        if isinstance(entries_path, str) and entries_path:
                            generator_entries[generator_name].extend(iter_jsonl(Path(entries_path)))
                    skipped_items += 1
                    continue

                candidate_rows: list[dict[str, object]]
                candidate_stats: dict[str, object]
                if (
                    isinstance(cached_output, dict)
                    and cached_output.get("llm_input_hash") == llm_input_hash
                ):
                    candidates_path_value = cached_output.get("candidates_path")
                    candidate_stats_value = cached_output.get("candidate_stats")
                    if (
                        isinstance(candidates_path_value, str)
                        and candidates_path_value
                        and isinstance(candidate_stats_value, dict)
                        and Path(candidates_path_value).exists()
                    ):
                        candidates_path = Path(candidates_path_value)
                        candidate_rows = [row for row in iter_jsonl(candidates_path)]
                        candidate_stats = candidate_stats_value
                        logger.info(
                            "Re-evaluating editorial thresholds from cached candidates for %s (%s)",
                            video_id,
                            title,
                        )
                    else:
                        candidate_stats, candidate_rows = _run_editorial_rewrite_for_video(
                            chunk_records=chunk_records,
                            cfg=context.config.generate.editorial_rewrite,
                            prompt_template=editorial_prompt,
                            client=client,
                            model=context.config.generate.model,
                            temperature=context.config.generate.temperature,
                            max_tokens=context.config.generate.max_tokens,
                        )
                        candidates_path = (
                            generator_dirs[generator_name]
                            / "videos"
                            / f"{video_id}_candidates.jsonl"
                        )
                        write_jsonl(candidates_path, candidate_rows)
                else:
                    candidate_stats, candidate_rows = _run_editorial_rewrite_for_video(
                        chunk_records=chunk_records,
                        cfg=context.config.generate.editorial_rewrite,
                        prompt_template=editorial_prompt,
                        client=client,
                        model=context.config.generate.model,
                        temperature=context.config.generate.temperature,
                        max_tokens=context.config.generate.max_tokens,
                    )
                    candidates_path = (
                        generator_dirs[generator_name]
                        / "videos"
                        / f"{video_id}_candidates.jsonl"
                    )
                    write_jsonl(candidates_path, candidate_rows)

                video_row, entries, decisions = _apply_editorial_quality_gates(
                    candidate_rows=candidate_rows,
                    candidate_stats=candidate_stats,
                    cfg=context.config.generate.editorial_rewrite,
                )
                video_row["video_id"] = video_id
                video_row["title"] = title

                entries_path = generator_dirs[generator_name] / "videos" / f"{video_id}.jsonl"
                decisions_path = (
                    generator_dirs[generator_name]
                    / "videos"
                    / f"{video_id}_decisions.jsonl"
                )
                write_jsonl(entries_path, entries)
                write_jsonl(decisions_path, decisions)
                generator_entries[generator_name].extend(entries)

                video_row["entries_path"] = str(entries_path)
                video_row["decisions_path"] = str(decisions_path)
                video_row["candidates_path"] = str(candidates_path)
                video_row["candidate_stats"] = candidate_stats
                video_row["llm_input_hash"] = llm_input_hash
                video_row["gates_input_hash"] = gates_input_hash

                if entries:
                    success_items += 1
                    stage_manifest.mark(
                        item_id,
                        status="success",
                        input_hash=combined_input_hash,
                        output=video_row,
                    )
                else:
                    skipped_items += 1
                    stage_manifest.mark(
                        item_id,
                        status="skipped",
                        input_hash=combined_input_hash,
                        output=video_row,
                        skipped_reason=str(video_row.get("skip_reason", "no_valid_examples")),
                    )

                generator_video_rows[generator_name].append(video_row)
            except Exception as exc:
                logger.exception("Generate failed for %s (%s) on %s", video_id, title, generator_name)
                failed_items += 1
                failed_row = {
                    "video_id": video_id,
                    "title": title,
                    "status": "failed",
                    "error": str(exc),
                    "metrics": {
                        "error": str(exc),
                    },
                }
                generator_video_rows[generator_name].append(failed_row)
                stage_manifest.mark(
                    item_id,
                    status="failed",
                    input_hash=stable_dict_hash(
                        {
                            "schema": GENERATE_SCHEMA_VERSION,
                            "generator": generator_name,
                            "video_id": video_id,
                            "error": str(exc),
                        }
                    ),
                    output=failed_row,
                    error=str(exc),
                )

    generator_summaries: dict[str, dict[str, object]] = {}
    for generator_name in enabled_generators:
        generator_dir = generator_dirs[generator_name]
        dataset_filename = _generator_dataset_filename(generator_name)
        dataset_path = generator_dir / dataset_filename
        videos_path = generator_dir / "videos.jsonl"
        metrics_path = generator_dir / "metrics.jsonl"

        rows_for_generator = generator_video_rows[generator_name]
        rows_for_generator.sort(key=lambda item: str(item.get("video_id", "")))

        write_jsonl(dataset_path, generator_entries[generator_name])
        write_jsonl(videos_path, rows_for_generator)
        write_jsonl(
            metrics_path,
            [
                {
                    "video_id": row.get("video_id"),
                    "title": row.get("title"),
                    "status": row.get("status"),
                    "metrics": _extract_metrics_from_row(row),
                }
                for row in rows_for_generator
            ],
        )

        ready_videos = sum(1 for row in rows_for_generator if row.get("status") == "ready")
        skipped_videos = sum(1 for row in rows_for_generator if row.get("status") == "skipped")
        failed_videos = sum(1 for row in rows_for_generator if row.get("status") == "failed")

        generator_summaries[generator_name] = {
            "dataset_path": str(dataset_path),
            "videos_path": str(videos_path),
            "metrics_path": str(metrics_path),
            "dataset_rows": len(generator_entries[generator_name]),
            "ready_videos": ready_videos,
            "skipped_videos": skipped_videos,
            "failed_videos": failed_videos,
        }

    run_manifest_path = context.root / "manifest.json"
    atomic_write_json(
        run_manifest_path,
        {
            "run_id": context.run_id,
            "config_hash": context.config_hash,
            "generate": {
                "schema_version": GENERATE_SCHEMA_VERSION,
                "model": context.config.generate.model,
                "temperature": context.config.generate.temperature,
                "max_tokens": context.config.generate.max_tokens,
                "generators": enabled_generators,
                "generator_outputs": generator_summaries,
            },
        },
    )

    total_items = len(kept_rows) * len(enabled_generators)
    report = StageReport(
        stage=stage_name,
        total=total_items,
        success=success_items,
        failed=failed_items,
        skipped=skipped_items,
        output_path=str(run_manifest_path),
    )
    context.update_stage_status(
        stage_name,
        "completed" if failed_items == 0 else "failed",
        details={
            "summary": {
                "total": report.total,
                "success": report.success,
                "failed": report.failed,
                "skipped": report.skipped,
            },
            "output_path": str(run_manifest_path),
            "generators": generator_summaries,
        },
    )
    return report
