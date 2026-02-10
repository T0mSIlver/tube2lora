from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import sha256_file, stable_dict_hash
from tube2lora.utils.io import atomic_write_json, iter_jsonl


def _video_level_split(
    entries: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        metadata = entry.get("metadata")
        video_id = "unknown"
        if isinstance(metadata, dict) and isinstance(metadata.get("video_id"), str):
            video_id = str(metadata["video_id"])
        by_video.setdefault(video_id, []).append(entry)

    video_ids = list(by_video.keys())
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    eval_video_count = int(len(video_ids) * eval_ratio)
    if eval_ratio > 0 and len(video_ids) > 1:
        eval_video_count = max(1, eval_video_count)

    eval_ids = set(video_ids[:eval_video_count])
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for vid, samples in by_video.items():
        if vid in eval_ids:
            eval_rows.extend(samples)
        else:
            train_rows.extend(samples)
    if not train_rows and eval_rows:
        train_rows, eval_rows = eval_rows, []
    return train_rows, eval_rows


def _sample_generations(
    *,
    context: RunContext,
    adapter_dir: Path,
    eval_entries: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, str]]:
    if not eval_entries:
        return []

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.warning("Generation sample dependencies are unavailable; skipping generation samples.")
        return []

    if not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; skipping generation samples.")
        return []

    base_model = context.config.train.base_model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    rng = random.Random(context.config.train.seed)
    picked = eval_entries[:]
    rng.shuffle(picked)
    picked = picked[: context.config.evaluate.num_generation_samples]

    results: list[dict[str, str]] = []
    for sample in picked:
        messages = sample.get("messages")
        if not isinstance(messages, list):
            continue

        prompt_messages = []
        expected_output = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role == "assistant":
                expected_output = content
                continue
            prompt_messages.append({"role": role, "content": content})

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=context.config.evaluate.max_new_tokens,
                temperature=context.config.evaluate.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded

        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        results.append(
            {
                "video_id": str(metadata.get("video_id", "")),
                "prompt": prompt,
                "expected": expected_output,
                "generated": generated_text,
            }
        )

    return results


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "evaluate"
    context.update_stage_status(stage_name, "running")

    if not context.config.evaluate.enabled:
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

    metrics_path = context.stage_dir("train") / "training_metrics.json"
    dataset_path = context.root / "talk_as_creator" / "dataset_messages.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError("Train stage output not found.")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Talk-as-creator dataset not found. Run generate with talk_as_creator enabled."
        )

    stage_dir = context.stage_dir(stage_name)
    output_path = stage_dir / "evaluation.json"
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    input_hash = stable_dict_hash(
        {
            "metrics_sha256": sha256_file(metrics_path),
            "dataset_sha256": sha256_file(dataset_path),
            "eval_cfg": context.config.evaluate.model_dump(mode="json"),
            "train_cfg": context.config.train.model_dump(mode="json"),
        }
    )

    if manifest.should_skip("evaluate", input_hash):
        report = StageReport(stage=stage_name, total=1, success=1, failed=0, skipped=1, output_path=str(output_path))
        context.update_stage_status(stage_name, "completed", details={"output_path": str(output_path), "cached": True})
        return report

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    eval_loss = float(metrics.get("eval_loss", 0.0))
    perplexity = float(metrics.get("eval_perplexity")) if metrics.get("eval_perplexity") is not None else (math.exp(eval_loss) if eval_loss > 0 else None)

    entries = [row for row in iter_jsonl(dataset_path)]
    _, eval_entries = _video_level_split(
        entries,
        eval_ratio=context.config.train.eval_split_ratio,
        seed=context.config.train.seed,
    )

    adapter_dir = Path(str(metrics.get("adapter_dir", "")))
    generation_samples: list[dict[str, str]] = []
    if adapter_dir.exists() and adapter_dir.is_dir():
        generation_samples = _sample_generations(
            context=context,
            adapter_dir=adapter_dir,
            eval_entries=eval_entries,
            logger=logger,
        )

    payload = {
        "eval_loss": eval_loss,
        "eval_perplexity": perplexity,
        "eval_entries": len(eval_entries),
        "generation_samples": generation_samples,
    }
    atomic_write_json(output_path, payload)

    manifest.mark(
        "evaluate",
        status="success",
        input_hash=input_hash,
        output={"evaluation_path": str(output_path)},
    )

    report = StageReport(
        stage=stage_name,
        total=1,
        success=1,
        failed=0,
        skipped=0,
        output_path=str(output_path),
    )
    context.update_stage_status(
        stage_name,
        "completed",
        details={"output_path": str(output_path), "summary": payload},
    )
    return report
