from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from tube2lora.cache.manager import ManifestStore, RunContext
from tube2lora.stages.common import StageReport
from tube2lora.utils.hashing import sha256_file, stable_dict_hash
from tube2lora.utils.io import atomic_write_json, iter_jsonl


def _video_level_split(
    entries: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not entries:
        return [], []

    by_video: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        metadata = entry.get("metadata")
        video_id = None
        if isinstance(metadata, dict):
            raw_video_id = metadata.get("video_id")
            if isinstance(raw_video_id, str):
                video_id = raw_video_id
        if not video_id:
            video_id = "unknown"
        by_video.setdefault(video_id, []).append(entry)

    video_ids = list(by_video.keys())
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    eval_video_count = int(len(video_ids) * eval_ratio)
    if eval_ratio > 0 and len(video_ids) > 1:
        eval_video_count = max(1, eval_video_count)
    eval_ids = set(video_ids[:eval_video_count])

    train_entries: list[dict[str, Any]] = []
    eval_entries: list[dict[str, Any]] = []

    for video_id, samples in by_video.items():
        if video_id in eval_ids:
            eval_entries.extend(samples)
        else:
            train_entries.extend(samples)

    if not train_entries and eval_entries:
        train_entries, eval_entries = eval_entries, []

    return train_entries, eval_entries


def run(context: RunContext, logger: logging.Logger) -> StageReport:
    stage_name = "train"
    context.update_stage_status(stage_name, "running")

    if not context.config.train.enabled:
        report = StageReport(stage=stage_name, total=0, success=0, failed=0, skipped=0, output_path=None)
        context.update_stage_status(stage_name, "completed", details={"summary": report.__dict__})
        return report

    dataset_path = context.stage_dir("generate") / "dataset_messages.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError("Generate stage output not found.")

    stage_dir = context.stage_dir(stage_name)
    artifact_dir = context.stage_artifact_dir(stage_name)
    manifest = ManifestStore(context.stage_manifest_path(stage_name))

    input_hash = stable_dict_hash(
        {
            "dataset_sha256": sha256_file(dataset_path),
            "train_cfg": context.config.train.model_dump(mode="json"),
        }
    )

    if manifest.should_skip("train", input_hash):
        cached = manifest.items["train"].get("output")
        report = StageReport(stage=stage_name, total=1, success=1, failed=0, skipped=1, output_path=None)
        context.update_stage_status(stage_name, "completed", details={"cached": cached})
        return report

    entries = [row for row in iter_jsonl(dataset_path)]
    if not entries:
        raise RuntimeError("No generated dataset entries found for training")

    train_entries, eval_entries = _video_level_split(
        entries,
        eval_ratio=context.config.train.eval_split_ratio,
        seed=context.config.train.seed,
    )

    split_path = stage_dir / "dataset_split.json"
    atomic_write_json(
        split_path,
        {
            "total_entries": len(entries),
            "train_entries": len(train_entries),
            "eval_entries": len(eval_entries),
            "eval_split_ratio": context.config.train.eval_split_ratio,
        },
    )

    try:
        import torch
        from datasets import Dataset
        from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies missing. Install with: pip install -e .[train]"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for current training implementation.")

    # Patch for distributed world size issue in some transformers versions.
    original_get_world_size = torch.distributed.get_world_size

    def patched_get_world_size(group=None):
        if not torch.distributed.is_initialized():
            return 1
        return original_get_world_size(group)

    torch.distributed.get_world_size = patched_get_world_size

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=context.config.train.base_model,
        max_seq_length=context.config.train.max_seq_length,
        dtype=None,
        load_in_4bit=context.config.train.load_in_4bit,
        device_map="cuda",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=context.config.train.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=context.config.train.lora_alpha,
        lora_dropout=context.config.train.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=context.config.train.seed,
        use_rslora=context.config.train.use_rslora,
        loftq_config=None,
    )

    eos_token = tokenizer.eos_token or ""

    def format_examples(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts: list[str] = []
        messages_list = examples["messages"]
        for messages in messages_list:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text + eos_token)
        return {"text": texts}

    train_dataset = Dataset.from_list(train_entries).map(format_examples, batched=True)
    eval_dataset = Dataset.from_list(eval_entries).map(format_examples, batched=True) if eval_entries else None

    logs_dir = artifact_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    training_args_kwargs: dict[str, Any] = {
        "per_device_train_batch_size": context.config.train.per_device_train_batch_size,
        "gradient_accumulation_steps": context.config.train.gradient_accumulation_steps,
        "warmup_steps": context.config.train.warmup_steps,
        "num_train_epochs": context.config.train.num_train_epochs,
        "learning_rate": context.config.train.learning_rate,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": context.config.train.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": context.config.train.weight_decay,
        "lr_scheduler_type": context.config.train.lr_scheduler_type,
        "seed": context.config.train.seed,
        "output_dir": str(artifact_dir / "trainer_outputs"),
        "save_strategy": "steps",
        "save_steps": context.config.train.save_steps,
        "report_to": "tensorboard",
        "logging_dir": str(logs_dir),
    }
    if eval_dataset is not None:
        training_args_kwargs["evaluation_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = context.config.train.save_steps

    training_args = UnslothTrainingArguments(**training_args_kwargs)

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=context.config.train.max_seq_length,
        dataset_num_proc=context.config.train.dataset_num_proc,
        packing=context.config.train.packing,
        args=training_args,
    )

    logger.info("Starting training with %s train examples and %s eval examples", len(train_dataset), len(eval_dataset) if eval_dataset else 0)
    train_result = trainer.train()

    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    eval_loss = float(eval_metrics.get("eval_loss", 0.0)) if eval_metrics else 0.0
    eval_perplexity = math.exp(eval_loss) if eval_loss > 0 else None

    adapter_dir = artifact_dir / context.config.train.output_name
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics_payload = {
        "training_loss": float(train_result.training_loss),
        "eval_loss": eval_loss,
        "eval_perplexity": eval_perplexity,
        "train_entries": len(train_entries),
        "eval_entries": len(eval_entries),
        "adapter_dir": str(adapter_dir),
        "tensorboard_log_dir": str(logs_dir),
    }
    metrics_path = stage_dir / "training_metrics.json"
    atomic_write_json(metrics_path, metrics_payload)

    manifest.mark(
        "train",
        status="success",
        input_hash=input_hash,
        output={
            "metrics_path": str(metrics_path),
            "adapter_dir": str(adapter_dir),
            "tensorboard_log_dir": str(logs_dir),
        },
    )

    report = StageReport(
        stage=stage_name,
        total=1,
        success=1,
        failed=0,
        skipped=0,
        output_path=str(metrics_path),
    )
    context.update_stage_status(
        stage_name,
        "completed",
        details={
            "summary": {
                "total": report.total,
                "success": report.success,
                "failed": report.failed,
                "skipped": report.skipped,
            },
            "output_path": str(metrics_path),
        },
    )
    return report
