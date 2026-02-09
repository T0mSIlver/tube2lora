from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from tube2lora.cache.manager import PIPELINE_STAGES, RunContext, RunManager
from tube2lora.config import AppConfig, compute_config_hash, load_config
from tube2lora.stages import analyze, download, evaluate, filter as filter_stage, generate, normalize, train, transcribe
from tube2lora.stages.common import StageReport


StageFunction = Callable[[RunContext, logging.Logger], StageReport]


class FailureThresholdExceeded(RuntimeError):
    pass


STAGE_FUNCTIONS: dict[str, StageFunction] = {
    "download": download.run,
    "transcribe": transcribe.run,
    "normalize": normalize.run,
    "analyze": analyze.run,
    "filter": filter_stage.run,
    "generate": generate.run,
    "train": train.run,
    "evaluate": evaluate.run,
}


def build_context(config_path: str | Path, run_id: str | None = None) -> RunContext:
    config = load_config(config_path)
    config_hash = compute_config_hash(config)
    manager = RunManager(config.paths.runs_dir)
    context = manager.resolve_run(config=config, config_hash=config_hash, run_id=run_id)
    context.config = config
    return context


def run_stage(context: RunContext, stage_name: str, logger: logging.Logger) -> StageReport:
    stage_fn = STAGE_FUNCTIONS.get(stage_name)
    if stage_fn is None:
        raise ValueError(f"Unknown stage: {stage_name}")
    if context.config is None:
        raise RuntimeError("RunContext has no config loaded")
    return stage_fn(context, logger)


def run_pipeline(
    context: RunContext,
    logger: logging.Logger,
    stages: list[str] | None = None,
) -> list[StageReport]:
    if context.config is None:
        raise RuntimeError("RunContext has no config loaded")

    requested_stages = stages or PIPELINE_STAGES
    reports: list[StageReport] = []

    for stage_name in requested_stages:
        logger.info("Starting stage: %s", stage_name)
        try:
            report = run_stage(context, stage_name, logger)
            reports.append(report)
            logger.info(
                "Stage %s completed (success=%s failed=%s skipped=%s)",
                stage_name,
                report.success,
                report.failed,
                report.skipped,
            )

            if report.failure_rate > context.config.pipeline.max_failure_rate:
                raise FailureThresholdExceeded(
                    f"Stage '{stage_name}' failure rate {report.failure_rate:.2f} exceeded max_failure_rate {context.config.pipeline.max_failure_rate:.2f}"
                )

        except Exception as exc:
            context.update_stage_status(stage_name, "failed", details={"error": str(exc)})
            logger.exception("Stage %s failed", stage_name)
            if isinstance(exc, FailureThresholdExceeded):
                raise
            if not context.config.pipeline.continue_on_error:
                raise

    # Mark run completed if all requested stages have status completed.
    meta = context.read_meta()
    stage_status = meta.get("stage_status", {})
    requested_ok = all(stage_status.get(stage, {}).get("status") == "completed" for stage in requested_stages)
    if requested_ok:
        meta["status"] = "completed"
        context.write_meta(meta)

    return reports
