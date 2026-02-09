from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from tube2lora.cache.manager import PIPELINE_STAGES, RunManager
from tube2lora.config import compute_config_hash, load_config
from tube2lora.pipeline import build_context, run_pipeline
from tube2lora.utils.logging import configure_logging, get_logger


app = typer.Typer(help="tube2lora: YouTube-to-LoRA pipeline")


def _echo_reports(run_id: str, reports):
    typer.echo(f"Run ID: {run_id}")
    for report in reports:
        typer.echo(
            f"- {report.stage}: total={report.total} success={report.success} failed={report.failed} skipped={report.skipped}"
        )
        if report.output_path:
            typer.echo(f"  output: {report.output_path}")


def _execute(config: Path, stages: list[str], run_id: Optional[str]) -> None:
    context = build_context(config, run_id=run_id)
    configure_logging(context.pipeline_log_path)
    logger = get_logger("tube2lora")

    reports = run_pipeline(context, logger=logger, stages=stages)
    _echo_reports(context.run_id, reports)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    """Run full pipeline."""
    _execute(config=config, stages=PIPELINE_STAGES, run_id=run_id)


@app.command()
def download(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    _execute(config=config, stages=["download"], run_id=run_id)


@app.command()
def transcribe(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    _execute(config=config, stages=["transcribe"], run_id=run_id)


@app.command()
def normalize(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    _execute(config=config, stages=["normalize"], run_id=run_id)


@app.command()
def analyze(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    _execute(config=config, stages=["analyze"], run_id=run_id)


@app.command("filter")
def filter_cmd(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    _execute(config=config, stages=["filter"], run_id=run_id)


@app.command()
def generate(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    _execute(config=config, stages=["generate"], run_id=run_id)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    _execute(config=config, stages=["train"], run_id=run_id)


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    _execute(config=config, stages=["evaluate"], run_id=run_id)


@app.command()
def status(
    config: Optional[Path] = typer.Option(None, "--config", exists=True, file_okay=True, dir_okay=False),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    """Show pipeline status and cached stage summaries."""

    resolved_runs_dir = runs_dir
    cfg_hash: Optional[str] = None
    if config is not None:
        cfg = load_config(config)
        resolved_runs_dir = cfg.paths.runs_dir
        cfg_hash = compute_config_hash(cfg)

    manager = RunManager(resolved_runs_dir)

    target_dir: Optional[Path] = None
    if run_id:
        candidate = resolved_runs_dir / run_id
        if candidate.exists():
            target_dir = candidate
    else:
        for candidate in manager.list_runs():
            meta_path = candidate / "run.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cfg_hash is None or meta.get("config_hash") == cfg_hash:
                target_dir = candidate
                break

    if target_dir is None:
        typer.echo("No matching runs found.")
        raise typer.Exit(code=1)

    meta = json.loads((target_dir / "run.json").read_text(encoding="utf-8"))
    typer.echo(f"Run ID: {meta.get('run_id', target_dir.name)}")
    typer.echo(f"Status: {meta.get('status', 'unknown')}")
    typer.echo(f"Config hash: {meta.get('config_hash', 'unknown')}")
    typer.echo(f"Run dir: {target_dir}")

    stage_status = meta.get("stage_status", {})
    for stage_name in PIPELINE_STAGES:
        stage_meta = stage_status.get(stage_name, {}) if isinstance(stage_status, dict) else {}
        status_value = stage_meta.get("status", "pending")
        summary = stage_meta.get("summary", {})
        if isinstance(summary, dict) and summary:
            typer.echo(
                f"- {stage_name}: {status_value} (success={summary.get('success', 0)}, failed={summary.get('failed', 0)}, skipped={summary.get('skipped', 0)})"
            )
        else:
            typer.echo(f"- {stage_name}: {status_value}")


if __name__ == "__main__":
    app()
