from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tube2lora.utils.hashing import sha256_bytes, stable_dict_hash


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class InputConfig(StrictModel):
    source_type: Literal["channel", "playlist", "video", "url_file"] = "channel"
    source: str


class DownloadConfig(StrictModel):
    include_shorts: bool = False
    include_live: bool = False
    media: Literal["audio", "video", "none"] = "audio"
    keep_video: bool = False
    yt_dlp_audio_format: str = "bestaudio/best"
    yt_dlp_video_format: str = "bestvideo+bestaudio/best"


class FasterWhisperConfig(StrictModel):
    model_size: str = "large-v3"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True


class TranscribeConfig(StrictModel):
    backend: Literal["faster_whisper", "youtube_native"] = "faster_whisper"
    prefer_manual_transcript: bool = True
    allow_auto_generated: bool = False
    language_allowlist: list[str] = Field(default_factory=lambda: ["en"])
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)


class LLMEndpointConfig(StrictModel):
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 120
    max_retries: int = 3


class NormalizeConfig(StrictModel):
    enabled: bool = True
    prompt_template: Path
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096

    @field_validator("prompt_template", mode="before")
    @classmethod
    def _coerce_prompt_template(cls, value: Path | str) -> Path:
        return Path(value)


class AnalyzeConfig(StrictModel):
    enabled: bool = True
    minhash_num_perm: int = 128


class FilterConfig(StrictModel):
    enabled: bool = True
    min_words: int = 100
    language_allowlist: list[str] = Field(default_factory=lambda: ["en"])
    dedup_threshold: float = 0.85
    min_quality_score: float = 0.2


class GenerateConfig(StrictModel):
    enabled: bool = True
    prompt_template: Path
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096
    buffer_chars: int = 7000
    fallback_min_chars: int = 2500
    fallback_max_chars: int = 3500
    min_tail_chars: int = 800

    @field_validator("prompt_template", mode="before")
    @classmethod
    def _coerce_prompt_template(cls, value: Path | str) -> Path:
        return Path(value)


class TrainConfig(StrictModel):
    enabled: bool = True
    base_model: str = "unsloth/Ministral-3-3B-Instruct-2512"
    load_in_4bit: bool = True
    max_seq_length: int = 16384
    output_name: str = "ministral3b_tube2lora_adapter"
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 15
    num_train_epochs: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 5
    save_steps: int = 50
    dataset_num_proc: int = 2
    packing: bool = False
    seed: int = 3407
    eval_split_ratio: float = 0.1


class EvaluateConfig(StrictModel):
    enabled: bool = True
    num_generation_samples: int = 3
    max_new_tokens: int = 200
    temperature: float = 0.7


class PipelineConfig(StrictModel):
    continue_on_error: bool = True
    max_failure_rate: float = 0.2


class PathsConfig(StrictModel):
    runs_dir: Path = Path("runs")

    @field_validator("runs_dir", mode="before")
    @classmethod
    def _coerce_runs_dir(cls, value: Path | str) -> Path:
        return Path(value)


class AppConfig(StrictModel):
    input: InputConfig
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    llm: LLMEndpointConfig
    normalize: NormalizeConfig
    analyze: AnalyzeConfig = Field(default_factory=AnalyzeConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    generate: GenerateConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    config_path: Path = Field(default=Path("."), exclude=True, repr=False)

    @field_validator("config_path", mode="before")
    @classmethod
    def _normalize_config_path(cls, value: Path | str) -> Path:
        return Path(value).resolve()

    @model_validator(mode="after")
    def _validate_rules(self) -> "AppConfig":
        if self.input.source_type == "url_file" and not self.input.source:
            raise ValueError("input.source must point to a URL file when source_type=url_file")
        if self.pipeline.max_failure_rate < 0 or self.pipeline.max_failure_rate > 1:
            raise ValueError("pipeline.max_failure_rate must be between 0 and 1")
        return self


class PromptTemplate(StrictModel):
    name: str
    system_message: str = ""
    user_message_template: str
    fact_extraction_prompt: str
    semantic_boundary_prompt: str


class NormalizePromptTemplate(StrictModel):
    name: str
    system_prompt: str
    user_prompt_template: str


def _resolve_path(base_dir: Path, maybe_path: Path) -> Path:
    if maybe_path.is_absolute():
        return maybe_path
    return (base_dir / maybe_path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def load_config(config_path: str | Path) -> AppConfig:
    resolved_path = Path(config_path).resolve()
    payload = load_yaml(resolved_path)
    cfg = AppConfig.model_validate(payload).model_copy(update={"config_path": resolved_path})

    base_dir = resolved_path.parent

    updated = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={"runs_dir": _resolve_path(base_dir, cfg.paths.runs_dir)}
            ),
            "normalize": cfg.normalize.model_copy(
                update={
                    "prompt_template": _resolve_path(base_dir, cfg.normalize.prompt_template)
                }
            ),
            "generate": cfg.generate.model_copy(
                update={
                    "prompt_template": _resolve_path(base_dir, cfg.generate.prompt_template)
                }
            ),
        }
    )

    if updated.input.source_type == "url_file":
        updated = updated.model_copy(
            update={
                "input": updated.input.model_copy(
                    update={"source": str(_resolve_path(base_dir, Path(updated.input.source)))}
                )
            }
        )

    return updated


def load_prompt_template(path: Path) -> PromptTemplate:
    payload = load_yaml(path)
    return PromptTemplate.model_validate(payload)


def load_normalize_prompt_template(path: Path) -> NormalizePromptTemplate:
    payload = load_yaml(path)
    return NormalizePromptTemplate.model_validate(payload)


def compute_config_hash(config: AppConfig) -> str:
    config_file_bytes = config.config_path.read_bytes()
    normalize_prompt_bytes = config.normalize.prompt_template.read_bytes()
    generate_prompt_bytes = config.generate.prompt_template.read_bytes()

    payload: dict[str, Any] = {
        "config_file_sha256": sha256_bytes(config_file_bytes),
        "normalize_prompt_sha256": sha256_bytes(normalize_prompt_bytes),
        "generate_prompt_sha256": sha256_bytes(generate_prompt_bytes),
        "resolved_config": config.model_dump(mode="json", exclude={"config_path"}),
    }

    if config.input.source_type == "url_file":
        source_path = Path(config.input.source)
        if source_path.exists():
            payload["url_file_sha256"] = sha256_bytes(source_path.read_bytes())

    return stable_dict_hash(payload)
