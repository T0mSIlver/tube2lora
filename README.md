# tube2lora

`tube2lora` is a config-driven CLI pipeline that goes from YouTube URLs to a LoRA adapter, with a Mistral-first workflow:

1. `download`
2. `transcribe`
3. `normalize`
4. `analyze`
5. `filter`
6. `generate`
7. `train`
8. `evaluate`

It is designed for resumability, per-stage caching, and reproducible per-run artifacts under `runs/<run_id>/`.

## Highlights

- Stage-by-stage CLI (`tube2lora download`, `tube2lora transcribe`, ...)
- Per-item manifests for cache + resume behavior
- New run directory on config/prompt hash change
- OpenAI-compatible LLM client for normalization + dataset generation
- Transcriber protocol with backends:
  - `faster_whisper`
  - `youtube_native`
- Voxtral is currently sidelined and planned through vLLM realtime API integration
- Video-level train/eval split (prevents chunk leakage)
- TensorBoard logging in training stage

## Install (uv)

```bash
uv sync
```

Training dependencies:

```bash
uv sync --extra train
```

## Quickstart

1. Copy and edit config:

```bash
cp configs/default.yaml my_config.yaml
```

2. Run full pipeline:

```bash
uv run tube2lora run --config my_config.yaml
```

3. Run individual stages:

```bash
uv run tube2lora download --config my_config.yaml
uv run tube2lora transcribe --config my_config.yaml
uv run tube2lora train --config my_config.yaml
```

4. Check status:

```bash
uv run tube2lora status --config my_config.yaml
```

## Run Artifacts

Each run has:

- `run.json` stage status + metadata
- `logs/pipeline.log`
- `manifests/<stage>.json` item-level cache/resume records
- `stages/<stage>/...` stage outputs
- `artifacts/<stage>/...` heavy artifacts (media, models, logs)

## Config and Prompt Templates

- Default config: `configs/default.yaml`
- Normalize prompt template: `configs/prompts/normalize_default.yaml`
- Dataset task prompt template: `configs/prompts/video_essay_example.yaml`

The generation prompt template is mandatory and controls task-specific SFT message creation.
