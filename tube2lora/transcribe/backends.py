from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tube2lora.config import FasterWhisperConfig, VoxtralConfig


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptionResult:
    source: str
    language: str | None
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        raise NotImplementedError


VIDEO_ID_PATTERNS = [
    r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?]|$)",
    r"youtu\.be/([0-9A-Za-z_-]{11})",
    r"embed/([0-9A-Za-z_-]{11})",
]


def extract_video_id(video_url: str) -> str | None:
    for pattern in VIDEO_ID_PATTERNS:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_transcript(
    video_id: str,
    language_allowlist: list[str],
    *,
    manual_only: bool,
    allow_auto_generated: bool,
    logger: logging.Logger,
) -> TranscriptionResult | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
    except ImportError as exc:
        raise RuntimeError("youtube-transcript-api is required for youtube_native backend") from exc

    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        return None

    selected = None

    # First preference: manually created transcripts in allowlist.
    for transcript in transcript_list:
        if getattr(transcript, "is_generated", False):
            continue
        if transcript.language_code in language_allowlist:
            selected = transcript
            break

    # Optional fallback: manually created transcript in any language.
    if selected is None and manual_only:
        for transcript in transcript_list:
            if getattr(transcript, "is_generated", False):
                continue
            selected = transcript
            break

    # Optional auto-generated fallback when explicitly enabled.
    if selected is None and allow_auto_generated:
        for transcript in transcript_list:
            if transcript.language_code in language_allowlist:
                selected = transcript
                break
        if selected is None:
            available = list(transcript_list)
            if available:
                selected = available[0]

    if selected is None:
        return None

    transcript_language = getattr(selected, "language_code", None)
    if transcript_language and transcript_language not in language_allowlist:
        logger.info(
            "Skipping video %s because manual transcript language '%s' is outside allowlist %s",
            video_id,
            transcript_language,
            language_allowlist,
        )
        return TranscriptionResult(source="youtube_manual_filtered", language=transcript_language, segments=[])

    fetched = selected.fetch()
    segments: list[TranscriptSegment] = []
    for item in fetched:
        text = getattr(item, "text", "").strip()
        if not text:
            continue
        start = float(getattr(item, "start", 0.0))
        duration = float(getattr(item, "duration", 0.0))
        end = start + duration
        segments.append(TranscriptSegment(start=start, end=end, text=text))

    source = "youtube_manual" if not getattr(selected, "is_generated", False) else "youtube_auto"
    return TranscriptionResult(source=source, language=transcript_language, segments=segments)


class FasterWhisperTranscriber:
    def __init__(self, cfg: FasterWhisperConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is required for faster_whisper backend") from exc

        device = self.cfg.device
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"

        compute_type = self.cfg.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self.logger.info(
            "Loading Faster Whisper model %s (device=%s, compute_type=%s)",
            self.cfg.model_size,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            self.cfg.model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        self._ensure_model()
        if self._model is None:  # pragma: no cover
            raise RuntimeError("Whisper model is not loaded")

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
            condition_on_previous_text=True,
        )

        segments: list[TranscriptSegment] = []
        for seg in segments_iter:
            text = str(getattr(seg, "text", "")).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=float(getattr(seg, "start", 0.0)),
                    end=float(getattr(seg, "end", 0.0)),
                    text=text,
                )
            )

        return TranscriptionResult(
            source="faster_whisper",
            language=str(getattr(info, "language", "")) or None,
            segments=segments,
        )


class VoxtralTranscriber:
    def __init__(self, cfg: VoxtralConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers+torch are required for voxtral backend"
            ) from exc

        device = -1
        if self.cfg.device == "cuda":
            device = 0
        elif self.cfg.device == "auto" and torch.cuda.is_available():
            device = 0

        torch_dtype = None
        if self.cfg.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.cfg.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.cfg.torch_dtype == "float32":
            torch_dtype = torch.float32

        self.logger.info("Loading Voxtral ASR pipeline (%s)", self.cfg.model_name)
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.cfg.model_name,
            device=device,
            torch_dtype=torch_dtype,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        self._ensure_pipeline()
        if self._pipeline is None:  # pragma: no cover
            raise RuntimeError("Voxtral pipeline was not initialized")

        output = self._pipeline(
            str(audio_path),
            return_timestamps=True,
            chunk_length_s=self.cfg.chunk_length_s,
            batch_size=self.cfg.batch_size,
        )

        segments: list[TranscriptSegment] = []
        chunks = output.get("chunks") if isinstance(output, dict) else None
        if isinstance(chunks, list) and chunks:
            for chunk in chunks:
                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue
                timestamp = chunk.get("timestamp") or (0.0, 0.0)
                start = float(timestamp[0] or 0.0)
                end = float(timestamp[1] or start)
                segments.append(TranscriptSegment(start=start, end=end, text=text))
        else:
            text = str(output.get("text", "")).strip() if isinstance(output, dict) else str(output)
            if text:
                segments.append(TranscriptSegment(start=0.0, end=0.0, text=text))

        return TranscriptionResult(source="voxtral", language=None, segments=segments)
