"""Speech recognition with faster-whisper."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .cues import Segment, Word
from .errors import VideoSubsError
from .gpu import choose_runtime, default_model, preload_cuda_libraries

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


@dataclass
class Transcript:
    """What was heard: the language, and the speech as timed segments."""

    language: str  # ISO 639-1 code
    probability: float
    duration: float
    segments: list[Segment] = field(default_factory=list)
    task: str = "transcribe"  # "translate" means the segments are already English

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "probability": self.probability,
            "duration": self.duration,
            "task": self.task,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "words": [[w.start, w.end, w.text] for w in s.words]}
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transcript:
        segments = [
            Segment(s["start"], s["end"], s["text"], tuple(Word(*w) for w in s.get("words", [])))
            for s in data["segments"]
        ]
        return cls(data["language"], data["probability"], data["duration"], segments, data.get("task", "transcribe"))


class Transcriber(Protocol):
    def transcribe(
        self, audio: Path, *, language: str | None, task: str = "transcribe", progress: ProgressCallback | None = None
    ) -> Transcript: ...


@dataclass
class WhisperSettings:
    model: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad: bool = True
    batch_size: int | None = None  # None: batch on a GPU, not on a CPU
    initial_prompt: str | None = None
    download_root: str | None = None
    cpu_threads: int = 0


class FasterWhisperTranscriber:
    """Loads the model on first use, so planning a run or skipping every file costs nothing."""

    def __init__(self, settings: WhisperSettings | None = None) -> None:
        self.settings = settings or WhisperSettings()
        self.device = self.settings.device
        self.compute_type = self.settings.compute_type
        self.model_name = self.settings.model
        self._model = None
        self._batched = None

    # -- loading -----------------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        preload_cuda_libraries()
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device, compute_type = choose_runtime(self.device, self.compute_type)
        name = self.model_name if self.model_name != "auto" else default_model(device)
        log.info("loading model %s on %s (%s)", name, device, compute_type)
        try:
            self._model = WhisperModel(
                name,
                device=device,
                compute_type=compute_type,
                download_root=self.settings.download_root,
                cpu_threads=self.settings.cpu_threads,
            )
        except Exception as error:
            if device == "cuda" and self.device == "auto":
                log.warning("the GPU could not be used (%s); falling back to the CPU", error)
                self.device, self.compute_type = "cpu", "auto"
                self._load()
                return
            raise VideoSubsError(f"could not load the speech model {name!r}: {error}") from error
        self.device, self.compute_type, self.model_name = device, compute_type, name
        batch = self.settings.batch_size if self.settings.batch_size is not None else (8 if device == "cuda" else 1)
        self._batched = BatchedInferencePipeline(model=self._model) if batch > 1 else None
        self._batch_size = batch

    def describe(self) -> str:
        """Model, device and precision in one line, such as "large-v3-turbo on cuda (float16)"."""
        return f"{self.model_name} on {self.device} ({self.compute_type})"

    def ensure_loaded(self) -> None:
        self._load()

    # -- running -----------------------------------------------------------------------------

    def _options(self, language: str | None, task: str) -> dict:
        return {
            "language": language,
            "task": task,
            "beam_size": self.settings.beam_size,
            "word_timestamps": True,
            "vad_filter": self.settings.vad,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 300},
            "initial_prompt": self.settings.initial_prompt,
            # Carrying text from one window to the next is what lets a hallucinated phrase repeat forever.
            "condition_on_previous_text": False,
            "hallucination_silence_threshold": 2.0,
            "language_detection_segments": 2,
        }

    def transcribe(
        self, audio: Path, *, language: str | None, task: str = "transcribe", progress: ProgressCallback | None = None
    ) -> Transcript:
        self._load()
        options = self._options(language, task)
        try:
            segments, info = self._run(audio, options)
            return self._collect(segments, info, task, progress)
        except RuntimeError as error:
            # Typical on small GPUs with a big model: retry once with a leaner precision.
            if "out of memory" in str(error).lower() and self.device == "cuda" and self.compute_type != "int8_float16":
                log.warning("the GPU ran out of memory; retrying with int8_float16")
                self._model = None
                self.compute_type = "int8_float16"
                self._load()
                segments, info = self._run(audio, options)
                return self._collect(segments, info, task, progress)
            raise

    def _run(self, audio: Path, options: dict):
        if self._batched is not None:
            return self._batched.transcribe(str(audio), batch_size=self._batch_size, **options)
        return self._model.transcribe(str(audio), **options)

    @staticmethod
    def _collect(segments, info, task: str, progress: ProgressCallback | None) -> Transcript:
        out: list[Segment] = []
        total = float(info.duration or 0.0)
        for segment in segments:  # recognition happens here, as the generator is consumed
            text = segment.text.strip()
            if not text:
                continue
            words = tuple(Word(w.start, w.end, w.word) for w in (segment.words or ()))
            out.append(Segment(float(segment.start), float(segment.end), text, words))
            if progress and total > 0:
                progress(min(1.0, float(segment.end) / total))
        if progress:
            progress(1.0)
        return Transcript(info.language, float(info.language_probability), total, out, task)
