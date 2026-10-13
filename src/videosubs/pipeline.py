"""The pipeline: for each file decide what is left to do, then recognise, translate, write and mux.

Design rules that make a long batch safe to leave running:

* One file failing never stops the rest (unless asked); every outcome is reported at the end.
* Work is skipped when its result already exists, so a run can be repeated or resumed.
* The slow step (recognition) is cached, so a failure further along does not cost it twice.
* Output files are written under a temporary name and moved into place, so a crash cannot
  leave a half-written subtitle file, and the source video is never modified.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .cache import TranscriptCache
from .cues import (
    FORMATTERS,
    Cue,
    Style,
    localise_punctuation,
    parse_subtitles,
    segments_to_cues,
    sentence_units,
    translated_to_cues,
    wrap,
)
from .errors import MediaError, TranslationError, VideoSubsError
from .languages import Language, find_language, from_whisper_code
from .media import (
    MediaFile,
    MediaInfo,
    choose_audio_stream,
    extract_audio,
    find_sidecars,
    probe,
    sidecar_path,
)
from .mux import SubtitleTrack, mux, output_path, replace_original
from .transcribe import Transcriber, Transcript
from .translate import TranslationSettings, Translator, choose_backend, create_translator

log = logging.getLogger(__name__)


@dataclass
class Options:
    languages: list[Language] = field(default_factory=list)  # empty: just the spoken language
    source_language: Language | None = None  # None: detected
    formats: list[str] = field(default_factory=lambda: ["srt"])
    embed: bool = False
    replace: bool = False
    keep_sidecars: bool = True
    keep_backup: bool = True
    default_subtitle: Language | None = None
    force: bool = False
    dry_run: bool = False
    stop_on_error: bool = False
    audio_track: int | None = None
    style: Style = field(default_factory=Style)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    cache: TranscriptCache = field(default_factory=lambda: TranscriptCache(enabled=False))
    model_key: str = ""  # identifies the recogniser settings in the cache key


@dataclass
class Outcome:
    media: MediaFile
    status: str  # "done", "skipped", "failed" or "planned"
    message: str = ""
    outputs: list[Path] = field(default_factory=list)
    language: str | None = None  # what was heard
    audio_seconds: float = 0.0
    elapsed: float = 0.0


class Reporter(Protocol):
    def file_started(self, index: int, total: int, media: MediaFile) -> None: ...
    def stage(self, name: str) -> None: ...
    def progress(self, fraction: float) -> None: ...
    def note(self, message: str) -> None: ...
    def info(self, message: str) -> None: ...
    def file_finished(self, outcome: Outcome) -> None: ...


class SilentReporter:
    """Does nothing; for tests and for calling the pipeline as a library."""

    def file_started(self, index: int, total: int, media: MediaFile) -> None: ...
    def stage(self, name: str) -> None: ...
    def progress(self, fraction: float) -> None: ...
    def note(self, message: str) -> None: ...
    def info(self, message: str) -> None: ...
    def file_finished(self, outcome: Outcome) -> None: ...


TranslatorFactory = Callable[[str, TranslationSettings], Translator]


def _write_atomically(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".videosubs-tmp")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def _base_stem(path: Path) -> str:
    """ "movie.en" -> "movie" when the last part of the name is a language tag."""
    stem, _, tag = path.stem.rpartition(".")
    return stem if stem and find_language(tag) else path.stem


class Pipeline:
    def __init__(
        self,
        options: Options,
        transcriber: Transcriber,
        *,
        reporter: Reporter | None = None,
        translator_factory: TranslatorFactory = create_translator,
    ) -> None:
        self.options = options
        self.transcriber = transcriber
        self.reporter = reporter or SilentReporter()
        self._factory = translator_factory
        self._translators: dict[str, Translator] = {}
        self._announced = False

    # -- running a batch ---------------------------------------------------------------------

    def run(self, files: Sequence[MediaFile]) -> list[Outcome]:
        outcomes: list[Outcome] = []
        for index, media in enumerate(files, 1):
            self.reporter.file_started(index, len(files), media)
            started = time.monotonic()
            try:
                outcome = self.process(media)
            except VideoSubsError as error:
                outcome = Outcome(media, "failed", str(error))
            except Exception as error:  # a bug or an odd file: report it, keep going
                log.exception("unexpected error on %s", media.path)
                outcome = Outcome(media, "failed", f"unexpected error: {error}")
            outcome.elapsed = time.monotonic() - started
            outcomes.append(outcome)
            self.reporter.file_finished(outcome)
            if outcome.status == "failed" and self.options.stop_on_error:
                break
        return outcomes

    def process(self, media: MediaFile) -> Outcome:
        if media.kind == "subtitle":
            return self._translate_subtitle_file(media)
        return self._process_media(media)

    # -- recognising speech ------------------------------------------------------------------

    def _process_media(self, media: MediaFile) -> Outcome:
        opts = self.options
        info = probe(media.path)
        position = choose_audio_stream(info, track=opts.audio_track, language=opts.source_language)
        formats = self._formats(media)

        skip = self._nothing_to_do(media, info, formats)
        if skip:
            return Outcome(media, "skipped", skip, audio_seconds=info.duration)
        if opts.dry_run:
            return Outcome(media, "planned", self._plan(media, formats), audio_seconds=info.duration)

        self.reporter.stage("Transcribing")
        transcript = self._transcript(media, info, position, task="transcribe")
        source = opts.source_language or from_whisper_code(transcript.language)
        if opts.source_language is None and transcript.probability < 0.6:
            self.reporter.note(
                f"not sure which language this is ({source.name}, {transcript.probability:.0%}); "
                "use --source-language if the subtitles look wrong"
            )
        if not transcript.segments:
            return Outcome(media, "skipped", "no speech was found", language=source.code, audio_seconds=info.duration)

        targets = opts.languages or [source]
        written: dict[str, Path] = {}  # language code -> the SRT file to embed
        outputs: list[Path] = []
        for language in targets:
            paths = {extension: sidecar_path(media.path, language, extension) for extension in formats}
            wanted = [extension for extension, path in paths.items() if opts.force or not path.exists()]
            # Translating is the slow part, so it is only done for files that will really be written.
            cues = self._cues_for(media, info, position, transcript, source, language) if wanted else []
            for extension, path in paths.items():
                if extension in wanted:
                    _write_atomically(path, FORMATTERS[extension](cues))
                    outputs.append(path)
                else:
                    self.reporter.note(f"kept the existing {path.name} (use --force to rewrite it)")
                if extension == "srt":
                    written[language.code] = path

        if opts.embed and media.kind == "video":
            outputs.extend(self._embed(media, info, targets, written))
        return Outcome(
            media,
            "done",
            ", ".join(language.name for language in targets),
            outputs=outputs,
            language=source.code,
            audio_seconds=info.duration,
        )

    def _formats(self, media: MediaFile) -> list[str]:
        formats = list(dict.fromkeys(self.options.formats))
        if self.options.embed and media.kind == "video" and "srt" not in formats:
            formats.insert(0, "srt")  # the muxer takes SRT
        return formats

    def _nothing_to_do(self, media: MediaFile, info: MediaInfo, formats: list[str]) -> str | None:
        """Why this file needs no work (so the run can say so), or None when it does."""
        opts = self.options
        if opts.force:
            return None
        if opts.languages:
            written = all(sidecar_path(media.path, lang, f).exists() for lang in opts.languages for f in formats)
            if written and self._embedded(media, info, opts.languages):
                return "already has subtitles in " + ", ".join(lang.name for lang in opts.languages)
            return None
        # No target languages: the spoken language is written, and any subtitle file this tool
        # could have made for the video means that was already done.
        if opts.source_language:
            existing = [
                p for lang in [opts.source_language] if (p := sidecar_path(media.path, lang, formats[0])).exists()
            ]
        else:
            existing = find_sidecars(media.path, formats[0])
        if existing and self._embedded(media, info, [opts.source_language] if opts.source_language else []):
            return f"already has {existing[0].name}"
        return None

    def _embedded(self, media: MediaFile, info: MediaInfo, languages: Sequence[Language]) -> bool:
        """Is the embedding step finished (or not wanted) for these languages?"""
        if not (self.options.embed and media.kind == "video"):
            return True
        if output_path(media.path).exists():
            return True
        present = {tag.whisper for s in info.subtitles if s.language and (tag := find_language(s.language))}
        return bool(languages) and all(lang.whisper in present for lang in languages)

    def _plan(self, media: MediaFile, formats: list[str]) -> str:
        languages = ", ".join(lang.name for lang in self.options.languages) or "the spoken language"
        files = f"{media.path.stem}.<lang>.{'/'.join(formats)}"
        extra = f", then {output_path(media.path).name}" if self.options.embed and media.kind == "video" else ""
        return f"would write {files} for {languages}{extra}"

    def _transcript(self, media: MediaFile, info: MediaInfo, position: int, *, task: str) -> Transcript:
        opts = self.options
        language = opts.source_language.whisper if opts.source_language else None
        key = opts.cache.key(media.path, recogniser=opts.model_key, task=task, language=language, track=position)
        cached = opts.cache.get(key)
        if cached is not None:
            self.reporter.note("reused the saved transcript of an earlier run")
            return cached

        with tempfile.TemporaryDirectory(prefix="videosubs-") as scratch:
            audio = media.path
            if position != 0:  # the recogniser reads the first audio track by itself
                audio = extract_audio(media.path, position, Path(scratch) / "audio.wav")
            transcript = self.transcriber.transcribe(
                audio, language=language, task=task, progress=self.reporter.progress
            )
            self._announce_model()
        if transcript.duration <= 0:
            transcript.duration = info.duration
        opts.cache.put(key, transcript)
        return transcript

    def _announce_model(self) -> None:
        """Tell the person once which model and device are doing the work."""
        if self._announced:
            return
        self._announced = True
        describe = getattr(self.transcriber, "describe", None)
        if callable(describe):
            self.reporter.info(f"speech model: {describe()}")

    # -- making the subtitles ----------------------------------------------------------------

    def _cues_for(
        self,
        media: MediaFile,
        info: MediaInfo,
        position: int,
        transcript: Transcript,
        source: Language,
        target: Language,
    ) -> list[Cue]:
        style = self.options.style
        if target.code == source.code:
            return segments_to_cues(transcript.segments, style, target.spaced)

        backend = choose_backend(self.options.translation, target)
        if backend == "whisper":
            if target.whisper != "en":
                raise TranslationError("Whisper can only translate into English. Use --translator nllb or deepl.")
            self.reporter.stage("Translating to English (Whisper)")
            english = self._transcript(media, info, position, task="translate")
            return segments_to_cues(english.segments, style, True)

        self.reporter.stage(f"Translating to {target.name} ({backend})")
        translator = self._translator(backend)
        # One sentence at a time: better translations, and each one keeps its own true timing.
        units = sentence_units(transcript.segments)
        texts = translator.translate([unit.text for unit in units], source, target, self.reporter.progress)
        texts = [localise_punctuation(text, target.code) for text in texts]
        return translated_to_cues(units, texts, style, target.spaced)

    def _translator(self, backend: str) -> Translator:
        if backend not in self._translators:
            self._translators[backend] = self._factory(backend, self.options.translation)
        return self._translators[backend]

    # -- muxing ------------------------------------------------------------------------------

    def _embed(
        self, media: MediaFile, info: MediaInfo, targets: list[Language], written: dict[str, Path]
    ) -> list[Path]:
        opts = self.options
        already = {tag.whisper for s in info.subtitles if s.language and (tag := find_language(s.language))}
        tracks = [
            SubtitleTrack(written[lang.code], lang)
            for lang in targets
            if lang.code in written and (opts.force or lang.whisper not in already)
        ]
        if not tracks:
            self.reporter.note("every requested language is already embedded in the video")
            return []
        destination = output_path(media.path)
        if destination.exists() and not opts.force:
            self.reporter.note(f"{destination.name} already exists (use --force to rewrite it)")
            return []
        self.reporter.stage("Adding subtitles to the video")
        mux(
            media.path,
            info,
            tracks,
            destination,
            default_language=opts.default_subtitle,
            on_progress=self.reporter.progress,
        )
        outputs = [destination]
        if opts.replace:
            outputs = [replace_original(media.path, destination, keep_backup=opts.keep_backup)]
        if not opts.keep_sidecars:
            for track in tracks:
                track.path.unlink(missing_ok=True)
        return outputs

    # -- translating an existing subtitle file -----------------------------------------------

    def _translate_subtitle_file(self, media: MediaFile) -> Outcome:
        opts = self.options
        if not opts.languages:
            return Outcome(media, "skipped", "a subtitle file is only translated: choose languages with -l")
        tag = media.path.suffixes[-2].lstrip(".") if len(media.path.suffixes) >= 2 else ""
        source = opts.source_language or find_language(tag)
        if source is None:
            raise MediaError("what language is it written in? Say so with --source-language")
        extension = media.path.suffix.lstrip(".").lower()
        base = media.path.with_name(_base_stem(media.path) + media.path.suffix)

        targets = [lang for lang in opts.languages if lang.code != source.code]
        todo = [lang for lang in targets if opts.force or not sidecar_path(base, lang, extension).exists()]
        if not todo:
            return Outcome(media, "skipped", "already translated into " + ", ".join(lang.name for lang in targets))
        if opts.dry_run:
            return Outcome(media, "planned", "would translate into " + ", ".join(lang.name for lang in todo))

        cues = parse_subtitles(media.path.read_text(encoding="utf-8-sig", errors="replace"))
        if not cues:
            raise MediaError("no subtitles were found in this file")
        outputs: list[Path] = []
        for language in todo:
            backend = choose_backend(opts.translation, language)
            if backend == "whisper":
                raise TranslationError(
                    "Whisper translates speech, not text: use --translator nllb or deepl for a subtitle file."
                )
            self.reporter.stage(f"Translating to {language.name} ({backend})")
            texts = self._translator(backend).translate(
                [" ".join(cue.text.splitlines()) for cue in cues], source, language, self.reporter.progress
            )
            texts = [localise_punctuation(text, language.code) for text in texts]
            width = opts.style.line_width(language.spaced)
            translated = [
                Cue(cue.start, cue.end, "\n".join(wrap(text.strip(), width, opts.style.max_lines, language.spaced)))
                for cue, text in zip(cues, texts, strict=True)
            ]
            path = sidecar_path(base, language, extension)
            _write_atomically(path, FORMATTERS[extension](translated))
            outputs.append(path)
        return Outcome(media, "done", ", ".join(lang.name for lang in todo), outputs=outputs, language=source.code)
