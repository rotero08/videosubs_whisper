"""Finding media files, and asking ffprobe and ffmpeg about them."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .errors import MediaError, ToolMissingError
from .languages import Language, find_language

VIDEO_EXTENSIONS = frozenset(
    {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".wmv", ".flv", ".mpg", ".mpeg",
     ".ts", ".m2ts", ".mts", ".3gp", ".ogv"}
)  # fmt: skip
AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".aac", ".flac", ".wav", ".ogg", ".oga", ".opus", ".wma"})
SUBTITLE_EXTENSIONS = frozenset({".srt", ".vtt"})

#: Files this tool wrote itself end like "movie.subs.mkv" and are never processed again.
OUTPUT_MARKER = ".subs"
BACKUP_DIR = ".videosubs-backup"


@dataclass(frozen=True)
class MediaFile:
    path: Path
    kind: str  # "video", "audio" or "subtitle"


def kind_of(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in SUBTITLE_EXTENSIONS:
        return "subtitle"
    return None


def is_own_output(path: Path) -> bool:
    return path.stem.lower().endswith(OUTPUT_MARKER)


def _natural_key(path: Path) -> list[object]:
    """Sort "episode 2" before "episode 10"."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def discover(inputs: Iterable[Path], *, recursive: bool = True) -> tuple[list[MediaFile], list[str]]:
    """Expand files and folders into media files, in natural order, with duplicates removed.

    A folder yields its videos and audio files (not subtitle files: those are only translated
    when named explicitly). Hidden folders, the tool's own outputs and its backup folder are
    skipped. The second value lists problems worth telling the user about.
    """
    found: dict[Path, MediaFile] = {}
    problems: list[str] = []

    def add(path: Path, *, explicit: bool) -> None:
        kind = kind_of(path)
        if kind is None or (kind == "subtitle" and not explicit) or is_own_output(path):
            return
        found.setdefault(path.resolve(), MediaFile(path, kind))

    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            walker = path.rglob("*") if recursive else path.glob("*")
            for child in walker:
                relative = child.relative_to(path).parts[:-1]
                if any(part.startswith(".") for part in relative) or not child.is_file():
                    continue
                add(child, explicit=False)
        elif path.is_file():
            if kind_of(path) is None:
                problems.append(f"{path}: not a video, audio or subtitle file I recognise")
            elif is_own_output(path):
                problems.append(f"{path}: this looks like a file I wrote earlier, so I skipped it")
            else:
                add(path, explicit=True)
        else:
            problems.append(f"{path}: no such file or folder")
    return sorted(found.values(), key=lambda m: _natural_key(m.path)), problems


# ---------------------------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamInfo:
    index: int  # position among all streams of the file
    kind: str  # "video", "audio", "subtitle", ...
    codec: str
    language: str | None
    title: str | None
    channels: int | None
    default: bool


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    streams: tuple[StreamInfo, ...]

    def of_kind(self, kind: str) -> list[StreamInfo]:
        return [s for s in self.streams if s.kind == kind]

    @property
    def audio(self) -> list[StreamInfo]:
        return self.of_kind("audio")

    @property
    def subtitles(self) -> list[StreamInfo]:
        return self.of_kind("subtitle")

    @property
    def has_video(self) -> bool:
        return any(s.kind == "video" for s in self.streams)


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise ToolMissingError(
            f"{name} is not installed or not on the PATH. Install FFmpeg "
            "(https://ffmpeg.org/download.html; on Linux `sudo apt install ffmpeg` or "
            "`sudo dnf install ffmpeg`, on macOS `brew install ffmpeg`, on Windows `winget install ffmpeg`)."
        )
    return found


def parse_probe(path: Path, data: dict) -> MediaInfo:
    """Turn ffprobe's JSON into a MediaInfo (separate from the call so it can be tested)."""
    streams = []
    for raw in data.get("streams", []):
        tags = {k.lower(): v for k, v in (raw.get("tags") or {}).items()}
        language = tags.get("language")
        streams.append(
            StreamInfo(
                index=int(raw.get("index", len(streams))),
                kind=raw.get("codec_type", "unknown"),
                codec=raw.get("codec_name", "unknown"),
                language=None if language in (None, "", "und") else language,
                title=tags.get("title"),
                channels=raw.get("channels"),
                default=bool((raw.get("disposition") or {}).get("default")),
            )
        )
    duration = 0.0
    for candidate in [
        (data.get("format") or {}).get("duration"),
        *(s.get("duration") for s in data.get("streams", [])),
    ]:
        try:
            duration = max(duration, float(candidate))
        except (TypeError, ValueError):
            continue
    return MediaInfo(path=path, duration=duration, streams=tuple(streams))


def probe(path: Path) -> MediaInfo:
    ffprobe = require_tool("ffprobe")
    result = subprocess.run(
        # -i, so that a relative name starting with "-" is not read as an option
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        lines = (result.stderr or "").strip().splitlines()
        reason = lines[-1] if lines else "unknown error"
        reason = reason.removeprefix(f"{path}: ")  # ffprobe starts its message with the path
        raise MediaError(f"ffprobe could not read this file ({reason})")
    try:
        return parse_probe(path, json.loads(result.stdout))
    except json.JSONDecodeError as error:
        raise MediaError(f"ffprobe returned something unreadable ({error})") from error


# ---------------------------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------------------------


def choose_audio_stream(info: MediaInfo, *, track: int | None = None, language: Language | None = None) -> int:
    """The position (among audio streams, counting from 0) of the track to transcribe.

    An explicit 1-based `track` wins. Otherwise a track tagged with the spoken language, then
    the file's default track, then the first one.
    """
    audio = info.audio
    if not audio:
        raise MediaError("there is no audio track to transcribe")
    if track is not None:
        if not 1 <= track <= len(audio):
            raise MediaError(f"audio track {track} does not exist (the file has {len(audio)})")
        return track - 1
    if language is not None:
        for position, stream in enumerate(audio):
            tagged = find_language(stream.language) if stream.language else None
            if tagged is not None and tagged.whisper == language.whisper:
                return position
    for position, stream in enumerate(audio):
        if stream.default:
            return position
    return 0


def extract_audio(path: Path, position: int, destination: Path) -> Path:
    """Write one audio track as 16 kHz mono WAV, the form speech recognition wants."""
    ffmpeg = require_tool("ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path), "-map", f"0:a:{position}",
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    if result.returncode != 0:
        raise MediaError(f"ffmpeg could not extract the audio ({result.stderr.strip()[-300:]})")
    return destination


def sidecar_path(media: Path, language: Language, extension: str = "srt") -> Path:
    """Where a subtitle file for `media` goes: movie.es.srt, the name players look for."""
    return media.with_name(f"{media.stem}.{language.code}.{extension}")


def find_sidecars(media: Path, extension: str = "srt") -> list[Path]:
    """Every "<name>.<language>.<extension>" next to `media`, whatever the language.

    Folder listings are compared by name rather than matched with a glob, because release
    names are full of characters a glob treats as patterns: "[Group] Show - 01 [1080p].mkv".
    """
    prefix, suffix = f"{media.stem}.", f".{extension}"
    found = []
    try:
        entries = list(media.parent.iterdir())
    except OSError:
        return []
    for entry in entries:
        name = entry.name
        is_candidate = name.startswith(prefix) and name.endswith(suffix) and len(name) > len(prefix) + len(suffix)
        if is_candidate and find_language(name[len(prefix) : -len(suffix)]) is not None:
            found.append(entry)
    return sorted(found)
