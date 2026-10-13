"""Putting subtitle files into a video.

Everything in the original file is kept (video, every audio track, existing subtitles, fonts,
chapters, metadata): only subtitle tracks are added. The result is written to a temporary
name, checked against the original, and only then moved into place. The source video is never
modified, and is replaced only when asked, and even then it is moved aside rather than deleted.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import MuxError
from .languages import Language
from .media import BACKUP_DIR, MediaInfo, probe, require_tool


@dataclass(frozen=True)
class SubtitleTrack:
    path: Path
    language: Language


def output_path(video: Path) -> Path:
    """movie.mp4 -> movie.subs.mkv, next to the original."""
    return video.with_name(f"{video.stem}.subs.mkv")


def build_command(
    ffmpeg: str,
    video: Path,
    info: MediaInfo,
    tracks: Sequence[SubtitleTrack],
    destination: Path,
    *,
    default_language: Language | None = None,
) -> list[str]:
    """The ffmpeg command line that adds `tracks` to `video` and writes Matroska to `destination`."""
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(video),
    ]
    for track in tracks:
        command += ["-i", str(track.path)]

    # Keep everything from the original; optional maps ("?") so a file without, say, attachments is fine.
    command += ["-map", "0:v?", "-map", "0:a?", "-map", "0:s?", "-map", "0:t?"]
    for i in range(len(tracks)):
        command += ["-map", f"{i + 1}:0"]
    command += ["-map_chapters", "0", "-map_metadata", "0", "-c", "copy"]

    existing = info.subtitles
    # Subtitle tracks that cannot be copied into Matroska (MP4's mov_text) are converted to SRT.
    for position, stream in enumerate(existing):
        if stream.codec == "mov_text":
            command += [f"-c:s:{position}", "srt"]
    for offset, track in enumerate(tracks):
        position = len(existing) + offset
        command += [
            f"-c:s:{position}", "srt",
            f"-metadata:s:s:{position}", f"language={track.language.iso2}",
            f"-metadata:s:s:{position}", f"title={track.language.name}",
        ]  # fmt: skip
        if default_language is not None:
            wanted = track.language.code == default_language.code
            command += [f"-disposition:s:{position}", "default" if wanted else "0"]
    command += ["-f", "matroska", str(destination)]
    return command


def _run(command: list[str], duration: float, on_progress: Callable[[float], None] | None) -> None:
    """Run ffmpeg, reporting progress from its machine-readable output."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        key, _, value = line.strip().partition("=")
        if key in ("out_time_us", "out_time_ms") and on_progress and duration > 0 and value.isdigit():
            on_progress(min(1.0, int(value) / 1_000_000 / duration))
    _, errors = process.communicate()
    if process.returncode != 0:
        raise MuxError(f"ffmpeg failed: {errors.strip()[-500:] or 'no message'}")
    if on_progress:
        on_progress(1.0)


def verify(original: MediaInfo, result: MediaInfo, added: int) -> None:
    """Refuse an output that lost anything: streams, or a noticeable amount of time."""
    problems = []
    for kind, expected in (("video", len(original.of_kind("video"))), ("audio", len(original.audio)),
                           ("subtitle", len(original.subtitles) + added)):  # fmt: skip
        got = len(result.of_kind(kind))
        if got != expected:
            problems.append(f"{kind}: expected {expected} stream(s), found {got}")
    if original.duration and abs(result.duration - original.duration) > max(2.0, original.duration * 0.01):
        problems.append(f"duration changed from {original.duration:.1f}s to {result.duration:.1f}s")
    if problems:
        raise MuxError("the new file does not match the original (" + "; ".join(problems) + ")")


def mux(
    video: Path,
    info: MediaInfo,
    tracks: Sequence[SubtitleTrack],
    destination: Path,
    *,
    default_language: Language | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Write `destination`: the video with the tracks added. Raises MuxError, leaving no partial file."""
    if not tracks:
        raise MuxError("there are no subtitle tracks to add")
    ffmpeg = require_tool("ffmpeg")
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    try:
        _run(
            build_command(ffmpeg, video, info, tracks, partial, default_language=default_language),
            info.duration,
            on_progress,
        )
        verify(info, probe(partial), len(tracks))
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def replace_original(original: Path, muxed: Path, *, keep_backup: bool = True) -> Path:
    """Put the muxed file where the original was. The original moves to a backup folder next to
    it (or is deleted, when `keep_backup` is False). Returns the final path (always .mkv)."""
    final = original.with_suffix(".mkv")
    if final.exists() and final.resolve() != original.resolve():
        raise MuxError(f"{final.name} already exists, so I will not replace {original.name} with it")
    if keep_backup:
        folder = original.parent / BACKUP_DIR
        folder.mkdir(exist_ok=True)
        target = folder / original.name
        if target.exists():
            target = folder / f"{original.stem}.{int(original.stat().st_mtime)}{original.suffix}"
        shutil.move(str(original), target)
    else:
        original.unlink()
    muxed.replace(final)
    return final
