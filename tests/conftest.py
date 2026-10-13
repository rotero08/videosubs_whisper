"""Shared fixtures: small real media files made with ffmpeg, and fakes for the slow parts."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from videosubs.cues import Segment, Word
from videosubs.languages import Language
from videosubs.transcribe import Transcript

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def pytest_collection_modifyitems(config, items):
    if HAS_FFMPEG:
        return
    skip = pytest.mark.skip(reason="ffmpeg and ffprobe are not installed")
    for item in items:
        if "needs_ffmpeg" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def make_media(tmp_path):
    """make_media("movie.mkv", audio=("eng", "jpn"), seconds=3) -> Path to a real tiny file."""

    def make(name="movie.mkv", *, audio=("eng",), seconds=3, subtitles=(), video=True) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
        inputs = 0
        maps = []
        if video:
            command += ["-f", "lavfi", "-i", f"color=c=black:s=64x64:r=10:d={seconds}"]
            maps += ["-map", f"{inputs}:v"]
            inputs += 1
        for i, _ in enumerate(audio):
            command += ["-f", "lavfi", "-i", f"sine=frequency={440 + 110 * i}:duration={seconds}"]
            maps += ["-map", f"{inputs}:a"]
            inputs += 1
        for i, language in enumerate(audio):
            maps += [f"-metadata:s:a:{i}", f"language={language}"]
        srt_files = []
        for i, language in enumerate(subtitles):
            srt = tmp_path / f"_embedded_{i}.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            srt_files.append(srt)
            command += ["-i", str(srt)]
            maps += ["-map", f"{inputs}:0", f"-metadata:s:s:{i}", f"language={language}"]
            inputs += 1
        command += [*maps, "-c:v", "mpeg4", "-c:a", "pcm_s16le", "-c:s", "srt"]
        if not video:
            command += ["-vn"]
        command += [str(path)]
        subprocess.run(command, check=True, capture_output=True)
        return path

    return make


def probe_streams(path: Path) -> dict[str, list[str | None]]:
    """{"video": [...], "audio": [languages], "subtitle": [languages]} straight from ffprobe."""
    import json

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, list[str | None]] = {"video": [], "audio": [], "subtitle": []}
    for stream in json.loads(out)["streams"]:
        result.setdefault(stream["codec_type"], []).append((stream.get("tags") or {}).get("language"))
    return result


def sentence(start: float, text: str, per_word: float = 0.3) -> Segment:
    """A segment with word timings, built from text."""
    t = start
    words = []
    for token in text.split():
        words.append(Word(t, t + per_word, f" {token}"))
        t += per_word
    return Segment(start, t, text, tuple(words))


class FakeTranscriber:
    """Returns a canned transcript and records how it was called."""

    def __init__(self, language="en", segments=None, probability=0.95, fail_on=()):
        self.language = language
        self.segments = (
            segments
            if segments is not None
            else [sentence(0.2, "Hello and welcome."), sentence(2.0, "This is a short test.")]
        )
        self.probability = probability
        self.fail_on = set(fail_on)
        self.calls: list[dict] = []

    def transcribe(self, audio, *, language, task="transcribe", progress=None):
        self.calls.append({"audio": Path(audio), "language": language, "task": task})
        if Path(audio).name in self.fail_on:
            raise RuntimeError("the model exploded")
        if progress:
            progress(0.5)
            progress(1.0)
        return Transcript(language or self.language, self.probability, 3.0, list(self.segments), task)


class FakeTranslator:
    """Translates "x" into "[es] x" so tests can see what happened to each text."""

    name = "fake"

    def __init__(self):
        self.calls: list[tuple[list[str], str, str]] = []

    def translate(self, texts, source: Language, target: Language, progress=None):
        self.calls.append((list(texts), source.code, target.code))
        if progress:
            progress(1.0)
        return [f"[{target.code}] {text}" for text in texts]


@pytest.fixture
def fake_transcriber():
    return FakeTranscriber()


@pytest.fixture
def fake_translator():
    return FakeTranslator()
