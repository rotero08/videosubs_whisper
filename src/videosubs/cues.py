"""Subtitle cues: turning recognised speech into readable, well-timed subtitles, and the file
formats around them.

Speech recognition returns sentences with a start and end, and (when asked) a time for every
word. A good subtitle is not that: it is at most two short lines, it stays on screen long
enough to read, it breaks at natural places, and it never overlaps the next one. This module
does that reshaping. It works on plain data and has no I/O, so every rule can be tested.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

SENTENCE_END = ".!?…。！？‼⁇⁈⁉"
CLAUSE_END = ",;:、，；：—–"
_CLOSERS = "\"'”’)]」』）»"


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str  # as recognised, usually with a leading space


@dataclass(frozen=True)
class Segment:
    """A stretch of speech, usually a sentence, as returned by the recogniser."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str  # lines separated by "\n"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Style:
    """How subtitles are laid out."""

    max_chars: int = 42  # per line, for languages written with spaces
    max_chars_wide: int = 22  # per line, for Japanese, Chinese, Thai
    max_lines: int = 2
    max_seconds: float = 7.0  # longest a cue may stay on screen
    min_seconds: float = 1.0  # shortest, when the next cue leaves room
    gap: float = 0.04  # least silence between two cues
    pause: float = 1.0  # a silence this long always starts a new cue
    sentence_min: int = 14  # characters before a sentence end is allowed to end a cue

    def line_width(self, spaced: bool) -> int:
        return self.max_chars if spaced else self.max_chars_wide

    def capacity(self, spaced: bool) -> int:
        return self.line_width(spaced) * self.max_lines


DEFAULT_STYLE = Style()


# ---------------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------------


def clean(text: str, spaced: bool = True) -> str:
    """Collapse whitespace; for unspaced languages also drop spaces between two wide characters."""
    text = re.sub(r"\s+", " ", text).strip()
    if not spaced:
        text = re.sub(r"(?<=[^\x00-\x7f]) (?=[^\x00-\x7f])", "", text)
    return text


def join_words(words: Sequence[Word], spaced: bool = True) -> str:
    return clean("".join(word.text for word in words), spaced)


def _strip_closers(text: str) -> str:
    return text.rstrip().rstrip(_CLOSERS)


def ends_sentence(text: str) -> bool:
    stripped = _strip_closers(text)
    return bool(stripped) and stripped[-1] in SENTENCE_END


def ends_clause(text: str) -> bool:
    stripped = _strip_closers(text)
    return bool(stripped) and (stripped[-1] in CLAUSE_END or stripped[-1] in SENTENCE_END)


def _balanced_split(words: list[str], lines: int, width: int, sep: str = " ") -> list[str] | None:
    """Split words into exactly `lines` lines no wider than `width`, as even as possible."""
    n = len(words)
    if lines > n:
        return None
    lengths = [len(w) for w in words]
    gap = len(sep)
    total = sum(lengths) + (n - 1) * gap
    ideal = total / lines
    inf = float("inf")

    def line_len(i: int, j: int) -> int:
        return sum(lengths[i:j]) + (j - i - 1) * gap

    # best[k][j]: lowest cost of putting the first j words on k lines.
    best = [[inf] * (n + 1) for _ in range(lines + 1)]
    back = [[0] * (n + 1) for _ in range(lines + 1)]
    best[0][0] = 0.0
    for k in range(1, lines + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                length = line_len(i, j)
                if length > width or best[k - 1][i] == inf:
                    continue
                cost = best[k - 1][i] + (length - ideal) ** 2
                if cost < best[k][j]:
                    best[k][j], back[k][j] = cost, i
    if best[lines][n] == inf:
        return None
    result, j = [], n
    for k in range(lines, 0, -1):
        i = back[k][j]
        result.append(sep.join(words[i:j]))
        j = i
    return result[::-1]


def _greedy_wrap(words: list[str], width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _wrap_wide(text: str, width: int, max_lines: int) -> list[str]:
    if len(text) <= width:
        return [text]
    count = max(math.ceil(len(text) / width), 1)
    size = math.ceil(len(text) / count)
    lines: list[str] = []
    i = 0
    while i < len(text):
        end = min(len(text), i + size)
        if end < len(text):
            # Prefer to break just after punctuation close to the planned point: a little before
            # it, or a little after it as long as the line still fits.
            for j in range(min(len(text), i + width), max(i + 1, end - 4), -1):
                if text[j - 1] in CLAUSE_END or text[j - 1] in SENTENCE_END:
                    end = j
                    break
        lines.append(text[i:end])
        i = end
    return lines


def wrap(text: str, width: int, max_lines: int = 2, spaced: bool = True) -> list[str]:
    """Break text into lines of at most `width` characters, balanced so they look even."""
    if len(text) <= width:
        return [text]
    if not spaced:
        return _wrap_wide(text, width, max_lines)
    words = text.split(" ")
    need = math.ceil(len(text) / width)
    for lines in range(need, max(need, max_lines) + 1):
        balanced = _balanced_split(words, lines, width)
        if balanced:
            return balanced
    return _greedy_wrap(words, width)


# ---------------------------------------------------------------------------------------------
# Building cues
# ---------------------------------------------------------------------------------------------


def _best_break(buffer: list[Word]) -> int:
    """Where to cut a buffer that has grown too long: after the last clause end in its back half."""
    for i in range(len(buffer) - 1, math.ceil(len(buffer) * 0.4) - 1, -1):
        if ends_clause(buffer[i].text):
            return i + 1
    return len(buffer)


def cues_from_words(words: Iterable[Word], style: Style = DEFAULT_STYLE, spaced: bool = True) -> list[Cue]:
    """Group timed words into cues that read well and keep to the style's limits."""
    width = style.line_width(spaced)
    capacity = style.capacity(spaced)
    cues: list[Cue] = []
    buffer: list[Word] = []

    def emit(chunk: list[Word]) -> None:
        text = join_words(chunk, spaced)
        if text:
            cues.append(Cue(chunk[0].start, chunk[-1].end, "\n".join(wrap(text, width, style.max_lines, spaced))))

    def too_big(chunk: list[Word], word: Word) -> bool:
        return len(join_words([*chunk, word], spaced)) > capacity or word.end - chunk[0].start > style.max_seconds

    for word in words:
        if not word.text.strip():
            continue
        while buffer:
            if word.start - buffer[-1].end > style.pause:
                emit(buffer)
                buffer = []
            elif too_big(buffer, word):
                cut = _best_break(buffer)
                emit(buffer[:cut])
                buffer = buffer[cut:]
            elif ends_sentence(buffer[-1].text) and len(join_words(buffer, spaced)) >= style.sentence_min:
                emit(buffer)
                buffer = []
            else:
                break
        buffer.append(word)
    if buffer:
        emit(buffer)
    return cues


_SENTENCES = re.compile(r".+?(?:[.!?…]+[\"'”’)\]]*(?=\s|$)|$)", re.S)
_SENTENCES_WIDE = re.compile(r".+?(?:[。！？!?…]+[」』）)]*|$)", re.S)


def split_sentences(text: str, spaced: bool = True) -> list[str]:
    pattern = _SENTENCES if spaced else _SENTENCES_WIDE
    return [part.strip() for part in pattern.findall(text) if part.strip()]


_CLAUSES = re.compile(r".+?(?:[,;:、，；：]+\s*|$)", re.S)


def _split_long(text: str, capacity: int, spaced: bool) -> list[str]:
    """Cut a sentence longer than one cue can hold into balanced pieces at its clause breaks, and
    cut any clause that is still too long by words."""
    sep = " " if spaced else ""
    clauses = [c.strip() for c in _CLAUSES.findall(text) if c.strip()]
    pieces: list[str] = []
    needed = max(1, math.ceil(len(clean(text, spaced)) / capacity))
    for count in range(needed, len(clauses) + 1):
        balanced = _balanced_split(clauses, count, capacity, sep)
        if balanced:
            pieces = balanced
            break
    else:
        pieces = clauses or [clean(text, spaced)]

    final: list[str] = []
    for piece in pieces:
        if len(piece) <= capacity:
            final.append(piece)
        elif spaced:
            final.extend(_greedy_wrap(piece.split(" "), capacity))
        else:
            final.extend(piece[i : i + capacity] for i in range(0, len(piece), capacity))
    return final


def cues_from_text(text: str, start: float, end: float, style: Style = DEFAULT_STYLE, spaced: bool = True) -> list[Cue]:
    """Cues for text that has no word timings (a translation): split it at sentence and clause
    breaks, and share the time between the pieces in proportion to their length."""
    text = clean(text, spaced)
    if not text or end <= start:
        return []
    capacity = style.capacity(spaced)
    width = style.line_width(spaced)

    pieces: list[str] = []
    for sentence in split_sentences(text, spaced):
        parts = [sentence] if len(sentence) <= capacity else _split_long(sentence, capacity, spaced)
        for part in parts:
            # A tiny fragment ("Yes.") reads better with its neighbour than alone.
            if pieces and len(part) <= 8 and len(clean(f"{pieces[-1]} {part}", spaced)) <= capacity:
                pieces[-1] = clean(f"{pieces[-1]} {part}", spaced)
            else:
                pieces.append(part)

    total = sum(len(p) for p in pieces) or 1
    cues: list[Cue] = []
    elapsed = 0
    for piece in pieces:
        t0 = start + (end - start) * elapsed / total
        elapsed += len(piece)
        t1 = start + (end - start) * elapsed / total
        cues.append(Cue(t0, t1, "\n".join(wrap(piece, width, style.max_lines, spaced))))
    return cues


def tidy(cues: Iterable[Cue], style: Style = DEFAULT_STYLE) -> list[Cue]:
    """Final clean-up over a whole file: order, no overlaps, no empty or repeated cues, and
    short cues given a little more time when the next cue leaves room for it."""
    ordered = sorted((c for c in cues if c.text.strip()), key=lambda c: (c.start, c.end))
    result: list[Cue] = []
    for cue in ordered:
        if cue.end <= cue.start:
            cue = replace(cue, end=cue.start + 0.5)
        if result:
            previous = result[-1]
            if cue.text == previous.text and cue.start - previous.end < 0.5:
                result[-1] = replace(previous, end=max(previous.end, cue.end))  # an echoed line
                continue
            if cue.start < previous.end + style.gap:
                # Overlap: trim the earlier cue, but never to nothing.
                trimmed = max(previous.start + 0.2, cue.start - style.gap)
                result[-1] = replace(previous, end=min(previous.end, trimmed))
                if cue.start < result[-1].end + style.gap:
                    cue = replace(
                        cue, start=result[-1].end + style.gap, end=max(cue.end, result[-1].end + style.gap + 0.2)
                    )
        result.append(cue)

    for i, cue in enumerate(result):
        if cue.duration < style.min_seconds:
            room = (result[i + 1].start - style.gap) if i + 1 < len(result) else cue.start + style.min_seconds
            result[i] = replace(cue, end=max(cue.end, min(cue.start + style.min_seconds, room)))
    return result


def segments_to_cues(segments: Iterable[Segment], style: Style = DEFAULT_STYLE, spaced: bool = True) -> list[Cue]:
    """Cues for recognised speech: word by word when the words are timed, else by text."""
    cues: list[Cue] = []
    for segment in segments:
        if segment.words:
            cues.extend(cues_from_words(segment.words, style, spaced))
        else:
            cues.extend(cues_from_text(segment.text, segment.start, segment.end, style, spaced))
    return tidy(cues, style)


def translated_to_cues(
    segments: Sequence[Segment], texts: Sequence[str], style: Style = DEFAULT_STYLE, spaced: bool = True
) -> list[Cue]:
    """Cues for translated text: each segment's translation is laid over that segment's time span."""
    cues: list[Cue] = []
    for segment, text in zip(segments, texts, strict=True):
        cues.extend(cues_from_text(text, segment.start, segment.end, style, spaced))
    return tidy(cues, style)


# ---------------------------------------------------------------------------------------------
# Sentences and typography, for translation
# ---------------------------------------------------------------------------------------------

# A full stop after these does not end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "sra.", "srta.", "jr.",
        "vs.", "etc.", "e.g.", "i.e.", "st.", "no.", "nr.", "mme.", "mlle.", "herr.", "frau.",
    }
)  # fmt: skip


def _ends_real_sentence(word: str) -> bool:
    return ends_sentence(word) and word.strip().lower() not in _ABBREVIATIONS


def sentence_units(segments: Iterable[Segment], spaced: bool = True) -> list[Segment]:
    """Cut segments into single sentences, each with its own true time span.

    Recognisers often return two sentences as one segment, and a translator does best with one
    sentence at a time. The word timings say exactly where each sentence begins and ends, so a
    translation can be laid over the right stretch of time. Segments without word timings are cut
    at sentence punctuation and the time shared by length, as a fallback.
    """
    units: list[Segment] = []
    for segment in segments:
        if segment.words:
            current: list[Word] = []
            for word in segment.words:
                if not word.text.strip():
                    continue
                current.append(word)
                if _ends_real_sentence(word.text):
                    units.append(_unit(current, spaced))
                    current = []
            if current:
                units.append(_unit(current, spaced))
        else:
            sentences = split_sentences(segment.text, spaced) or [segment.text]
            total = sum(len(x) for x in sentences) or 1
            elapsed = 0
            for sentence in sentences:
                begin = segment.start + (segment.end - segment.start) * elapsed / total
                elapsed += len(sentence)
                finish = segment.start + (segment.end - segment.start) * elapsed / total
                units.append(Segment(begin, finish, sentence))
    return [unit for unit in units if unit.text]


def _unit(words: list[Word], spaced: bool) -> Segment:
    return Segment(words[0].start, words[-1].end, join_words(words, spaced), tuple(words))


_FULLWIDTH = {
    "ja": {".": "。", ",": "、", "?": "？", "!": "！", ":": "：", ";": "；"},
    "zh": {".": "。", ",": "，", "?": "？", "!": "！", ":": "：", ";": "；"},
}
_ASCII_AFTER_WIDE = re.compile(r"(?<=[^\x00-\x7f])\s*([.,?!:;])\s*")


def localise_punctuation(text: str, language: str) -> str:
    """Use Japanese or Chinese punctuation, and no spaces around it, in translated text.

    Translators trained mostly on spaced languages tend to leave "ようこそ. 今日," behind. Only
    punctuation that follows a wide character is converted, so "Python 3.5" keeps its dot.
    """
    table = _FULLWIDTH.get(language.split("-")[0])
    if table is None:
        return text
    converted = _ASCII_AFTER_WIDE.sub(lambda m: table[m.group(1)], text)
    return clean(converted, spaced=False)


# ---------------------------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------------------------


def _stamp(seconds: float, separator: str) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def format_srt(cues: Iterable[Cue]) -> str:
    blocks = [
        f"{i}\n{_stamp(cue.start, ',')} --> {_stamp(cue.end, ',')}\n{cue.text}\n" for i, cue in enumerate(cues, 1)
    ]
    return "\n".join(blocks)


def format_vtt(cues: Iterable[Cue]) -> str:
    blocks = [f"{_stamp(cue.start, '.')} --> {_stamp(cue.end, '.')}\n{cue.text}\n" for cue in cues]
    return "WEBVTT\n\n" + "\n".join(blocks)


def format_txt(cues: Iterable[Cue]) -> str:
    """The words only, one cue per line, for reading or searching."""
    return "".join(f"{' '.join(cue.text.splitlines())}\n" for cue in cues)


FORMATTERS = {"srt": format_srt, "vtt": format_vtt, "txt": format_txt}

_TIMING = re.compile(
    r"(?P<a>\d+:\d{2}:\d{2}[,.]\d{1,3}|\d{1,2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<b>\d+:\d{2}:\d{2}[,.]\d{1,3}|\d{1,2}:\d{2}[,.]\d{1,3})"
)


def _seconds(stamp: str) -> float:
    stamp = stamp.replace(",", ".")
    parts = stamp.split(":")
    hours = float(parts[-3]) if len(parts) == 3 else 0.0
    return hours * 3600 + float(parts[-2]) * 60 + float(parts[-1])


def parse_subtitles(text: str) -> list[Cue]:
    """Read SRT or WebVTT. Tolerant: a byte-order mark, Windows line ends, missing cue numbers
    and a header block are all fine; blocks without a timing line are ignored."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            match = _TIMING.search(line)
            if match:
                body = "\n".join(lines[i + 1 :]).strip()
                if body:
                    cues.append(Cue(_seconds(match["a"]), _seconds(match["b"]), body))
                break
    return cues
