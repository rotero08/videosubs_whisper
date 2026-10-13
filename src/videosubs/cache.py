"""A small on-disk cache of transcripts, so a failed or repeated run does not redo the slow part.

The key covers the file (path, size, modification time) and every setting that changes what
the recogniser hears, so editing the file or switching model never serves a stale transcript.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from .transcribe import Transcript


def default_cache_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "videosubs"


class TranscriptCache:
    def __init__(self, directory: Path | None = None, *, enabled: bool = True) -> None:
        self.directory = directory or default_cache_dir()
        self.enabled = enabled

    def key(self, media: Path, **settings: object) -> str:
        stat = media.stat()
        parts = [str(media.resolve()), str(stat.st_size), str(stat.st_mtime_ns)]
        parts += [f"{k}={settings[k]}" for k in sorted(settings)]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]

    def _file(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Transcript | None:
        if not self.enabled:
            return None
        try:
            return Transcript.from_dict(json.loads(self._file(key).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, key: str, transcript: Transcript) -> None:
        if not self.enabled:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            partial = self._file(key).with_suffix(".tmp")
            partial.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False), encoding="utf-8")
            partial.replace(self._file(key))
        except OSError:
            pass  # a cache that cannot be written is only a missed optimisation

    def clear(self) -> int:
        removed = 0
        if self.directory.is_dir():
            for file in self.directory.glob("*.json"):
                file.unlink(missing_ok=True)
                removed += 1
        return removed
