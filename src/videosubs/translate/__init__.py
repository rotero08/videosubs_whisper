"""Translating subtitle text into other languages.

Three ways, chosen with --translator:

  whisper  Whisper's own "translate" task. Free and built in, but English is the only target.
  nllb     Offline neural translation (Meta's NLLB-200 through CTranslate2). Any of ~200
           languages, no key, no network after the one-time model download. The model is
           licensed for non-commercial use (CC-BY-NC-4.0).
  deepl    The DeepL API. The best quality for the languages it covers; needs your own key.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..errors import TranslationError
from ..languages import Language

ProgressCallback = Callable[[float], None]
BACKENDS = ("auto", "whisper", "nllb", "deepl")


class Translator(Protocol):
    name: str

    def translate(
        self, texts: Sequence[str], source: Language, target: Language, progress: ProgressCallback | None = None
    ) -> list[str]:
        """One translation per input text, in the same order."""
        ...


@dataclass
class TranslationSettings:
    backend: str = "auto"
    deepl_key: str | None = None
    formality: str = "default"  # DeepL: default, more or less
    nllb_model: str = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"
    device: str = "auto"
    compute_type: str = "auto"
    download_root: str | None = None

    def resolved_deepl_key(self) -> str | None:
        return self.deepl_key or os.environ.get("DEEPL_API_KEY") or os.environ.get("DEEPL_AUTH_KEY")


def choose_backend(settings: TranslationSettings, target: Language) -> str:
    """Resolve "auto": DeepL when a key is available and it offers the language, else NLLB."""
    if settings.backend != "auto":
        return settings.backend
    if settings.resolved_deepl_key() and target.deepl:
        return "deepl"
    return "nllb"


def create_translator(backend: str, settings: TranslationSettings) -> Translator:
    if backend == "deepl":
        from .deepl_backend import DeepLTranslator

        key = settings.resolved_deepl_key()
        if not key:
            raise TranslationError(
                "DeepL needs an API key: set the DEEPL_API_KEY environment variable "
                "(a free key from https://www.deepl.com/pro-api works)."
            )
        return DeepLTranslator(key, formality=settings.formality)
    if backend == "nllb":
        from .nllb import NllbTranslator

        return NllbTranslator(
            settings.nllb_model,
            device=settings.device,
            compute_type=settings.compute_type,
            download_root=settings.download_root,
        )
    raise TranslationError(f"unknown translation backend {backend!r}: use one of {', '.join(BACKENDS[1:])}")
