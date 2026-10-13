"""Translation through the DeepL API (https://www.deepl.com/pro-api)."""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import TranslationError
from ..languages import Language
from . import ProgressCallback

#: Languages DeepL accepts as a *source* (its target codes also carry regions, e.g. EN-US).
_SOURCES = frozenset(
    [
        "AR",
        "BG",
        "CS",
        "DA",
        "DE",
        "EL",
        "EN",
        "ES",
        "ET",
        "FI",
        "FR",
        "HU",
        "ID",
        "IT",
        "JA",
        "KO",
        "LT",
        "LV",
        "NB",
        "NL",
        "PL",
        "PT",
        "RO",
        "RU",
        "SK",
        "SL",
        "SV",
        "TR",
        "UK",
        "ZH",
    ]
)
_BATCH = 50
_FORMALITY = {"default": "default", "more": "prefer_more", "less": "prefer_less"}


def source_code(language: Language) -> str | None:
    """DeepL's source code for a language, or None to let DeepL detect it."""
    code = language.whisper.upper()
    code = "NB" if code == "NO" else code
    return code if code in _SOURCES else None


class DeepLTranslator:
    name = "deepl"

    def __init__(self, auth_key: str, *, formality: str = "default", client=None) -> None:
        if formality not in _FORMALITY:
            raise TranslationError(f"unknown formality {formality!r}: use default, more or less")
        self.formality = _FORMALITY[formality]
        if client is None:
            try:
                import deepl
            except ImportError as error:
                raise TranslationError(
                    "DeepL support is not installed: run `pip install 'videosubs[deepl]'`"
                ) from error
            client = deepl.Translator(auth_key)
        self.client = client

    def translate(
        self, texts: Sequence[str], source: Language, target: Language, progress: ProgressCallback | None = None
    ) -> list[str]:
        if not target.deepl:
            raise TranslationError(f"DeepL does not offer {target.name}. Try --translator nllb.")
        results: list[str] = []
        for start in range(0, len(texts), _BATCH):
            batch = list(texts[start : start + _BATCH])
            try:
                response = self.client.translate_text(
                    batch,
                    source_lang=source_code(source),
                    target_lang=target.deepl,
                    formality=self.formality,
                    preserve_formatting=True,
                )
            except Exception as error:
                raise self._explain(error) from error
            results.extend(item.text for item in (response if isinstance(response, list) else [response]))
            if progress:
                progress(min(1.0, (start + len(batch)) / max(1, len(texts))))
        if len(results) != len(texts):
            raise TranslationError("DeepL returned a different number of translations than were sent")
        return results

    @staticmethod
    def _explain(error: Exception) -> TranslationError:
        kind = type(error).__name__
        if kind == "AuthorizationException":
            return TranslationError("DeepL rejected the API key. Check DEEPL_API_KEY.")
        if kind == "QuotaExceededException":
            return TranslationError("The DeepL quota for this key is used up.")
        if kind == "TooManyRequestsException":
            return TranslationError("DeepL says there are too many requests right now; try again in a minute.")
        return TranslationError(f"DeepL could not translate ({kind}: {error})")
