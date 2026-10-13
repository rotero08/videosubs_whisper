"""The languages the tool knows, and how each one is spelled in the places that matter.

One row per language: the ISO 639-1 code Whisper uses, the ISO 639-2 code a Matroska file
stores, the code NLLB (the offline translator) expects and the one DeepL expects.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from .errors import UnknownLanguageError


@dataclass(frozen=True)
class Language:
    code: str  # ISO 639-1 ("es"); a script is added only where it matters ("zh-Hant")
    name: str  # English name
    native: str
    iso2: str  # ISO 639-2/B, as stored in Matroska
    nllb: str  # FLORES-200 code used by NLLB
    deepl: str | None  # DeepL target code, None if DeepL does not offer it
    spaced: bool = True  # words are separated by spaces (False for Japanese, Chinese, Thai)

    @property
    def whisper(self) -> str:
        """The code Whisper uses; it has no script variants."""
        return self.code.split("-")[0]


# fmt: off
LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "English", "eng", "eng_Latn", "EN-US"),
    Language("es", "Spanish", "Español", "spa", "spa_Latn", "ES"),
    Language("fr", "French", "Français", "fre", "fra_Latn", "FR"),
    Language("de", "German", "Deutsch", "ger", "deu_Latn", "DE"),
    Language("it", "Italian", "Italiano", "ita", "ita_Latn", "IT"),
    Language("pt", "Portuguese", "Português", "por", "por_Latn", "PT-BR"),
    Language("ru", "Russian", "Русский", "rus", "rus_Cyrl", "RU"),
    Language("ja", "Japanese", "日本語", "jpn", "jpn_Jpan", "JA", spaced=False),
    Language("ko", "Korean", "한국어", "kor", "kor_Hang", "KO"),
    Language("zh", "Chinese", "中文", "chi", "zho_Hans", "ZH", spaced=False),
    Language("zh-Hant", "Chinese (Traditional)", "繁體中文", "chi", "zho_Hant", "ZH-HANT", spaced=False),
    Language("ar", "Arabic", "العربية", "ara", "arb_Arab", "AR"),
    Language("hi", "Hindi", "हिन्दी", "hin", "hin_Deva", None),
    Language("tr", "Turkish", "Türkçe", "tur", "tur_Latn", "TR"),
    Language("pl", "Polish", "Polski", "pol", "pol_Latn", "PL"),
    Language("nl", "Dutch", "Nederlands", "dut", "nld_Latn", "NL"),
    Language("sv", "Swedish", "Svenska", "swe", "swe_Latn", "SV"),
    Language("da", "Danish", "Dansk", "dan", "dan_Latn", "DA"),
    Language("no", "Norwegian", "Norsk", "nor", "nob_Latn", "NB"),
    Language("fi", "Finnish", "Suomi", "fin", "fin_Latn", "FI"),
    Language("cs", "Czech", "Čeština", "cze", "ces_Latn", "CS"),
    Language("el", "Greek", "Ελληνικά", "gre", "ell_Grek", "EL"),
    Language("he", "Hebrew", "עברית", "heb", "heb_Hebr", None),
    Language("hu", "Hungarian", "Magyar", "hun", "hun_Latn", "HU"),
    Language("ro", "Romanian", "Română", "rum", "ron_Latn", "RO"),
    Language("uk", "Ukrainian", "Українська", "ukr", "ukr_Cyrl", "UK"),
    Language("vi", "Vietnamese", "Tiếng Việt", "vie", "vie_Latn", None),
    Language("th", "Thai", "ไทย", "tha", "tha_Thai", None, spaced=False),
    Language("id", "Indonesian", "Bahasa Indonesia", "ind", "ind_Latn", "ID"),
    Language("ms", "Malay", "Bahasa Melayu", "may", "zsm_Latn", None),
    Language("bg", "Bulgarian", "Български", "bul", "bul_Cyrl", "BG"),
    Language("hr", "Croatian", "Hrvatski", "hrv", "hrv_Latn", None),
    Language("sk", "Slovak", "Slovenčina", "slo", "slk_Latn", "SK"),
    Language("sl", "Slovenian", "Slovenščina", "slv", "slv_Latn", "SL"),
    Language("lt", "Lithuanian", "Lietuvių", "lit", "lit_Latn", "LT"),
    Language("lv", "Latvian", "Latviešu", "lav", "lvs_Latn", "LV"),
    Language("et", "Estonian", "Eesti", "est", "est_Latn", "ET"),
    Language("ca", "Catalan", "Català", "cat", "cat_Latn", None),
    Language("fa", "Persian", "فارسی", "per", "pes_Arab", None),
    Language("bn", "Bengali", "বাংলা", "ben", "ben_Beng", None),
    Language("ta", "Tamil", "தமிழ்", "tam", "tam_Taml", None),
    Language("ur", "Urdu", "اردو", "urd", "urd_Arab", None),
    Language("sr", "Serbian", "Српски", "srp", "srp_Cyrl", None),
    Language("is", "Icelandic", "Íslenska", "ice", "isl_Latn", None),
    Language("af", "Afrikaans", "Afrikaans", "afr", "afr_Latn", None),
    Language("sw", "Swahili", "Kiswahili", "swa", "swh_Latn", None),
    Language("tl", "Tagalog", "Tagalog", "tgl", "tgl_Latn", None),
    Language("gl", "Galician", "Galego", "glg", "glg_Latn", None),
    Language("eu", "Basque", "Euskara", "baq", "eus_Latn", None),
    Language("cy", "Welsh", "Cymraeg", "wel", "cym_Latn", None),
)
# fmt: on

_BY_CODE = {language.code.lower(): language for language in LANGUAGES}

# Other ways people write the same languages.
_ALIASES = {
    # ISO 639-2/T and other three-letter forms
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "fre": "fr",
    "deu": "de",
    "ger": "de",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "jpn": "ja",
    "kor": "ko",
    "zho": "zh",
    "chi": "zh",
    "ara": "ar",
    "hin": "hi",
    "tur": "tr",
    "pol": "pl",
    "nld": "nl",
    "dut": "nl",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    "nob": "no",
    "nb": "no",
    "fin": "fi",
    "ces": "cs",
    "cze": "cs",
    "ell": "el",
    "gre": "el",
    "heb": "he",
    "hun": "hu",
    "ron": "ro",
    "rum": "ro",
    "ukr": "uk",
    "vie": "vi",
    "tha": "th",
    "ind": "id",
    "msa": "ms",
    "may": "ms",
    "bul": "bg",
    "hrv": "hr",
    "slk": "sk",
    "slo": "sk",
    "slv": "sl",
    "lit": "lt",
    "lav": "lv",
    "est": "et",
    "cat": "ca",
    "fas": "fa",
    "per": "fa",
    "ben": "bn",
    "tam": "ta",
    "urd": "ur",
    "srp": "sr",
    "isl": "is",
    "ice": "is",
    "afr": "af",
    "swa": "sw",
    "tgl": "tl",
    "glg": "gl",
    "eus": "eu",
    "baq": "eu",
    "cym": "cy",
    "wel": "cy",
    # Regional and script forms
    "zh-hans": "zh",
    "zh-cn": "zh",
    "zh-sg": "zh",
    "zh-tw": "zh-hant",
    "zh-hk": "zh-hant",
    "cmn": "zh",
    "yue": "zh-hant",
}


def _normalise(text: str) -> str:
    return text.strip().lower().replace("_", "-")


def _build_names() -> dict[str, Language]:
    names: dict[str, Language] = {}
    for language in LANGUAGES:
        names[language.name.lower()] = language
        names[language.native.lower()] = language
        names[language.iso2] = language
    names["traditional chinese"] = _BY_CODE["zh-hant"]
    names["simplified chinese"] = _BY_CODE["zh"]
    names["castilian"] = _BY_CODE["es"]
    names["flemish"] = _BY_CODE["nl"]
    return names


_BY_NAME = _build_names()


def find_language(text: str) -> Language | None:
    """The language for a code ("es", "spa", "pt-BR"), name ("Spanish") or native name, or None."""
    key = _normalise(text)
    if not key:
        return None
    if key in _BY_CODE:
        return _BY_CODE[key]
    if key in _ALIASES:
        return _BY_CODE[_ALIASES[key]]
    if key in _BY_NAME:
        return _BY_NAME[key]
    # A regional form of a language we know: "pt-br", "en-gb", "es-419".
    base = key.split("-")[0]
    if "-" in key and base in _BY_CODE:
        return _BY_CODE[base]
    return None


def resolve_language(text: str) -> Language:
    """Like find_language, but an unknown language raises an error that suggests what was meant."""
    language = find_language(text)
    if language is not None:
        return language
    candidates = list(_BY_NAME) + list(_BY_CODE)
    close = difflib.get_close_matches(_normalise(text), candidates, n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise UnknownLanguageError(
        f"I do not know the language {text!r}.{hint} Use a name such as 'Spanish' or a code "
        "such as 'es' (see `videosubs --list-languages`)."
    )


def resolve_languages(values: list[str]) -> list[Language]:
    """Resolve each value and drop repeats, keeping the order given."""
    seen: dict[str, Language] = {}
    for value in values:
        language = resolve_language(value)
        seen.setdefault(language.code, language)
    return list(seen.values())


def from_whisper_code(code: str) -> Language:
    """The language for a code Whisper reported; unknown ones become a bare placeholder."""
    language = find_language(code)
    if language is not None:
        return language
    return Language(code.lower(), code, code, code[:3].lower(), "", None)
