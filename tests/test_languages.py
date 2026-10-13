import pytest

from videosubs.errors import UnknownLanguageError
from videosubs.languages import (
    LANGUAGES,
    find_language,
    from_whisper_code,
    resolve_language,
    resolve_languages,
)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("es", "es"),
        ("Spanish", "es"),
        ("SPANISH", "es"),
        ("español", "es"),
        ("spa", "es"),
        ("pt-BR", "pt"),
        ("pt_br", "pt"),
        ("en-GB", "en"),
        ("ja", "ja"),
        ("Japanese", "ja"),
        ("日本語", "ja"),
        ("deu", "de"),
        ("zh", "zh"),
        ("zh-TW", "zh-Hant"),
        ("Traditional Chinese", "zh-Hant"),
        ("  Fr  ", "fr"),
    ],
)
def test_finds_a_language_however_it_is_written(text, code):
    assert find_language(text).code == code


def test_unknown_language_suggests_a_close_name():
    with pytest.raises(UnknownLanguageError, match="Spanish"):
        resolve_language("Spanis")
    with pytest.raises(UnknownLanguageError, match="do not know"):
        resolve_language("klingon-ish")
    assert find_language("") is None


def test_resolving_a_list_drops_repeats_but_keeps_order():
    assert [lang.code for lang in resolve_languages(["english", "es", "EN", "spa", "ja"])] == ["en", "es", "ja"]


def test_every_language_has_the_codes_the_backends_need():
    codes = [lang.code for lang in LANGUAGES]
    assert len(codes) == len(set(codes)), "duplicate language code"
    for lang in LANGUAGES:
        assert len(lang.iso2) == 3, lang
        assert "_" in lang.nllb, lang  # FLORES-200 codes look like spa_Latn
        assert lang.native and lang.name


def test_whisper_sees_no_script_variants():
    assert find_language("zh-Hant").whisper == "zh"
    assert find_language("pt").whisper == "pt"


def test_japanese_and_chinese_are_not_spaced_but_korean_is():
    assert not find_language("ja").spaced
    assert not find_language("zh").spaced
    assert find_language("ko").spaced


def test_unknown_whisper_codes_become_a_placeholder_instead_of_crashing():
    placeholder = from_whisper_code("haw")
    assert placeholder.code == "haw"
    assert placeholder.nllb == ""
    assert from_whisper_code("es").name == "Spanish"
