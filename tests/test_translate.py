import pytest

from videosubs.errors import TranslationError
from videosubs.languages import find_language
from videosubs.translate import TranslationSettings, choose_backend, create_translator
from videosubs.translate.deepl_backend import DeepLTranslator, source_code
from videosubs.translate.nllb import NllbTranslator

EN, ES, JA, HI = (find_language(c) for c in ("en", "es", "ja", "hi"))


class Result:
    def __init__(self, text):
        self.text = text


class FakeDeepL:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def translate_text(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if self.error:
            raise self.error
        return [Result(f"<{kwargs['target_lang']}>{t}") for t in texts]


class TestDeepL:
    def test_sends_codes_and_returns_translations_in_order(self):
        client = FakeDeepL()
        out = DeepLTranslator("key", client=client).translate(["a", "b"], EN, ES)
        assert out == ["<ES>a", "<ES>b"]
        _, kwargs = client.calls[0]
        assert kwargs["source_lang"] == "EN" and kwargs["target_lang"] == "ES"
        assert kwargs["preserve_formatting"] is True and kwargs["formality"] == "default"

    def test_batches_large_requests_and_reports_progress(self):
        client = FakeDeepL()
        seen = []
        out = DeepLTranslator("key", client=client).translate([str(i) for i in range(120)], EN, ES, seen.append)
        assert len(out) == 120
        assert [len(call[0]) for call in client.calls] == [50, 50, 20]
        assert seen[-1] == 1.0

    def test_unsupported_source_languages_are_left_to_deepl_to_detect(self):
        assert source_code(HI) is None
        assert source_code(find_language("no")) == "NB"
        assert source_code(find_language("zh-Hant")) == "ZH"

    def test_formality_maps_to_deepls_forgiving_options(self):
        client = FakeDeepL()
        DeepLTranslator("key", formality="more", client=client).translate(["x"], EN, ES)
        assert client.calls[0][1]["formality"] == "prefer_more"
        with pytest.raises(TranslationError, match="formality"):
            DeepLTranslator("key", formality="loud", client=client)

    def test_a_language_deepl_lacks_is_refused_before_any_request(self):
        client = FakeDeepL()
        with pytest.raises(TranslationError, match="does not offer Hindi"):
            DeepLTranslator("key", client=client).translate(["x"], EN, HI)
        assert client.calls == []

    @pytest.mark.parametrize(
        ("name", "message"),
        [
            ("AuthorizationException", "rejected the API key"),
            ("QuotaExceededException", "quota"),
            ("TooManyRequestsException", "too many"),
        ],
    )
    def test_api_problems_become_readable_errors(self, name, message):
        error = type(name, (Exception,), {})("boom")
        with pytest.raises(TranslationError, match=message):
            DeepLTranslator("key", client=FakeDeepL(error)).translate(["x"], EN, ES)

    def test_a_short_answer_is_caught(self):
        class Short(FakeDeepL):
            def translate_text(self, texts, **kwargs):
                return [Result("only one")]

        with pytest.raises(TranslationError, match="different number"):
            DeepLTranslator("key", client=Short()).translate(["a", "b"], EN, ES)


class FakeTokenizer:
    """Splits on spaces; token ids are just positions in a growing table."""

    def __init__(self):
        self.table = {}

    def encode(self, text, add_special_tokens=True):
        class Encoding:
            tokens = text.split()

        return Encoding()

    def token_to_id(self, token):
        return self.table.setdefault(token, len(self.table))

    def decode(self, ids, skip_special_tokens=True):
        reverse = {v: k for k, v in self.table.items()}
        return " ".join(reverse[i] for i in ids)


class FakeCT2:
    def __init__(self):
        self.batches = []

    def translate_batch(self, batch, target_prefix, **options):
        self.batches.append((batch, target_prefix, options))

        class Out:
            def __init__(self, tokens):
                self.hypotheses = [tokens]

        return [
            Out([prefix[0], *[t.upper() for t in tokens[1:-1]]])
            for tokens, prefix in zip(batch, target_prefix, strict=True)
        ]


class TestNllb:
    def make(self):
        translator = NllbTranslator("some/model")
        translator._translator, translator._tokenizer = FakeCT2(), FakeTokenizer()
        return translator

    def test_wraps_source_text_in_language_tokens_and_strips_the_target_token(self):
        translator = self.make()
        out = translator.translate(["hello there"], EN, ES)
        batch, prefix, options = translator._translator.batches[0]
        assert batch == [["eng_Latn", "hello", "there", "</s>"]]
        assert prefix == [["spa_Latn"]]
        assert out == ["HELLO THERE"]
        assert options["beam_size"] == 4

    def test_keeps_order_across_chunks_even_though_it_sorts_by_length(self):
        translator = self.make()
        texts = [f"{'word ' * (i % 7 + 1)}{i}" for i in range(70)]
        out = translator.translate(texts, EN, ES)
        assert out == [t.upper().strip() for t in texts]
        assert len(translator._translator.batches) == 3

    def test_lines_without_letters_are_passed_through_untouched(self):
        translator = self.make()
        out = translator.translate(["♪", "12", "real words"], EN, ES)
        assert out == ["♪", "12", "REAL WORDS"]
        assert len(translator._translator.batches[0][0]) == 1

    def test_a_language_without_a_nllb_code_is_refused(self):
        translator = self.make()
        placeholder = type(ES)("xx", "Made-up", "Made-up", "xxx", "", None)
        with pytest.raises(TranslationError, match="does not know Made-up"):
            translator.translate(["hi"], EN, placeholder)

    def test_progress_ends_at_one(self):
        seen = []
        self.make().translate(["a b", "c d"], EN, JA, seen.append)
        assert seen[-1] == 1.0


class TestChoosingABackend:
    def test_explicit_choices_win(self):
        assert choose_backend(TranslationSettings(backend="nllb", deepl_key="k"), ES) == "nllb"

    def test_auto_prefers_deepl_when_there_is_a_key_and_it_offers_the_language(self, monkeypatch):
        monkeypatch.delenv("DEEPL_API_KEY", raising=False)
        monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
        assert choose_backend(TranslationSettings(), ES) == "nllb"
        assert choose_backend(TranslationSettings(deepl_key="k"), ES) == "deepl"
        assert choose_backend(TranslationSettings(deepl_key="k"), HI) == "nllb"
        monkeypatch.setenv("DEEPL_API_KEY", "from-env")
        assert choose_backend(TranslationSettings(), ES) == "deepl"

    def test_deepl_without_a_key_explains_how_to_get_one(self, monkeypatch):
        monkeypatch.delenv("DEEPL_API_KEY", raising=False)
        monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
        with pytest.raises(TranslationError, match="DEEPL_API_KEY"):
            create_translator("deepl", TranslationSettings())

    def test_unknown_backends_are_refused(self):
        with pytest.raises(TranslationError, match="unknown translation backend"):
            create_translator("babelfish", TranslationSettings())
