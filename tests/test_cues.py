from videosubs.cues import (
    Cue,
    Segment,
    Style,
    Word,
    clean,
    cues_from_text,
    cues_from_words,
    format_srt,
    format_txt,
    format_vtt,
    localise_punctuation,
    parse_subtitles,
    segments_to_cues,
    sentence_units,
    split_sentences,
    tidy,
    translated_to_cues,
    wrap,
)


def words(text: str, start: float = 0.0, per_word: float = 0.4, gap: float = 0.0) -> list[Word]:
    """Timed words the way a recogniser reports them: each with a leading space."""
    out, t = [], start
    for token in text.split():
        out.append(Word(t, t + per_word, f" {token}"))
        t += per_word + gap
    return out


class TestWrap:
    def test_short_text_stays_on_one_line(self):
        assert wrap("Hello there.", 42) == ["Hello there."]

    def test_long_text_splits_into_two_balanced_lines(self):
        lines = wrap("This is a rather long sentence that needs two lines to fit", 42, 2)
        assert len(lines) == 2
        assert all(len(line) <= 42 for line in lines)
        assert abs(len(lines[0]) - len(lines[1])) <= 12

    def test_a_word_longer_than_the_line_does_not_loop_or_vanish(self):
        lines = wrap("Supercalifragilisticexpialidocious is long", 20, 2)
        assert "Supercalifragilisticexpialidocious" in lines[0]
        assert "".join(" ".join(lines).split()) == "Supercalifragilisticexpialidociousislong"

    def test_wide_text_breaks_after_punctuation_near_the_middle(self):
        text = "今日はとても良い天気ですね、散歩に行きましょう"
        lines = wrap(text, 14, 2, spaced=False)
        assert len(lines) == 2
        assert "".join(lines) == text
        assert lines[0].endswith("、")


class TestWords:
    def test_a_short_sentence_is_one_cue_with_the_words_times(self):
        cues = cues_from_words(words("Hello and welcome."))
        assert [c.text for c in cues] == ["Hello and welcome."]
        assert cues[0].start == 0.0
        assert cues[0].end == 1.2000000000000002 or abs(cues[0].end - 1.2) < 1e-9

    def test_a_pause_starts_a_new_cue(self):
        first = words("First part here.", start=0)
        second = words("Second part later.", start=first[-1].end + 2.5)
        assert len(cues_from_words(first + second)) == 2

    def test_sentence_ends_split_cues_once_there_is_enough_text(self):
        text = "We went to the market yesterday. Then we came home and slept."
        cues = cues_from_words(words(text))
        assert [c.text.replace("\n", " ") for c in cues] == [
            "We went to the market yesterday.",
            "Then we came home and slept.",
        ]

    def test_a_tiny_sentence_does_not_get_its_own_cue(self):
        cues = cues_from_words(words("Yes. We can go there tomorrow if you like."))
        assert len(cues) == 1

    def test_a_long_run_of_words_splits_at_a_clause_break_not_mid_phrase(self):
        text = "The long road home was quiet, and nobody spoke until morning came again"
        cues = cues_from_words(words(text), Style(max_chars=20))
        assert len(cues) >= 2
        assert cues[0].text.replace("\n", " ") == "The long road home was quiet,"

    def test_no_cue_exceeds_the_style_limits(self):
        text = " ".join(f"word{i}" for i in range(60))
        style = Style(max_chars=30, max_lines=2, max_seconds=5.0)
        for cue in cues_from_words(words(text), style):
            assert cue.duration <= 5.0 + 0.4
            assert all(len(line) <= 30 for line in cue.text.split("\n"))
            assert len(cue.text.split("\n")) <= 2

    def test_blank_words_are_ignored(self):
        assert cues_from_words([Word(0, 1, " "), Word(1, 2, " hi")])[0].text == "hi"

    def test_japanese_words_are_joined_without_spaces(self):
        ja = [
            Word(0.0, 0.5, "こんにちは"),
            Word(0.5, 1.0, "、"),
            Word(1.0, 1.6, "元気"),
            Word(1.6, 2.0, "ですか"),
            Word(2.0, 2.2, "？"),
        ]
        assert cues_from_words(ja, spaced=False)[0].text == "こんにちは、元気ですか？"


class TestText:
    def test_splits_sentences_but_not_decimals(self):
        assert split_sentences("It costs 3.5 euros. Really? Yes!") == ["It costs 3.5 euros.", "Really?", "Yes!"]

    def test_splits_japanese_sentences(self):
        assert split_sentences("今日は晴れです。明日は雨です！", spaced=False) == ["今日は晴れです。", "明日は雨です！"]

    def test_time_is_shared_in_proportion_to_length(self):
        cues = cues_from_text("Short one. A much longer second sentence follows here.", 10.0, 20.0)
        assert len(cues) == 2
        assert cues[0].start == 10.0
        assert abs(cues[-1].end - 20.0) < 1e-9
        assert cues[0].duration < cues[1].duration
        assert abs(cues[0].end - cues[1].start) < 1e-9

    def test_a_tiny_fragment_joins_its_neighbour(self):
        cues = cues_from_text("We are leaving now. Yes.", 0, 4)
        assert len(cues) == 1

    def test_a_long_sentence_is_cut_at_commas_to_fit(self):
        text = (
            "On the first day we walked along the river, on the second we climbed the hill, and on the third we rested"
        )
        cues = cues_from_text(text, 0, 12, Style(max_chars=30))
        assert len(cues) >= 3
        assert all(len(line) <= 30 for cue in cues for line in cue.text.split("\n"))

    def test_empty_text_or_an_empty_span_gives_nothing(self):
        assert cues_from_text("   ", 0, 5) == []
        assert cues_from_text("Hi there", 5, 5) == []


class TestTidy:
    def test_overlaps_are_trimmed_not_dropped(self):
        cues = tidy([Cue(0, 3, "a"), Cue(2, 5, "b")], Style())
        assert cues[0].end <= cues[1].start
        assert [c.text for c in cues] == ["a", "b"]

    def test_short_cues_get_more_time_when_there_is_room(self):
        cues = tidy([Cue(0, 0.3, "Hi"), Cue(5, 6, "Later")], Style(min_seconds=1.0))
        assert cues[0].duration >= 1.0 - 1e-9

    def test_short_cues_are_not_stretched_into_the_next_one(self):
        cues = tidy([Cue(0, 0.3, "Hi"), Cue(0.5, 3, "Next")], Style(min_seconds=1.0, gap=0.04))
        assert cues[0].end <= cues[1].start

    def test_an_echoed_line_is_merged(self):
        cues = tidy([Cue(0, 1, "Thank you."), Cue(1.1, 2, "Thank you.")])
        assert len(cues) == 1
        assert cues[0].end == 2

    def test_empty_and_unordered_cues_are_handled(self):
        cues = tidy([Cue(5, 6, "late"), Cue(1, 2, " "), Cue(0, 1, "early")])
        assert [c.text for c in cues] == ["early", "late"]

    def test_zero_length_cues_get_a_duration(self):
        assert tidy([Cue(3, 3, "x")])[0].end > 3


def test_segments_use_word_times_when_present_and_text_when_not():
    timed = Segment(0, 2, "Hello world.", tuple(words("Hello world.")))
    plain = Segment(5, 9, "No words here for this one.")
    cues = segments_to_cues([timed, plain])
    assert [c.text for c in cues] == ["Hello world.", "No words here for this one."]
    assert cues[1].start == 5


def test_translation_is_laid_over_each_segments_span():
    segs = [Segment(0, 4, "Hello. How are you?"), Segment(6, 8, "Fine.")]
    cues = translated_to_cues(segs, ["Hola. ¿Cómo estás?", "Bien."])
    assert cues[0].start == 0
    assert cues[-1].start >= 6
    assert [c.text for c in cues][-1] == "Bien."


SAMPLE = [Cue(0.0, 1.16, "Hello and welcome."), Cue(3725.5, 3727.005, "Line one\nLine two")]


class TestFormats:
    cues = SAMPLE

    def test_srt(self):
        assert format_srt(self.cues) == (
            "1\n00:00:00,000 --> 00:00:01,160\nHello and welcome.\n\n"
            "2\n01:02:05,500 --> 01:02:07,005\nLine one\nLine two\n"
        )

    def test_vtt(self):
        out = format_vtt(self.cues)
        assert out.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.160\n")
        assert "01:02:05.500 --> 01:02:07.005" in out

    def test_txt(self):
        assert format_txt(self.cues) == "Hello and welcome.\nLine one Line two\n"

    def test_timestamps_carry_correctly(self):
        assert "00:00:01,000" in format_srt([Cue(0.9996, 2, "x")])

    def test_srt_round_trips(self):
        assert parse_subtitles(format_srt(self.cues)) == self.cues

    def test_vtt_round_trips(self):
        assert parse_subtitles(format_vtt(self.cues)) == self.cues

    def test_parser_is_tolerant(self):
        messy = "﻿1\r\n00:00:01,000 --> 00:00:02,500\r\nHi\r\n\r\n\r\nNOTE a comment\r\n\r\n00:00:03.000 --> 00:00:04.000\r\nNo number\r\n"
        cues = parse_subtitles(messy)
        assert [(c.start, c.end, c.text) for c in cues] == [(1.0, 2.5, "Hi"), (3.0, 4.0, "No number")]

    def test_parser_returns_nothing_for_junk(self):
        assert parse_subtitles("not subtitles at all") == []


def test_clean_collapses_whitespace_and_wide_spaces():
    assert clean("  a   b \n c ") == "a b c"
    assert clean("今日は 天気", spaced=False) == "今日は天気"


class TestSentenceUnits:
    def test_a_two_sentence_segment_becomes_two_units_with_their_own_times(self):
        segment = Segment(
            0.0, 3.2, "Hello there. How are you?", tuple(words("Hello there. How are you?", per_word=0.4, gap=0.0))
        )
        units = sentence_units([segment])
        assert [u.text for u in units] == ["Hello there.", "How are you?"]
        assert (units[0].start, round(units[0].end, 2)) == (0.0, 0.8)
        assert (round(units[1].start, 2), round(units[1].end, 2)) == (0.8, 2.0)

    def test_abbreviations_do_not_end_a_sentence(self):
        segment = Segment(0, 3, "Dr. Smith will see you now.", tuple(words("Dr. Smith will see you now.")))
        assert [u.text for u in sentence_units([segment])] == ["Dr. Smith will see you now."]

    def test_a_trailing_fragment_without_punctuation_is_kept(self):
        segment = Segment(0, 3, "It ended. and then", tuple(words("It ended. and then")))
        assert [u.text for u in sentence_units([segment])] == ["It ended.", "and then"]

    def test_segments_without_word_times_are_split_by_punctuation_and_share_the_time(self):
        units = sentence_units([Segment(10, 20, "Short one. A much longer second sentence follows here.")])
        assert len(units) == 2
        assert units[0].start == 10 and abs(units[-1].end - 20) < 1e-9
        assert units[0].end == units[1].start

    def test_japanese_sentences_split_at_the_ideographic_stop(self):
        units = sentence_units([Segment(0, 4, "今日は晴れです。明日は雨です。")], spaced=False)
        assert [u.text for u in units] == ["今日は晴れです。", "明日は雨です。"]


class TestPunctuation:
    def test_japanese_gets_ideographic_punctuation_and_loses_the_spaces(self):
        assert localise_punctuation("この短いデモにようこそ.今日,", "ja") == "この短いデモにようこそ。今日、"
        assert (
            localise_punctuation("します. オーディオを聴き,文を見つけ,", "ja")
            == "します。オーディオを聴き、文を見つけ、"
        )

    def test_chinese_uses_its_own_comma(self):
        assert localise_punctuation("你好, 世界.", "zh") == "你好，世界。"
        assert localise_punctuation("你好, 世界.", "zh-Hant") == "你好，世界。"

    def test_ascii_text_inside_japanese_keeps_its_dots(self):
        assert localise_punctuation("Python 3.5を使います.", "ja") == "Python 3.5を使います。"
        assert localise_punctuation("e.g. これです.", "ja") == "e.g. これです。"

    def test_other_languages_are_untouched(self):
        assert localise_punctuation("Hola. ¿Qué tal?", "es") == "Hola. ¿Qué tal?"


class TestLongSentences:
    def test_an_overlong_sentence_splits_into_balanced_pieces_at_clause_breaks(self):
        text = "First we open the box, then we read the card, after that we check the list, and finally we close everything."
        cues = cues_from_text(text, 0.0, 10.0, Style(max_chars=42))
        first, second = (c.text.replace("\n", " ") for c in cues)
        # Greedy packing would give 75 + 32 characters; balancing gives 45 + 62.
        assert first == "First we open the box, then we read the card,"
        assert second == "after that we check the list, and finally we close everything."

    def test_a_clause_longer_than_a_cue_is_still_cut_by_words(self):
        text = "word " * 40
        cues = cues_from_text(text, 0, 20, Style(max_chars=20))
        assert len(cues) >= 2
        assert all(len(line) <= 20 for cue in cues for line in cue.text.split("\n"))
