import pytest

from conftest import FakeTranscriber, FakeTranslator, probe_streams, sentence
from videosubs.cache import TranscriptCache
from videosubs.cues import parse_subtitles
from videosubs.errors import MediaError, TranslationError
from videosubs.languages import find_language
from videosubs.media import MediaFile
from videosubs.pipeline import Options, Pipeline, SilentReporter
from videosubs.translate import TranslationSettings

pytestmark = pytest.mark.needs_ffmpeg
EN, ES, JA = (find_language(c) for c in ("en", "es", "ja"))


class Recorder(SilentReporter):
    def __init__(self):
        self.stages, self.notes, self.finished = [], [], []

    def stage(self, name):
        self.stages.append(name)

    def note(self, message):
        self.notes.append(message)

    def file_finished(self, outcome):
        self.finished.append(outcome)


def build(options=None, transcriber=None, translator=None, reporter=None):
    translator = translator or FakeTranslator()
    pipeline = Pipeline(
        options or Options(),
        transcriber or FakeTranscriber(),
        reporter=reporter or Recorder(),
        translator_factory=lambda backend, settings: translator,
    )
    return pipeline, translator


def video(path):
    return MediaFile(path, "video")


def texts(path):
    return [c.text for c in parse_subtitles(path.read_text(encoding="utf-8"))]


class TestWritingSubtitles:
    def test_writes_the_spoken_language_beside_the_video(self, make_media):
        path = make_media("movie.mkv")
        pipeline, _ = build()
        outcome = pipeline.process(video(path))
        assert outcome.status == "done"
        assert outcome.language == "en"
        sidecar = path.with_name("movie.en.srt")
        assert texts(sidecar) == ["Hello and welcome.", "This is a short test."]
        assert outcome.outputs == [sidecar]

    def test_the_source_language_can_be_forced_and_is_passed_to_the_recogniser(self, make_media):
        path = make_media("movie.mkv")
        transcriber = FakeTranscriber(language="ja")
        pipeline, _ = build(Options(source_language=JA), transcriber)
        pipeline.process(video(path))
        assert transcriber.calls[0]["language"] == "ja"
        assert path.with_name("movie.ja.srt").exists()

    def test_translations_are_made_per_segment_and_laid_over_the_segments_time(self, make_media):
        path = make_media("movie.mkv")
        pipeline, translator = build(Options(languages=[EN, ES]))
        outcome = pipeline.process(video(path))
        assert [p.name for p in outcome.outputs] == ["movie.en.srt", "movie.es.srt"]
        sent, source, target = translator.calls[0]
        assert sent == ["Hello and welcome.", "This is a short test."] and (source, target) == ("en", "es")
        cues = parse_subtitles(path.with_name("movie.es.srt").read_text())
        assert [c.text for c in cues] == ["[es] Hello and welcome.", "[es] This is a short test."]
        assert cues[0].start == pytest.approx(0.2, abs=0.05) and cues[1].start == pytest.approx(2.0, abs=0.05)

    def test_only_the_requested_languages_are_written(self, make_media):
        path = make_media("movie.mkv")
        build(Options(languages=[ES]))[0].process(video(path))
        assert path.with_name("movie.es.srt").exists() and not path.with_name("movie.en.srt").exists()

    def test_other_formats(self, make_media):
        path = make_media("movie.mkv")
        build(Options(formats=["srt", "vtt", "txt"]))[0].process(video(path))
        assert path.with_name("movie.en.vtt").read_text().startswith("WEBVTT")
        assert path.with_name("movie.en.txt").read_text() == "Hello and welcome.\nThis is a short test.\n"

    def test_no_temporary_files_are_left_behind(self, make_media):
        path = make_media("movie.mkv")
        build()[0].process(video(path))
        assert sorted(p.name for p in path.parent.iterdir() if "tmp" in p.name) == []

    def test_a_video_with_no_speech_is_skipped_not_failed(self, make_media):
        path = make_media("silent.mkv")
        outcome = build(transcriber=FakeTranscriber(segments=[]))[0].process(video(path))
        assert outcome.status == "skipped" and "no speech" in outcome.message
        assert not list(path.parent.glob("silent.*.srt"))

    def test_a_low_confidence_language_guess_is_flagged(self, make_media):
        reporter = Recorder()
        build(transcriber=FakeTranscriber(probability=0.4), reporter=reporter)[0].process(video(make_media()))
        assert any("not sure which language" in n for n in reporter.notes)


class TestSkippingAndForcing:
    def test_a_second_run_does_nothing(self, make_media):
        path = make_media("movie.mkv")
        transcriber = FakeTranscriber()
        pipeline, _ = build(transcriber=transcriber)
        pipeline.process(video(path))
        again = pipeline.process(video(path))
        assert again.status == "skipped" and "already has" in again.message
        assert len(transcriber.calls) == 1

    def test_only_the_missing_languages_are_made_but_existing_files_are_never_overwritten(self, make_media):
        path = make_media("movie.mkv")
        existing = path.with_name("movie.en.srt")
        existing.write_text("1\n00:00:00,000 --> 00:00:01,000\nMY OWN EDIT\n")
        reporter = Recorder()
        outcome = build(Options(languages=[EN, ES]), reporter=reporter)[0].process(video(path))
        assert outcome.status == "done"
        assert texts(existing) == ["MY OWN EDIT"]
        assert path.with_name("movie.es.srt").exists()
        assert any("kept the existing movie.en.srt" in n for n in reporter.notes)

    def test_force_rewrites(self, make_media):
        path = make_media("movie.mkv")
        existing = path.with_name("movie.en.srt")
        existing.write_text("1\n00:00:00,000 --> 00:00:01,000\nOLD\n")
        build(Options(force=True))[0].process(video(path))
        assert "Hello and welcome." in texts(existing)

    def test_names_full_of_glob_characters_are_recognised_as_done(self, make_media):
        path = make_media("[Group] Show - 01 [1080p].mkv")
        pipeline, _ = build()
        pipeline.process(video(path))
        assert pipeline.process(video(path)).status == "skipped"

    def test_dry_run_changes_nothing_and_never_calls_the_recogniser(self, make_media):
        path = make_media("movie.mkv")
        transcriber = FakeTranscriber()
        outcome = build(Options(dry_run=True, languages=[EN, ES], embed=True), transcriber)[0].process(video(path))
        assert outcome.status == "planned"
        assert "movie.subs.mkv" in outcome.message and "English, Spanish" in outcome.message
        assert transcriber.calls == [] and sorted(p.name for p in path.parent.iterdir()) == ["movie.mkv"]


class TestBatches:
    def test_one_failure_does_not_stop_the_others(self, make_media):
        a, b, c = (make_media(f"{n}.mkv") for n in "abc")
        pipeline, _ = build(transcriber=FakeTranscriber(fail_on={"b.mkv"}))
        outcomes = pipeline.run([video(a), video(b), video(c)])
        assert [o.status for o in outcomes] == ["done", "failed", "done"]
        assert "exploded" in outcomes[1].message

    def test_stop_on_error_stops(self, make_media):
        a, b, c = (make_media(f"{n}.mkv") for n in "abc")
        pipeline, _ = build(Options(stop_on_error=True), FakeTranscriber(fail_on={"b.mkv"}))
        assert [o.status for o in pipeline.run([video(a), video(b), video(c)])] == ["done", "failed"]

    def test_a_file_ffprobe_cannot_read_is_a_failure_with_a_reason(self, tmp_path):
        junk = tmp_path / "junk.mkv"
        junk.write_text("not a video")
        outcome = build()[0].run([video(junk)])[0]
        assert outcome.status == "failed" and "could not read" in outcome.message

    def test_the_reporter_hears_about_every_file(self, make_media):
        reporter = Recorder()
        build(reporter=reporter)[0].run([video(make_media("a.mkv")), video(make_media("b.mkv"))])
        assert [o.media.path.name for o in reporter.finished] == ["a.mkv", "b.mkv"]


class TestTranscriptCache:
    def test_a_repeat_run_after_a_failed_translation_does_not_transcribe_again(self, make_media, tmp_path):
        path = make_media("movie.mkv")
        transcriber = FakeTranscriber()
        cache = TranscriptCache(tmp_path / "cache")

        class Failing(FakeTranslator):
            def translate(self, *args, **kwargs):
                raise TranslationError("the translator is down")

        options = Options(languages=[ES], cache=cache, model_key="small")
        first = build(options, transcriber, Failing())[0].run([video(path)])[0]
        assert first.status == "failed"
        second = build(options, transcriber)[0].run([video(path)])[0]
        assert second.status == "done"
        assert len(transcriber.calls) == 1

    def test_a_different_model_does_not_reuse_the_transcript(self, make_media, tmp_path):
        path = make_media("movie.mkv")
        transcriber = FakeTranscriber()
        cache = TranscriptCache(tmp_path / "cache")
        build(Options(cache=cache, model_key="small"), transcriber)[0].process(video(path))
        path.with_name("movie.en.srt").unlink()
        build(Options(cache=cache, model_key="large"), transcriber)[0].process(video(path))
        assert len(transcriber.calls) == 2


class TestAudioTracks:
    def test_the_track_in_the_spoken_language_is_extracted_and_used(self, make_media):
        path = make_media("movie.mkv", audio=("eng", "jpn"))
        transcriber = FakeTranscriber(language="ja")
        build(Options(source_language=JA), transcriber)[0].process(video(path))
        used = transcriber.calls[0]["audio"]
        assert used != path and used.suffix == ".wav"  # the second track had to be extracted

    def test_the_first_track_is_read_directly_without_extraction(self, make_media):
        path = make_media("movie.mkv", audio=("eng", "jpn"))
        transcriber = FakeTranscriber()
        build(transcriber=transcriber)[0].process(video(path))
        assert transcriber.calls[0]["audio"] == path

    def test_a_bad_track_number_is_an_error_with_the_count(self, make_media):
        path = make_media("movie.mkv")
        with pytest.raises(MediaError, match="does not exist"):
            build(Options(audio_track=3))[0].process(video(path))

    def test_a_video_without_audio_fails_clearly(self, make_media):
        path = make_media("mute.mkv", audio=())
        with pytest.raises(MediaError, match="no audio track"):
            build()[0].process(video(path))


class TestWhisperTranslation:
    def test_english_from_another_language_is_a_second_pass_of_the_recogniser(self, make_media):
        path = make_media("pelicula.mkv")
        english = [sentence(0.1, "Hello from the English pass.")]

        class TwoPass(FakeTranscriber):
            def transcribe(self, audio, *, language, task="transcribe", progress=None):
                result = super().transcribe(audio, language=language, task=task, progress=progress)
                if task == "translate":
                    result.segments = english
                return result

        transcriber = TwoPass(language="es", segments=[sentence(0.1, "Hola desde el audio.")])
        options = Options(languages=[EN], translation=TranslationSettings(backend="whisper"))
        build(options, transcriber)[0].process(video(path))
        assert [c["task"] for c in transcriber.calls] == ["transcribe", "translate"]
        assert texts(path.with_name("pelicula.en.srt")) == ["Hello from the English pass."]

    def test_whisper_cannot_translate_into_other_languages(self, make_media):
        path = make_media("movie.mkv")
        options = Options(languages=[ES], translation=TranslationSettings(backend="whisper"))
        with pytest.raises(TranslationError, match="only translate into English"):
            build(options)[0].process(video(path))


class TestEmbedding:
    def test_writes_a_new_mkv_with_the_subtitles_and_leaves_the_original(self, make_media):
        path = make_media("movie.mp4", audio=("eng",))
        outcome = build(Options(languages=[EN, ES], embed=True))[0].process(video(path))
        out = path.with_name("movie.subs.mkv")
        assert out in outcome.outputs
        assert probe_streams(out)["subtitle"] == ["eng", "spa"]
        assert path.exists() and path.with_name("movie.en.srt").exists()

    def test_embedding_always_produces_srt_even_if_only_other_formats_were_asked_for(self, make_media):
        path = make_media("movie.mkv")
        build(Options(embed=True, formats=["vtt"]))[0].process(video(path))
        assert path.with_name("movie.en.srt").exists() and path.with_name("movie.subs.mkv").exists()

    def test_languages_already_in_the_video_are_not_added_twice(self, make_media):
        path = make_media("movie.mkv", subtitles=("eng",))
        reporter = Recorder()
        outcome = build(Options(languages=[EN], embed=True), reporter=reporter)[0].process(video(path))
        assert not path.with_name("movie.subs.mkv").exists()
        assert any("already embedded" in n for n in reporter.notes)
        assert outcome.status == "done"

    def test_no_sidecar_removes_the_srt_files_afterwards(self, make_media):
        path = make_media("movie.mkv")
        build(Options(embed=True, keep_sidecars=False))[0].process(video(path))
        assert path.with_name("movie.subs.mkv").exists() and not path.with_name("movie.en.srt").exists()

    def test_the_default_subtitle_flag_reaches_the_file(self, make_media):
        import json
        import subprocess

        path = make_media("movie.mkv")
        build(Options(languages=[EN, ES], embed=True, default_subtitle=ES))[0].process(video(path))
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "s", str(path.with_name("movie.subs.mkv"))],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        defaults = {s["tags"]["language"]: s["disposition"]["default"] for s in json.loads(out)["streams"]}
        assert defaults == {"eng": 0, "spa": 1}

    def test_replace_swaps_the_file_and_keeps_a_backup(self, make_media):
        path = make_media("movie.mp4")
        outcome = build(Options(embed=True, replace=True))[0].process(video(path))
        final = path.with_suffix(".mkv")
        assert outcome.outputs[-1] == final and final.exists()
        assert not path.exists() and (path.parent / ".videosubs-backup" / "movie.mp4").exists()
        assert probe_streams(final)["subtitle"] == ["eng"]
        # And the run is repeatable: the file now has the subtitles, so there is nothing to do.
        assert build(Options(embed=True, replace=True, languages=[EN]))[0].process(video(final)).status == "skipped"

    def test_audio_files_are_never_embedded(self, make_media):
        path = make_media("podcast.mka", video=False)
        outcome = build(Options(embed=True))[0].process(MediaFile(path, "audio"))
        assert outcome.status == "done" and not path.with_name("podcast.subs.mkv").exists()


class TestSubtitleFiles:
    def write(self, tmp_path, name="movie.en.srt"):
        path = tmp_path / name
        path.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nGood morning.\n\n2\n00:00:04,000 --> 00:00:05,500\nSee you later.\n",
            encoding="utf-8",
        )
        return MediaFile(path, "subtitle")

    def test_translates_keeping_every_cue_and_its_timing(self, tmp_path):
        source = self.write(tmp_path)
        outcome = build(Options(languages=[ES]))[0].process(source)
        out = tmp_path / "movie.es.srt"
        assert outcome.status == "done" and outcome.outputs == [out]
        cues = parse_subtitles(out.read_text())
        assert [(c.start, c.end, c.text) for c in cues] == [
            (1.0, 3.0, "[es] Good morning."),
            (4.0, 5.5, "[es] See you later."),
        ]

    def test_the_source_language_comes_from_the_name_or_the_flag(self, tmp_path):
        untagged = self.write(tmp_path, "movie.srt")
        with pytest.raises(MediaError, match="--source-language"):
            build(Options(languages=[ES]))[0].process(untagged)
        assert build(Options(languages=[ES], source_language=EN))[0].process(untagged).status == "done"
        assert (tmp_path / "movie.es.srt").exists()

    def test_without_languages_there_is_nothing_to_do(self, tmp_path):
        outcome = build()[0].process(self.write(tmp_path))
        assert outcome.status == "skipped" and "-l" in outcome.message

    def test_an_existing_translation_is_left_alone(self, tmp_path):
        source = self.write(tmp_path)
        pipeline, translator = build(Options(languages=[ES]))
        pipeline.process(source)
        assert pipeline.process(source).status == "skipped"
        assert len(translator.calls) == 1

    def test_translating_into_its_own_language_is_a_skip(self, tmp_path):
        assert build(Options(languages=[EN]))[0].process(self.write(tmp_path)).status == "skipped"

    def test_dry_run(self, tmp_path):
        outcome = build(Options(languages=[ES], dry_run=True))[0].process(self.write(tmp_path))
        assert outcome.status == "planned" and not (tmp_path / "movie.es.srt").exists()


class TestWorkIsNotWastedOnFilesThatAlreadyExist:
    def test_a_language_whose_file_exists_is_not_translated_again(self, make_media):
        path = make_media("movie.mkv")
        path.with_name("movie.es.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nMI VERSION\n", encoding="utf-8")
        pipeline, translator = build(Options(languages=[EN, ES]))
        outcome = pipeline.process(video(path))
        assert translator.calls == []  # the slow step never ran: English is the spoken language
        assert texts(path.with_name("movie.es.srt")) == ["MI VERSION"]
        assert outcome.status == "done" and [p.name for p in outcome.outputs] == ["movie.en.srt"]

    def test_everything_already_written_is_a_skip_before_any_work(self, make_media):
        path = make_media("movie.mkv")
        path.with_name("movie.es.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nMI VERSION\n", encoding="utf-8")
        transcriber = FakeTranscriber()
        outcome = build(Options(languages=[ES]), transcriber)[0].process(video(path))
        assert outcome.status == "skipped" and transcriber.calls == []

    def test_an_existing_file_is_still_the_one_that_gets_embedded(self, make_media):
        path = make_media("movie.mkv")
        path.with_name("movie.es.srt").write_text("1\n00:00:00,500 --> 00:00:01,500\nMI VERSION\n", encoding="utf-8")
        pipeline, translator = build(Options(languages=[ES], embed=True))
        pipeline.process(video(path))
        assert translator.calls == []
        assert probe_streams(path.with_name("movie.subs.mkv"))["subtitle"] == ["spa"]

    def test_only_the_missing_language_is_translated(self, make_media):
        path = make_media("movie.mkv")
        path.with_name("movie.es.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nMI VERSION\n", encoding="utf-8")
        pipeline, translator = build(Options(languages=[ES, JA]))
        pipeline.process(video(path))
        assert [call[2] for call in translator.calls] == ["ja"]
