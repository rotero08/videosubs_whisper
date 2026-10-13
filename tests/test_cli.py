import pytest

from conftest import FakeTranscriber
from videosubs import cli


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the real recogniser, so the command line can be run without a model."""
    transcriber = FakeTranscriber()
    monkeypatch.setattr(cli, "FasterWhisperTranscriber", lambda settings: transcriber)
    return transcriber


class TestInformation:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as stop:
            cli.main(["--version"])
        assert stop.value.code == 0
        assert capsys.readouterr().out.startswith("videosubs ")

    def test_list_languages(self, capsys):
        assert cli.main(["--list-languages"]) == 0
        out = capsys.readouterr().out
        assert "Spanish" in out and "Japanese" in out and "zh-Hant" in out

    def test_help_documents_the_commands_a_person_needs(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        out = capsys.readouterr().out
        for flag in ("--languages", "--embed", "--replace", "--dry-run", "--translator", "--model", "--device"):
            assert flag in out
        assert "examples:" in out

    def test_no_arguments_asks_for_input(self, capsys):
        with pytest.raises(SystemExit) as stop:
            cli.main([])
        assert stop.value.code == 2
        assert "at least one" in capsys.readouterr().err


class TestErrors:
    def test_an_unknown_language_is_an_error_not_a_traceback(self, tmp_path, capsys, fake_model):
        video = tmp_path / "a.mkv"
        video.write_bytes(b"")
        assert cli.main([str(video), "-l", "Klingon"]) == 2
        assert "do not know the language" in capsys.readouterr().out

    def test_options_that_need_each_other(self, tmp_path, capsys, fake_model):
        video = tmp_path / "a.mkv"
        video.write_bytes(b"")
        assert cli.main([str(video), "--no-sidecar"]) == 2
        assert "--embed" in capsys.readouterr().out
        assert cli.main([str(video), "--delete-original"]) == 2
        assert cli.main([str(video), "--default-subtitle", "es"]) == 2

    def test_nothing_found(self, tmp_path, capsys, fake_model):
        assert cli.main([str(tmp_path)]) == 2
        assert "Nothing to do" in capsys.readouterr().out

    def test_a_missing_path_is_reported(self, tmp_path, capsys, fake_model):
        assert cli.main([str(tmp_path / "nope.mkv")]) == 2
        assert "no such file" in capsys.readouterr().out

    def test_nonsense_numbers(self, tmp_path, capsys, fake_model):
        video = tmp_path / "a.mkv"
        video.write_bytes(b"")
        assert cli.main([str(video), "--max-chars", "0"]) == 2
        assert cli.main([str(video), "--max-duration", "0"]) == 2


@pytest.mark.needs_ffmpeg
class TestRuns:
    def test_a_folder_run_writes_subtitles_and_succeeds(self, make_media, fake_model, capsys):
        a = make_media("shows/a.mkv")
        b = make_media("shows/b.mkv")
        assert cli.main([str(a.parent), "--no-cache"]) == 0
        assert a.with_name("a.en.srt").exists() and b.with_name("b.en.srt").exists()
        out = capsys.readouterr().out
        assert "2 files: 2 done" in out

    def test_dry_run_leaves_the_folder_alone(self, make_media, fake_model, capsys):
        a = make_media("a.mkv")
        assert cli.main([str(a), "--dry-run", "-l", "en", "es"]) == 0
        assert fake_model.calls == [] and [p.name for p in a.parent.iterdir()] == ["a.mkv"]
        assert "planned" in capsys.readouterr().out

    def test_the_exit_code_is_1_when_a_file_fails_and_the_rest_still_run(self, make_media, capsys):
        a, b = make_media("a.mkv"), make_media("b.mkv")
        broken = FakeTranscriber(fail_on={"a.mkv"})
        import videosubs.cli as module

        original = module.FasterWhisperTranscriber
        module.FasterWhisperTranscriber = lambda settings: broken
        try:
            code = module.main([str(a), str(b), "--no-cache"])
        finally:
            module.FasterWhisperTranscriber = original
        assert code == 1
        assert b.with_name("b.en.srt").exists()
        assert "Problems" in capsys.readouterr().out

    def test_the_old_flags_still_work(self, make_media, fake_model):
        a = make_media("old/a.mkv")
        # -f folder, -s languages, -du spoken language, -de device, as in the first version
        assert (
            cli.main(
                [
                    "-f",
                    str(a.parent),
                    "-s",
                    "english",
                    "spanish",
                    "-du",
                    "english",
                    "-de",
                    "cpu",
                    "--no-cache",
                    "--dry-run",
                ]
            )
            == 0
        )
        assert fake_model.calls == []

    def test_quiet_prints_only_the_summary(self, make_media, fake_model, capsys):
        a = make_media("a.mkv")
        assert cli.main([str(a), "-q", "--no-cache"]) == 0
        out = capsys.readouterr().out
        assert "a.mkv" not in out and "1 file: 1 done" in out


@pytest.mark.needs_ffmpeg
def test_file_names_with_brackets_are_shown_as_they_are(make_media, fake_model, capsys):
    """Rich reads [text] as a style tag; a release name like "Show [1080p].mkv" must survive."""
    path = make_media("Show [1080p] [ABC123].mkv")
    assert cli.main([str(path), "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert "Show [1080p] [ABC123].mkv" in out
    assert "Show [1080p] [ABC123].en.srt" in out
