from pathlib import Path

import pytest

from conftest import probe_streams
from videosubs.errors import MediaError
from videosubs.languages import find_language
from videosubs.media import (
    choose_audio_stream,
    discover,
    extract_audio,
    find_sidecars,
    is_own_output,
    kind_of,
    parse_probe,
    probe,
    sidecar_path,
)

needs_ffmpeg = pytest.mark.needs_ffmpeg


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class TestDiscover:
    def test_finds_videos_in_natural_order_and_ignores_other_files(self, tmp_path):
        for name in ["ep 10.mkv", "ep 2.mkv", "ep 1.mp4", "notes.txt", "cover.jpg"]:
            touch(tmp_path / name)
        files, problems = discover([tmp_path])
        assert [f.path.name for f in files] == ["ep 1.mp4", "ep 2.mkv", "ep 10.mkv"]
        assert problems == []

    def test_recurses_by_default_and_can_be_told_not_to(self, tmp_path):
        touch(tmp_path / "a.mkv")
        touch(tmp_path / "sub" / "b.mkv")
        assert len(discover([tmp_path])[0]) == 2
        assert len(discover([tmp_path], recursive=False)[0]) == 1

    def test_skips_its_own_outputs_hidden_folders_and_the_backup_folder(self, tmp_path):
        touch(tmp_path / "movie.mkv")
        touch(tmp_path / "movie.subs.mkv")
        touch(tmp_path / ".videosubs-backup" / "movie.mkv")
        touch(tmp_path / ".hidden" / "x.mkv")
        assert [f.path.name for f in discover([tmp_path])[0]] == ["movie.mkv"]

    def test_a_folder_does_not_yield_subtitle_files_but_naming_one_does(self, tmp_path):
        touch(tmp_path / "movie.mkv")
        srt = touch(tmp_path / "movie.en.srt")
        assert [f.kind for f in discover([tmp_path])[0]] == ["video"]
        assert [f.kind for f in discover([srt])[0]] == ["subtitle"]

    def test_the_same_file_named_twice_counts_once(self, tmp_path):
        video = touch(tmp_path / "a.mkv")
        assert len(discover([video, tmp_path])[0]) == 1

    def test_reports_things_that_cannot_be_used(self, tmp_path):
        text = touch(tmp_path / "readme.txt")
        own = touch(tmp_path / "movie.subs.mkv")
        _, problems = discover([text, own, tmp_path / "missing.mkv"])
        assert len(problems) == 3
        assert any("no such file" in p for p in problems)

    def test_kinds(self, tmp_path):
        assert kind_of(tmp_path / "A.MKV") == "video"
        assert kind_of(tmp_path / "a.flac") == "audio"
        assert kind_of(tmp_path / "a.srt") == "subtitle"
        assert kind_of(tmp_path / "a.txt") is None
        assert is_own_output(tmp_path / "x.subs.mkv")
        assert not is_own_output(tmp_path / "x.mkv")


class TestSidecars:
    def test_names_follow_the_player_convention(self, tmp_path):
        assert sidecar_path(tmp_path / "movie.mkv", find_language("es")).name == "movie.es.srt"
        assert sidecar_path(tmp_path / "movie.mkv", find_language("zh-TW"), "vtt").name == "movie.zh-Hant.vtt"

    def test_finds_existing_ones_even_when_the_name_is_full_of_glob_characters(self, tmp_path):
        video = touch(tmp_path / "[Group] Show - 01 [1080p].mkv")
        touch(tmp_path / "[Group] Show - 01 [1080p].en.srt")
        touch(tmp_path / "[Group] Show - 01 [1080p].notalanguage.srt")
        touch(tmp_path / "[Group] Show - 02 [1080p].en.srt")
        assert [p.name for p in find_sidecars(video)] == ["[Group] Show - 01 [1080p].en.srt"]

    def test_a_missing_folder_is_not_an_error(self, tmp_path):
        assert find_sidecars(tmp_path / "gone" / "x.mkv") == []


PROBE_DATA = {
    "format": {"duration": "120.5"},
    "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 2, "tags": {"language": "jpn", "TITLE": "Original"}, "disposition": {"default": 1}},
        {"index": 2, "codec_type": "audio", "codec_name": "ac3", "tags": {"language": "und"}},
        {"index": 3, "codec_type": "subtitle", "codec_name": "ass", "tags": {"language": "eng"}},
    ],
}  # fmt: skip


class TestProbeParsing:
    def test_reads_streams_tags_and_duration(self, tmp_path):
        info = parse_probe(tmp_path / "x.mkv", PROBE_DATA)
        assert info.duration == 120.5
        assert info.has_video
        assert [s.language for s in info.audio] == ["jpn", None]
        assert info.audio[0].title == "Original"
        assert info.audio[0].default
        assert info.subtitles[0].codec == "ass"

    def test_missing_duration_falls_back_to_a_stream(self, tmp_path):
        info = parse_probe(tmp_path / "x.mkv", {"streams": [{"codec_type": "audio", "duration": "7.5"}]})
        assert info.duration == 7.5

    def test_chooses_the_track_in_the_spoken_language_then_the_default_then_the_first(self, tmp_path):
        info = parse_probe(tmp_path / "x.mkv", PROBE_DATA)
        assert choose_audio_stream(info) == 0  # the default
        assert choose_audio_stream(info, language=find_language("ja")) == 0
        assert choose_audio_stream(info, track=2) == 1
        plain = parse_probe(tmp_path / "x.mkv", {"streams": [{"codec_type": "audio"}, {"codec_type": "audio"}]})
        assert choose_audio_stream(plain) == 0

    def test_a_track_tagged_with_the_language_beats_the_default(self, tmp_path):
        data = {"streams": [
            {"codec_type": "audio", "tags": {"language": "eng"}, "disposition": {"default": 1}},
            {"codec_type": "audio", "tags": {"language": "jpn"}},
        ]}  # fmt: skip
        info = parse_probe(tmp_path / "x.mkv", data)
        assert choose_audio_stream(info, language=find_language("ja")) == 1

    def test_errors_are_clear(self, tmp_path):
        silent = parse_probe(tmp_path / "x.mkv", {"streams": [{"codec_type": "video"}]})
        with pytest.raises(MediaError, match="no audio track"):
            choose_audio_stream(silent)
        info = parse_probe(tmp_path / "x.mkv", PROBE_DATA)
        with pytest.raises(MediaError, match="does not exist"):
            choose_audio_stream(info, track=5)


@needs_ffmpeg
class TestWithRealFiles:
    def test_probe_reads_a_real_file(self, make_media):
        path = make_media("a.mkv", audio=("eng", "jpn"), seconds=2)
        info = probe(path)
        assert info.has_video
        assert [s.language for s in info.audio] == ["eng", "jpn"]
        assert 1.5 < info.duration < 2.6

    def test_probe_rejects_junk(self, tmp_path):
        junk = tmp_path / "junk.mkv"
        junk.write_text("this is not a video")
        with pytest.raises(MediaError, match="could not read"):
            probe(junk)

    def test_extracts_the_chosen_track_as_16k_mono_wav(self, make_media, tmp_path):
        path = make_media("a.mkv", audio=("eng", "jpn"), seconds=2)
        out = extract_audio(path, 1, tmp_path / "out.wav")
        streams = probe(out)
        assert streams.audio[0].codec == "pcm_s16le"
        assert streams.audio[0].channels == 1
        assert probe_streams(out)["audio"] == [None]


@needs_ffmpeg
def test_a_relative_file_name_starting_with_a_dash_is_not_taken_for_an_option(make_media, monkeypatch):
    path = make_media("-dash.mkv", seconds=2)
    monkeypatch.chdir(path.parent)
    assert probe(Path("-dash.mkv")).has_video
