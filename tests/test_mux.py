import pytest

from conftest import probe_streams
from videosubs.errors import MuxError
from videosubs.languages import find_language
from videosubs.media import MediaInfo, StreamInfo, parse_probe, probe
from videosubs.mux import SubtitleTrack, build_command, mux, output_path, replace_original, verify

needs_ffmpeg = pytest.mark.needs_ffmpeg
ES, JA = find_language("es"), find_language("ja")


def info_with(tmp_path, *, subs=()):
    streams = [{"codec_type": "video", "codec_name": "h264"}, {"codec_type": "audio", "codec_name": "aac"}]
    streams += [{"codec_type": "subtitle", "codec_name": codec} for codec in subs]
    return parse_probe(tmp_path / "movie.mp4", {"format": {"duration": "60"}, "streams": streams})


class TestCommand:
    def test_keeps_everything_and_adds_each_track_with_language_and_title(self, tmp_path):
        info = info_with(tmp_path)
        tracks = [SubtitleTrack(tmp_path / "m.es.srt", ES), SubtitleTrack(tmp_path / "m.ja.srt", JA)]
        cmd = build_command("ffmpeg", tmp_path / "movie.mp4", info, tracks, tmp_path / "out.part")
        text = " ".join(cmd)
        assert "-map 0:v? -map 0:a? -map 0:s? -map 0:t? -map 1:0 -map 2:0" in text
        assert "-map_chapters 0" in text and "-c copy" in text
        assert "-metadata:s:s:0 language=spa" in text and "-metadata:s:s:0 title=Spanish" in text
        assert "-metadata:s:s:1 language=jpn" in text
        assert "-c:s:0 srt" in text and "-c:s:1 srt" in text
        assert cmd[-3:] == ["-f", "matroska", str(tmp_path / "out.part")]

    def test_new_tracks_come_after_existing_subtitles(self, tmp_path):
        info = info_with(tmp_path, subs=("ass", "mov_text"))
        cmd = " ".join(
            build_command("ffmpeg", tmp_path / "m.mp4", info, [SubtitleTrack(tmp_path / "a.srt", ES)], tmp_path / "o")
        )
        assert "-metadata:s:s:2 language=spa" in cmd
        assert "-c:s:1 srt" in cmd  # the existing mov_text track cannot be copied into Matroska
        assert "-c:s:0" not in cmd  # an existing ASS track is copied untouched, styling and all

    def test_default_subtitle_flags_exactly_one_track(self, tmp_path):
        info = info_with(tmp_path)
        tracks = [SubtitleTrack(tmp_path / "a.srt", ES), SubtitleTrack(tmp_path / "b.srt", JA)]
        cmd = " ".join(build_command("ffmpeg", tmp_path / "m.mp4", info, tracks, tmp_path / "o", default_language=JA))
        assert "-disposition:s:0 0" in cmd and "-disposition:s:1 default" in cmd

    def test_output_name(self, tmp_path):
        assert output_path(tmp_path / "movie.mp4").name == "movie.subs.mkv"


class TestVerify:
    def make(self, *, video=1, audio=1, subs=0, duration=60.0):
        streams = (
            [StreamInfo(i, "video", "h264", None, None, None, False) for i in range(video)]
            + [StreamInfo(10 + i, "audio", "aac", None, None, 2, False) for i in range(audio)]
            + [StreamInfo(20 + i, "subtitle", "srt", None, None, None, False) for i in range(subs)]
        )
        return MediaInfo(path=None, duration=duration, streams=tuple(streams))

    def test_accepts_a_faithful_copy(self):
        verify(self.make(), self.make(subs=2, duration=60.4), added=2)

    def test_rejects_lost_audio_or_video_or_subtitles(self):
        with pytest.raises(MuxError, match="audio: expected 2"):
            verify(self.make(audio=2), self.make(audio=1, subs=1), added=1)
        with pytest.raises(MuxError, match="subtitle: expected 1"):
            verify(self.make(), self.make(subs=0), added=1)

    def test_rejects_a_truncated_file(self):
        with pytest.raises(MuxError, match="duration"):
            verify(self.make(duration=3600), self.make(subs=1, duration=1800), added=1)


@needs_ffmpeg
class TestRealMux:
    def srt(self, tmp_path, name="movie.es.srt"):
        path = tmp_path / name
        path.write_text("1\n00:00:00,500 --> 00:00:02,000\nHola\n", encoding="utf-8")
        return path

    def test_adds_a_subtitle_track_and_keeps_every_original_stream(self, make_media, tmp_path):
        video = make_media("movie.mkv", audio=("eng", "jpn"), subtitles=("eng",), seconds=3)
        info = probe(video)
        out = mux(video, info, [SubtitleTrack(self.srt(tmp_path), ES)], output_path(video), default_language=ES)
        streams = probe_streams(out)
        assert len(streams["video"]) == 1
        assert streams["audio"] == ["eng", "jpn"]
        assert streams["subtitle"] == ["eng", "spa"]  # the original one first, ours after it
        assert video.exists()  # the original is untouched
        assert not out.with_name(out.name + ".part").exists()

    def test_works_from_an_mp4(self, make_media, tmp_path):
        video = make_media("clip.mp4", audio=("eng",), seconds=2)
        out = mux(video, probe(video), [SubtitleTrack(self.srt(tmp_path), ES)], output_path(video))
        assert out.suffix == ".mkv"
        assert probe_streams(out)["subtitle"] == ["spa"]

    def test_a_failure_leaves_no_partial_file(self, make_media, tmp_path):
        video = make_media("movie.mkv", seconds=2)
        broken = tmp_path / "broken.srt"  # does not exist, so ffmpeg fails
        destination = output_path(video)
        with pytest.raises(MuxError):
            mux(video, probe(video), [SubtitleTrack(broken, ES)], destination)
        assert not destination.exists()
        assert not destination.with_name(destination.name + ".part").exists()

    def test_nothing_to_add_is_an_error(self, make_media):
        video = make_media("movie.mkv", seconds=2)
        with pytest.raises(MuxError, match="no subtitle tracks"):
            mux(video, probe(video), [], output_path(video))

    def test_reports_progress(self, make_media, tmp_path):
        video = make_media("movie.mkv", seconds=3)
        seen = []
        mux(video, probe(video), [SubtitleTrack(self.srt(tmp_path), ES)], output_path(video), on_progress=seen.append)
        assert seen and seen[-1] == 1.0


class TestReplaceOriginal:
    def test_the_original_moves_to_a_backup_folder(self, tmp_path):
        original = tmp_path / "movie.mp4"
        original.write_bytes(b"old")
        muxed = tmp_path / "movie.subs.mkv"
        muxed.write_bytes(b"new")
        final = replace_original(original, muxed)
        assert final == tmp_path / "movie.mkv" and final.read_bytes() == b"new"
        assert (tmp_path / ".videosubs-backup" / "movie.mp4").read_bytes() == b"old"
        assert not original.exists() and not muxed.exists()

    def test_a_second_backup_of_the_same_name_does_not_overwrite_the_first(self, tmp_path):
        folder = tmp_path / ".videosubs-backup"
        folder.mkdir()
        (folder / "movie.mkv").write_bytes(b"first")
        original = tmp_path / "movie.mkv"
        original.write_bytes(b"second")
        muxed = tmp_path / "movie.subs.mkv"
        muxed.write_bytes(b"new")
        replace_original(original, muxed)
        assert (folder / "movie.mkv").read_bytes() == b"first"
        assert len(list(folder.iterdir())) == 2

    def test_can_delete_instead_of_keeping_a_backup(self, tmp_path):
        original = tmp_path / "movie.mkv"
        original.write_bytes(b"old")
        muxed = tmp_path / "movie.subs.mkv"
        muxed.write_bytes(b"new")
        replace_original(original, muxed, keep_backup=False)
        assert not (tmp_path / ".videosubs-backup").exists()
        assert original.read_bytes() == b"new"

    def test_refuses_to_clobber_an_unrelated_file(self, tmp_path):
        original = tmp_path / "movie.mp4"
        original.write_bytes(b"old")
        other = tmp_path / "movie.mkv"
        other.write_bytes(b"someone else's")
        muxed = tmp_path / "movie.subs.mkv"
        muxed.write_bytes(b"new")
        with pytest.raises(MuxError, match="already exists"):
            replace_original(original, muxed)
        assert original.read_bytes() == b"old" and other.read_bytes() == b"someone else's"
