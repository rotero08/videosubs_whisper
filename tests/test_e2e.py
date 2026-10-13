"""A real run with nothing faked: synthesised speech, the real model, the real muxer.

It downloads a model (the 75 MB "tiny" one) and needs espeak-ng to make speech, so it runs only
when asked:   VIDEOSUBS_E2E=1 pytest -m slow
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from conftest import probe_streams
from videosubs import cli

ESPEAK = shutil.which("espeak-ng") or shutil.which("espeak")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.needs_ffmpeg,
    pytest.mark.skipif(not os.environ.get("VIDEOSUBS_E2E"), reason="set VIDEOSUBS_E2E=1 to run the real-model test"),
    pytest.mark.skipif(not ESPEAK, reason="espeak-ng is needed to synthesise speech"),
]

SPEECH = "Welcome to this short demonstration. Today we will test a tool that writes subtitles automatically."


def test_synthesised_speech_becomes_embedded_subtitles(tmp_path, capsys):
    wav = tmp_path / "speech.wav"
    subprocess.run([ESPEAK, "-v", "en-us", "-s", "140", "-w", str(wav), SPEECH], check=True)
    video = tmp_path / "Demo [e2e].mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x203040:s=160x90:r=10",
         "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "aac", "-shortest",
         "-metadata:s:a:0", "language=eng", str(video)],
        check=True,
        capture_output=True,
    )  # fmt: skip

    code = cli.main([str(video), "-m", "tiny", "--device", "cpu", "--no-cache", "--embed"])
    assert code == 0, capsys.readouterr().out

    heard = (tmp_path / "Demo [e2e].en.srt").read_text(encoding="utf-8").lower()
    keywords = ["welcome", "short", "demonstration", "today", "tool", "subtitles"]
    assert sum(word in heard for word in keywords) >= 3, heard
    assert probe_streams(tmp_path / "Demo [e2e].subs.mkv")["subtitle"] == ["eng"]
