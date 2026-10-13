import os

import pytest

from conftest import FakeTranscriber
from videosubs import gpu
from videosubs.cache import TranscriptCache
from videosubs.errors import VideoSubsError
from videosubs.transcribe import Transcript


def sample() -> Transcript:
    return FakeTranscriber().transcribe("x.wav", language=None)


class TestCache:
    def test_a_transcript_survives_a_round_trip(self, tmp_path):
        media = tmp_path / "movie.mkv"
        media.write_bytes(b"data")
        cache = TranscriptCache(tmp_path / "cache")
        key = cache.key(media, model="small")
        assert cache.get(key) is None
        original = sample()
        cache.put(key, original)
        restored = cache.get(key)
        assert restored.language == original.language
        assert restored.segments == original.segments  # including every word timing

    def test_the_key_changes_with_the_file_and_the_settings(self, tmp_path):
        media = tmp_path / "movie.mkv"
        media.write_bytes(b"data")
        cache = TranscriptCache(tmp_path / "cache")
        first = cache.key(media, model="small")
        assert cache.key(media, model="large") != first
        assert cache.key(media, model="small", task="translate") != first
        media.write_bytes(b"different and longer")
        os.utime(media, (1, 1))
        assert cache.key(media, model="small") != first

    def test_a_disabled_cache_stores_and_returns_nothing(self, tmp_path):
        cache = TranscriptCache(tmp_path / "cache", enabled=False)
        cache.put("k", sample())
        assert cache.get("k") is None
        assert not (tmp_path / "cache").exists()

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        cache = TranscriptCache(tmp_path)
        (tmp_path / "bad.json").write_text("{not json")
        assert cache.get("bad") is None

    def test_clear_removes_entries(self, tmp_path):
        cache = TranscriptCache(tmp_path)
        cache.put("a", sample())
        cache.put("b", sample())
        assert cache.clear() == 2
        assert cache.get("a") is None


class TestRuntimeChoice:
    def test_auto_picks_the_gpu_and_half_precision_when_there_is_one(self, monkeypatch):
        monkeypatch.setattr(gpu, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(gpu, "supported_compute_types", lambda device: {"float16", "int8", "float32"})
        assert gpu.choose_runtime() == ("cuda", "float16")

    def test_a_gpu_without_float16_falls_back(self, monkeypatch):
        monkeypatch.setattr(gpu, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(gpu, "supported_compute_types", lambda device: {"int8_float16", "float32"})
        assert gpu.choose_runtime() == ("cuda", "int8_float16")

    def test_auto_uses_the_cpu_with_int8_when_there_is_no_gpu(self, monkeypatch):
        monkeypatch.setattr(gpu, "cuda_device_count", lambda: 0)
        assert gpu.choose_runtime() == ("cpu", "int8")

    def test_asking_for_a_gpu_that_is_not_there_explains_what_to_do(self, monkeypatch):
        monkeypatch.setattr(gpu, "cuda_device_count", lambda: 0)
        with pytest.raises(VideoSubsError, match=r"videosubs\[gpu\]"):
            gpu.choose_runtime("cuda")

    def test_an_explicit_precision_is_respected(self, monkeypatch):
        monkeypatch.setattr(gpu, "cuda_device_count", lambda: 0)
        assert gpu.choose_runtime("cpu", "float32") == ("cpu", "float32")

    def test_unknown_devices_are_refused(self):
        with pytest.raises(VideoSubsError, match="unknown device"):
            gpu.choose_runtime("tpu")

    def test_default_model_depends_on_the_device(self):
        assert gpu.default_model("cuda") == "large-v3-turbo"
        assert gpu.default_model("cpu") == "small"
