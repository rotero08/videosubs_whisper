"""Choosing where speech recognition runs, and making an NVIDIA GPU usable without a system CUDA.

CTranslate2 (the engine under faster-whisper) needs the CUDA 12 runtime libraries cuBLAS and
cuDNN 9. `pip install videosubs[gpu]` installs them as Python packages, but the dynamic loader
does not look inside site-packages, so they are loaded explicitly, before CTranslate2 first
needs them.
"""

from __future__ import annotations

import ctypes
import glob
import os
import site
import sys
from functools import cache

from .errors import VideoSubsError

_LIBRARIES = (
    ("nvidia", "cublas", "lib", "libcublas.so*"),
    ("nvidia", "cudnn", "lib", "libcudnn*.so*"),
)


@cache
def preload_cuda_libraries() -> tuple[str, ...]:
    """Load cuBLAS and cuDNN from pip packages if present. Returns what was loaded."""
    loaded: list[str] = []
    if sys.platform.startswith("win"):
        for root in site.getsitepackages():
            for package in ("cublas", "cudnn"):
                folder = os.path.join(root, "nvidia", package, "bin")
                if os.path.isdir(folder):
                    os.add_dll_directory(folder)
                    loaded.append(folder)
        return tuple(loaded)
    for root in [*site.getsitepackages(), site.getusersitepackages()]:
        for *folders, pattern in _LIBRARIES:
            for library in sorted(glob.glob(os.path.join(root, *folders, pattern))):
                try:
                    ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(library)
                except OSError:
                    continue
    return tuple(loaded)


def cuda_device_count() -> int:
    preload_cuda_libraries()
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # no driver, no GPU, or a CTranslate2 built without CUDA
        return 0


def supported_compute_types(device: str) -> set[str]:
    import ctranslate2

    return set(ctranslate2.get_supported_compute_types(device))


def choose_runtime(device: str = "auto", compute_type: str = "auto") -> tuple[str, str]:
    """Resolve "auto" to a device and a numeric precision that this machine supports."""
    if device not in ("auto", "cuda", "cpu"):
        raise VideoSubsError(f"unknown device {device!r}: use auto, cuda or cpu")
    has_gpu = cuda_device_count() > 0
    if device == "cuda" and not has_gpu:
        raise VideoSubsError(
            "No usable NVIDIA GPU was found. Install the CUDA libraries with `pip install "
            "'videosubs[gpu]'` and check `nvidia-smi`, or use --device cpu."
        )
    resolved = "cuda" if device == "cuda" or (device == "auto" and has_gpu) else "cpu"

    if compute_type != "auto":
        return resolved, compute_type
    if resolved == "cpu":
        return "cpu", "int8"
    available = supported_compute_types("cuda")
    for candidate in ("float16", "int8_float16", "float32"):
        if candidate in available:
            return "cuda", candidate
    return "cuda", "float32"


def default_model(device: str) -> str:
    """Large-v3-turbo is close to large-v3 in accuracy at a fraction of the cost on a GPU; on a
    CPU a smaller model keeps a whole library within reach."""
    return "large-v3-turbo" if device == "cuda" else "small"
