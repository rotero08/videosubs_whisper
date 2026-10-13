"""Offline translation with NLLB-200 running on CTranslate2.

The model is a CTranslate2 conversion of Meta's NLLB-200 distilled (about 1.4 GB for the 1.3B
model, 0.6 GB for the 600M one), fetched once from Hugging Face. Its licence is CC-BY-NC-4.0:
free for personal and research use, not for commercial use.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from ..errors import TranslationError
from ..gpu import choose_runtime, preload_cuda_libraries
from ..languages import Language
from . import ProgressCallback

log = logging.getLogger(__name__)
_CHUNK = 32


class NllbTranslator:
    name = "nllb"

    def __init__(
        self,
        model: str = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        download_root: str | None = None,
        beam_size: int = 4,
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.beam_size = beam_size
        self._translator = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._translator is not None:
            return
        preload_cuda_libraries()
        import ctranslate2
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        path = Path(self.model)
        if not path.is_dir():
            log.info("fetching translation model %s (once; it is large)", self.model)
            try:
                path = Path(
                    snapshot_download(
                        self.model,
                        cache_dir=self.download_root,
                        allow_patterns=["*.json", "*.bin", "*.model", "*.txt"],
                    )
                )
            except Exception as error:
                raise TranslationError(f"could not download the translation model {self.model!r}: {error}") from error
        device, compute_type = choose_runtime(self.device, self.compute_type)
        try:
            self._translator = ctranslate2.Translator(str(path), device=device, compute_type=compute_type)
            self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        except Exception as error:
            raise TranslationError(f"could not load the translation model from {path}: {error}") from error

    def translate(
        self, texts: Sequence[str], source: Language, target: Language, progress: ProgressCallback | None = None
    ) -> list[str]:
        for language in (source, target):
            if not language.nllb:
                raise TranslationError(f"the offline translator does not know {language.name} ({language.code})")
        self._load()
        results = [""] * len(texts)
        # Lines with no letters in them ("♪", "...", "12") are not worth translating, and the model
        # tends to invent text for them.
        work = [i for i, text in enumerate(texts) if any(ch.isalpha() for ch in text)]
        translatable = set(work)
        for i, text in enumerate(texts):
            if i not in translatable:
                results[i] = text
        work.sort(key=lambda i: len(texts[i]))  # similar lengths batch efficiently

        done = 0
        for start in range(0, len(work), _CHUNK):
            chunk = work[start : start + _CHUNK]
            for index, translated in zip(
                chunk, self._translate_batch([texts[i] for i in chunk], source, target), strict=True
            ):
                results[index] = translated
            done += len(chunk)
            if progress:
                progress(done / max(1, len(work)))
        if progress:
            progress(1.0)
        return results

    def _translate_batch(self, texts: list[str], source: Language, target: Language) -> list[str]:
        encode = self._tokenizer.encode
        batch = [[source.nllb, *encode(text, add_special_tokens=False).tokens, "</s>"] for text in texts]
        outputs = self._translator.translate_batch(
            batch,
            target_prefix=[[target.nllb] for _ in batch],
            beam_size=self.beam_size,
            max_decoding_length=256,
            repetition_penalty=1.05,
        )
        translated = []
        for output in outputs:
            tokens = output.hypotheses[0]
            if tokens and tokens[0] == target.nllb:
                tokens = tokens[1:]
            ids = [i for token in tokens if (i := self._tokenizer.token_to_id(token)) is not None]
            translated.append(self._tokenizer.decode(ids, skip_special_tokens=True).strip())
        return translated
