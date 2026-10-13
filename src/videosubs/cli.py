"""The command line."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape

from . import __version__
from .cache import TranscriptCache, default_cache_dir
from .cues import FORMATTERS, Style
from .errors import VideoSubsError
from .languages import LANGUAGES, resolve_language, resolve_languages
from .media import discover, require_tool
from .pipeline import Options, Pipeline
from .report import RichReporter, print_summary
from .transcribe import FasterWhisperTranscriber, WhisperSettings
from .translate import BACKENDS, TranslationSettings

EXAMPLES = """\
examples:
  videosubs movie.mkv                         write movie.<language>.srt in the spoken language
  videosubs ~/Videos -l en es                 a whole folder, with English and Spanish subtitles
  videosubs ~/Videos -l en --embed            ... and also add them to a copy named *.subs.mkv
  videosubs lecture.mp4 -m small --device cpu on a computer without a GPU
  videosubs movie.en.srt -l es ja             translate an existing subtitle file
  videosubs ~/Videos --dry-run                show what would be done and change nothing
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videosubs",
        description="Subtitles for a whole video library: speech recognition with Whisper, translation "
        "into other languages, and (optionally) adding the subtitles to the video file.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*", type=Path, metavar="PATH", help="videos, folders of videos, or subtitle files"
    )
    parser.add_argument(
        "-f", "--folder", dest="folders", action="append", type=Path, default=[], help=argparse.SUPPRESS
    )
    parser.add_argument("--version", action="version", version=f"videosubs {__version__}")
    parser.add_argument("--list-languages", action="store_true", help="show the languages I know and exit")
    parser.add_argument("--clear-cache", action="store_true", help="forget saved transcripts and exit")

    lang = parser.add_argument_group("languages")
    lang.add_argument(
        "-l",
        "--languages",
        nargs="+",
        action="extend",
        default=[],
        metavar="LANG",
        help="languages to write subtitles in, by name or code (default: only the spoken language)",
    )
    lang.add_argument(
        "--source-language",
        "--from",
        dest="source",
        metavar="LANG",
        help="the language spoken in the video (default: detected)",
    )
    # The flags of the first version still work, but are not advertised.
    parser.add_argument("-s", "--subs", dest="languages", nargs="+", action="extend", help=argparse.SUPPRESS)
    parser.add_argument("-du", "--dubs", dest="source", help=argparse.SUPPRESS)

    rec = parser.add_argument_group("speech recognition")
    rec.add_argument(
        "-m",
        "--model",
        default="auto",
        help="Whisper model: tiny, base, small, medium, large-v3, large-v3-turbo, distil-large-v3 ... "
        "(default: auto, the turbo model on a GPU and small on a CPU)",
    )
    rec.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="where to run (default: auto)")
    rec.add_argument("--compute-type", default="auto", help="numeric precision, e.g. float16, int8 (default: auto)")
    rec.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="search width; higher is a little more accurate and slower (default: 5)",
    )
    rec.add_argument(
        "--batch-size", type=int, default=None, help="how many chunks to recognise at once on a GPU (default: 8)"
    )
    rec.add_argument(
        "--no-vad", action="store_true", help="do not skip silence before recognising (voice activity detection)"
    )
    rec.add_argument("--prompt", default=None, help="text that primes the recogniser: names, spellings, jargon")
    rec.add_argument(
        "--audio-track", type=int, default=None, metavar="N", help="which audio track to use, counting from 1"
    )
    rec.add_argument("--model-dir", default=None, help="where downloaded models are kept")  # fmt: skip

    tr = parser.add_argument_group("translation")
    tr.add_argument(
        "--translator",
        choices=BACKENDS,
        default="auto",
        help="how to translate (default: auto, DeepL if DEEPL_API_KEY is set, else the offline model)",
    )
    tr.add_argument(
        "--nllb-model",
        default=TranslationSettings.nllb_model,
        help="the offline translation model, a Hugging Face name or a folder "
        "(JustFrederik/nllb-200-distilled-600M-ct2-int8 is a smaller one)",
    )
    tr.add_argument(
        "--formality", choices=("default", "more", "less"), default="default", help="DeepL formal or informal address"
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--format",
        nargs="+",
        choices=sorted(FORMATTERS),
        default=["srt"],
        dest="formats",
        help="file formats (default: srt)",
    )
    out.add_argument(
        "--embed", action="store_true", help="add the subtitles to the video: writes NAME.subs.mkv beside the original"
    )
    out.add_argument("--replace", action="store_true", help="with the above, replace the original by the new file "
                     "(it is checked first, and the original is kept in a .videosubs-backup folder)")  # fmt: skip
    out.add_argument(
        "--delete-original", action="store_true", help="with --replace, delete the original instead of keeping a backup"
    )
    out.add_argument("--no-sidecar", action="store_true", help="with --embed, remove the .srt files afterwards")
    out.add_argument(
        "--default-subtitle", metavar="LANG", help="with --embed, the track players should turn on by themselves"
    )
    out.add_argument(
        "--max-chars", type=int, default=Style.max_chars, help="longest subtitle line (default: %(default)s)"
    )
    out.add_argument(
        "--max-lines", type=int, default=Style.max_lines, help="most lines in a subtitle (default: %(default)s)"
    )
    out.add_argument(
        "--max-duration",
        type=float,
        default=Style.max_seconds,
        help="most seconds a subtitle stays (default: %(default)s)",
    )

    parser.add_argument("-o", "--overwrite", dest="replace", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-d", "--delete", dest="no_sidecar", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-de", dest="device", choices=("auto", "cuda", "cpu"), help=argparse.SUPPRESS)

    run = parser.add_argument_group("running")
    run.add_argument("--force", action="store_true", help="redo work even when its result already exists")
    run.add_argument("--dry-run", action="store_true", help="say what would be done, and do nothing")
    run.add_argument("--no-recursive", action="store_true", help="do not look inside sub-folders")
    run.add_argument("--stop-on-error", action="store_true", help="stop at the first file that fails")
    run.add_argument("--no-cache", action="store_true", help="do not save or reuse transcripts")
    run.add_argument("-v", "--verbose", action="store_true", help="more detail, for troubleshooting")
    run.add_argument("-q", "--quiet", action="store_true", help="print only problems and the summary")
    return parser


def _list_languages(console: Console) -> None:
    console.print("[bold]Languages[/]  (use a name or a code; * = offered by DeepL)\n")
    for language in LANGUAGES:
        star = "*" if language.deepl else " "
        console.print(f"  {language.code:<8} {star} {language.name:<24} {language.native}")


def _options(args: argparse.Namespace, model_key: str) -> Options:
    if args.replace:
        args.embed = True
    if args.no_sidecar and not args.embed:
        raise VideoSubsError("--no-sidecar only makes sense together with --embed")
    if args.default_subtitle and not args.embed:
        raise VideoSubsError("--default-subtitle only makes sense together with --embed")
    if args.delete_original and not args.replace:
        raise VideoSubsError("--delete-original only makes sense together with --replace")
    for name in ("max_chars", "max_lines", "beam_size"):
        if getattr(args, name) < 1:
            raise VideoSubsError(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_duration <= 0:
        raise VideoSubsError("--max-duration must be above zero")

    return Options(
        languages=resolve_languages(args.languages),
        source_language=resolve_language(args.source) if args.source else None,
        formats=list(dict.fromkeys(args.formats)),
        embed=args.embed,
        replace=args.replace,
        keep_sidecars=not args.no_sidecar,
        keep_backup=not args.delete_original,
        default_subtitle=resolve_language(args.default_subtitle) if args.default_subtitle else None,
        force=args.force,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        audio_track=args.audio_track,
        style=Style(max_chars=args.max_chars, max_lines=args.max_lines, max_seconds=args.max_duration),
        translation=TranslationSettings(
            backend=args.translator,
            formality=args.formality,
            nllb_model=args.nllb_model,
            device=args.device,
            download_root=args.model_dir,
        ),
        cache=TranscriptCache(enabled=not args.no_cache),
        model_key=model_key,
    )


def _quiet_hub(verbose: bool) -> None:
    """Hugging Face prints a warning on every download when no token is set; it only matters to
    people who hit rate limits, so it is shown only with --verbose."""
    if not verbose:
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


def main(argv: list[str] | None = None) -> int:
    console = Console(highlight=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    _quiet_hub(args.verbose)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=args.verbose, markup=False)],
    )
    # --verbose is about this tool; the HTTP libraries underneath stay quiet either way.
    logging.getLogger("videosubs").setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    try:
        if args.list_languages:
            _list_languages(console)
            return 0
        if args.clear_cache:
            removed = TranscriptCache().clear()
            console.print(f"Removed {removed} saved transcript{'s' if removed != 1 else ''} from {default_cache_dir()}")
            return 0

        inputs = [*args.folders, *args.paths]
        if not inputs:
            parser.error("give me at least one video, folder or subtitle file")

        model_key = f"{args.model}|beam={args.beam_size}|vad={not args.no_vad}|prompt={args.prompt or ''}"
        options = _options(args, model_key)
        files, problems = discover(inputs, recursive=not args.no_recursive)
        for problem in problems:
            console.print(f"[yellow]note[/] {escape(problem)}")
        if not files:
            console.print("[red]Nothing to do: no videos, audio or subtitle files were found.[/]")
            return 2
        if any(f.kind != "subtitle" for f in files):
            require_tool("ffprobe")
            if options.embed:
                require_tool("ffmpeg")

        transcriber = FasterWhisperTranscriber(
            WhisperSettings(
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                vad=not args.no_vad,
                batch_size=args.batch_size,
                initial_prompt=args.prompt,
                download_root=args.model_dir,
            )
        )
        pipeline = Pipeline(options, transcriber, reporter=RichReporter(console, quiet=args.quiet))
        started = time.monotonic()
        outcomes = pipeline.run(files)
        print_summary(console, outcomes, time.monotonic() - started)
        return 1 if any(o.status == "failed" for o in outcomes) else 0
    except VideoSubsError as error:
        console.print(f"[red]Error:[/] {escape(str(error))}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped. Anything already written is complete; run again to continue.[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
