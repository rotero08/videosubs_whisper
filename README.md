# videosubs

**Subtitles for a whole video library.** Point it at a folder; it listens to every video with
[Whisper](https://github.com/openai/whisper), writes timed subtitles next to each file, translates them into
as many languages as you like, and can add them to the video itself. It runs on your own computer, keeps going
when one file fails, and never touches a video unless you ask it to.

```console
$ videosubs demo -l en es --embed
  speech model: large-v3-turbo on cuda (float16)
✓ [1/3] Show S01E01.mp4  English, Spanish
    → Show S01E01.en.srt
    → Show S01E01.es.srt
    → Show S01E01.subs.mkv
↷ [2/3] Show S01E02.mp4  already has subtitles in English, Spanish
✗ [3/3] Show S01E03.mkv  ffprobe could not read this file (Invalid data found when processing input)
Problems
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ File                 ┃ What went wrong                                                           ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ demo/Show S01E03.mkv │ ffprobe could not read this file (Invalid data found when processing      │
│                      │ input)                                                                    │
└──────────────────────┴───────────────────────────────────────────────────────────────────────────┘
3 files: 1 done, 1 skipped, 1 failed in 9s (22s of audio, 2.5× real time)
```

## What it does

For each video (or audio file) it:

1. **Listens.** Speech recognition with [faster-whisper](https://github.com/SYSTRAN/faster-whisper): the language
   is detected for you, silence is skipped, and every word gets a time.
2. **Lays out subtitles that read well.** At most two lines of 42 characters, broken at natural places, on screen
   long enough to read, never overlapping. (Japanese and Chinese use shorter lines and their own punctuation.)
3. **Translates**, one sentence at a time so that each translation lands exactly where it was spoken. Offline with
   Meta's NLLB-200 (about 200 languages, no account), or with DeepL if you have a key.
4. **Writes** `Movie.en.srt`, `Movie.es.srt` … beside the video, the names Plex, Jellyfin, VLC and mpv look for.
5. **Optionally adds them to the video** as new subtitle tracks (language-tagged, nothing else changed).

It can also translate a subtitle file you already have: `videosubs Movie.en.srt -l es ja`.

## Install

You need **Python 3.10+** and **FFmpeg** (`ffmpeg` and `ffprobe` on your PATH):

| | |
| --- | --- |
| Linux | `sudo apt install ffmpeg` or `sudo dnf install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Windows | `winget install ffmpeg` |

Then, from a clone of this repository:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install ".[gpu]"          # with an NVIDIA graphics card (installs the CUDA 12 libraries too)
pip install .                 # without one: runs on the CPU
pip install ".[gpu,deepl]"    # add DeepL translation
```

`uv tool install ".[gpu]"` or `pipx install ".[gpu]"` put the `videosubs` command on your PATH instead.
The first run downloads the models it needs (the speech model, and the translation model only if you translate)
into the Hugging Face cache:

| Model | Size | Notes |
| --- | --- | --- |
| `tiny`, `base`, `small` | 75 MB, 145 MB, 465 MB | fast; `small` is the CPU default |
| `medium` | 1.5 GB | |
| `large-v3-turbo` | 1.6 GB | **the default on a GPU**: close to large-v3, much faster |
| `large-v3` | 3 GB | the most accurate |
| NLLB-200 1.3B (translation) | 1.4 GB | the default offline translator |
| NLLB-200 600M (translation) | 0.6 GB | `--nllb-model JustFrederik/nllb-200-distilled-600M-ct2-int8` |

## Use it

```bash
videosubs movie.mkv                          # movie.en.srt, in the language spoken
videosubs ~/Videos                           # every video under it, sub-folders included
videosubs ~/Videos -l en es ja               # English, Spanish and Japanese subtitles for each
videosubs ~/Videos -l en --embed             # ... and also write movie.subs.mkv with them inside
videosubs ~/Videos -l en --embed --replace   # ... and swap it in for the original (kept in a backup folder)
videosubs lecture.mp4 -m small --device cpu  # no GPU
videosubs film.mkv --audio-track 2           # the second audio track
videosubs movie.en.srt -l es                 # translate a subtitle file
videosubs ~/Videos --dry-run                 # show what would happen and change nothing
```

A run can be repeated or resumed at any time: whatever already exists is skipped, and the slow step
(recognition) is remembered, so a failure further along never costs it twice.

### How fast is it?

Measured on a laptop with an RTX 4070 (8 GB) and 16 CPU cores, on 6 minutes 42 seconds of clean speech, model
loading included:

| Setup | Time | Speed |
| --- | --- | --- |
| GPU, `large-v3-turbo` (the default) | 7 s | 56× real time |
| CPU only, `small` | 38 s | 10.5× real time |

Real recordings with music and noise are slower and less accurate than clean speech; treat these as an upper bound.

## What you get

Subtitle files are written next to the video, named `<video name>.<language>.<format>`:

```
Movie.mkv
Movie.en.srt        English
Movie.es.srt        Spanish
Movie.subs.mkv      with --embed: the video with both added as tracks
```

* **Formats**: `--format srt vtt txt` (SRT by default; `txt` is the words only, for reading or searching).
* **Embedding** (`--embed`) keeps everything in the original (video, every audio track, existing subtitles, fonts,
  chapters, metadata) and adds one language-tagged subtitle track per language. MP4 input becomes MKV. The result
  is checked against the original (same streams, same length) before it is moved into place, and the original is
  left alone. `--default-subtitle es` marks which track players should switch on by themselves.
* **Replacing** (`--replace`) puts that checked file where the original was. The original is moved to a
  `.videosubs-backup` folder beside it, never deleted (`--delete-original` if you really want that).
* **Nothing is overwritten**: an existing subtitle file you edited is kept (`--force` to redo). Files are written
  under a temporary name and moved into place, so an interrupted run leaves no half-written files.

## Languages and translation

Name a language however you like: `es`, `spa`, `Spanish`, `español`, `pt-BR` all work (`--list-languages` shows
what is known; Whisper itself hears about a hundred). `--source-language` tells it what is spoken when detection
guesses wrong.

| `--translator` | Quality | Needs | Notes |
| --- | --- | --- | --- |
| `nllb` | good | nothing, offline | About 200 languages. The model is licensed **CC-BY-NC-4.0: personal and research use, not commercial.** |
| `deepl` | best for the languages it covers | an API key in `DEEPL_API_KEY` | Free keys work. `--formality more/less`. |
| `whisper` | good | nothing | Whisper's own translation, **into English only**, straight from the audio. |
| `auto` (default) | | | DeepL when a key is set and it offers the language, otherwise `nllb`. |

## All options

| Option | What it does |
| --- | --- |
| `-l, --languages LANG...` | Languages to write (default: only the spoken one) |
| `--source-language LANG` | The spoken language (default: detected) |
| `-m, --model NAME` | `tiny` `base` `small` `medium` `large-v3` `large-v3-turbo` `distil-large-v3` … (default `auto`) |
| `--device auto\|cuda\|cpu`, `--compute-type` | Where and at what precision to run (default: the GPU if there is one) |
| `--beam-size`, `--batch-size` | Accuracy against speed; batching applies on a GPU |
| `--no-vad` | Do not skip silences before recognising |
| `--prompt TEXT` | Prime the recogniser with names, spellings or jargon |
| `--audio-track N` | Which audio track to use, counting from 1 (default: the one tagged with the spoken language, else the default track) |
| `--translator`, `--nllb-model`, `--formality` | See above |
| `--format srt vtt txt` | File formats |
| `--embed`, `--replace`, `--delete-original`, `--no-sidecar`, `--default-subtitle` | Adding to the video, see above (`--no-sidecar` removes the `.srt` files afterwards) |
| `--max-chars`, `--max-lines`, `--max-duration` | Subtitle layout (42, 2, 7 s) |
| `--force`, `--dry-run`, `--no-recursive`, `--stop-on-error` | Running |
| `--no-cache`, `--clear-cache` | The saved transcripts (kept in your user cache folder) |
| `-v`, `-q` | More detail, or only problems and the summary |

Exit status: `0` when everything went well or was skipped, `1` when some file failed (the others still ran),
`2` for a usage or setup problem, `130` when interrupted.

## Troubleshooting

* **"ffprobe is not installed"** – install FFmpeg (above).
* **The GPU is not used** – `pip install ".[gpu]"` adds the CUDA 12 libraries, and `nvidia-smi` must show your card.
  Without them the tool falls back to the CPU and tells you so. `--device cuda` fails loudly instead.
* **Out of GPU memory** – it retries by itself with a leaner precision; otherwise try `-m small` or `--compute-type int8_float16`.
* **Wrong language, or text in the wrong language** – pass `--source-language`. A video that opens with a long
  silence or music can fool detection.
* **Repeated or invented lines** – `--prompt` with the names that appear usually helps; recognition already skips silence
  and does not carry text from one window to the next, the usual cause of loops.
* **`TypeError: open() got an unexpected keyword argument 'metadata_errors'`** – PyAV 19 broke faster-whisper; this
  project pins `av<19`, so it only appears if you installed the pieces by hand.
* **A file is skipped as "already has…"** – a subtitle file for it exists. `--force` redoes it.

## Coming from the first version

The command is now `videosubs` (or `python -m videosubs`), and the old flags still work: `-f` folder, `-s/--subs`
languages, `-du/--dubs` spoken language, `-de` device, `-d/--delete` (now `--no-sidecar`) and `-o/--overwrite`
(now `--replace`). What changed underneath:

* Recognition moved from `openai-whisper` to `faster-whisper` (CTranslate2), which its authors report as up to four times
  faster with less memory; on a GPU the work is also batched.
* The model is loaded once for the whole batch, not once per file, and audio is read directly instead of being
  converted to MP3 first.
* Translation no longer drives a web browser through Tor to use DeepL's website. That depended on one Windows
  machine's paths and on the page's markup, and a website is not what a service offers to programs; the official
  DeepL API is. It is replaced by that API, an offline model, or Whisper's own English translation.
* The first version could delete the original video before checking that the new one existed. Now the result is
  verified first and the original is moved aside, not deleted.
* The default options of the first version were broken (a string was treated as a list, so `-m` defaulted to the model
  `"m"`); every default is tested now.

## Development

```bash
pip install -e ".[dev]"
pytest                       # 190 tests; those that need ffmpeg make small real media files
ruff check . && ruff format --check .
VIDEOSUBS_E2E=1 pytest -m slow   # also run real speech through the real model (needs espeak-ng)
```

```
src/videosubs/
  cli.py          the command line                 pipeline.py   decides what to do, per file
  cues.py         subtitle layout and file formats media.py      file discovery, ffprobe, audio tracks
  languages.py    names, ISO codes, NLLB/DeepL codes mux.py        adding tracks, verifying, replacing
  transcribe.py   faster-whisper                   translate/    DeepL and NLLB backends
  gpu.py          device and CUDA library setup    cache.py      saved transcripts
```

Credits: [Whisper](https://github.com/openai/whisper) (OpenAI), [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and [CTranslate2](https://github.com/OpenNMT/CTranslate2), [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb)
(Meta), [FFmpeg](https://ffmpeg.org/), [Rich](https://github.com/Textualize/rich).
