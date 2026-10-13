"""Errors the tool raises on purpose, each with a message meant to be read by a person."""


class VideoSubsError(Exception):
    """Base class: the command line prints the message and exits without a traceback."""


class UnknownLanguageError(VideoSubsError):
    """A language name or code that the tool does not know."""


class MediaError(VideoSubsError):
    """A file that cannot be read as media, or has nothing to transcribe."""


class ToolMissingError(VideoSubsError):
    """ffmpeg or ffprobe is not installed."""


class TranslationError(VideoSubsError):
    """A translation backend failed or cannot handle the requested languages."""


class MuxError(VideoSubsError):
    """Writing or verifying the output video failed."""
