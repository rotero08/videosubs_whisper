"""What the person running the tool sees: progress while it works and a summary at the end."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .media import MediaFile
from .pipeline import Outcome

_ICONS = {"done": "[green]✓[/]", "skipped": "[cyan]↷[/]", "failed": "[red]✗[/]", "planned": "[blue]•[/]"}


def clock(seconds: float) -> str:
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class RichReporter:
    """Implements the pipeline's Reporter with a live progress bar for the file in hand."""

    def __init__(self, console: Console, *, quiet: bool = False) -> None:
        self.console = console
        self.quiet = quiet
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=28),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            disable=quiet,
        )
        self._task = None
        self._label = ""

    def file_started(self, index: int, total: int, media: MediaFile) -> None:
        # File names are data, not markup: "Show [1080p].mkv" must not be read as a style tag.
        self._label = escape(f"[{index}/{total}] {media.path.name}")
        if not self.quiet:
            self._progress.start()
        self._task = self._progress.add_task(self._label, total=100, completed=0)

    def stage(self, name: str) -> None:
        if self._task is not None:
            self._progress.update(self._task, description=f"{self._label}  [dim]{name}[/]", completed=0)

    def progress(self, fraction: float) -> None:
        if self._task is not None:
            self._progress.update(self._task, completed=max(0.0, min(1.0, fraction)) * 100)

    def note(self, message: str) -> None:
        self._progress.console.print(f"  [yellow]note[/] {escape(message)}")

    def info(self, message: str) -> None:
        if not self.quiet:
            self._progress.console.print(f"  [dim]{escape(message)}[/]")

    def file_finished(self, outcome: Outcome) -> None:
        if self._task is not None:
            self._progress.remove_task(self._task)
            self._task = None
        self._progress.stop()
        if self.quiet and outcome.status != "failed":
            return
        detail = f"[dim]{escape(outcome.message)}[/]" if outcome.message else ""
        self.console.print(f"{_ICONS[outcome.status]} {self._label}  {detail}")
        for path in outcome.outputs:
            self.console.print(f"    [dim]→ {escape(path.name)}[/]")


def print_summary(console: Console, outcomes: Sequence[Outcome], elapsed: float) -> None:
    if not outcomes:
        return
    counts = {status: sum(o.status == status for o in outcomes) for status in ("done", "skipped", "failed", "planned")}
    failed = [o for o in outcomes if o.status == "failed"]

    if failed:
        table = Table(title="Problems", title_justify="left", show_lines=False, header_style="bold")
        table.add_column("File", overflow="fold")
        table.add_column("What went wrong", overflow="fold")
        for outcome in failed:
            table.add_row(escape(str(outcome.media.path)), escape(outcome.message))
        console.print(table)

    audio = sum(o.audio_seconds for o in outcomes if o.status == "done")
    parts = [f"{counts[s]} {s}" for s in ("done", "planned", "skipped", "failed") if counts[s]]
    line = f"{len(outcomes)} file{'s' if len(outcomes) != 1 else ''}: {', '.join(parts)}"
    if elapsed >= 1:
        line += f" in {clock(elapsed)}"
    if audio and elapsed >= 1:
        line += f" ({clock(audio)} of audio, {audio / elapsed:.1f}× real time)"
    console.print(f"\n[bold]{line}[/]")
