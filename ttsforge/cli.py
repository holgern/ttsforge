"""CLI interface for ttsforge - convert EPUB to audiobooks."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.prompt import Confirm, IntPrompt
from rich import print as rprint

from .constants import (
    DEFAULT_CONFIG,
    LANGUAGE_DESCRIPTIONS,
    PROGRAM_DESCRIPTION,
    PROGRAM_NAME,
    SUPPORTED_OUTPUT_FORMATS,
    VOICES,
    VOICE_PREFIX_TO_LANG,
    DEFAULT_VOICE_FOR_LANG,
)
from .conversion import (
    ConversionOptions,
    ConversionProgress,
    SPLIT_MODES,
    TTSConverter,
    detect_language_from_iso,
    get_default_voice_for_language,
)
from .utils import (
    format_duration,
    format_size,
    get_gpu_info,
    load_config,
    reset_config,
    save_config,
)

console = Console()


def get_version() -> str:
    """Get the package version."""
    try:
        from ._version import version

        return version
    except ImportError:
        return "0.0.0+unknown"


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit.")
@click.pass_context
def main(ctx: click.Context, version: bool) -> None:
    """ttsforge - Generate audiobooks from EPUB files with TTS."""
    if version:
        console.print(f"[bold]{PROGRAM_NAME}[/bold] version {get_version()}")
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("epub_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file path. Defaults to input filename with new extension.",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(SUPPORTED_OUTPUT_FORMATS),
    help="Output audio format.",
)
@click.option(
    "-v",
    "--voice",
    type=click.Choice(VOICES),
    help="Voice to use for TTS.",
)
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    help="Language code (a=American English, b=British English, etc.).",
)
@click.option(
    "-s",
    "--speed",
    type=float,
    help="Speech speed (0.5 to 2.0).",
)
@click.option(
    "--gpu/--no-gpu",
    "use_gpu",
    default=None,
    help="Enable/disable GPU acceleration.",
)
@click.option(
    "--chapters",
    type=str,
    help="Chapters to convert (e.g., '1-5', '1,3,5', 'all'). Interactive if not specified.",
)
@click.option(
    "--silence",
    type=float,
    help="Silence duration between chapters in seconds.",
)
@click.option(
    "--title",
    type=str,
    help="Title metadata for the audiobook.",
)
@click.option(
    "--author",
    type=str,
    help="Author metadata for the audiobook.",
)
@click.option(
    "--cover",
    type=click.Path(exists=True, path_type=Path),
    help="Cover image for m4b format.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Skip confirmation prompts.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed output.",
)
@click.option(
    "--split-mode",
    "split_mode",
    type=click.Choice(["auto", "line", "paragraph", "sentence", "clause"]),
    default=None,
    help="Text splitting mode: auto (language-based), line (newlines), paragraph, sentence (spaCy), clause (sentence+comma).",
)
@click.option(
    "--resume/--no-resume",
    "resume",
    default=True,
    help="Enable/disable resume capability for interrupted conversions (default: enabled).",
)
@click.option(
    "--keep-chapters",
    "keep_chapter_files",
    is_flag=True,
    help="Keep individual chapter audio files after conversion.",
)
def convert(
    epub_file: Path,
    output: Optional[Path],
    output_format: Optional[str],
    voice: Optional[str],
    language: Optional[str],
    speed: Optional[float],
    use_gpu: Optional[bool],
    chapters: Optional[str],
    silence: Optional[float],
    title: Optional[str],
    author: Optional[str],
    cover: Optional[Path],
    yes: bool,
    verbose: bool,
    split_mode: Optional[str],
    resume: bool,
    keep_chapter_files: bool,
) -> None:
    """Convert an EPUB file to an audiobook.

    EPUB_FILE is the path to the EPUB file to convert.
    """
    config = load_config()

    # Get format first (needed for output path construction)
    fmt = output_format or config.get("default_format", "m4b")

    # Determine output path
    if output is None:
        output = epub_file.with_suffix(f".{fmt}")
    elif output.is_dir():
        # If output is a directory, construct filename from epub name
        output = output / f"{epub_file.stem}.{fmt}"

    # Get format from output extension if not specified
    if output_format is None:
        output_format = output.suffix.lstrip(".") or config.get("default_format", "m4b")

    # Load chapters from EPUB
    console.print(f"[bold]Loading:[/bold] {epub_file}")

    try:
        from epub2text import EPUBParser
    except ImportError:
        console.print(
            "[red]Error:[/red] epub2text is not installed. Run: pip install epub2text"
        )
        sys.exit(1)

    # Parse EPUB
    try:
        parser = EPUBParser(str(epub_file))
    except Exception as e:
        console.print(f"[red]Error loading EPUB:[/red] {e}")
        sys.exit(1)

    # Get EPUB metadata
    metadata = parser.get_metadata()
    epub_title = metadata.title or epub_file.stem
    epub_author = metadata.authors[0] if metadata.authors else "Unknown"
    epub_language = metadata.language

    # Extract chapters
    with console.status("Extracting chapters..."):
        epub_chapters = parser.get_chapters()

    if not epub_chapters:
        console.print("[red]Error:[/red] No chapters found in EPUB file.")
        sys.exit(1)

    console.print(f"[green]Found {len(epub_chapters)} chapters[/green]")

    # Auto-detect language if not specified
    if language is None:
        if epub_language:
            language = detect_language_from_iso(epub_language)
            console.print(
                f"[dim]Auto-detected language: {LANGUAGE_DESCRIPTIONS.get(language, language)}[/dim]"
            )
        else:
            language = config.get("default_language", "a")

    # Get voice
    if voice is None:
        voice = config.get("default_voice")
        # Ensure voice matches language
        if voice:
            voice_lang = VOICE_PREFIX_TO_LANG.get(voice[:2], "a")
            if voice_lang != language:
                voice = get_default_voice_for_language(language)
        else:
            voice = get_default_voice_for_language(language)

    # Chapter selection
    selected_indices: Optional[list[int]] = None
    if chapters:
        selected_indices = _parse_chapter_selection(chapters, len(epub_chapters))
    elif not yes:
        selected_indices = _interactive_chapter_selection(epub_chapters)

    if selected_indices is not None and len(selected_indices) == 0:
        console.print("[yellow]No chapters selected. Exiting.[/yellow]")
        return

    # Show conversion summary
    _show_conversion_summary(
        epub_file=epub_file,
        output=output,
        output_format=output_format,
        voice=voice or "af_bella",
        language=language or "a",
        speed=speed or config.get("default_speed", 1.0),
        use_gpu=use_gpu if use_gpu is not None else config.get("use_gpu", True),
        num_chapters=len(selected_indices) if selected_indices else len(epub_chapters),
        title=title or epub_title,
        author=author or epub_author,
    )

    # Confirm
    if not yes:
        if not Confirm.ask("Proceed with conversion?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Create conversion options
    options = ConversionOptions(
        voice=voice,
        language=language,
        speed=speed or config.get("default_speed", 1.0),
        output_format=output_format,
        output_dir=output.parent,
        use_gpu=use_gpu if use_gpu is not None else config.get("use_gpu", True),
        silence_between_chapters=silence or config.get("silence_between_chapters", 2.0),
        split_mode=split_mode or config.get("default_split_mode", "auto"),
        resume=resume,
        keep_chapter_files=keep_chapter_files,
        title=title or epub_title,
        author=author or epub_author,
        cover_image=cover,
    )

    # Set up progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    task_id = None
    current_chapter_text = ""

    def progress_callback(prog: ConversionProgress) -> None:
        nonlocal task_id, current_chapter_text
        if task_id is not None:
            progress.update(task_id, completed=prog.chars_processed)
            current_chapter_text = f"Chapter {prog.current_chapter}/{prog.total_chapters}: {prog.chapter_name}"
            progress.update(task_id, description=current_chapter_text[:50])

    def log_callback(message: str, level: str) -> None:
        if verbose:
            if level == "error":
                console.print(f"[red]{message}[/red]")
            elif level == "warning":
                console.print(f"[yellow]{message}[/yellow]")
            else:
                console.print(f"[dim]{message}[/dim]")

    # Run conversion
    converter = TTSConverter(
        options=options,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )

    # Calculate total characters for progress
    total_chars = sum(
        ch.char_count
        for i, ch in enumerate(epub_chapters)
        if selected_indices is None or i in selected_indices
    )

    with progress:
        task_id = progress.add_task("Converting...", total=total_chars)

        result = converter.convert_epub(
            epub_path=epub_file,
            output_path=output,
            selected_chapters=selected_indices,
        )

        progress.update(task_id, completed=total_chars)

    # Show result
    if result.success:
        console.print()
        console.print(
            Panel(
                f"[green]Audiobook saved to:[/green]\n{result.output_path}",
                title="[bold green]Conversion Complete[/bold green]",
            )
        )
    else:
        console.print()
        console.print(
            Panel(
                f"[red]{result.error_message}[/red]",
                title="[bold red]Conversion Failed[/bold red]",
            )
        )
        sys.exit(1)


@main.command("list")
@click.argument("epub_file", type=click.Path(exists=True, path_type=Path))
def list_chapters(epub_file: Path) -> None:
    """List chapters in an EPUB file.

    EPUB_FILE is the path to the EPUB file.
    """
    try:
        from epub2text import EPUBParser
    except ImportError:
        console.print(
            "[red]Error:[/red] epub2text is not installed. Run: pip install epub2text"
        )
        sys.exit(1)

    with console.status("Loading EPUB..."):
        try:
            parser = EPUBParser(str(epub_file))
            chapters = parser.get_chapters()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    if not chapters:
        console.print("[yellow]No chapters found in EPUB file.[/yellow]")
        return

    table = Table(title=f"Chapters in {epub_file.name}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="bold")
    table.add_column("Characters", justify="right")

    total_chars = 0
    for i, ch in enumerate(chapters, 1):
        char_count = ch.char_count
        total_chars += char_count
        table.add_row(str(i), ch.title[:60], f"{char_count:,}")

    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {len(chapters)} chapters, {total_chars:,} characters"
    )


@main.command()
@click.argument("epub_file", type=click.Path(exists=True, path_type=Path))
def info(epub_file: Path) -> None:
    """Show metadata and information about an EPUB file.

    EPUB_FILE is the path to the EPUB file.
    """
    try:
        from epub2text import EPUBParser
    except ImportError:
        console.print(
            "[red]Error:[/red] epub2text is not installed. Run: pip install epub2text"
        )
        sys.exit(1)

    # Parse EPUB
    with console.status("Loading EPUB..."):
        try:
            parser = EPUBParser(str(epub_file))
            metadata = parser.get_metadata()
            chapters = parser.get_chapters()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    total_chars = sum(ch.char_count for ch in chapters) if chapters else 0

    # Display info
    console.print(Panel(f"[bold]{epub_file.name}[/bold]", title="EPUB Information"))

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    if metadata:
        if metadata.title:
            table.add_row("Title", metadata.title)
        if metadata.authors:
            table.add_row("Author", ", ".join(metadata.authors))
        if metadata.language:
            lang = metadata.language
            lang_desc = LANGUAGE_DESCRIPTIONS.get(detect_language_from_iso(lang), lang)
            table.add_row("Language", f"{lang} ({lang_desc})")
        if metadata.publisher:
            table.add_row("Publisher", metadata.publisher)
        if metadata.publication_year:
            table.add_row("Year", str(metadata.publication_year))

    table.add_row("Chapters", str(len(chapters)) if chapters else "0")
    table.add_row("Characters", f"{total_chars:,}")
    table.add_row("File Size", format_size(epub_file.stat().st_size))

    console.print(table)


@main.command()
@click.option(
    "--language",
    "-l",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    help="Filter voices by language.",
)
def voices(language: Optional[str]) -> None:
    """List available TTS voices."""
    table = Table(title="Available Voices")
    table.add_column("Voice", style="bold")
    table.add_column("Language")
    table.add_column("Gender")
    table.add_column("Default", style="dim")

    for voice in VOICES:
        prefix = voice[:2]
        lang_code = VOICE_PREFIX_TO_LANG.get(prefix, "?")

        if language and lang_code != language:
            continue

        lang_name = LANGUAGE_DESCRIPTIONS.get(lang_code, "Unknown")
        gender = "Female" if prefix[1] == "f" else "Male"
        is_default = "Yes" if DEFAULT_VOICE_FOR_LANG.get(lang_code) == voice else ""

        table.add_row(voice, lang_name, gender, is_default)

    console.print(table)


@main.command()
@click.option("--show", is_flag=True, help="Show current configuration.")
@click.option("--reset", is_flag=True, help="Reset configuration to defaults.")
@click.option(
    "--set",
    "set_option",
    nargs=2,
    multiple=True,
    metavar="KEY VALUE",
    help="Set a configuration option.",
)
def config(show: bool, reset: bool, set_option: tuple[tuple[str, str], ...]) -> None:
    """Manage ttsforge configuration.

    Configuration is stored in ~/.config/ttsforge/config.json
    """
    if reset:
        reset_config()
        console.print("[green]Configuration reset to defaults.[/green]")
        return

    if set_option:
        current_config = load_config()
        for key, value in set_option:
            if key not in DEFAULT_CONFIG:
                console.print(f"[yellow]Warning:[/yellow] Unknown option '{key}'")
                continue

            # Type conversion
            default_type = type(DEFAULT_CONFIG[key])
            try:
                if default_type == bool:
                    typed_value = value.lower() in ("true", "1", "yes")
                elif default_type == float:
                    typed_value = float(value)
                elif default_type == int:
                    typed_value = int(value)
                else:
                    typed_value = value

                current_config[key] = typed_value
                console.print(f"[green]Set {key} = {typed_value}[/green]")
            except ValueError:
                console.print(f"[red]Invalid value for {key}: {value}[/red]")

        save_config(current_config)
        return

    # Show configuration
    current_config = load_config()

    table = Table(title="Current Configuration")
    table.add_column("Option", style="bold")
    table.add_column("Value")
    table.add_column("Default", style="dim")

    for key, default_value in DEFAULT_CONFIG.items():
        current_value = current_config.get(key, default_value)
        is_default = current_value == default_value
        table.add_row(
            key,
            str(current_value),
            str(default_value) if not is_default else "",
        )

    console.print(table)

    # Show GPU status
    gpu_status, gpu_enabled = get_gpu_info(current_config.get("use_gpu", True))
    console.print(f"\n[bold]GPU Status:[/bold] {gpu_status}")


def _parse_chapter_selection(selection: str, total_chapters: int) -> list[int]:
    """Parse chapter selection string into list of indices."""
    if selection.lower() == "all":
        return list(range(total_chapters))

    indices: set[int] = set()

    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            # Range
            try:
                start, end = part.split("-")
                start_idx = int(start) - 1
                end_idx = int(end)
                indices.update(range(max(0, start_idx), min(total_chapters, end_idx)))
            except ValueError:
                console.print(f"[yellow]Invalid range: {part}[/yellow]")
        else:
            # Single number
            try:
                idx = int(part) - 1
                if 0 <= idx < total_chapters:
                    indices.add(idx)
            except ValueError:
                console.print(f"[yellow]Invalid chapter number: {part}[/yellow]")

    return sorted(indices)


def _interactive_chapter_selection(chapters: list) -> Optional[list[int]]:
    """Interactive chapter selection using Rich."""
    console.print("\n[bold]Available Chapters:[/bold]")

    table = Table(show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Title")
    table.add_column("Chars", justify="right")

    for i, ch in enumerate(chapters, 1):
        table.add_row(str(i), ch.title[:50], f"{ch.char_count:,}")

    console.print(table)

    console.print("\n[dim]Enter chapter selection:[/dim]")
    console.print("[dim]  - 'all' for all chapters[/dim]")
    console.print("[dim]  - '1-5' for range[/dim]")
    console.print("[dim]  - '1,3,5' for specific chapters[/dim]")
    console.print("[dim]  - Press Enter for all chapters[/dim]")

    selection = console.input("\n[bold]Selection:[/bold] ").strip()

    if not selection:
        return None  # All chapters

    return _parse_chapter_selection(selection, len(chapters))


def _show_conversion_summary(
    epub_file: Path,
    output: Path,
    output_format: str,
    voice: str,
    language: str,
    speed: float,
    use_gpu: bool,
    num_chapters: int,
    title: str,
    author: str,
) -> None:
    """Show conversion summary before starting."""
    console.print()

    table = Table(title="Conversion Summary", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Input", str(epub_file))
    table.add_row("Output", str(output))
    table.add_row("Format", output_format.upper())
    table.add_row("Chapters", str(num_chapters))
    table.add_row("Voice", voice)
    table.add_row("Language", LANGUAGE_DESCRIPTIONS.get(language, language))
    table.add_row("Speed", f"{speed}x")
    table.add_row("GPU", "Enabled" if use_gpu else "Disabled")
    table.add_row("Title", title)
    table.add_row("Author", author)

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
