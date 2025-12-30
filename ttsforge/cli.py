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
from rich.prompt import Confirm
from rich.table import Table

from .constants import (
    DEFAULT_CONFIG,
    DEFAULT_VOICE_FOR_LANG,
    LANGUAGE_DESCRIPTIONS,
    PROGRAM_NAME,
    SUPPORTED_OUTPUT_FORMATS,
    VOICE_PREFIX_TO_LANG,
    VOICES,
)
from .conversion import (
    ConversionOptions,
    ConversionProgress,
    TTSConverter,
    detect_language_from_iso,
    get_default_voice_for_language,
)
from .onnx_backend import (
    ONNX_MODEL_FILES,
    are_models_downloaded,
    download_model,
    get_model_dir,
    get_model_path,
    is_model_downloaded,
)
from .utils import (
    format_chapters_range,
    format_filename_template,
    format_size,
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
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to custom kokoro.onnx model file.",
)
@click.option(
    "--voices",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to custom voices.bin file.",
)
@click.pass_context
def main(
    ctx: click.Context, version: bool, model: Optional[Path], voices: Optional[Path]
) -> None:
    """ttsforge - Generate audiobooks from EPUB files with TTS."""
    ctx.ensure_object(dict)
    ctx.obj["model_path"] = model
    ctx.obj["voices_path"] = voices
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
    help="Chapters to convert (e.g., '1-5', '1,3,5', 'all').",
)
@click.option(
    "--silence",
    type=float,
    help="Silence duration between chapters in seconds.",
)
@click.option(
    "--segment-pause-min",
    type=float,
    default=None,
    help="Minimum pause between sentences in seconds (default: 0.1).",
)
@click.option(
    "--segment-pause-max",
    type=float,
    default=None,
    help="Maximum pause between sentences in seconds (default: 0.3).",
)
@click.option(
    "--paragraph-pause-min",
    type=float,
    default=None,
    help="Minimum pause between paragraphs in seconds (default: 0.5).",
)
@click.option(
    "--paragraph-pause-max",
    type=float,
    default=None,
    help="Maximum pause between paragraphs in seconds (default: 1.0).",
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
    help="Text splitting mode: auto, line, paragraph, sentence, clause.",
)
@click.option(
    "--resume/--no-resume",
    "resume",
    default=True,
    help="Enable/disable resume capability (default: enabled).",
)
@click.option(
    "--fresh",
    is_flag=True,
    help="Discard any previous progress and start conversion from scratch.",
)
@click.option(
    "--keep-chapters",
    "keep_chapter_files",
    is_flag=True,
    help="Keep individual chapter audio files after conversion.",
)
@click.option(
    "--voice-blend",
    "voice_blend",
    type=str,
    help="Blend multiple voices (e.g., 'af_nicole:50,am_michael:50').",
)
@click.option(
    "--voice-db",
    "voice_database",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom voice database (SQLite).",
)
@click.pass_context
def convert(
    ctx: click.Context,
    epub_file: Path,
    output: Optional[Path],
    output_format: Optional[str],
    voice: Optional[str],
    language: Optional[str],
    speed: Optional[float],
    use_gpu: Optional[bool],
    chapters: Optional[str],
    silence: Optional[float],
    segment_pause_min: Optional[float],
    segment_pause_max: Optional[float],
    paragraph_pause_min: Optional[float],
    paragraph_pause_max: Optional[float],
    title: Optional[str],
    author: Optional[str],
    cover: Optional[Path],
    yes: bool,
    verbose: bool,
    split_mode: Optional[str],
    resume: bool,
    fresh: bool,
    keep_chapter_files: bool,
    voice_blend: Optional[str],
    voice_database: Optional[Path],
) -> None:
    """Convert an EPUB file to an audiobook.

    EPUB_FILE is the path to the EPUB file to convert.
    """
    config = load_config()
    model_path = ctx.obj.get("model_path") if ctx.obj else None
    voices_path = ctx.obj.get("voices_path") if ctx.obj else None

    # Get format first (needed for output path construction)
    fmt = output_format or config.get("default_format", "m4b")

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
    default_title = config.get("default_title", "Untitled")
    epub_title = metadata.title or default_title
    epub_author = metadata.authors[0] if metadata.authors else "Unknown"
    epub_language = metadata.language

    # Use CLI title/author if provided, otherwise use EPUB metadata
    effective_title = title or epub_title
    effective_author = author or epub_author

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
            lang_desc = LANGUAGE_DESCRIPTIONS.get(language, language)
            console.print(f"[dim]Auto-detected language: {lang_desc}[/dim]")
        else:
            language = config.get("default_language", "a")

    # Get voice
    if voice is None:
        voice = config.get("default_voice")
        # Ensure voice matches language
        if voice and language:
            voice_lang = VOICE_PREFIX_TO_LANG.get(voice[:2], "a")
            if voice_lang != language:
                voice = get_default_voice_for_language(language)
        elif language:
            voice = get_default_voice_for_language(language)
        else:
            voice = "af_heart"

    # Ensure language has a default
    if language is None:
        language = "a"

    # Chapter selection
    selected_indices: Optional[list[int]] = None
    if chapters:
        selected_indices = _parse_chapter_selection(chapters, len(epub_chapters))
    elif not yes:
        selected_indices = _interactive_chapter_selection(epub_chapters)

    if selected_indices is not None and len(selected_indices) == 0:
        console.print("[yellow]No chapters selected. Exiting.[/yellow]")
        return

    # Determine output path using filename template
    if output is None:
        output_template = config.get("output_filename_template", "{book_title}")
        chapters_range = format_chapters_range(
            selected_indices or list(range(len(epub_chapters))), len(epub_chapters)
        )
        output_filename = format_filename_template(
            output_template,
            book_title=effective_title,
            author=effective_author,
            input_stem=epub_file.stem,
            chapters_range=chapters_range,
            default_title=default_title,
        )
        # Append chapters range to filename if partial selection
        if chapters_range:
            output_filename = f"{output_filename}_{chapters_range}"
        output = epub_file.parent / f"{output_filename}.{fmt}"
    elif output.is_dir():
        # If output is a directory, construct filename using template
        output_template = config.get("output_filename_template", "{book_title}")
        chapters_range = format_chapters_range(
            selected_indices or list(range(len(epub_chapters))), len(epub_chapters)
        )
        output_filename = format_filename_template(
            output_template,
            book_title=effective_title,
            author=effective_author,
            input_stem=epub_file.stem,
            chapters_range=chapters_range,
            default_title=default_title,
        )
        if chapters_range:
            output_filename = f"{output_filename}_{chapters_range}"
        output = output / f"{output_filename}.{fmt}"

    # Get format from output extension if not specified
    if output_format is None:
        output_format = output.suffix.lstrip(".") or config.get("default_format", "m4b")

    # Show conversion summary
    _show_conversion_summary(
        epub_file=epub_file,
        output=output,
        output_format=output_format or config.get("default_format", "m4b"),
        voice=voice or "af_bella",
        language=language or "a",
        speed=speed or config.get("default_speed", 1.0),
        use_gpu=use_gpu if use_gpu is not None else config.get("use_gpu", False),
        num_chapters=len(selected_indices) if selected_indices else len(epub_chapters),
        title=effective_title,
        author=effective_author,
    )

    # Confirm
    if not yes:
        if not Confirm.ask("Proceed with conversion?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Handle --fresh flag: delete existing progress
    if fresh:
        import shutil

        from .utils import sanitize_filename

        safe_book_title = sanitize_filename(effective_title)[:50]
        work_dir = output.parent / f".{safe_book_title}_chapters"
        if work_dir.exists():
            console.print(f"[yellow]Removing previous progress:[/yellow] {work_dir}")
            shutil.rmtree(work_dir)
        # Fresh start means we don't try to resume
        resume = False

    # Create conversion options
    options = ConversionOptions(
        voice=voice or config.get("default_voice", "af_heart"),
        language=language or config.get("default_language", "a"),
        speed=speed or config.get("default_speed", 1.0),
        output_format=output_format or config.get("default_format", "m4b"),
        output_dir=output.parent,
        use_gpu=use_gpu if use_gpu is not None else config.get("use_gpu", False),
        silence_between_chapters=silence or config.get("silence_between_chapters", 2.0),
        segment_pause_min=(
            segment_pause_min
            if segment_pause_min is not None
            else config.get("segment_pause_min", 0.1)
        ),
        segment_pause_max=(
            segment_pause_max
            if segment_pause_max is not None
            else config.get("segment_pause_max", 0.3)
        ),
        paragraph_pause_min=(
            paragraph_pause_min
            if paragraph_pause_min is not None
            else config.get("paragraph_pause_min", 0.5)
        ),
        paragraph_pause_max=(
            paragraph_pause_max
            if paragraph_pause_max is not None
            else config.get("paragraph_pause_max", 1.0)
        ),
        split_mode=split_mode or config.get("default_split_mode", "auto"),
        resume=resume,
        keep_chapter_files=keep_chapter_files,
        title=effective_title,
        author=effective_author,
        cover_image=cover,
        voice_blend=voice_blend,
        voice_database=voice_database,
        chapter_filename_template=config.get(
            "chapter_filename_template",
            "{chapter_num:03d}_{book_title}_{chapter_title}",
        ),
        model_path=model_path,
        voices_path=voices_path,
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
            ch = prog.current_chapter
            total = prog.total_chapters
            current_chapter_text = f"Chapter {ch}/{total}: {prog.chapter_name}"
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


# Default sample text for testing TTS settings
DEFAULT_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sample text demonstrates the text-to-speech capabilities, "
    "including punctuation handling, and natural speech flow."
)


@main.command()
@click.argument("text", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file path (default: ./sample.wav).",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(SUPPORTED_OUTPUT_FORMATS),
    default="wav",
    help="Output audio format.",
)
@click.option("-v", "--voice", type=click.Choice(VOICES), help="TTS voice to use.")
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    help="Language for TTS.",
)
@click.option("-s", "--speed", type=float, help="Speech speed (default: 1.0).")
@click.option(
    "--gpu/--no-gpu",
    "use_gpu",
    default=None,
    help="Use GPU acceleration if available.",
)
@click.option(
    "--split-mode",
    "split_mode",
    type=click.Choice(["auto", "line", "paragraph", "sentence", "clause"]),
    help="Text splitting mode for processing.",
)
@click.option("--verbose", is_flag=True, help="Show detailed output.")
@click.pass_context
def sample(
    ctx: click.Context,
    text: Optional[str],
    output: Optional[Path],
    output_format: str,
    voice: Optional[str],
    language: Optional[str],
    speed: Optional[float],
    use_gpu: Optional[bool],
    split_mode: Optional[str],
    verbose: bool,
) -> None:
    """Generate a sample audio file to test TTS settings.

    If no TEXT is provided, uses a default sample text.

    Examples:

        ttsforge sample

        ttsforge sample "Hello, this is a test."

        ttsforge sample --voice am_adam --speed 1.2 -o test.wav
    """
    from .conversion import ConversionOptions, TTSConverter

    # Get model path from global context
    model_path = ctx.obj.get("model_path") if ctx.obj else None
    voices_path = ctx.obj.get("voices_path") if ctx.obj else None

    # Use default text if none provided
    sample_text = text or DEFAULT_SAMPLE_TEXT

    # Determine output path
    if output is None:
        output = Path(f"./sample.{output_format}")
    elif output.suffix == "":
        # If no extension provided, add the format
        output = output.with_suffix(f".{output_format}")

    # Load config for defaults
    user_config = load_config()

    # Build conversion options (use ConversionOptions defaults if not specified)
    options = ConversionOptions(
        voice=voice or user_config.get("voice") or "af_bella",
        language=language or user_config.get("language") or "a",
        speed=speed or user_config.get("speed", 1.0),
        output_format=output_format,
        use_gpu=use_gpu if use_gpu is not None else user_config.get("use_gpu", True),
        split_mode=split_mode or user_config.get("split_mode", "auto"),
        model_path=model_path,
        voices_path=voices_path,
    )

    # Always show settings
    console.print(f"[dim]Voice:[/dim] {options.voice}")
    lang_desc = LANGUAGE_DESCRIPTIONS.get(options.language, "Unknown")
    console.print(f"[dim]Language:[/dim] {options.language} ({lang_desc})")
    console.print(f"[dim]Speed:[/dim] {options.speed}")
    console.print(f"[dim]Format:[/dim] {options.output_format}")
    console.print(f"[dim]Split mode:[/dim] {options.split_mode}")
    console.print(f"[dim]GPU:[/dim] {'enabled' if options.use_gpu else 'disabled'}")

    if verbose:
        text_preview = sample_text[:100]
        ellipsis = "..." if len(sample_text) > 100 else ""
        console.print(f"[dim]Text:[/dim] {text_preview}{ellipsis}")
        console.print(f"[dim]Output:[/dim] {output}")

    try:
        converter = TTSConverter(options)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Generating audio...", total=None)
            result = converter.convert_text(sample_text, output)

        if result.success:
            console.print(f"[green]Sample saved to:[/green] {output}")
        else:
            console.print(f"[red]Error:[/red] {result.error_message}")
            raise SystemExit(1)

    except Exception as e:
        console.print(f"[red]Error generating sample:[/red] {e}")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise SystemExit(1) from None


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


# Demo sample text per language
DEMO_TEXT = {
    "a": "Hello! This audio was generated by {voice}. How do you like it?",
    "b": "Hello! This audio was generated by {voice}. How do you like it?",
    "e": "Hola! Este audio fue generado por {voice}. Que te parece?",
    "f": "Bonjour! Cet audio a ete genere par {voice}. Comment le trouvez-vous?",
    "h": "Namaste! Yah audio {voice} dwara banaya gaya hai. Aapko kaisa laga?",
    "i": "Ciao! Questo audio e stato generato da {voice}. Ti piace?",
    "j": "Konnichiwa! Kono onsei wa {voice} ni yotte sakusei saremashita.",
    "p": "Ola! Este audio foi gerado por {voice}. O que voce achou?",
    "z": "Ni hao! Zhe ge yinpin shi you {voice} shengcheng de.",
}

# Preset voice blends for demo command
# Format: (blend_string, description)
VOICE_BLEND_PRESETS = [
    # Same language, different gender
    ("af_nicole:50,am_michael:50", "American female + male blend"),
    ("bf_emma:50,bm_george:50", "British female + male blend"),
    # Same gender, different accent
    ("af_heart:50,bf_emma:50", "American + British female blend"),
    ("am_adam:50,bm_daniel:50", "American + British male blend"),
    # Same gender, different voice
    ("af_nicole:50,af_bella:50", "Two American females blend"),
    ("am_adam:50,am_eric:50", "Two American males blend"),
    # Multi-voice blend
    ("af_heart:33,af_nicole:33,af_bella:34", "Three American females blend"),
]


@main.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: ./voices_demo.wav).",
)
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    default=None,
    help="Filter voices by language (default: all languages).",
)
@click.option(
    "-v",
    "--voice",
    "voices_filter",
    type=str,
    default=None,
    help="Specific voices to include (comma-separated, e.g., 'af_heart,am_adam').",
)
@click.option(
    "-s",
    "--speed",
    type=float,
    default=1.0,
    help="Speech speed (default: 1.0).",
)
@click.option(
    "--gpu/--no-gpu",
    "use_gpu",
    default=None,
    help="Enable/disable GPU acceleration.",
)
@click.option(
    "--silence",
    type=float,
    default=0.5,
    help="Silence between voice samples in seconds (default: 0.5).",
)
@click.option(
    "--text",
    type=str,
    default=None,
    help="Custom text to use (use {voice} placeholder for voice name).",
)
@click.option(
    "--separate",
    is_flag=True,
    help="Save each voice as a separate file instead of concatenating.",
)
@click.option(
    "--blend",
    type=str,
    default=None,
    help="Voice blend to demo (e.g., 'af_nicole:50,am_michael:50').",
)
@click.option(
    "--blend-presets",
    is_flag=True,
    help="Demo a curated set of voice blend combinations.",
)
@click.pass_context
def demo(
    ctx: click.Context,
    output: Optional[Path],
    language: Optional[str],
    voices_filter: Optional[str],
    speed: float,
    use_gpu: Optional[bool],
    silence: float,
    text: Optional[str],
    separate: bool,
    blend: Optional[str],
    blend_presets: bool,
) -> None:
    """Generate a demo audio file with all available voices.

    Creates a single audio file with samples from each voice, or separate files
    for each voice with --separate. Great for previewing and comparing voices.

    Supports voice blending with --blend or --blend-presets options.

    Examples:

        ttsforge demo

        ttsforge demo -l a  # Only American English voices

        ttsforge demo -v af_heart,am_adam  # Specific voices

        ttsforge demo --separate -o ./voices/  # Separate files in directory

        ttsforge demo --text "Custom message from {voice}!"

        ttsforge demo --blend "af_nicole:50,am_michael:50"  # Custom voice blend

        ttsforge demo --blend-presets  # Demo all preset voice blends
    """
    import numpy as np

    from .onnx_backend import KokoroONNX, VoiceBlend

    config = load_config()
    gpu = use_gpu if use_gpu is not None else config.get("use_gpu", False)
    model_path = ctx.obj.get("model_path") if ctx.obj else None
    voices_path = ctx.obj.get("voices_path") if ctx.obj else None

    # Helper function to create filename from blend string
    def blend_to_filename(blend_str: str) -> str:
        """Convert blend string to filename-safe format."""
        # e.g., "af_nicole:50,am_michael:50" -> "blend_af_nicole_50_am_michael_50"
        parts = []
        for part in blend_str.split(","):
            part = part.strip()
            if ":" in part:
                voice_name, weight = part.split(":", 1)
                parts.append(f"{voice_name.strip()}_{weight.strip()}")
            else:
                parts.append(part.strip())
        return "blend_" + "_".join(parts)

    # Handle blend modes (--blend or --blend-presets)
    if blend or blend_presets:
        # Collect blends to process
        blends_to_process: list[tuple[str, str]] = []  # (blend_string, description)

        if blend:
            # Custom blend specified
            blends_to_process.append((blend, f"Custom blend: {blend}"))

        if blend_presets:
            # Add all preset blends
            blends_to_process.extend(VOICE_BLEND_PRESETS)

        # Determine output directory
        if output is None:
            output = Path("./voice_blends")
        output.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold]Output directory:[/bold] {output}")
        console.print(f"[dim]Voice blends: {len(blends_to_process)}[/dim]")
        console.print(f"[dim]Speed: {speed}x[/dim]")
        console.print(f"[dim]GPU: {'enabled' if gpu else 'disabled'}[/dim]")

        # Initialize TTS engine
        try:
            kokoro = KokoroONNX(
                model_path=model_path,
                voices_path=voices_path,
                use_gpu=gpu,
            )
        except Exception as e:
            console.print(f"[red]Error initializing TTS engine:[/red] {e}")
            sys.exit(1)

        sample_rate = 24000

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Generating voice blend demos...", total=len(blends_to_process)
            )

            for blend_str, description in blends_to_process:
                try:
                    # Parse the blend
                    voice_blend = VoiceBlend.parse(blend_str)

                    # Create demo text describing the blend
                    voice_names = [v for v, _ in voice_blend.voices]
                    if text:
                        demo_text = text.format(voice=" and ".join(voice_names))
                    else:
                        demo_text = f"This is a blend of {' and '.join(voice_names)} speaking together."

                    # Generate audio with blended voice
                    samples, sr = kokoro.create(
                        demo_text, voice=voice_blend, speed=speed
                    )

                    # Save to file
                    import soundfile as sf

                    filename = blend_to_filename(blend_str) + ".wav"
                    voice_file = output / filename
                    sf.write(str(voice_file), samples, sr)
                    progress.console.print(
                        f"  [green]{description}[/green]: {voice_file}"
                    )

                except Exception as e:
                    console.print(f"  [red]{blend_str}[/red]: Failed - {e}")

                progress.advance(task)

        console.print(
            f"\n[green]Saved {len(blends_to_process)} voice blend demos to:[/green] {output}"
        )
        return

    # Regular voice demo mode (no blending)
    # Determine which voices to use
    selected_voices: list[str] = []

    if voices_filter:
        # Specific voices requested
        for v in voices_filter.split(","):
            v = v.strip()
            if v in VOICES:
                selected_voices.append(v)
            else:
                console.print(f"[yellow]Warning:[/yellow] Unknown voice '{v}'")
    elif language:
        # Filter by language
        for v in VOICES:
            prefix = v[:2]
            lang_code = VOICE_PREFIX_TO_LANG.get(prefix, "?")
            if lang_code == language:
                selected_voices.append(v)
    else:
        # All voices
        selected_voices = list(VOICES)

    if not selected_voices:
        console.print("[red]Error:[/red] No voices selected.")
        sys.exit(1)

    # Determine output path
    if separate:
        if output is None:
            output = Path("./voice_demos")
        output.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Output directory:[/bold] {output}")
    else:
        if output is None:
            output = Path("./voices_demo.wav")
        console.print(f"[bold]Output file:[/bold] {output}")

    console.print(f"[dim]Voices: {len(selected_voices)}[/dim]")
    console.print(f"[dim]Speed: {speed}x[/dim]")
    console.print(f"[dim]GPU: {'enabled' if gpu else 'disabled'}[/dim]")

    # Initialize TTS engine
    try:
        kokoro = KokoroONNX(
            model_path=model_path,
            voices_path=voices_path,
            use_gpu=gpu,
        )
    except Exception as e:
        console.print(f"[red]Error initializing TTS engine:[/red] {e}")
        sys.exit(1)

    # Generate samples
    all_samples: list[np.ndarray] = []
    sample_rate = 24000  # Kokoro sample rate

    # Create silence array for gaps between samples
    silence_samples = np.zeros(int(silence * sample_rate), dtype=np.float32)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Generating voice demos...", total=len(selected_voices)
        )

        for voice in selected_voices:
            # Determine language and text for this voice
            prefix = voice[:2]
            lang_code = VOICE_PREFIX_TO_LANG.get(prefix, "a")

            if text:
                demo_text = text.format(voice=voice)
            else:
                demo_text = DEMO_TEXT.get(lang_code, DEMO_TEXT["a"]).format(voice=voice)

            try:
                samples, sr = kokoro.create(demo_text, voice=voice, speed=speed)

                if separate:
                    # Save individual file
                    import soundfile as sf

                    voice_file = output / f"{voice}.wav"
                    sf.write(str(voice_file), samples, sr)
                    progress.console.print(f"  [green]{voice}[/green]: {voice_file}")
                else:
                    all_samples.append(samples)
                    if voice != selected_voices[-1]:
                        all_samples.append(silence_samples)

            except Exception as e:
                console.print(f"  [red]{voice}[/red]: Failed - {e}")

            progress.advance(task)

    # Concatenate and save if not separate
    if not separate and all_samples:
        import soundfile as sf

        combined = np.concatenate(all_samples)
        sf.write(str(output), combined, sample_rate)
        console.print(f"\n[green]Demo saved to:[/green] {output}")

        # Show duration
        duration_secs = len(combined) / sample_rate
        mins, secs = divmod(int(duration_secs), 60)
        console.print(f"[dim]Duration: {mins}m {secs}s[/dim]")
    elif separate:
        console.print(
            f"\n[green]Saved {len(selected_voices)} voice demos to:[/green] {output}"
        )


@main.command()
@click.option("--force", is_flag=True, help="Force re-download even if files exist.")
def download(force: bool) -> None:
    """Download ONNX model files required for TTS.

    Downloads the kokoro-onnx model files (~330MB total) to the cache directory.
    This is required before using ttsforge for the first time.

    The models will be downloaded automatically on first use, but you can use
    this command to download them proactively.
    """
    model_dir = get_model_dir()
    console.print(f"[bold]Model directory:[/bold] {model_dir}")

    if are_models_downloaded() and not force:
        console.print("[green]All model files are already downloaded.[/green]")
        for filename in ONNX_MODEL_FILES:
            path = get_model_path(filename)
            size = format_size(path.stat().st_size) if path.exists() else "N/A"
            console.print(f"  {filename}: {size}")
        return

    console.print("Downloading ONNX model files...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        for filename in ONNX_MODEL_FILES:
            if is_model_downloaded(filename) and not force:
                console.print(f"  [dim]{filename}: already downloaded[/dim]")
                continue

            current_task_id = progress.add_task(f"Downloading {filename}...", total=100)

            def update_progress(
                current: int, total: int, tid: int = current_task_id
            ) -> None:
                if total > 0:
                    percent = (current / total) * 100
                    progress.update(tid, completed=percent)

            try:
                download_model(filename, progress_callback=update_progress, force=force)
                progress.update(current_task_id, completed=100)
                path = get_model_path(filename)
                size = format_size(path.stat().st_size)
                console.print(f"  [green]{filename}: {size}[/green]")
            except Exception as e:
                console.print(f"  [red]{filename}: Failed - {e}[/red]")
                sys.exit(1)

    console.print("\n[green]All model files downloaded successfully![/green]")


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
                if default_type is bool:
                    typed_value = value.lower() in ("true", "1", "yes")
                elif default_type is float:
                    typed_value = float(value)
                elif default_type is int:
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

    # Show model status
    if are_models_downloaded():
        model_dir = get_model_dir()
        console.print(f"\n[bold]ONNX Models:[/bold] Downloaded ({model_dir})")
    else:
        console.print("\n[bold]ONNX Models:[/bold] [yellow]Not downloaded[/yellow]")
        console.print("[dim]Run 'ttsforge download' to download models[/dim]")


# Phonemes command group
@main.group()
def phonemes() -> None:
    """Commands for working with phonemes and pre-tokenized content.

    The phonemes subcommand allows you to:
    - Export EPUB books as pre-tokenized phoneme data (JSON)
    - Export human-readable phoneme representations for review
    - Convert pre-tokenized phoneme files to audiobooks
    - Preview phonemes for given text

    This is useful for:
    - Reviewing and editing pronunciation before generating audio
    - Faster repeated conversions (skip phonemization step)
    - Archiving phoneme data for different vocabulary versions
    """
    pass


@phonemes.command("export")
@click.argument("epub_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file path. Defaults to input filename with .phonemes.json extension.",
)
@click.option(
    "--readable",
    is_flag=True,
    help="Export as human-readable text format instead of JSON.",
)
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    default="a",
    help="Language code for phonemization.",
)
@click.option(
    "--chapters",
    type=str,
    help="Chapters to export (e.g., '1-5', '1,3,5', 'all').",
)
@click.option(
    "--vocab-version",
    type=str,
    default="v1.0",
    help="Vocabulary version to use for tokenization.",
)
@click.option(
    "--split-mode",
    "split_mode",
    type=click.Choice(["paragraph", "sentence", "clause"]),
    default="sentence",
    help="Split mode: paragraph (newlines), sentence (spaCy), clause (+ commas).",
)
@click.option(
    "--max-chars",
    type=int,
    default=300,
    help="Maximum characters per segment (for additional splitting of long segments).",
)
def phonemes_export(
    epub_file: Path,
    output: Optional[Path],
    readable: bool,
    language: str,
    chapters: Optional[str],
    vocab_version: str,
    split_mode: str,
    max_chars: int,
) -> None:
    """Export an EPUB as pre-tokenized phoneme data.

    This creates a JSON file containing the book's text converted to
    phonemes and tokens, which can be later converted to audio without
    re-running the phonemization step.

    Split modes:
    - paragraph: Split only on double newlines (fewer, longer segments)
    - sentence: Split on sentence boundaries using spaCy (recommended)
    - clause: Split on sentences + commas (more, shorter segments)

    Examples:

        ttsforge phonemes export book.epub

        ttsforge phonemes export book.epub --readable -o book.readable.txt

        ttsforge phonemes export book.epub --language b --chapters 1-5

        ttsforge phonemes export book.epub --split-mode clause
    """
    from .phonemes import PhonemeBook
    from .tokenizer import Tokenizer

    config = load_config()

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
        metadata = parser.get_metadata()
        epub_chapters = parser.get_chapters()
    except Exception as e:
        console.print(f"[red]Error loading EPUB:[/red] {e}")
        sys.exit(1)

    if not epub_chapters:
        console.print("[red]Error:[/red] No chapters found in EPUB file.")
        sys.exit(1)

    # Chapter selection
    selected_indices: Optional[list[int]] = None
    if chapters:
        selected_indices = _parse_chapter_selection(chapters, len(epub_chapters))

    # Get effective title and author
    default_title = config.get("default_title", "Untitled")
    effective_title = metadata.title or default_title
    effective_author = metadata.authors[0] if metadata.authors else "Unknown"

    # Compute chapters range for filename and metadata
    chapters_range = format_chapters_range(
        selected_indices or list(range(len(epub_chapters))), len(epub_chapters)
    )

    # Determine output path using template
    if output is None:
        output_template = config.get("phoneme_export_template", "{book_title}")
        output_filename = format_filename_template(
            output_template,
            book_title=effective_title,
            author=effective_author,
            input_stem=epub_file.stem,
            chapters_range=chapters_range,
            default_title=default_title,
        )
        # Append chapters range to filename if partial selection
        if chapters_range:
            output_filename = f"{output_filename}_{chapters_range}"
        suffix = ".readable.txt" if readable else ".phonemes.json"
        output = epub_file.parent / f"{output_filename}{suffix}"

    # Get language code for espeak
    from .onnx_backend import LANG_CODE_TO_ONNX

    espeak_lang = LANG_CODE_TO_ONNX.get(language, "en-us")

    # Initialize tokenizer
    console.print(f"[dim]Initializing tokenizer (vocab: {vocab_version})...[/dim]")
    try:
        tokenizer = Tokenizer(vocab_version=vocab_version)
    except Exception as e:
        console.print(f"[red]Error initializing tokenizer:[/red] {e}")
        sys.exit(1)

    # Create PhonemeBook with chapters_range in metadata
    book = PhonemeBook(
        title=effective_title,
        vocab_version=vocab_version,
        lang=espeak_lang,
        metadata={
            "source": str(epub_file),
            "author": effective_author,
            "split_mode": split_mode,
            "chapters_range": chapters_range,
            "total_source_chapters": len(epub_chapters),
        },
    )

    console.print(f"[dim]Split mode: {split_mode}, Max chars: {max_chars}[/dim]")

    # Track warnings for long phonemes
    phoneme_warnings: list[str] = []

    def warn_callback(msg: str) -> None:
        """Collect phoneme length warnings."""
        phoneme_warnings.append(msg)

    # Process chapters
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        num_chapters = len(selected_indices) if selected_indices else len(epub_chapters)
        task = progress.add_task("Phonemizing chapters...", total=num_chapters)

        for i, ch in enumerate(epub_chapters):
            if selected_indices is not None and i not in selected_indices:
                continue

            chapter = book.create_chapter(ch.title)

            # Pass entire chapter text - add_text handles splitting based on split_mode
            if ch.text.strip():
                chapter.add_text(
                    ch.text,
                    tokenizer,
                    lang=espeak_lang,
                    split_mode=split_mode,
                    max_chars=max_chars,
                    warn_callback=warn_callback,
                )

            progress.advance(task)

    # Show warnings if any
    if phoneme_warnings:
        console.print(
            f"\n[yellow]Warning:[/yellow] {len(phoneme_warnings)} segment(s) had "
            f"phonemes exceeding the 510 character limit and were truncated."
        )
        if len(phoneme_warnings) <= 5:
            for w in phoneme_warnings:
                console.print(f"  [dim]{w}[/dim]")
        else:
            for w in phoneme_warnings[:3]:
                console.print(f"  [dim]{w}[/dim]")
            console.print(f"  [dim]... and {len(phoneme_warnings) - 3} more[/dim]")

    # Save output
    if readable:
        book.save_readable(output)
    else:
        book.save(output)

    console.print(f"[green]Exported to:[/green] {output}")
    console.print(
        f"[dim]Chapters: {len(book.chapters)}, "
        f"Segments: {book.total_segments}, "
        f"Tokens: {book.total_tokens:,}[/dim]"
    )


@phonemes.command("convert")
@click.argument("phoneme_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file path. Defaults to input filename with audio extension.",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(SUPPORTED_OUTPUT_FORMATS),
    help="Output audio format.",
)
@click.option("-v", "--voice", type=click.Choice(VOICES), help="Voice to use for TTS.")
@click.option("-s", "--speed", type=float, default=1.0, help="Speech speed.")
@click.option(
    "--gpu/--no-gpu",
    "use_gpu",
    default=None,
    help="Enable/disable GPU acceleration.",
)
@click.option(
    "--silence",
    type=float,
    default=2.0,
    help="Silence between chapters in seconds.",
)
@click.option(
    "--segment-pause-min",
    type=float,
    default=None,
    help="Minimum pause between sentences in seconds (default: 0.1).",
)
@click.option(
    "--segment-pause-max",
    type=float,
    default=None,
    help="Maximum pause between sentences in seconds (default: 0.3).",
)
@click.option(
    "--paragraph-pause-min",
    type=float,
    default=None,
    help="Minimum pause between paragraphs in seconds (default: 0.5).",
)
@click.option(
    "--paragraph-pause-max",
    type=float,
    default=None,
    help="Maximum pause between paragraphs in seconds (default: 1.0).",
)
@click.option(
    "--chapters",
    type=str,
    default=None,
    help="Select chapters to convert (1-based). E.g., '1-5', '3,5,7', or '1-3,7'.",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Audiobook title (for m4b metadata).",
)
@click.option(
    "--author",
    type=str,
    default=None,
    help="Audiobook author (for m4b metadata).",
)
@click.option(
    "--cover",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Cover image path (for m4b format).",
)
@click.option(
    "--voice-blend",
    type=str,
    default=None,
    help="Blend multiple voices. E.g., 'af_nicole:50,am_michael:50'.",
)
@click.option(
    "--voice-database",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to custom voice database (SQLite).",
)
@click.option(
    "--streaming/--no-streaming",
    "streaming",
    default=False,
    help="Use streaming mode (faster, no resume). Default: resumable.",
)
@click.option(
    "--keep-chapters",
    is_flag=True,
    help="Keep intermediate chapter files after merging.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Skip confirmation prompts.",
)
@click.pass_context
def phonemes_convert(
    ctx: click.Context,
    phoneme_file: Path,
    output: Optional[Path],
    output_format: Optional[str],
    voice: Optional[str],
    speed: float,
    use_gpu: Optional[bool],
    silence: float,
    segment_pause_min: Optional[float],
    segment_pause_max: Optional[float],
    paragraph_pause_min: Optional[float],
    paragraph_pause_max: Optional[float],
    chapters: Optional[str],
    title: Optional[str],
    author: Optional[str],
    cover: Optional[Path],
    voice_blend: Optional[str],
    voice_database: Optional[Path],
    streaming: bool,
    keep_chapters: bool,
    yes: bool,
) -> None:
    """Convert a pre-tokenized phoneme file to audio.

    PHONEME_FILE should be a JSON file created by 'ttsforge phonemes export'.

    By default, conversion is resumable (chapter-at-a-time mode). If interrupted,
    re-running the same command will resume from the last completed chapter.

    Use --streaming for faster conversion without resume capability.

    Examples:

        ttsforge phonemes convert book.phonemes.json

        ttsforge phonemes convert book.phonemes.json -v am_adam -o book.m4b

        ttsforge phonemes convert book.phonemes.json --chapters 1-5

        ttsforge phonemes convert book.phonemes.json --streaming
    """
    from .phoneme_conversion import (
        PhonemeConversionOptions,
        PhonemeConversionProgress,
        PhonemeConverter,
        parse_chapter_selection,
    )
    from .phonemes import PhonemeBook

    console.print(f"[bold]Loading:[/bold] {phoneme_file}")

    try:
        book = PhonemeBook.load(phoneme_file)
    except Exception as e:
        console.print(f"[red]Error loading phoneme file:[/red] {e}")
        sys.exit(1)

    # Load config for defaults
    config = load_config()
    model_path = ctx.obj.get("model_path") if ctx.obj else None
    voices_path = ctx.obj.get("voices_path") if ctx.obj else None

    # Get book info and metadata
    book_info = book.get_info()
    book_metadata = book_info.get("metadata", {})
    default_title = config.get("default_title", "Untitled")

    # Use CLI title/author if provided, otherwise use book metadata
    effective_title = (
        title if title is not None else book_info.get("title", default_title)
    )
    effective_author = (
        author if author is not None else book_metadata.get("author", "Unknown")
    )

    # Validate chapter selection if provided
    selected_indices: list[int] = []
    if chapters:
        try:
            selected_indices = parse_chapter_selection(chapters, len(book.chapters))
        except ValueError as e:
            console.print(f"[red]Invalid chapter selection:[/red] {e}")
            sys.exit(1)

    # Compute chapters range for filename
    # Use metadata chapters_range if converting all chapters from a partial export
    stored_chapters_range = book_metadata.get("chapters_range", "")
    if selected_indices:
        # New selection on top of potentially partial export
        chapters_range = format_chapters_range(selected_indices, len(book.chapters))
    else:
        # Use stored range if available
        chapters_range = stored_chapters_range

    # Determine output format and path
    fmt = output_format or config.get("default_format", "m4b")
    if output is None:
        output_template = config.get("output_filename_template", "{book_title}")
        output_filename = format_filename_template(
            output_template,
            book_title=effective_title,
            author=effective_author,
            input_stem=phoneme_file.stem,
            chapters_range=chapters_range,
            default_title=default_title,
        )
        # Append chapters range to filename if partial selection
        if chapters_range:
            output_filename = f"{output_filename}_{chapters_range}"
        output = phoneme_file.parent / f"{output_filename}.{fmt}"

    # Get voice
    if voice is None:
        voice = config.get("default_voice", "af_heart")

    # Get GPU setting
    gpu = use_gpu if use_gpu is not None else config.get("use_gpu", False)

    # Calculate total segments for selected chapters
    if selected_indices:
        selected_chapter_count = len(selected_indices)
        total_segments = sum(len(book.chapters[i].segments) for i in selected_indices)
    else:
        selected_chapter_count = len(book.chapters)
        total_segments = book.total_segments

    # Show info
    console.print(f"[dim]Title: {effective_title}[/dim]")
    if selected_indices:
        ch_count = f"{selected_chapter_count}/{book_info['chapters']}"
        console.print(
            f"[dim]Chapters: {ch_count} (selected), Segments: {total_segments}[/dim]"
        )
    else:
        console.print(
            f"[dim]Chapters: {book_info['chapters']}, "
            f"Segments: {book_info['segments']}, "
            f"Tokens: {book_info['tokens']:,}[/dim]"
        )

    if voice_blend:
        console.print(f"[dim]Voice blend: {voice_blend}[/dim]")
    else:
        console.print(f"[dim]Voice: {voice}, Speed: {speed}x[/dim]")

    console.print(f"[dim]Output: {output} (format: {fmt})[/dim]")
    mode_str = "streaming" if streaming else "resumable (chapter-at-a-time)"
    console.print(f"[dim]Mode: {mode_str}[/dim]")

    if not yes:
        if not Confirm.ask("Proceed with conversion?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Create conversion options
    options = PhonemeConversionOptions(
        voice=voice or config.get("default_voice", "af_heart"),
        speed=speed,
        output_format=fmt,
        use_gpu=gpu,
        silence_between_chapters=silence,
        segment_pause_min=(
            segment_pause_min
            if segment_pause_min is not None
            else config.get("segment_pause_min", 0.1)
        ),
        segment_pause_max=(
            segment_pause_max
            if segment_pause_max is not None
            else config.get("segment_pause_max", 0.3)
        ),
        paragraph_pause_min=(
            paragraph_pause_min
            if paragraph_pause_min is not None
            else config.get("paragraph_pause_min", 0.5)
        ),
        paragraph_pause_max=(
            paragraph_pause_max
            if paragraph_pause_max is not None
            else config.get("paragraph_pause_max", 1.0)
        ),
        title=effective_title,
        author=effective_author,
        cover_image=cover,
        voice_blend=voice_blend,
        voice_database=voice_database,
        chapters=chapters,
        resume=not streaming,  # Resume only in chapter-at-a-time mode
        keep_chapter_files=keep_chapters,
        chapter_filename_template=config.get(
            "chapter_filename_template",
            "{chapter_num:03d}_{book_title}_{chapter_title}",
        ),
        model_path=model_path,
        voices_path=voices_path,
    )

    # Progress tracking with Rich
    progress_bar: Optional[Progress] = None
    task_id = None

    def log_callback(message: str, level: str) -> None:
        """Handle log messages."""
        if level == "warning":
            console.print(f"[yellow]{message}[/yellow]")
        elif level == "error":
            console.print(f"[red]{message}[/red]")
        else:
            console.print(f"[dim]{message}[/dim]")

    def progress_callback(prog: PhonemeConversionProgress) -> None:
        """Update progress display."""
        nonlocal progress_bar, task_id
        if progress_bar is not None and task_id is not None:
            ch_progress = f"Ch {prog.current_chapter}/{prog.total_chapters}"
            progress_bar.update(
                task_id,
                completed=prog.segments_processed,
                description=f"[cyan]{ch_progress}[/cyan]",
            )

    # Create converter
    converter = PhonemeConverter(
        book=book,
        options=options,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )

    # Run conversion with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[segment_info]}[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        progress_bar = progress
        task_id = progress.add_task(
            "[cyan]Converting...[/cyan]",
            total=total_segments,
            segment_info="",
        )

        # Choose conversion mode
        if streaming:
            result = converter.convert_streaming(output)
        else:
            result = converter.convert(output)

        # Mark complete
        progress.update(task_id, completed=total_segments)

    # Show result
    if result.success:
        console.print("\n[green]Conversion complete![/green]")
        console.print(f"[bold]Output:[/bold] {result.output_path}")
        if result.duration > 0:
            from .utils import format_duration

            console.print(f"[dim]Duration: {format_duration(result.duration)}[/dim]")
    else:
        console.print(f"\n[red]Conversion failed:[/red] {result.error_message}")
        sys.exit(1)


@phonemes.command("preview")
@click.argument("text")
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    default="a",
    help="Language code for phonemization.",
)
@click.option(
    "--tokens",
    is_flag=True,
    help="Show token IDs in addition to phonemes.",
)
@click.option(
    "--vocab-version",
    type=str,
    default="v1.0",
    help="Vocabulary version to use.",
)
def phonemes_preview(
    text: str,
    language: str,
    tokens: bool,
    vocab_version: str,
) -> None:
    """Preview phonemes for given text.

    Shows how text will be converted to phonemes and optionally tokens.

    Examples:

        ttsforge phonemes preview "Hello world"

        ttsforge phonemes preview "Hello world" --tokens

        ttsforge phonemes preview "Hello world" --language b
    """
    from .onnx_backend import LANG_CODE_TO_ONNX
    from .tokenizer import Tokenizer

    espeak_lang = LANG_CODE_TO_ONNX.get(language, "en-us")

    try:
        tokenizer = Tokenizer(vocab_version=vocab_version)
    except Exception as e:
        console.print(f"[red]Error initializing tokenizer:[/red] {e}")
        sys.exit(1)

    phonemes = tokenizer.phonemize(text, lang=espeak_lang)
    readable = tokenizer.format_readable(text, lang=espeak_lang)

    console.print(f"[bold]Text:[/bold] {text}")
    lang_desc = LANGUAGE_DESCRIPTIONS.get(language, language)
    console.print(f"[bold]Language:[/bold] {lang_desc} ({espeak_lang})")
    console.print(f"[bold]Phonemes:[/bold] {phonemes}")
    console.print(f"[bold]Readable:[/bold] {readable}")

    if tokens:
        token_ids = tokenizer.tokenize(phonemes)
        console.print(f"[bold]Tokens:[/bold] {token_ids}")
        console.print(f"[dim]Token count: {len(token_ids)}[/dim]")


@phonemes.command("info")
@click.argument("phoneme_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--stats",
    is_flag=True,
    help="Show detailed token statistics.",
)
def phonemes_info(phoneme_file: Path, stats: bool) -> None:
    """Show information about a phoneme file.

    PHONEME_FILE should be a JSON file created by 'ttsforge phonemes export'.

    Use --stats to show detailed token statistics (min, median, mean, max).
    """
    from .phonemes import PhonemeBook

    try:
        book = PhonemeBook.load(phoneme_file)
    except Exception as e:
        console.print(f"[red]Error loading phoneme file:[/red] {e}")
        sys.exit(1)

    info = book.get_info()

    table = Table(title=f"Phoneme File: {phoneme_file.name}")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Title", info["title"])
    table.add_row("Vocabulary", info["vocab_version"])
    table.add_row("Language", info["lang"])
    table.add_row("Chapters", str(info["chapters"]))
    table.add_row("Segments", str(info["segments"]))
    table.add_row("Tokens", f"{info['tokens']:,}")
    table.add_row("Phonemes", f"{info['phonemes']:,}")

    if info.get("metadata"):
        for key, value in info["metadata"].items():
            table.add_row(f"Meta: {key}", str(value))

    console.print(table)

    # Collect token counts per segment for statistics
    token_counts = [len(seg.tokens) for _, seg in book.iter_segments()]
    char_counts = [len(seg.text) for _, seg in book.iter_segments()]
    phoneme_counts = [len(seg.phonemes) for _, seg in book.iter_segments()]

    if token_counts and stats:
        import statistics

        # Token statistics
        console.print("\n[bold]Segment Statistics:[/bold]")
        stats_table = Table(show_header=True)
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Tokens", justify="right")
        stats_table.add_column("Characters", justify="right")
        stats_table.add_column("Phonemes", justify="right")

        stats_table.add_row(
            "Count",
            str(len(token_counts)),
            str(len(char_counts)),
            str(len(phoneme_counts)),
        )
        stats_table.add_row(
            "Min",
            str(min(token_counts)),
            str(min(char_counts)),
            str(min(phoneme_counts)),
        )
        stats_table.add_row(
            "Max",
            str(max(token_counts)),
            str(max(char_counts)),
            str(max(phoneme_counts)),
        )
        stats_table.add_row(
            "Mean",
            f"{statistics.mean(token_counts):.1f}",
            f"{statistics.mean(char_counts):.1f}",
            f"{statistics.mean(phoneme_counts):.1f}",
        )
        stats_table.add_row(
            "Median",
            f"{statistics.median(token_counts):.1f}",
            f"{statistics.median(char_counts):.1f}",
            f"{statistics.median(phoneme_counts):.1f}",
        )
        if len(token_counts) > 1:
            stats_table.add_row(
                "Std Dev",
                f"{statistics.stdev(token_counts):.1f}",
                f"{statistics.stdev(char_counts):.1f}",
                f"{statistics.stdev(phoneme_counts):.1f}",
            )

        console.print(stats_table)

        # Token distribution histogram (simple text-based)
        console.print("\n[bold]Token Distribution:[/bold]")
        # Create buckets
        buckets = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, float("inf")]
        bucket_labels = [
            "0-49",
            "50-99",
            "100-149",
            "150-199",
            "200-249",
            "250-299",
            "300-349",
            "350-399",
            "400-449",
            "450-499",
            "500+",
        ]
        bucket_counts = [0] * (len(buckets) - 1)

        for count in token_counts:
            for i in range(len(buckets) - 1):
                if buckets[i] <= count < buckets[i + 1]:
                    bucket_counts[i] += 1
                    break

        max_count = max(bucket_counts) if bucket_counts else 1
        bar_width = 30

        for label, count in zip(bucket_labels, bucket_counts):
            if count > 0 or label in [
                "0-49",
                "50-99",
                "100-149",
            ]:  # Always show first few
                bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
                bar = "█" * bar_len
                console.print(f"  {label:>8} │ {bar:<{bar_width}} {count:>4}")

    # Show chapters
    console.print("\n[bold]Chapters:[/bold]")
    chapter_table = Table(show_header=True)
    chapter_table.add_column("#", style="dim", width=4)
    chapter_table.add_column("Title")
    chapter_table.add_column("Segments", justify="right")
    chapter_table.add_column("Tokens", justify="right")

    for i, chapter in enumerate(book.chapters, 1):
        chapter_table.add_row(
            str(i),
            chapter.title[:50],
            str(len(chapter.segments)),
            f"{chapter.total_tokens:,}",
        )

    console.print(chapter_table)


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
