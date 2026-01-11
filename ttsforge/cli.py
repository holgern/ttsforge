"""CLI interface for ttsforge - convert EPUB to audiobooks."""

import re
import sys
from pathlib import Path
from typing import Optional, cast

import click
from pykokoro.onnx_backend import (
    DEFAULT_MODEL_QUALITY,
    MODEL_QUALITY_FILES,
    ModelQuality,
    are_models_downloaded,
    are_voices_downloaded,
    download_all_voices,
    download_config,
    download_model,
    get_config_path,
    get_model_dir,
    get_model_path,
    get_voices_bin_path,
    is_config_downloaded,
    is_model_downloaded,
)
from pykokoro.onnx_backend import (
    VOICE_NAMES_V1_0 as VOICE_NAMES,
)
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
from .utils import (
    format_chapters_range,
    format_filename_template,
    format_size,
    load_config,
    reset_config,
    save_config,
)

console = Console()


def parse_voice_parameter(voice: str) -> tuple[str | None, str | None]:
    """Parse voice parameter to detect if it's a single voice or a blend.

    Args:
        voice: Voice parameter (e.g., 'af_sky' or 'af_nicole:50,am_michael:50')

    Returns:
        Tuple of (voice, voice_blend) where one will be None

    Examples:
        >>> parse_voice_parameter('af_sky')
        ('af_sky', None)
        >>> parse_voice_parameter('af_nicole:50,am_michael:50')
        (None, 'af_nicole:50,am_michael:50')
    """
    # Detect if it's a blend (contains both : and ,)
    if ":" in voice and "," in voice:
        return (None, voice)
    else:
        return (voice, None)


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
    "--lang",
    type=str,
    default=None,
    help="Override language for phonemization (e.g., 'de', 'fr', 'en-us'). "
    "By default, language is determined from the voice.",
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
    "--pause-clause",
    type=float,
    default=None,
    help="Pause after clauses in seconds (default: 0.25).",
)
@click.option(
    "--pause-sentence",
    type=float,
    default=None,
    help="Pause after sentences in seconds (default: 0.2).",
)
@click.option(
    "--pause-paragraph",
    type=float,
    default=None,
    help="Pause after paragraphs in seconds (default: 0.75).",
)
@click.option(
    "--pause-variance",
    type=float,
    default=None,
    help="Random variance added to pauses in seconds (default: 0.05).",
)
@click.option(
    "--trim-silence/--no-trim-silence",
    "trim_silence",
    default=None,
    help="Trim leading/trailing silence from audio (default: enabled).",
)
@click.option(
    "--announce-chapters/--no-announce-chapters",
    "announce_chapters",
    default=None,
    help="Read chapter titles aloud before chapter content (default: enabled).",
)
@click.option(
    "--chapter-pause",
    type=float,
    default=None,
    help="Pause duration after chapter title announcement in seconds (default: 2.0).",
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
    "--generate-ssmd",
    "generate_ssmd_only",
    is_flag=True,
    help="Generate only SSMD files without creating audio (for manual editing).",
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
@click.option(
    "--use-mixed-language",
    "use_mixed_language",
    is_flag=True,
    help="Enable mixed-language support (auto-detect multiple languages in text).",
)
@click.option(
    "--mixed-language-primary",
    "mixed_language_primary",
    type=str,
    help="Primary language for mixed-language mode (e.g., 'de', 'en-us').",
)
@click.option(
    "--mixed-language-allowed",
    "mixed_language_allowed",
    type=str,
    help="Comma-separated list of allowed languages (e.g., 'de,en-us').",
)
@click.option(
    "--mixed-language-confidence",
    "mixed_language_confidence",
    type=float,
    help=(
        "Detection confidence threshold for mixed-language mode "
        "(0.0-1.0, default: 0.7)."
    ),
)
@click.option(
    "--phoneme-dict",
    "phoneme_dictionary_path",
    type=click.Path(exists=True),
    help="Path to custom phoneme dictionary JSON file for pronunciation overrides.",
)
@click.option(
    "--phoneme-dict-case-sensitive",
    "phoneme_dict_case_sensitive",
    is_flag=True,
    help="Make phoneme dictionary matching case-sensitive (default: case-insensitive).",
)
@click.pass_context
def convert(
    ctx: click.Context,
    epub_file: Path,
    output: Optional[Path],
    output_format: Optional[str],
    voice: Optional[str],
    language: Optional[str],
    lang: Optional[str],
    speed: Optional[float],
    use_gpu: Optional[bool],
    chapters: Optional[str],
    silence: Optional[float],
    pause_clause: Optional[float],
    pause_sentence: Optional[float],
    pause_paragraph: Optional[float],
    pause_variance: Optional[float],
    trim_silence: Optional[bool],
    announce_chapters: Optional[bool],
    chapter_pause: Optional[float],
    title: Optional[str],
    author: Optional[str],
    cover: Optional[Path],
    yes: bool,
    verbose: bool,
    split_mode: Optional[str],
    resume: bool,
    generate_ssmd_only: bool,
    fresh: bool,
    keep_chapter_files: bool,
    voice_blend: Optional[str],
    voice_database: Optional[Path],
    use_mixed_language: bool,
    mixed_language_primary: Optional[str],
    mixed_language_allowed: Optional[str],
    mixed_language_confidence: Optional[float],
    phoneme_dictionary_path: Optional[str],
    phoneme_dict_case_sensitive: bool,
) -> None:
    """Convert an EPUB file to an audiobook.

    EPUB_FILE is the path to the EPUB file to convert.
    """
    config = load_config()
    model_path = ctx.obj.get("model_path") if ctx.obj else None
    voices_path = ctx.obj.get("voices_path") if ctx.obj else None

    # Get format first (needed for output path construction)
    fmt = output_format or config.get("default_format", "m4b")

    # Load chapters from input file
    console.print(f"[bold]Loading:[/bold] {epub_file}")

    from .input_reader import InputReader

    # Parse input file
    try:
        reader = InputReader(epub_file)
    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        sys.exit(1)

    # Get metadata
    metadata = reader.get_metadata()
    default_title = config.get("default_title", "Untitled")
    epub_title = metadata.title or default_title
    epub_author = metadata.authors[0] if metadata.authors else "Unknown"
    epub_language = metadata.language

    # Use CLI title/author if provided, otherwise use metadata
    effective_title = title or epub_title
    effective_author = author or epub_author

    # Extract chapters
    with console.status("Extracting chapters..."):
        epub_chapters = reader.get_chapters()

    if not epub_chapters:
        console.print("[red]Error:[/red] No chapters found in file.")
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

    # Parse mixed_language_allowed from comma-separated string
    parsed_mixed_language_allowed = None
    if mixed_language_allowed:
        parsed_mixed_language_allowed = [
            lang.strip() for lang in mixed_language_allowed.split(",")
        ]

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
        lang=lang,
        use_mixed_language=use_mixed_language
        or config.get("use_mixed_language", False),
        mixed_language_primary=mixed_language_primary
        or config.get("mixed_language_primary"),
        mixed_language_allowed=parsed_mixed_language_allowed
        or config.get("mixed_language_allowed"),
        mixed_language_confidence=mixed_language_confidence
        if mixed_language_confidence is not None
        else config.get("mixed_language_confidence", 0.7),
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
        lang=lang or config.get("phonemization_lang"),
        use_mixed_language=(
            use_mixed_language or config.get("use_mixed_language", False)
        ),
        mixed_language_primary=(
            mixed_language_primary or config.get("mixed_language_primary")
        ),
        mixed_language_allowed=(
            parsed_mixed_language_allowed or config.get("mixed_language_allowed")
        ),
        mixed_language_confidence=(
            mixed_language_confidence
            if mixed_language_confidence is not None
            else config.get("mixed_language_confidence", 0.7)
        ),
        phoneme_dictionary_path=(
            phoneme_dictionary_path or config.get("phoneme_dictionary_path")
        ),
        phoneme_dict_case_sensitive=(
            phoneme_dict_case_sensitive
            or config.get("phoneme_dict_case_sensitive", False)
        ),
        pause_clause=(
            pause_clause
            if pause_clause is not None
            else config.get("pause_clause", 0.25)
        ),
        pause_sentence=(
            pause_sentence
            if pause_sentence is not None
            else config.get("pause_sentence", 0.2)
        ),
        pause_paragraph=(
            pause_paragraph
            if pause_paragraph is not None
            else config.get("pause_paragraph", 0.75)
        ),
        pause_variance=(
            pause_variance
            if pause_variance is not None
            else config.get("pause_variance", 0.05)
        ),
        trim_silence=(
            trim_silence
            if trim_silence is not None
            else config.get("trim_silence", True)
        ),
        announce_chapters=(
            announce_chapters
            if announce_chapters is not None
            else config.get("announce_chapters", True)
        ),
        chapter_pause_after_title=(
            chapter_pause
            if chapter_pause is not None
            else config.get("chapter_pause_after_title", 2.0)
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
        generate_ssmd_only=generate_ssmd_only,
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
    """List chapters in a file.

    EPUB_FILE is the path to the file (EPUB, TXT, or PDF).
    """
    from .input_reader import InputReader

    with console.status("Loading file..."):
        try:
            reader = InputReader(epub_file)
            chapters = reader.get_chapters()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    if not chapters:
        console.print("[yellow]No chapters found in file.[/yellow]")
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
    """Show metadata and information about a file.

    EPUB_FILE is the path to the file (EPUB, TXT, or PDF).
    """
    from .input_reader import InputReader

    # Parse file
    with console.status("Loading file..."):
        try:
            reader = InputReader(epub_file)
            metadata = reader.get_metadata()
            chapters = reader.get_chapters()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    total_chars = sum(ch.char_count for ch in chapters) if chapters else 0

    # Display info
    console.print(Panel(f"[bold]{epub_file.name}[/bold]", title="File Information"))

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
@click.option(
    "-v",
    "--voice",
    type=str,
    help=(
        "TTS voice to use or voice blend "
        "(e.g., 'af_sky' or 'af_nicole:50,am_michael:50')."
    ),
)
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    help="Language for TTS.",
)
@click.option(
    "--lang",
    type=str,
    default=None,
    help="Override language for phonemization (e.g., 'de', 'fr', 'en-us').",
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
@click.option(
    "-p",
    "--play",
    "play_audio",
    is_flag=True,
    help="Play audio directly (also saves to file if -o specified).",
)
@click.option(
    "--use-mixed-language",
    "use_mixed_language",
    is_flag=True,
    help="Enable mixed-language support (auto-detect multiple languages in text).",
)
@click.option(
    "--mixed-language-primary",
    "mixed_language_primary",
    type=str,
    help="Primary language for mixed-language mode (e.g., 'de', 'en-us').",
)
@click.option(
    "--mixed-language-allowed",
    "mixed_language_allowed",
    type=str,
    help="Comma-separated list of allowed languages (e.g., 'de,en-us').",
)
@click.option(
    "--mixed-language-confidence",
    "mixed_language_confidence",
    type=float,
    help=(
        "Detection confidence threshold for mixed-language mode "
        "(0.0-1.0, default: 0.7)."
    ),
)
@click.option(
    "--phoneme-dict",
    "phoneme_dictionary_path",
    type=click.Path(exists=True),
    help="Path to custom phoneme dictionary JSON file for pronunciation overrides.",
)
@click.option(
    "--phoneme-dict-case-sensitive",
    "phoneme_dict_case_sensitive",
    is_flag=True,
    help="Make phoneme dictionary matching case-sensitive (default: case-insensitive).",
)
@click.pass_context
def sample(
    ctx: click.Context,
    text: Optional[str],
    output: Optional[Path],
    output_format: str,
    voice: Optional[str],
    language: Optional[str],
    lang: Optional[str],
    speed: Optional[float],
    use_gpu: Optional[bool],
    split_mode: Optional[str],
    play_audio: bool,
    verbose: bool,
    use_mixed_language: bool,
    mixed_language_primary: Optional[str],
    mixed_language_allowed: Optional[str],
    mixed_language_confidence: Optional[float],
    phoneme_dictionary_path: Optional[str],
    phoneme_dict_case_sensitive: bool,
) -> None:
    """Generate a sample audio file to test TTS settings.

    If no TEXT is provided, uses a default sample text.

    Examples:

        ttsforge sample

        ttsforge sample "Hello, this is a test."

        ttsforge sample --voice am_adam --speed 1.2 -o test.wav

        ttsforge sample --play  # Play directly without saving

        ttsforge sample --play -o test.wav  # Play and save to file
    """
    import tempfile

    from .conversion import ConversionOptions, TTSConverter

    # Get model path from global context
    model_path = ctx.obj.get("model_path") if ctx.obj else None
    voices_path = ctx.obj.get("voices_path") if ctx.obj else None

    # Use default text if none provided
    sample_text = text or DEFAULT_SAMPLE_TEXT

    # Handle output path for playback mode
    temp_dir: Optional[str] = None
    save_output = output is not None or not play_audio

    if play_audio and output is None:
        # Create temp file for playback only
        temp_dir = tempfile.mkdtemp()
        output = Path(temp_dir) / "sample.wav"
        output_format = "wav"  # Force WAV for playback
    elif output is None:
        output = Path(f"./sample.{output_format}")
    elif output.suffix == "":
        # If no extension provided, add the format
        output = output.with_suffix(f".{output_format}")

    # Load config for defaults
    user_config = load_config()

    # Parse mixed_language_allowed from comma-separated string
    parsed_mixed_language_allowed = None
    if mixed_language_allowed:
        parsed_mixed_language_allowed = [
            lang_item.strip() for lang_item in mixed_language_allowed.split(",")
        ]

    # Auto-detect if voice is a blend
    voice_value = voice or user_config.get("voice") or "af_bella"
    parsed_voice, parsed_voice_blend = parse_voice_parameter(voice_value)

    # Build conversion options (use ConversionOptions defaults if not specified)
    options = ConversionOptions(
        voice=parsed_voice or "af_bella",
        voice_blend=parsed_voice_blend,
        language=language or user_config.get("language") or "a",
        speed=speed or user_config.get("speed", 1.0),
        output_format=output_format,
        use_gpu=use_gpu if use_gpu is not None else user_config.get("use_gpu", True),
        split_mode=split_mode or user_config.get("split_mode", "auto"),
        lang=lang or user_config.get("phonemization_lang"),
        use_mixed_language=(
            use_mixed_language or user_config.get("use_mixed_language", False)
        ),
        mixed_language_primary=(
            mixed_language_primary or user_config.get("mixed_language_primary")
        ),
        mixed_language_allowed=(
            parsed_mixed_language_allowed or user_config.get("mixed_language_allowed")
        ),
        mixed_language_confidence=(
            mixed_language_confidence
            if mixed_language_confidence is not None
            else user_config.get("mixed_language_confidence", 0.7)
        ),
        phoneme_dictionary_path=(
            phoneme_dictionary_path or user_config.get("phoneme_dictionary_path")
        ),
        phoneme_dict_case_sensitive=(
            phoneme_dict_case_sensitive
            or user_config.get("phoneme_dict_case_sensitive", False)
        ),
        model_path=model_path,
        voices_path=voices_path,
    )

    # Always show settings
    if options.voice_blend:
        console.print(f"[dim]Voice Blend:[/dim] {options.voice_blend}")
    else:
        console.print(f"[dim]Voice:[/dim] {options.voice}")
    lang_desc = LANGUAGE_DESCRIPTIONS.get(options.language, "Unknown")
    console.print(f"[dim]Language:[/dim] {options.language} ({lang_desc})")
    if options.lang:
        console.print(f"[dim]Phonemization Lang:[/dim] {options.lang} (override)")
    console.print(f"[dim]Speed:[/dim] {options.speed}")
    console.print(f"[dim]Format:[/dim] {options.output_format}")
    console.print(f"[dim]Split mode:[/dim] {options.split_mode}")
    console.print(f"[dim]GPU:[/dim] {'enabled' if options.use_gpu else 'disabled'}")

    if verbose:
        text_preview = sample_text[:100]
        ellipsis = "..." if len(sample_text) > 100 else ""
        console.print(f"[dim]Text:[/dim] {text_preview}{ellipsis}")
        if save_output:
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
            # Handle playback if requested
            if play_audio:
                import sounddevice as sd  # type: ignore[import-untyped]
                import soundfile as sf

                audio_data, sample_rate = sf.read(str(output))
                console.print("[dim]Playing audio...[/dim]")
                sd.play(audio_data, sample_rate)
                sd.wait()
                console.print("[green]Playback complete.[/green]")

            # Report save location (if not temp file)
            if save_output:
                console.print(f"[green]Sample saved to:[/green] {output}")

            # Cleanup temp file if needed
            if temp_dir is not None:
                import shutil

                shutil.rmtree(temp_dir)
        else:
            console.print(f"[red]Error:[/red] {result.error_message}")
            raise SystemExit(1)

    except Exception as e:
        console.print(f"[red]Error generating sample:[/red] {e}")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        # Cleanup temp dir on error
        if temp_dir is not None:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
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
@click.option(
    "-p",
    "--play",
    "play_audio",
    is_flag=True,
    help="Play audio directly (also saves to file if -o specified).",
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
    play_audio: bool,
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

        ttsforge demo --play  # Play directly without saving

        ttsforge demo -v af_heart --play  # Play a single voice demo
    """
    import numpy as np
    from pykokoro import Kokoro, VoiceBlend

    config = load_config()
    gpu = use_gpu if use_gpu is not None else config.get("use_gpu", False)
    model_path = ctx.obj.get("model_path") if ctx.obj else None
    voices_path = ctx.obj.get("voices_path") if ctx.obj else None

    # Playback is not compatible with --separate or --blend-presets (multiple files)
    if play_audio and separate:
        console.print(
            "[red]Error:[/red] --play is not compatible with --separate. "
            "Use --play without --separate to play a combined demo."
        )
        sys.exit(1)
    if play_audio and blend_presets:
        console.print(
            "[red]Error:[/red] --play is not compatible with --blend-presets. "
            "Use --play with a single --blend instead."
        )
        sys.exit(1)

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

        # For playback with single blend, we don't need an output directory
        save_output = output is not None or not play_audio

        if save_output:
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
            kokoro = Kokoro(
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
                        voices_str = " and ".join(voice_names)
                        demo_text = (
                            f"This is a blend of {voices_str} speaking together."
                        )

                    # Generate audio with blended voice
                    samples, sr = kokoro.create(
                        demo_text, voice=voice_blend, speed=speed
                    )

                    # Handle playback
                    if play_audio:
                        import sounddevice as sd  # type: ignore[import-untyped]

                        progress.console.print(f"  [dim]Playing {description}...[/dim]")
                        sd.play(samples, sr)
                        sd.wait()
                        progress.console.print(
                            f"  [green]{description}[/green]: Playback complete"
                        )

                    # Save to file if output specified
                    if save_output and output is not None:
                        import soundfile as sf

                        filename = blend_to_filename(blend_str) + ".wav"
                        voice_file = output / filename
                        sf.write(str(voice_file), samples, sr)
                        if not play_audio:
                            progress.console.print(
                                f"  [green]{description}[/green]: {voice_file}"
                            )

                except Exception as e:
                    console.print(f"  [red]{blend_str}[/red]: Failed - {e}")

                progress.advance(task)

        if save_output:
            num_saved = len(blends_to_process)
            console.print(
                f"\n[green]Saved {num_saved} voice blend demos to:[/green] {output}"
            )
        elif play_audio:
            console.print("\n[green]Playback complete.[/green]")
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

    # Determine output path and whether to save
    save_output = output is not None or not play_audio

    if separate:
        if output is None:
            output = Path("./voice_demos")
        output.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Output directory:[/bold] {output}")
    elif save_output:
        if output is None:
            output = Path("./voices_demo.wav")
        console.print(f"[bold]Output file:[/bold] {output}")

    console.print(f"[dim]Voices: {len(selected_voices)}[/dim]")
    console.print(f"[dim]Speed: {speed}x[/dim]")
    console.print(f"[dim]GPU: {'enabled' if gpu else 'disabled'}[/dim]")

    # Initialize TTS engine
    try:
        kokoro = Kokoro(
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

                if separate and output is not None:
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

    # Handle combined output (not separate mode)
    if not separate and all_samples:
        combined = np.concatenate(all_samples)

        # Play audio if requested
        if play_audio:
            import sounddevice as sd  # type: ignore[import-untyped]

            console.print("[dim]Playing audio...[/dim]")
            sd.play(combined, sample_rate)
            sd.wait()
            console.print("[green]Playback complete.[/green]")

        # Save to file if output specified or not in play-only mode
        if save_output and output is not None:
            import soundfile as sf

            sf.write(str(output), combined, sample_rate)
            console.print(f"[green]Demo saved to:[/green] {output}")

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
@click.option(
    "--quality",
    "-q",
    type=click.Choice(list(MODEL_QUALITY_FILES.keys())),
    default=None,
    help="Model quality/quantization level. Default: from config or fp32.",
)
def download(force: bool, quality: Optional[str]) -> None:
    """Download ONNX model and voice files required for TTS.

    Downloads from Hugging Face (onnx-community/Kokoro-82M-v1.0-ONNX).

    Quality options:
      fp32     - Full precision (326 MB) - Best quality, default
      fp16     - Half precision (163 MB) - Good quality, smaller
      q8       - 8-bit quantized (92 MB) - Good quality, compact
      q8f16    - 8-bit with fp16 (86 MB) - Smallest file
      q4       - 4-bit quantized (305 MB)
      q4f16    - 4-bit with fp16 (155 MB)
      uint8    - Unsigned 8-bit (177 MB)
      uint8f16 - Unsigned 8-bit with fp16 (114 MB)
    """
    # Get quality from config if not specified
    if quality is None:
        cfg = load_config()
        quality = cfg.get("model_quality", DEFAULT_MODEL_QUALITY)

    # Cast to ModelQuality - safe because click.Choice validates input
    # and config uses a valid default
    model_quality = cast(ModelQuality, quality)

    model_dir = get_model_dir()
    console.print(f"[bold]Model directory:[/bold] {model_dir}")
    console.print(f"[bold]Model quality:[/bold] {model_quality}")

    # Check if already downloaded
    if are_models_downloaded(model_quality) and not force:
        console.print("[green]All model files are already downloaded.[/green]")
        model_path = get_model_path(model_quality)
        voices_path = get_voices_bin_path()
        config_path = get_config_path()

        if model_path.exists():
            console.print(
                f"  {model_path.name}: {format_size(model_path.stat().st_size)}"
            )
        if voices_path.exists():
            console.print(
                f"  voices.bin.npz: {format_size(voices_path.stat().st_size)}"
            )
        if config_path.exists():
            console.print(f"  config.json: {format_size(config_path.stat().st_size)}")
        return

    console.print("Downloading ONNX model files from Hugging Face...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Download config.json
        if not is_config_downloaded() or force:
            config_task = progress.add_task("Downloading config.json...", total=100)

            def config_progress(current: int, total: int) -> None:
                if total > 0:
                    progress.update(config_task, completed=(current / total) * 100)

            try:
                download_config(progress_callback=config_progress, force=force)
                progress.update(config_task, completed=100)
                config_path = get_config_path()
                size = format_size(config_path.stat().st_size)
                console.print(f"  [green]config.json: {size}[/green]")
            except Exception as e:
                console.print(f"  [red]config.json: Failed - {e}[/red]")
                sys.exit(1)
        else:
            console.print("  [dim]config.json: already downloaded[/dim]")

        # Download model
        model_path = get_model_path(model_quality)
        model_filename = model_path.name
        if not is_model_downloaded(model_quality) or force:
            model_task = progress.add_task(
                f"Downloading {model_filename}...", total=100
            )

            def model_progress(current: int, total: int) -> None:
                if total > 0:
                    progress.update(model_task, completed=(current / total) * 100)

            try:
                download_model(
                    model_quality,
                    progress_callback=model_progress,
                    force=force,
                )
                progress.update(model_task, completed=100)
                model_path = get_model_path(model_quality)
                size = format_size(model_path.stat().st_size)
                console.print(f"  [green]{model_filename}: {size}[/green]")
            except Exception as e:
                console.print(f"  [red]{model_filename}: Failed - {e}[/red]")
                sys.exit(1)
        else:
            console.print(f"  [dim]{model_filename}: already downloaded[/dim]")

        # Download voices
        if not are_voices_downloaded() or force:
            voices_task = progress.add_task(
                f"Downloading voices (0/{len(VOICE_NAMES)})...", total=100
            )

            def voices_progress(voice_name: str, current: int, total: int) -> None:
                progress.update(
                    voices_task,
                    description=f"Downloading voices ({current}/{total})...",
                    completed=(current / total) * 100,
                )

            try:
                download_all_voices(progress_callback=voices_progress, force=force)
                progress.update(voices_task, completed=100)
                voices_path = get_voices_bin_path()
                size = format_size(voices_path.stat().st_size)
                console.print(f"  [green]voices.bin.npz: {size}[/green]")
            except Exception as e:
                console.print(f"  [red]voices: Failed - {e}[/red]")
                sys.exit(1)
        else:
            console.print("  [dim]voices.bin.npz: already downloaded[/dim]")

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
    from pykokoro import Tokenizer

    from .input_reader import InputReader
    from .phonemes import PhonemeBook

    config = load_config()

    console.print(f"[bold]Loading:[/bold] {epub_file}")

    # Parse file
    try:
        reader = InputReader(epub_file)
        metadata = reader.get_metadata()
        epub_chapters = reader.get_chapters()
    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        sys.exit(1)

    if not epub_chapters:
        console.print("[red]Error:[/red] No chapters found in file.")
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
    from pykokoro.onnx_backend import LANG_CODE_TO_ONNX

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

            # Remove <<CHAPTER: ...>> markers that epub2text adds
            # at the start of content since we now announce chapter titles
            # separately
            content = ch.text
            content = re.sub(
                r"^\s*<<CHAPTER:[^>]*>>\s*\n*", "", content, count=1, flags=re.MULTILINE
            )

            # Pass entire chapter text - add_text handles splitting based on split_mode
            if content.strip():
                chapter.add_text(
                    content,
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
    "--pause-clause",
    type=float,
    default=None,
    help="Pause after clauses in seconds (default: 0.25).",
)
@click.option(
    "--pause-sentence",
    type=float,
    default=None,
    help="Pause after sentences in seconds (default: 0.2).",
)
@click.option(
    "--pause-paragraph",
    type=float,
    default=None,
    help="Pause after paragraphs in seconds (default: 0.75).",
)
@click.option(
    "--pause-variance",
    type=float,
    default=None,
    help="Random variance added to pauses in seconds (default: 0.05).",
)
@click.option(
    "--trim-silence/--no-trim-silence",
    "trim_silence",
    default=None,
    help="Trim leading/trailing silence from audio (default: enabled).",
)
@click.option(
    "--announce-chapters/--no-announce-chapters",
    "announce_chapters",
    default=None,
    help="Read chapter titles aloud before chapter content (default: enabled).",
)
@click.option(
    "--chapter-pause",
    type=float,
    default=None,
    help="Pause duration after chapter title announcement in seconds (default: 2.0).",
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
    pause_clause: Optional[float],
    pause_sentence: Optional[float],
    pause_paragraph: Optional[float],
    pause_variance: Optional[float],
    trim_silence: Optional[bool],
    announce_chapters: Optional[bool],
    chapter_pause: Optional[float],
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
        pause_clause=(
            pause_clause
            if pause_clause is not None
            else config.get("pause_clause", 0.25)
        ),
        pause_sentence=(
            pause_sentence
            if pause_sentence is not None
            else config.get("pause_sentence", 0.2)
        ),
        pause_paragraph=(
            pause_paragraph
            if pause_paragraph is not None
            else config.get("pause_paragraph", 0.75)
        ),
        pause_variance=(
            pause_variance
            if pause_variance is not None
            else config.get("pause_variance", 0.05)
        ),
        trim_silence=(
            trim_silence
            if trim_silence is not None
            else config.get("trim_silence", True)
        ),
        announce_chapters=(
            announce_chapters
            if announce_chapters is not None
            else config.get("announce_chapters", True)
        ),
        chapter_pause_after_title=(
            chapter_pause
            if chapter_pause is not None
            else config.get("chapter_pause_after_title", 2.0)
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
    type=str,
    default="a",
    help="Language code for phonemization (e.g., 'de', 'en-us', 'a' for auto).",
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
@click.option(
    "-p",
    "--play",
    is_flag=True,
    help="Play audio preview of the text.",
)
@click.option(
    "-v",
    "--voice",
    type=str,
    default="af_sky",
    help=(
        "Voice to use for audio preview, or voice blend "
        "(e.g., 'af_nicole:50,am_michael:50')."
    ),
)
@click.option(
    "--phoneme-dict",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to custom phoneme dictionary file.",
)
def phonemes_preview(
    text: str,
    language: str,
    tokens: bool,
    vocab_version: str,
    play: bool,
    voice: str,
    phoneme_dict: Path | None,
) -> None:
    """Preview phonemes for given text.

    Shows how text will be converted to phonemes and optionally tokens.
    Use --play to hear the audio output.

    Examples:

        ttsforge phonemes preview "Hello world"

        ttsforge phonemes preview "Hello world" --tokens

        ttsforge phonemes preview "Hello world" --language de

        ttsforge phonemes preview "König" --language de --play

        ttsforge phonemes preview "Hermione" --play --phoneme-dict custom.json

        ttsforge phonemes preview "Hello" --play --voice "af_nicole:50,am_michael:50"
    """
    from pykokoro import Tokenizer
    from pykokoro.onnx_backend import LANG_CODE_TO_ONNX

    # Map language code - support both short codes and ISO codes
    if language in LANG_CODE_TO_ONNX:
        espeak_lang = LANG_CODE_TO_ONNX[language]
    else:
        # Assume it's already an ISO code like 'de', 'en-us', etc.
        espeak_lang = language

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

    # Audio preview
    if play:
        import tempfile

        from .conversion import ConversionOptions, TTSConverter

        console.print("\n[bold]Generating audio preview...[/bold]")

        try:
            # Auto-detect if voice is a blend
            parsed_voice, parsed_voice_blend = parse_voice_parameter(voice)

            # Initialize converter
            options = ConversionOptions(
                phoneme_dictionary_path=str(phoneme_dict) if phoneme_dict else None,
                voice=parsed_voice or "af_sky",  # Fallback to default if blend
                voice_blend=parsed_voice_blend,
                language=language,
                output_format="wav",  # Explicitly set WAV format
            )
            converter = TTSConverter(options)

            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_output = Path(tmp.name)

            try:
                # Generate audio
                result = converter.convert_text(text, temp_output)

                if result.success:
                    # Play the audio
                    import sounddevice as sd  # type: ignore[import-untyped]
                    import soundfile as sf

                    audio_data, sample_rate = sf.read(str(temp_output))
                    console.print("[dim]▶ Playing...[/dim]")
                    sd.play(audio_data, sample_rate)
                    sd.wait()
                    console.print("[green]✓ Playback complete[/green]")
                else:
                    console.print(f"[red]Error:[/red] {result.error_message}")

            finally:
                # Cleanup temp file
                if temp_output.exists():
                    temp_output.unlink()

        except Exception as e:
            console.print(f"[red]Error playing audio:[/red] {e}")
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            sys.exit(1)


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
    lang: Optional[str] = None,
    use_mixed_language: bool = False,
    mixed_language_primary: Optional[str] = None,
    mixed_language_allowed: Optional[list[str]] = None,
    mixed_language_confidence: float = 0.7,
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
    if lang:
        table.add_row("Phonemization Lang", f"{lang} (override)")
    if use_mixed_language:
        table.add_row("Mixed-Language", "Enabled")
        if mixed_language_primary:
            table.add_row("  Primary Lang", mixed_language_primary)
        if mixed_language_allowed:
            table.add_row("  Allowed Langs", ", ".join(mixed_language_allowed))
        table.add_row("  Confidence", f"{mixed_language_confidence:.2f}")
    table.add_row("Speed", f"{speed}x")
    table.add_row("GPU", "Enabled" if use_gpu else "Disabled")
    table.add_row("Title", title)
    table.add_row("Author", author)

    console.print(table)
    console.print()


@main.command()
@click.argument(
    "input_file",
    type=click.Path(path_type=Path),
    required=False,
    default=None,
)
@click.option(
    "-v",
    "--voice",
    type=click.Choice(VOICES),
    help="TTS voice to use.",
)
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    help="Language for TTS.",
)
@click.option(
    "-s",
    "--speed",
    type=float,
    help="Speech speed (default: 1.0).",
)
@click.option(
    "--gpu/--no-gpu",
    "use_gpu",
    default=None,
    help="Use GPU acceleration if available.",
)
@click.option(
    "--mode",
    "content_mode",
    type=click.Choice(["chapters", "pages"]),
    default=None,
    help="Split content by chapters or pages (default: chapters).",
)
@click.option(
    "-c",
    "--chapters",
    type=str,
    help="Chapter selection (e.g., '1-5', '1,3,5', '3-'). Use with --mode chapters.",
)
@click.option(
    "-p",
    "--pages",
    type=str,
    help="Page selection (e.g., '1-50', '10,20,30'). Use with --mode pages.",
)
@click.option(
    "--start-chapter",
    type=int,
    help="Start from specific chapter number (1-indexed).",
)
@click.option(
    "--start-page",
    type=int,
    help="Start from specific page number (1-indexed).",
)
@click.option(
    "--page-size",
    type=int,
    default=None,
    help="Synthetic page size in characters (default: 2000). Only for --mode pages.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from last saved position.",
)
@click.option(
    "--list",
    "list_content",
    is_flag=True,
    help="List chapters/pages and exit without reading.",
)
@click.option(
    "--split",
    "split_mode",
    type=click.Choice(["sentence", "paragraph"]),
    default=None,
    help="Text splitting mode: sentence (shorter) or paragraph (grouped).",
)
@click.option(
    "--pause-clause",
    type=float,
    default=None,
    help="Pause after clauses in seconds.",
)
@click.option(
    "--pause-sentence",
    type=float,
    default=None,
    help="Pause after sentences in seconds.",
)
@click.option(
    "--pause-paragraph",
    type=float,
    default=None,
    help="Pause after paragraphs in seconds.",
)
@click.option(
    "--pause-variance",
    type=float,
    default=None,
    help="Random variance added to pauses in seconds.",
)
@click.option(
    "--trim-silence/--no-trim-silence",
    "trim_silence",
    default=None,
    help="Trim leading/trailing silence from audio.",
)
@click.pass_context
def read(
    ctx: click.Context,
    input_file: Optional[Path],
    voice: Optional[str],
    language: Optional[str],
    speed: Optional[float],
    use_gpu: Optional[bool],
    content_mode: Optional[str],
    chapters: Optional[str],
    pages: Optional[str],
    start_chapter: Optional[int],
    start_page: Optional[int],
    page_size: Optional[int],
    resume: bool,
    list_content: bool,
    split_mode: Optional[str],
    pause_clause: Optional[float],
    pause_sentence: Optional[float],
    pause_paragraph: Optional[float],
    pause_variance: Optional[float],
    trim_silence: Optional[bool],
) -> None:
    """Read an EPUB or text file aloud with streaming playback.

    Streams audio in real-time without creating output files.
    Supports chapter/page selection, position saving, and resume.

    \b
    Examples:
        ttsforge read book.epub
        ttsforge read book.epub --chapters "1-5"
        ttsforge read book.epub --mode pages --pages "1-50"
        ttsforge read book.epub --mode pages --start-page 10
        ttsforge read book.epub --start-chapter 3
        ttsforge read book.epub --resume
        ttsforge read book.epub --split sentence
        ttsforge read book.epub --list
        ttsforge read story.txt
        cat story.txt | ttsforge read -

    \b
    Controls:
        Ctrl+C - Stop reading (position is saved for resume)
    """
    import random
    import signal
    import sys
    import time

    from pykokoro import Kokoro
    from pykokoro.onnx_backend import LANG_CODE_TO_ONNX

    from .audio_player import (
        PlaybackPosition,
        clear_playback_position,
        load_playback_position,
        save_playback_position,
    )

    # Get model path from global context
    model_path = ctx.obj.get("model_path") if ctx.obj else None
    voices_path = ctx.obj.get("voices_path") if ctx.obj else None

    # Load config for defaults
    config = load_config()
    effective_voice = voice or config.get("default_voice", "af_heart")
    effective_language = language or config.get("default_language", "a")
    effective_speed = speed if speed is not None else config.get("default_speed", 1.0)
    effective_use_gpu = (
        use_gpu if use_gpu is not None else config.get("default_use_gpu", False)
    )
    # Content mode: chapters or pages
    effective_content_mode = content_mode or config.get(
        "default_content_mode", "chapters"
    )
    effective_page_size = page_size or config.get("default_page_size", 2000)
    # Use default_split_mode from config, map "auto" to "sentence" for streaming
    config_split_mode = split_mode or config.get("default_split_mode", "sentence")
    # Map auto/clause/line to sentence for the read command
    if config_split_mode in ("auto", "clause", "line"):
        effective_split_mode = "sentence"
    else:
        effective_split_mode = config_split_mode
    # Pause settings
    effective_pause_sentence = (
        pause_sentence
        if pause_sentence is not None
        else config.get("pause_sentence", 0.2)
    )
    effective_pause_paragraph = (
        pause_paragraph
        if pause_paragraph is not None
        else config.get("pause_paragraph", 0.75)
    )
    effective_pause_variance = (
        pause_variance
        if pause_variance is not None
        else config.get("pause_variance", 0.05)
    )

    # Get language code for TTS
    espeak_lang = LANG_CODE_TO_ONNX.get(effective_language, "en-us")

    # Validate conflicting options
    if effective_content_mode == "chapters" and (pages or start_page):
        console.print(
            "[yellow]Warning:[/yellow] --pages/--start-page ignored in chapters mode. "
            "Use --mode pages to read by pages."
        )
    if effective_content_mode == "pages" and (chapters or start_chapter):
        console.print(
            "[yellow]Warning:[/yellow] --chapters/--start-chapter ignored in "
            "pages mode. Use --mode chapters to read by chapters."
        )

    # Handle stdin input
    if input_file is None or str(input_file) == "-":
        if sys.stdin.isatty():
            console.print(
                "[red]Error:[/red] No input provided. Provide a file or pipe text."
            )
            console.print("[dim]Usage: ttsforge read book.epub[/dim]")
            console.print("[dim]       cat story.txt | ttsforge read -[/dim]")
            sys.exit(1)

        # Read from stdin
        text_content = sys.stdin.read().strip()
        if not text_content:
            console.print("[red]Error:[/red] No text received from stdin.")
            sys.exit(1)

        # Create a simple structure for stdin text
        content_data = [{"title": "Text", "text": text_content, "index": 0}]
        file_identifier = "stdin"
        content_label = "section"  # Generic label for stdin
    else:
        # Validate file exists (removed exists=True from click.Path for stdin)
        if not input_file.exists():
            console.print(f"[red]Error:[/red] File not found: {input_file}")
            sys.exit(1)

        file_identifier = str(input_file.resolve())

        # Handle different file types using InputReader
        try:
            from .input_reader import InputReader

            reader = InputReader(input_file)
            metadata = reader.get_metadata()
        except Exception as e:
            console.print(f"[red]Error loading file:[/red] {e}")
            sys.exit(1)

        # Show book info
        title = metadata.title or input_file.stem
        author = metadata.authors[0] if metadata.authors else "Unknown"
        console.print(f"[bold]{title}[/bold] by {author}")

        # For EPUB files, check if we can use pages mode
        if input_file.suffix.lower() == ".epub":
            # Load content based on mode (chapters or pages)
            if effective_content_mode == "pages":
                try:
                    from epub2text import EPUBParser

                    parser = EPUBParser(str(input_file))
                    epub_pages = parser.get_pages(
                        synthetic_page_size=effective_page_size
                    )
                except Exception as e:
                    console.print(f"[red]Error loading pages:[/red] {e}")
                    sys.exit(1)

                if not epub_pages:
                    console.print("[red]Error:[/red] No pages found in EPUB file.")
                    sys.exit(1)

                # Check if using native or synthetic pages
                has_native = parser.has_page_list()
                page_type = "native" if has_native else "synthetic"
                console.print(f"[dim]{len(epub_pages)} pages ({page_type})[/dim]")

                # Convert to our format
                content_data = [
                    {
                        "title": f"Page {p.page_number}",
                        "text": p.text,
                        "index": i,
                        "page_number": p.page_number,
                    }
                    for i, p in enumerate(epub_pages)
                ]
                content_label = "page"
            else:
                # Default: chapters mode
                epub_chapters = reader.get_chapters()

                if not epub_chapters:
                    console.print("[red]Error:[/red] No chapters found in file.")
                    sys.exit(1)

                console.print(f"[dim]{len(epub_chapters)} chapters[/dim]")

                # Convert to our format - remove chapter markers
                content_data = [
                    {
                        "title": ch.title or f"Chapter {i + 1}",
                        "text": re.sub(
                            r"^\s*<<CHAPTER:[^>]*>>\s*\n*",
                            "",
                            ch.text,
                            count=1,
                            flags=re.MULTILINE,
                        ),
                        "index": i,
                    }
                    for i, ch in enumerate(epub_chapters)
                ]
                content_label = "chapter"

        elif input_file.suffix.lower() in (".txt", ".text"):
            # Plain text file - use InputReader's chapters
            text_chapters = reader.get_chapters()

            if not text_chapters:
                console.print("[red]Error:[/red] No content found in file.")
                sys.exit(1)

            # If it's a single chapter, use it as-is
            # If multiple chapters detected, use them
            content_data = [
                {"title": ch.title or input_file.stem, "text": ch.text, "index": i}
                for i, ch in enumerate(text_chapters)
            ]
            content_label = "chapter" if len(text_chapters) > 1 else "section"
        else:
            console.print(
                f"[red]Error:[/red] Unsupported file type: {input_file.suffix}"
            )
            console.print("[dim]Supported formats: .epub, .txt[/dim]")
            sys.exit(1)

    # List content and exit if requested
    if list_content:
        console.print()
        for item in content_data:
            idx = item["index"] + 1
            item_title = item["title"]
            text_preview = item["text"][:80].replace("\n", " ").strip()
            if len(item["text"]) > 80:
                text_preview += "..."
            console.print(f"[bold]{idx:3}.[/bold] {item_title}")
            console.print(f"     [dim]{text_preview}[/dim]")
        return

    # Content selection (chapters or pages)
    selected_indices: Optional[list[int]] = None

    if effective_content_mode == "pages":
        # Page selection
        if pages:
            selected_indices = _parse_chapter_selection(pages, len(content_data))
        elif start_page:
            if start_page < 1 or start_page > len(content_data):
                console.print(
                    f"[red]Error:[/red] Invalid page number {start_page}. "
                    f"Valid range: 1-{len(content_data)}"
                )
                sys.exit(1)
            selected_indices = list(range(start_page - 1, len(content_data)))
    else:
        # Chapter selection
        if chapters:
            selected_indices = _parse_chapter_selection(chapters, len(content_data))
        elif start_chapter:
            if start_chapter < 1 or start_chapter > len(content_data):
                console.print(
                    f"[red]Error:[/red] Invalid chapter number {start_chapter}. "
                    f"Valid range: 1-{len(content_data)}"
                )
                sys.exit(1)
            selected_indices = list(range(start_chapter - 1, len(content_data)))

    # Handle resume
    start_segment_index = 0
    if resume:
        saved_position = load_playback_position()
        if saved_position and saved_position.file_path == file_identifier:
            # Resume from saved position
            resume_index = saved_position.chapter_index
            start_segment_index = saved_position.segment_index

            if selected_indices is None:
                selected_indices = list(range(resume_index, len(content_data)))
            else:
                # Filter to only include items from resume point
                selected_indices = [i for i in selected_indices if i >= resume_index]

            console.print(
                f"[yellow]Resuming from {content_label} {resume_index + 1}, "
                f"segment {start_segment_index + 1}[/yellow]"
            )
        else:
            console.print(
                "[dim]No saved position found for this file, "
                "starting from beginning.[/dim]"
            )

    # Final selection
    if selected_indices is None:
        selected_indices = list(range(len(content_data)))

    if not selected_indices:
        console.print(f"[yellow]No {content_label}s to read.[/yellow]")
        return

    console.print()
    lang_desc = LANGUAGE_DESCRIPTIONS.get(effective_language, effective_language)
    console.print(
        f"[dim]Voice: {effective_voice} | Language: {lang_desc} | "
        f"Speed: {effective_speed}x[/dim]"
    )
    console.print()

    # Initialize TTS model
    console.print("[dim]Loading TTS model...[/dim]")
    try:
        kokoro = Kokoro(
            model_path=model_path,
            voices_path=voices_path,
            use_gpu=effective_use_gpu,
        )
    except Exception as e:
        console.print(f"[red]Error initializing TTS:[/red] {e}")
        sys.exit(1)

    # Track current position for saving
    current_content_idx = selected_indices[0]
    current_segment_idx = 0
    stop_requested = False

    def signal_handler(signum, frame):
        """Handle Ctrl+C gracefully."""
        nonlocal stop_requested
        console.print("\n[yellow]Stopping... (position saved)[/yellow]")
        stop_requested = True

    # Set up signal handler
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        import concurrent.futures

        import sounddevice as sd

        # Create a thread pool for TTS generation (1 worker for lookahead)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def generate_audio(text_segment: str) -> tuple:
            """Generate audio for a text segment."""
            return kokoro.create(
                text_segment,
                voice=effective_voice,
                speed=effective_speed,
                lang=espeak_lang,
            )

        # Collect all segments across content items with their metadata
        all_segments: list[
            tuple[int, int, str, str]
        ] = []  # (content_idx, seg_idx, text, display)

        for content_position, content_idx in enumerate(selected_indices):
            content_item = content_data[content_idx]
            text = content_item["text"].strip()
            if not text:
                continue

            segments = _split_text_into_segments(text, split_mode=effective_split_mode)

            # Skip segments if resuming mid-content
            seg_offset = 0
            if content_position == 0 and start_segment_index > 0:
                segments = segments[start_segment_index:]
                seg_offset = start_segment_index

            for seg_idx, segment in enumerate(segments):
                actual_seg_idx = seg_idx + seg_offset
                # Clean up text for display (normalize whitespace)
                display_text = " ".join(segment.split())
                all_segments.append(
                    (content_idx, actual_seg_idx, segment, display_text)
                )

        if not all_segments:
            console.print("[yellow]No text to read.[/yellow]")
            return

        # Pre-generate first segment
        current_future = executor.submit(generate_audio, all_segments[0][2])
        next_future = None

        last_content_idx = -1

        for i, (content_idx, seg_idx, _segment_text, display_text) in enumerate(
            all_segments
        ):
            if stop_requested:
                break

            current_content_idx = content_idx
            current_segment_idx = seg_idx

            # Detect content change for paragraph pause
            content_changed = content_idx != last_content_idx

            # Show header when content item changes
            if content_changed:
                content_item = content_data[content_idx]
                console.print()
                label = content_label.capitalize()
                console.print(
                    f"[bold cyan]{label} {content_idx + 1}:[/bold cyan] "
                    f"{content_item['title']}"
                )
                console.print("-" * 60)
                if last_content_idx == -1 and start_segment_index > 0:
                    console.print(
                        f"[dim](resuming from segment {start_segment_index + 1})[/dim]"
                    )
                last_content_idx = content_idx

            # Display current segment
            console.print(f"[dim]{display_text}[/dim]")

            # Start generating next segment while we wait for current
            if i + 1 < len(all_segments):
                next_future = executor.submit(generate_audio, all_segments[i + 1][2])

            # Wait for current audio to be ready
            try:
                audio, sample_rate = current_future.result(timeout=60)
            except Exception as e:
                console.print(f"[red]TTS error:[/red] {e}")
                # Move to next segment's future
                if next_future:
                    current_future = next_future
                    next_future = None
                continue

            # Play audio
            if not stop_requested:
                sd.play(audio, sample_rate)
                sd.wait()

                # Add pause after segment (if not the last segment)
                if i + 1 < len(all_segments) and not stop_requested:
                    next_content_idx = all_segments[i + 1][0]
                    if next_content_idx != content_idx:
                        # Paragraph pause (between content items)
                        pause = effective_pause_paragraph + random.uniform(
                            -effective_pause_variance, effective_pause_variance
                        )
                    else:
                        # Segment pause (within content item)
                        pause = effective_pause_sentence + random.uniform(
                            -effective_pause_variance, effective_pause_variance
                        )
                    time.sleep(max(0, pause))  # Ensure non-negative

            # Swap futures: next becomes current
            if next_future:
                current_future = next_future
                next_future = None

        executor.shutdown(wait=False)

        # Finished
        if not stop_requested:
            # Clear saved position on successful completion
            clear_playback_position()
            console.print("\n[green]Finished reading.[/green]")
        else:
            # Save position for resume
            position = PlaybackPosition(
                file_path=file_identifier,
                chapter_index=current_content_idx,
                segment_index=current_segment_idx,
            )
            save_playback_position(position)
            label = content_label.capitalize()
            console.print(
                f"[dim]Position saved: {label} {current_content_idx + 1}, "
                f"Segment {current_segment_idx + 1}[/dim]"
            )
            console.print("[dim]Use --resume to continue from this position.[/dim]")

    except Exception as e:
        console.print(f"[red]Error during playback:[/red] {e}")
        # Save position on error too
        position = PlaybackPosition(
            file_path=file_identifier,
            chapter_index=current_content_idx,
            segment_index=current_segment_idx,
        )
        save_playback_position(position)
        raise
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)
        kokoro.close()


def _split_text_into_segments(
    text: str, split_mode: str = "paragraph", max_length: int = 500
) -> list[str]:
    """Split text into readable segments for streaming.

    Args:
        text: Text to split
        split_mode: "sentence" for individual sentences, "paragraph" for grouped
        max_length: Maximum segment length (used for paragraph mode)

    Returns:
        List of text segments
    """
    import re

    # First split on sentence-ending punctuation
    sentence_pattern = r"(?<=[.!?])\s+"
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if split_mode == "sentence":
        # Return individual sentences, but split very long ones
        result = []
        for sentence in sentences:
            if len(sentence) > max_length:
                # Split long sentences on clause boundaries
                clause_parts = re.split(r"(?<=[,;:])\s+", sentence)
                for part in clause_parts:
                    part = part.strip()
                    if part:
                        result.append(part)
            else:
                result.append(sentence)
        return result

    # Paragraph mode: group sentences up to max_length
    segments = []
    current_segment = ""

    for sentence in sentences:
        # If adding this sentence would exceed max_length
        if len(current_segment) + len(sentence) + 1 > max_length:
            if current_segment:
                segments.append(current_segment.strip())

            # If single sentence is too long, split it further
            if len(sentence) > max_length:
                # Split on clause boundaries
                clause_parts = re.split(r"(?<=[,;:])\s+", sentence)
                for part in clause_parts:
                    part = part.strip()
                    if len(part) > max_length:
                        # Last resort: split at word boundaries
                        words = part.split()
                        sub_segment = ""
                        for word in words:
                            if len(sub_segment) + len(word) + 1 > max_length:
                                if sub_segment:
                                    segments.append(sub_segment.strip())
                                sub_segment = word
                            else:
                                sub_segment = (
                                    f"{sub_segment} {word}" if sub_segment else word
                                )
                        if sub_segment:
                            current_segment = sub_segment
                    else:
                        segments.append(part)
                current_segment = ""
            else:
                current_segment = sentence
        else:
            current_segment = (
                f"{current_segment} {sentence}" if current_segment else sentence
            )

    if current_segment.strip():
        segments.append(current_segment.strip())

    return [s for s in segments if s.strip()]


@main.command(name="extract-names")
@click.argument(
    "input_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="custom_phonemes.json",
    help="Output JSON file path (default: custom_phonemes.json).",
)
@click.option(
    "--min-count",
    type=int,
    default=3,
    help="Minimum occurrences for a name to be included (default: 3).",
)
@click.option(
    "--max-names",
    type=int,
    default=100,
    help="Maximum number of names to extract (default: 100).",
)
@click.option(
    "-l",
    "--language",
    type=str,
    default="en-us",
    help="Language for phoneme generation (default: en-us).",
)
@click.option(
    "--include-all",
    is_flag=True,
    help="Include all capitalized proper nouns, not just PERSON entities.",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Show extracted names without saving to file.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=100000,
    help="Size of text chunks to process at once in characters (default: 100000).",
)
@click.option(
    "--chapters",
    type=str,
    default=None,
    help="Chapters to extract from (e.g., '1-5', '1,3,5', 'all').",
)
def extract_names(
    input_file: Path,
    output: Path,
    min_count: int,
    max_names: int,
    language: str,
    include_all: bool,
    preview: bool,
    chunk_size: int,
    chapters: str | None,
) -> None:
    """Extract proper names from a book and generate phoneme dictionary.

    Scans INPUT_FILE (EPUB or TXT) for proper names and creates a JSON phoneme
    dictionary with auto-generated pronunciation suggestions. You can then review
    and edit the suggestions before using them for TTS conversion.

    Examples:

        \b
        # Extract names and save to default file
        ttsforge extract-names mybook.epub

        \b
        # Preview names without saving
        ttsforge extract-names mybook.epub --preview

        \b
        # Extract frequent names only (10+ occurrences)
        ttsforge extract-names mybook.epub --min-count 10 -o names.json

        \b
        # Extract from specific chapters
        ttsforge extract-names mybook.epub --chapters 1,3,5-10

        \b
        # Extract from chapter range
        ttsforge extract-names mybook.epub --start 5 --end 15

        \b
        # Then use the dictionary for conversion
        ttsforge convert mybook.epub --phoneme-dict custom_phonemes.json
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from .input_reader import InputReader
    from .name_extractor import (
        extract_names_from_text,
        generate_phoneme_suggestions,
        save_phoneme_dictionary,
    )

    console.print(f"[bold]Extracting names from:[/bold] {input_file}")

    # Read file content
    try:
        reader = InputReader(input_file)
        all_chapters = reader.get_chapters()

        # Determine which chapters to process
        if chapters is not None:
            # Parse chapter selection (supports 'all', ranges, and specific chapters)
            if chapters.lower() == "all":
                selected_chapters = all_chapters
            else:
                selected_indices = _parse_chapter_selection(chapters, len(all_chapters))
                selected_chapters = [all_chapters[i] for i in selected_indices]

            if len(selected_chapters) < len(all_chapters):
                console.print(
                    f"[dim]Processing {len(selected_chapters)} of "
                    f"{len(all_chapters)} chapters[/dim]"
                )
        else:
            # Use all chapters by default
            selected_chapters = all_chapters

        # Remove chapter markers before joining text
        text = "\n\n".join(
            re.sub(
                r"^\s*<<CHAPTER:[^>]*>>\s*\n*",
                "",
                chapter.text,
                count=1,
                flags=re.MULTILINE,
            )
            for chapter in selected_chapters
        )

    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        raise SystemExit(1) from None

    # Check if spaCy is available
    try:
        import spacy  # noqa: F401
    except ImportError:
        console.print(
            "[red]Error:[/red] spaCy is required for name extraction.\n"
            "[yellow]Install with:[/yellow]\n"
            "  pip install spacy\n"
            "  python -m spacy download en_core_web_sm"
        )
        raise SystemExit(1) from None

    # Extract names
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing text and extracting names...", total=None)

        # Progress callback to update progress bar
        def update_progress(current: int, total: int) -> None:
            progress.update(
                task,
                description=f"Processing chunk {current}/{total}...",
                completed=current,
                total=total,
            )

        try:
            names = extract_names_from_text(
                text,
                min_count=min_count,
                max_names=max_names,
                include_all=include_all,
                chunk_size=chunk_size,
                progress_callback=update_progress,
            )
        except ImportError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None

    if not names:
        console.print(
            f"[yellow]No names found[/yellow] (min_count={min_count}). "
            f"Try lowering --min-count or using --include-all."
        )
        return

    console.print(f"\n[green]Found {len(names)} proper names[/green]\n")

    # Generate phoneme suggestions
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Generating phoneme suggestions...", total=None)
        suggestions = generate_phoneme_suggestions(names, language)

    # Display results in a table
    table = Table(title=f"Extracted Names (≥{min_count} occurrences)")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Suggested Phoneme", style="green")

    for name in sorted(names.keys(), key=lambda n: names[n], reverse=True):
        entry = suggestions[name]
        phoneme = entry["phoneme"]
        count = entry["occurrences"]

        # Highlight errors
        if entry.get("suggestion_quality") == "error":
            phoneme_display = f"[red]{phoneme}[/red]"
        else:
            phoneme_display = phoneme

        table.add_row(name, str(count), phoneme_display)

    console.print(table)

    # Save or preview
    if preview:
        console.print("\n[dim]Preview mode - no file saved.[/dim]")
        console.print("[dim]To save, run without --preview flag.[/dim]")
    else:
        save_phoneme_dictionary(
            suggestions, output, source_file=str(input_file.name), language=language
        )
        console.print(f"\n[green]✓ Saved to:[/green] {output}")
        console.print(
            "\n[dim]Next steps:[/dim]\n"
            f"  1. Review and edit {output} to fix any incorrect phonemes\n"
            f"  2. Use with: [cyan]ttsforge convert {input_file} "
            f"--phoneme-dict {output}[/cyan]"
        )


@main.command(name="list-names")
@click.argument(
    "phoneme_dict",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--sort-by",
    type=click.Choice(["name", "count", "alpha"]),
    default="count",
    help="Sort by: name (same as alpha), count (occurrences), alpha (alphabetical).",
)
@click.option(
    "--play",
    is_flag=True,
    help="Play audio preview for each name (interactive mode).",
)
@click.option(
    "-v",
    "--voice",
    type=str,
    default="af_sky",
    help="Voice to use for audio preview (default: af_sky).",
)
@click.option(
    "-l",
    "--language",
    type=str,
    default="a",
    help=(
        "Language code for audio preview "
        "(e.g., 'de', 'en-us', 'a' for auto, default: a)."
    ),
)
def list_names(
    phoneme_dict: Path, sort_by: str, play: bool, voice: str, language: str
) -> None:
    """List all names in a phoneme dictionary for review.

    Displays the contents of a phoneme dictionary in a readable table format,
    making it easy to review and identify names that need phoneme corrections.

    Use --play to interactively listen to each name pronunciation.

    Examples:

        \b
        # List names sorted by frequency
        ttsforge list-names custom_phonemes.json

        \b
        # List names alphabetically
        ttsforge list-names custom_phonemes.json --sort-by alpha

        \b
        # Interactive audio preview
        ttsforge list-names custom_phonemes.json --play

        \b
        # Audio preview with different voice and language
        ttsforge list-names custom_phonemes.json --play --voice af_bella --language de
    """
    import json

    from rich.table import Table

    # Load dictionary
    try:
        with open(phoneme_dict, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading dictionary:[/red] {e}")
        raise SystemExit(1) from None

    # Parse dictionary format
    if "entries" in data:
        # Metadata format
        entries = data["entries"]
        metadata = data.get("_metadata", {})
    else:
        # Simple format
        entries = data
        metadata = {}

    if not entries:
        console.print("[yellow]Dictionary is empty.[/yellow]")
        return

    # Show metadata if available
    if metadata:
        console.print("[dim]Dictionary info:[/dim]")
        if "generated_from" in metadata:
            console.print(f"  Generated from: {metadata['generated_from']}")
        if "generated_at" in metadata:
            console.print(f"  Generated at: {metadata['generated_at']}")
        if "language" in metadata:
            console.print(f"  Language: {metadata['language']}")
        console.print()

    # Create table
    table = Table(title=f"Phoneme Dictionary: {phoneme_dict.name}")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Phoneme", style="green")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Status", style="yellow")

    # Sort entries
    items = list(entries.items())
    if sort_by in ["name", "alpha"]:
        items.sort(key=lambda x: x[0].lower())
    elif sort_by == "count":
        items.sort(
            key=lambda x: (x[1].get("occurrences", 0) if isinstance(x[1], dict) else 0),
            reverse=True,
        )

    # Add rows
    for name, value in items:
        if isinstance(value, str):
            # Simple format
            phoneme = value
            count = "-"
            status = "manual"
        else:
            # Metadata format
            phoneme = value.get("phoneme", "-")
            count = str(value.get("occurrences", "-"))

            # Determine status
            if value.get("verified"):
                status = "✓ verified"
            elif value.get("suggestion_quality") == "error":
                status = "⚠ error"
            elif value.get("suggestion_quality") == "auto":
                status = "auto"
            else:
                status = "manual"

        # Highlight issues
        if phoneme == "/FIXME/":
            phoneme = "[red]/FIXME/[/red]"
            status = "[red]needs fix[/red]"

        table.add_row(name, phoneme, count, status)

    console.print(table)
    console.print(f"\n[green]Total entries:[/green] {len(entries)}")

    # Interactive audio preview mode
    if play:
        import tempfile

        console.print("\n[bold]Audio Preview Mode[/bold]")
        console.print(
            "[dim]Press Enter to play each name, or type a number to jump "
            "to that entry.[/dim]"
        )
        console.print("[dim]Type 'q' to quit, 's' to skip, 'r' to replay.[/dim]\n")

        from .conversion import ConversionOptions, TTSConverter

        # Initialize converter with phoneme dictionary
        try:
            # Auto-detect if voice is a blend
            parsed_voice, parsed_voice_blend = parse_voice_parameter(voice)

            options = ConversionOptions(
                phoneme_dictionary_path=str(phoneme_dict),
                voice=parsed_voice or "af_sky",
                voice_blend=parsed_voice_blend,
                language=language,
            )
            converter = TTSConverter(options)

            idx = 0
            while idx < len(items):
                name, value = items[idx]

                if isinstance(value, str):
                    phoneme = value
                else:
                    phoneme = value.get("phoneme", "")

                console.print(
                    f"\n[cyan]{idx + 1}/{len(items)}:[/cyan] "
                    f"[bold]{name}[/bold] → [green]{phoneme}[/green]"
                )

                # Get user input
                user_input = (
                    input("  [Enter=play, 's'=skip, 'r'=replay, 'q'=quit]: ")
                    .strip()
                    .lower()
                )

                if user_input == "q":
                    console.print("[dim]Exiting preview mode.[/dim]")
                    break
                elif user_input == "s":
                    idx += 1
                    continue
                elif user_input.isdigit():
                    # Jump to specific entry
                    target_idx = int(user_input) - 1
                    if 0 <= target_idx < len(items):
                        idx = target_idx
                        console.print(f"[dim]Jumping to entry {user_input}...[/dim]")
                        continue
                    else:
                        console.print(
                            f"[yellow]Invalid entry number. "
                            f"Must be 1-{len(items)}[/yellow]"
                        )
                        continue

                # Play audio (Enter or 'r')
                try:
                    # Create a test sentence with the name
                    test_text = f"The name {name} appears in the story."

                    # Create temp file
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp:
                        temp_output = Path(tmp.name)

                    try:
                        with console.status(f"Generating audio for '{name}'..."):
                            result = converter.convert_text(test_text, temp_output)

                        if result.success:
                            # Play the audio
                            import sounddevice as sd  # type: ignore[import-untyped]
                            import soundfile as sf

                            audio_data, sample_rate = sf.read(str(temp_output))
                            console.print("[dim]▶ Playing...[/dim]")
                            sd.play(audio_data, sample_rate)
                            sd.wait()
                            console.print("[green]✓ Done[/green]")
                        else:
                            console.print(f"[red]Error:[/red] {result.error_message}")

                    finally:
                        # Cleanup temp file
                        if temp_output.exists():
                            temp_output.unlink()

                    # Don't auto-advance on 'r' (replay)
                    if user_input != "r":
                        idx += 1

                except Exception as e:
                    console.print(f"[red]Error playing audio:[/red] {e}")
                    idx += 1
                    continue

        except Exception as e:
            console.print(f"[red]Error initializing audio preview:[/red] {e}")
            console.print("[yellow]Make sure you have the TTS model loaded.[/yellow]")

    # Show suggestions
    needs_review = sum(
        1
        for entry in entries.values()
        if isinstance(entry, dict)
        and entry.get("suggestion_quality") == "auto"
        and not entry.get("verified")
    )

    if needs_review > 0 and not play:
        console.print(
            f"\n[yellow]⚠ {needs_review} entries need review[/yellow] "
            f"(auto-generated, not verified)"
        )
        console.print(
            f"\n[dim]Tip:[/dim] Listen to samples with:\n"
            f"  [cyan]ttsforge list-names {phoneme_dict} --play[/cyan]"
        )

    # Show suggestions
    needs_review = sum(
        1
        for entry in entries.values()
        if isinstance(entry, dict)
        and entry.get("suggestion_quality") == "auto"
        and not entry.get("verified")
    )

    if needs_review > 0:
        console.print(
            f"\n[yellow]⚠ {needs_review} entries need review[/yellow] "
            f"(auto-generated, not verified)"
        )
        console.print(
            f"\n[dim]Tip:[/dim] Listen to samples with:\n"
            f"  [cyan]ttsforge list-names {phoneme_dict} --play[/cyan]"
        )


if __name__ == "__main__":
    main()
