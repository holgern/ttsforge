"""Conversion commands for ttsforge CLI.

Commands for converting EPUB/text files to audiobooks:
- convert: Main EPUB to audiobook conversion
- list: List chapters in a file
- info: Show file metadata
- sample: Generate TTS samples
- read: Interactive read command
"""

import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click
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

from ..constants import (
    LANGUAGE_DESCRIPTIONS,
    SUPPORTED_OUTPUT_FORMATS,
    VOICE_PREFIX_TO_LANG,
    VOICES,
)
from ..conversion import (
    Chapter,
    ConversionOptions,
    ConversionProgress,
    TTSConverter,
    detect_language_from_iso,
    get_default_voice_for_language,
)
from ..utils import (
    format_chapters_range,
    format_filename_template,
    format_size,
    load_config,
    resolve_conversion_defaults,
)
from .helpers import DEFAULT_SAMPLE_TEXT, console, parse_voice_parameter


@click.command()
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
    "--pause-mode",
    type=str,
    default=None,
    help="Pause mode: 'tts', 'manual', or 'auto' (default: auto).",
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
    "--detect-emphasis/--no-detect-emphasis",
    "detect_emphasis",
    default=False,
    help=(
        "Detect emphasis (italic/bold) from HTML tags in EPUB files "
        "(default: disabled)."
    ),
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
def convert(  # noqa: C901
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
    pause_mode: Optional[str],
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
    detect_emphasis: bool,
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

    from ..input_reader import InputReader

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
        if detect_emphasis and reader.file_type == "epub":
            # Get chapters with HTML for emphasis detection
            chapters_with_html = reader.get_chapters_with_html()
            epub_chapters = [ch for ch, _ in chapters_with_html]
            html_contents = [html for _, html in chapters_with_html]
        else:
            # Just get plain text chapters
            epub_chapters = reader.get_chapters()
            html_contents = None

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

        from ..utils import sanitize_filename

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
        pause_mode=(
            pause_mode if pause_mode is not None else config.get("pause_mode", "auto")
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
        detect_emphasis=detect_emphasis,
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

    # Filter chapters if selection provided
    if selected_indices:
        filtered_chapters = [
            ch for i, ch in enumerate(epub_chapters) if i in selected_indices
        ]
        if html_contents:
            filtered_html = [
                html for i, html in enumerate(html_contents) if i in selected_indices
            ]
        else:
            filtered_html = None
    else:
        filtered_chapters = epub_chapters
        filtered_html = html_contents

    # Convert input_reader.Chapter to conversion.Chapter
    chapters_to_convert: list[Chapter] = []
    for i, ch in enumerate(filtered_chapters):
        html_content = filtered_html[i] if filtered_html else None
        chapters_to_convert.append(
            Chapter(
                title=ch.title,
                content=ch.text,
                index=ch.index,
                html_content=html_content,
                is_ssmd=ch.is_ssmd,
            )
        )

    with progress:
        task_id = progress.add_task("Converting...", total=total_chars)

        result = converter.convert_chapters_resumable(
            chapters=chapters_to_convert,
            output_path=output,
            source_file=epub_file,
            resume=resume,
        )

        progress.update(task_id, completed=total_chars)

    # Show result
    if result.success:
        console.print()
        if generate_ssmd_only:
            console.print(
                Panel(
                    f"[green]SSMD files generated in:[/green]\n{result.chapters_dir}",
                    title="[bold green]SSMD Generation Complete[/bold green]",
                )
            )
        else:
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


@click.command("list")
@click.argument("epub_file", type=click.Path(exists=True, path_type=Path))
def list_chapters(epub_file: Path) -> None:
    """List chapters in a file.

    EPUB_FILE is the path to the file (EPUB, TXT, or PDF).
    """
    from ..input_reader import InputReader

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


@click.command()
@click.argument("epub_file", type=click.Path(exists=True, path_type=Path))
def info(epub_file: Path) -> None:
    """Show metadata and information about a file.

    EPUB_FILE is the path to the file (EPUB, TXT, or PDF).
    """
    from ..input_reader import InputReader

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


@click.command()
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

    from ..conversion import ConversionOptions, TTSConverter

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
    resolved_defaults = resolve_conversion_defaults(
        user_config,
        {
            "voice": voice,
            "language": language,
            "speed": speed,
            "split_mode": split_mode,
            "use_gpu": use_gpu,
            "lang": lang,
        },
    )

    # Parse mixed_language_allowed from comma-separated string
    parsed_mixed_language_allowed = None
    if mixed_language_allowed:
        parsed_mixed_language_allowed = [
            lang_item.strip() for lang_item in mixed_language_allowed.split(",")
        ]

    # Auto-detect if voice is a blend
    voice_value = resolved_defaults["voice"]
    parsed_voice, parsed_voice_blend = parse_voice_parameter(voice_value)

    # Build conversion options (use ConversionOptions defaults if not specified)
    options = ConversionOptions(
        voice=parsed_voice or "af_bella",
        voice_blend=parsed_voice_blend,
        language=resolved_defaults["language"],
        speed=resolved_defaults["speed"],
        output_format=output_format,
        use_gpu=resolved_defaults["use_gpu"],
        split_mode=resolved_defaults["split_mode"],
        lang=resolved_defaults["lang"],
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


@click.command()
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
    "--pause-mode",
    type=str,
    default=None,
    help="Trim leading/trailing silence from audio.",
)
@click.pass_context
def read(  # noqa: C901
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
    pause_mode: Optional[str],
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

    from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
    from pykokoro.onnx_backend import LANG_CODE_TO_ONNX, Kokoro
    from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
    from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
    from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter

    from ..audio_player import (
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
    resolved_defaults = resolve_conversion_defaults(
        config,
        {
            "voice": voice,
            "language": language,
            "speed": speed,
            "split_mode": split_mode,
            "use_gpu": use_gpu,
            "lang": None,
        },
    )
    effective_voice = resolved_defaults["voice"]
    effective_language = resolved_defaults["language"]
    effective_speed = resolved_defaults["speed"]
    effective_use_gpu = resolved_defaults["use_gpu"]
    # Content mode: chapters or pages
    effective_content_mode = content_mode or config.get(
        "default_content_mode", "chapters"
    )
    effective_page_size = page_size or config.get("default_page_size", 2000)
    # Use default_split_mode from config, map "auto" to "sentence" for streaming
    config_split_mode = resolved_defaults["split_mode"]
    # Map auto/clause/line to sentence for the read command
    if config_split_mode in ("auto", "clause", "line"):
        effective_split_mode = "sentence"
    else:
        effective_split_mode = config_split_mode
    # Pause settings
    effective_pause_clause = (
        pause_clause if pause_clause is not None else config.get("pause_clause", 0.25)
    )
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
    effective_pause_mode = (
        pause_mode if pause_mode is not None else config.get("pause_mode", True)
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
            from ..input_reader import InputReader

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

        elif input_file.suffix.lower() in (".txt", ".text", ".ssmd"):
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

    # Initialize TTS pipeline
    console.print("[dim]Loading TTS model...[/dim]")
    try:
        kokoro = Kokoro(
            model_path=model_path,
            voices_path=voices_path,
            use_gpu=effective_use_gpu,
        )
        generation = GenerationConfig(
            speed=effective_speed,
            lang=espeak_lang,
            pause_mode=effective_pause_mode,
            pause_clause=effective_pause_clause,
            pause_sentence=effective_pause_sentence,
            pause_paragraph=effective_pause_paragraph,
            pause_variance=effective_pause_variance,
        )
        pipeline_config = PipelineConfig(
            voice=effective_voice,
            generation=generation,
            model_path=model_path,
            voices_path=voices_path,
        )
        pipeline = KokoroPipeline(
            pipeline_config,
            phoneme_processing=OnnxPhonemeProcessorAdapter(kokoro),
            audio_generation=OnnxAudioGenerationAdapter(kokoro),
            audio_postprocessing=OnnxAudioPostprocessingAdapter(kokoro),
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
            print(text_segment)
            result = pipeline.run(text_segment)
            return result.audio, result.sample_rate

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
