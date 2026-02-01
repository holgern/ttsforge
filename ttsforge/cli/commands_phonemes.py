"""Phoneme-related CLI commands for ttsforge.

This module contains commands for working with phonemes and pre-tokenized content:
- export: Export EPUB books as pre-tokenized phoneme data (JSON)
- convert: Convert pre-tokenized phoneme files to audiobooks
- preview: Preview phonemes for given text
- info: Show information about a phoneme file
"""

import re
import sys
from pathlib import Path
from typing import Any

import click
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.table import Table

from ..chapter_selection import parse_chapter_selection
from ..constants import (
    LANGUAGE_DESCRIPTIONS,
    SUPPORTED_OUTPUT_FORMATS,
    VOICES,
)
from ..utils import (
    format_chapters_range,
    format_filename_template,
    load_config,
)
from .helpers import console, parse_voice_parameter


def _require_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError:
        console.print(
            "[red]Error:[/red] Audio playback requires the optional dependency "
            "'sounddevice'."
        )
        console.print(
            "[yellow]Install with:[/yellow]\n"
            "  pip install ttsforge[audio]\n"
            "  pip install sounddevice"
        )
        raise SystemExit(1) from None
    return sd


@click.group()
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
    output: Path | None,
    readable: bool,
    language: str,
    chapters: str | None,
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
    from pykokoro.tokenizer import Tokenizer

    from ..input_reader import InputReader
    from ..phonemes import PhonemeBook

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
    selected_indices: list[int] | None = None
    if chapters:
        try:
            selected_indices = parse_chapter_selection(chapters, len(epub_chapters))
        except ValueError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            sys.exit(1)

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
    "--pause-mode",
    type=str,
    default=None,
    help="auto, manual or tts (default: auto).",
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
    output: Path | None,
    output_format: str | None,
    voice: str | None,
    speed: float,
    use_gpu: bool | None,
    silence: float,
    pause_clause: float | None,
    pause_sentence: float | None,
    pause_paragraph: float | None,
    pause_variance: float | None,
    pause_mode: str | None,
    announce_chapters: bool | None,
    chapter_pause: float | None,
    chapters: str | None,
    title: str | None,
    author: str | None,
    cover: Path | None,
    voice_blend: str | None,
    voice_database: Path | None,
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
    from ..phoneme_conversion import (
        PhonemeConversionOptions,
        PhonemeConversionProgress,
        PhonemeConverter,
        parse_chapter_selection,
    )
    from ..phonemes import PhonemeBook

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
            else config.get("pause_clause", 0.3)
        ),
        pause_sentence=(
            pause_sentence
            if pause_sentence is not None
            else config.get("pause_sentence", 0.5)
        ),
        pause_paragraph=(
            pause_paragraph
            if pause_paragraph is not None
            else config.get("pause_paragraph", 0.9)
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
    progress_bar: Progress | None = None
    task_id: TaskID | None = None

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
            from ..utils import format_duration

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
    from pykokoro.onnx_backend import LANG_CODE_TO_ONNX
    from pykokoro.tokenizer import Tokenizer

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

        from ..conversion import ConversionOptions, TTSConverter

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
                    import soundfile as sf

                    audio_data, sample_rate = sf.read(str(temp_output))
                    console.print("[dim]▶ Playing...[/dim]")
                    sd = _require_sounddevice()
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
    from ..phonemes import PhonemeBook

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

        for label, count in zip(bucket_labels, bucket_counts, strict=False):
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
