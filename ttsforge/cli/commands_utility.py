"""Utility commands for ttsforge CLI."""

import re
import sys
import tempfile
from pathlib import Path
from typing import cast

import click
import numpy as np
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import (
    DEFAULT_MODEL_QUALITY,
    LANG_CODE_TO_ONNX,
    MODEL_QUALITY_FILES,
    Kokoro,
    ModelQuality,
    VoiceBlend,
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
from pykokoro.onnx_backend import VOICE_NAMES_V1_0 as VOICE_NAMES
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..chapter_selection import parse_chapter_selection
from ..constants import (
    DEFAULT_CONFIG,
    DEFAULT_VOICE_FOR_LANG,
    LANGUAGE_DESCRIPTIONS,
    VOICE_PREFIX_TO_LANG,
    VOICES,
)
from ..utils import format_size, load_config, reset_config, save_config
from .helpers import DEMO_TEXT, VOICE_BLEND_PRESETS, console, parse_voice_parameter


@click.command()
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    default=None,
    help="Filter voices by language (default: all languages).",
)
def voices(language: str | None) -> None:
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


@click.command()
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
def demo(  # noqa: C901
    ctx: click.Context,
    output: Path | None,
    language: str | None,
    voices_filter: str | None,
    speed: float,
    use_gpu: bool | None,
    silence: float,
    text: str | None,
    separate: bool,
    blend: str | None,
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

        # Initialize TTS pipeline
        try:
            kokoro = Kokoro(
                model_path=model_path,
                voices_path=voices_path,
                use_gpu=gpu,
            )
            generation = GenerationConfig(speed=speed, lang="en-us")
            pipeline_config = PipelineConfig(
                voice=DEFAULT_CONFIG.get("default_voice", "af_heart"),
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
                    blend_lang = VOICE_PREFIX_TO_LANG.get(voice_names[0][:2], "a")
                    onnx_lang = LANG_CODE_TO_ONNX.get(blend_lang, "en-us")
                    result = pipeline.run(demo_text, voice=voice_blend, lang=onnx_lang)
                    samples = result.audio
                    sr = result.sample_rate

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

    # Initialize TTS pipeline
    try:
        kokoro = Kokoro(
            model_path=model_path,
            voices_path=voices_path,
            use_gpu=gpu,
        )
        generation = GenerationConfig(speed=speed, lang="en-us")
        pipeline_config = PipelineConfig(
            voice=DEFAULT_CONFIG.get("default_voice", "af_heart"),
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
                onnx_lang = LANG_CODE_TO_ONNX.get(lang_code, "en-us")
                result = pipeline.run(demo_text, voice=voice, lang=onnx_lang)
                samples = result.audio
                sr = result.sample_rate

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


@click.command()
@click.option("--force", is_flag=True, help="Force re-download even if files exist.")
@click.option(
    "--quality",
    "-q",
    type=click.Choice(list(MODEL_QUALITY_FILES.keys())),
    default=None,
    help="Model quality/quantization level. Default: from config or fp32.",
)
def download(force: bool, quality: str | None) -> None:
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


@click.command()
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


@click.command(name="extract-names")
@click.argument(
    "input_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file path (default: INPUT_FILE_custom_phonemes.json).",
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
    default=500,
    help="Maximum number of names to extract (default: 500).",
)
@click.option(
    "-l",
    "--language",
    type=click.Choice(list(LANGUAGE_DESCRIPTIONS.keys())),
    default="a",
    help="Language for phoneme generation (default: a).",
)
@click.option(
    "--include-all",
    is_flag=True,
    help="Include all detected proper nouns (ignore min-count).",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Preview extracted names without saving to file.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=100000,
    help="Characters per chunk for processing (default: 100000).",
)
@click.option(
    "--chapters",
    type=str,
    default=None,
    help="Specific chapters to process (e.g., '1,3,5-10' or 'all'). Default: all.",
)
def extract_names(
    input_file: Path,
    output: Path | None,
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
    from rich.table import Table

    from ..input_reader import InputReader
    from ..name_extractor import (
        extract_names_from_text,
        generate_phoneme_suggestions,
        save_phoneme_dictionary,
    )

    # Set default output filename
    if output is None:
        output = input_file.with_name(f"{input_file.stem}_custom_phonemes.json")

    console.print(f"[bold]Extracting names from:[/bold] {input_file}")

    # Read file content
    try:
        reader = InputReader(input_file)
        all_chapters = reader.get_chapters()

        # Determine which chapters to process
        if chapters is not None:
            # Parse chapter selection (supports 'all', ranges, and specific chapters)
            try:
                selected_indices = parse_chapter_selection(chapters, len(all_chapters))
                selected_chapters = [all_chapters[i] for i in selected_indices]
            except ValueError as exc:
                console.print(f"[yellow]{exc}[/yellow]")
                sys.exit(1)

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


@click.command(name="list-names")
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
def list_names(  # noqa: C901
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

    from ..conversion import ConversionOptions, TTSConverter

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
        console.print("\n[bold]Audio Preview Mode[/bold]")
        console.print(
            "[dim]Press Enter to play each name, or type a number to jump "
            "to that entry.[/dim]"
        )
        console.print("[dim]Type 'q' to quit, 's' to skip, 'r' to replay.[/dim]\n")

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
