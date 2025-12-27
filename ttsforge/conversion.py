"""TTS conversion module for ttsforge - converts text/EPUB to audiobooks."""

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import soundfile as sf

from .constants import (
    DEFAULT_VOICE_FOR_LANG,
    ISO_TO_LANG_CODE,
    LANGUAGE_DESCRIPTIONS,
    SAMPLE_RATE,
    SUPPORTED_OUTPUT_FORMATS,
    VOICE_PREFIX_TO_LANG,
)
from .utils import (
    create_process,
    detect_encoding,
    ensure_ffmpeg,
    format_duration,
    get_device,
    prevent_sleep_end,
    prevent_sleep_start,
    sanitize_filename,
)


@dataclass
class Chapter:
    """Represents a chapter from an EPUB or text file."""

    title: str
    content: str
    index: int = 0

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class ConversionProgress:
    """Progress information during conversion."""

    current_chapter: int = 0
    total_chapters: int = 0
    chapter_name: str = ""
    chars_processed: int = 0
    total_chars: int = 0
    current_text: str = ""
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0

    @property
    def percent(self) -> int:
        if self.total_chars == 0:
            return 0
        return min(int(self.chars_processed / self.total_chars * 100), 99)

    @property
    def etr_formatted(self) -> str:
        return format_duration(self.estimated_remaining)


@dataclass
class ConversionResult:
    """Result of a conversion operation."""

    success: bool
    output_path: Optional[Path] = None
    subtitle_path: Optional[Path] = None
    error_message: Optional[str] = None
    chapters_dir: Optional[Path] = None


# Split mode options
SPLIT_MODES = ["auto", "line", "paragraph", "sentence", "clause"]


@dataclass
class ConversionOptions:
    """Options for TTS conversion."""

    voice: str = "af_bella"
    language: str = "a"
    speed: float = 1.0
    output_format: str = "m4b"
    output_dir: Optional[Path] = None
    use_gpu: bool = True
    silence_between_chapters: float = 2.0
    save_chapters_separately: bool = False
    merge_at_end: bool = True
    # Split mode: auto, line, paragraph, sentence, clause
    split_mode: str = "auto"
    # Metadata for m4b
    title: Optional[str] = None
    author: Optional[str] = None
    cover_image: Optional[Path] = None


# Pattern to detect chapter markers in text
CHAPTER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Chapter|CHAPTER|Ch\.?|Kapitel|Chapitre|Capitulo|Capitolo)\s*"
    r"(?:[IVXLCDM]+|\d+)"
    r"(?:\s*[:\-\.\s]\s*.*)?"
    r"|"
    r"(?:Prologue|PROLOGUE|Epilogue|EPILOGUE|Introduction|INTRODUCTION)"
    r"(?:\s*[:\-\.\s]\s*.*)?"
    r")\s*(?:\n|$)",
    re.MULTILINE | re.IGNORECASE,
)


def detect_language_from_iso(iso_code: Optional[str]) -> str:
    """Convert ISO language code to ttsforge language code."""
    if not iso_code:
        return "a"  # Default to American English
    iso_lower = iso_code.lower().strip()
    return ISO_TO_LANG_CODE.get(iso_lower, ISO_TO_LANG_CODE.get(iso_lower[:2], "a"))


def get_voice_language(voice: str) -> str:
    """Get the language code from a voice name."""
    prefix = voice[:2] if len(voice) >= 2 else ""
    return VOICE_PREFIX_TO_LANG.get(prefix, "a")


def get_default_voice_for_language(lang_code: str) -> str:
    """Get the default voice for a language."""
    return DEFAULT_VOICE_FOR_LANG.get(lang_code, "af_bella")


class TTSConverter:
    """Converts text to speech using Kokoro TTS."""

    # Split patterns for different languages
    PUNCTUATION_SENTENCE = ".!?।。！？"
    PUNCTUATION_SENTENCE_COMMA = ".!?,।。！？、，"

    def __init__(
        self,
        options: ConversionOptions,
        progress_callback: Optional[Callable[[ConversionProgress], None]] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Initialize the TTS converter.

        Args:
            options: Conversion options
            progress_callback: Called with progress updates
            log_callback: Called with log messages (message, level)
        """
        self.options = options
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self._cancelled = False
        self._pipeline: Any = None
        self._np: Any = None

    def log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        if self.log_callback:
            self.log_callback(message, level)

    def cancel(self) -> None:
        """Request cancellation of the conversion."""
        self._cancelled = True

    def _init_pipeline(self) -> None:
        """Initialize the TTS pipeline."""
        if self._pipeline is not None:
            return

        self.log("Initializing TTS pipeline...")
        import numpy as np
        from kokoro import KPipeline

        self._np = np
        device = get_device(self.options.use_gpu)
        self.log(f"Using device: {device}")

        self._pipeline = KPipeline(
            lang_code=self.options.language,
            repo_id="hexgrad/Kokoro-82M",
            device=device,
        )

    def _get_split_pattern(self) -> str:
        """Get the split pattern based on language (used for auto mode)."""
        lang = self.options.language

        # English languages use newline splitting
        if lang in ["a", "b"]:
            return "\n"

        # CJK languages use no spacing
        spacing = r"\s*" if lang in ["z", "j"] else r"\s+"

        # Default to sentence-based splitting for non-English
        return rf"(?<=[{self.PUNCTUATION_SENTENCE}]){spacing}|\n+"

    def _get_spacy_model(self) -> str:
        """Get the appropriate spaCy model for the current language."""
        lang = self.options.language
        # Map ttsforge language codes to spaCy models
        lang_to_spacy = {
            "a": "en_core_web_sm",  # American English
            "b": "en_core_web_sm",  # British English
            "e": "es_core_news_sm",  # Spanish
            "f": "fr_core_news_sm",  # French
            "i": "it_core_news_sm",  # Italian
            "p": "pt_core_news_sm",  # Portuguese
            "z": "zh_core_web_sm",  # Chinese
            "j": "ja_core_news_sm",  # Japanese
            "k": "ko_core_news_sm",  # Korean
            "h": "de_core_news_sm",  # German (Hindi not well supported)
        }
        return lang_to_spacy.get(lang, "en_core_web_sm")

    def _split_text_with_phrasplit(self, text: str) -> list[str]:
        """
        Split text using phrasplit based on split_mode.

        Args:
            text: Text to split

        Returns:
            List of text segments
        """
        mode = self.options.split_mode

        if mode == "line":
            # Simple line splitting
            return [line for line in text.split("\n") if line.strip()]

        if mode == "paragraph":
            try:
                from phrasplit import split_paragraphs

                return split_paragraphs(text)
            except ImportError:
                self.log(
                    "phrasplit not installed, falling back to line mode", "warning"
                )
                return [line for line in text.split("\n") if line.strip()]

        if mode == "sentence":
            try:
                from phrasplit import split_sentences

                spacy_model = self._get_spacy_model()
                return split_sentences(text, language_model=spacy_model)
            except ImportError:
                self.log(
                    "phrasplit not installed, falling back to line mode", "warning"
                )
                return [line for line in text.split("\n") if line.strip()]
            except OSError as e:
                self.log(
                    f"spaCy model error: {e}, falling back to line mode", "warning"
                )
                return [line for line in text.split("\n") if line.strip()]

        if mode == "clause":
            try:
                from phrasplit import split_clauses

                spacy_model = self._get_spacy_model()
                return split_clauses(text, language_model=spacy_model)
            except ImportError:
                self.log(
                    "phrasplit not installed, falling back to line mode", "warning"
                )
                return [line for line in text.split("\n") if line.strip()]
            except OSError as e:
                self.log(
                    f"spaCy model error: {e}, falling back to line mode", "warning"
                )
                return [line for line in text.split("\n") if line.strip()]

        # "auto" mode - return None to use pattern-based splitting
        return []

    def _use_phrasplit_mode(self) -> bool:
        """Check if we should use phrasplit for text splitting."""
        return self.options.split_mode in ["paragraph", "sentence", "clause"]

    def _generate_silence(self, duration: float) -> Any:
        """Generate silence audio of given duration."""
        samples = int(duration * SAMPLE_RATE)
        return self._np.zeros(samples, dtype="float32")

    def _write_audio_chunk(
        self,
        audio: Any,
        out_file: Optional[Any],
        ffmpeg_proc: Optional[subprocess.Popen[bytes]],
    ) -> None:
        """Write audio chunk to file or ffmpeg process."""
        if out_file is not None:
            out_file.write(audio)
        elif ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
            if hasattr(audio, "numpy"):
                audio_bytes = audio.numpy().astype("float32").tobytes()
            else:
                audio_bytes = audio.astype("float32").tobytes()
            ffmpeg_proc.stdin.write(audio_bytes)

    def _setup_output(
        self, output_path: Path
    ) -> tuple[Optional[Any], Optional[subprocess.Popen[bytes]]]:
        """Set up output file or ffmpeg process based on format."""
        fmt = self.options.output_format

        if fmt in ["wav", "mp3", "flac"]:
            out_file = sf.SoundFile(
                str(output_path),
                "w",
                samplerate=SAMPLE_RATE,
                channels=1,
                format=fmt,
            )
            return out_file, None

        # Formats requiring ffmpeg
        ensure_ffmpeg()
        import static_ffmpeg

        static_ffmpeg.add_paths()

        cmd = [
            "ffmpeg",
            "-y",
            "-thread_queue_size",
            "32768",
            "-f",
            "f32le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
        ]

        if fmt == "m4b":
            # Add cover image if provided
            if self.options.cover_image and self.options.cover_image.exists():
                cmd.extend(
                    [
                        "-i",
                        str(self.options.cover_image),
                        "-map",
                        "0:a",
                        "-map",
                        "1",
                        "-c:v",
                        "copy",
                        "-disposition:v",
                        "attached_pic",
                    ]
                )
            cmd.extend(
                [
                    "-c:a",
                    "aac",
                    "-q:a",
                    "2",
                    "-movflags",
                    "+faststart+use_metadata_tags",
                ]
            )
            # Add metadata
            if self.options.title:
                cmd.extend(["-metadata", f"title={self.options.title}"])
            if self.options.author:
                cmd.extend(["-metadata", f"artist={self.options.author}"])
        elif fmt == "opus":
            cmd.extend(["-c:a", "libopus", "-b:a", "24000"])

        cmd.append(str(output_path))

        ffmpeg_proc = create_process(
            cmd, stdin=subprocess.PIPE, text=False, suppress_output=True
        )
        return None, ffmpeg_proc  # type: ignore[return-value]

    def _finalize_output(
        self,
        out_file: Optional[Any],
        ffmpeg_proc: Optional[subprocess.Popen[bytes]],
    ) -> None:
        """Finalize and close output file/process."""
        if out_file is not None:
            out_file.close()
        elif ffmpeg_proc is not None:
            if ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

    def _add_chapters_to_m4b(
        self,
        output_path: Path,
        chapters: list[dict[str, Any]],
    ) -> None:
        """Add chapter markers to an m4b file."""
        if self.options.output_format != "m4b" or len(chapters) <= 1:
            return

        import static_ffmpeg

        static_ffmpeg.add_paths()

        # Create chapters metadata file
        chapters_file = output_path.with_suffix(".chapters.txt")
        with open(chapters_file, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            for ch in chapters:
                title = ch["title"].replace("=", "\\=")
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={int(ch['start'] * 1000)}\n")
                f.write(f"END={int(ch['end'] * 1000)}\n")
                f.write(f"title={title}\n\n")

        # Mux chapters into the file
        tmp_path = output_path.with_suffix(".tmp.m4b")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(output_path),
            "-i",
            str(chapters_file),
            "-map",
            "0:a",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:a",
            "copy",
        ]

        # Re-add cover if exists
        if self.options.cover_image and self.options.cover_image.exists():
            cmd.extend(
                [
                    "-i",
                    str(self.options.cover_image),
                    "-map",
                    "2",
                    "-c:v",
                    "copy",
                    "-disposition:v",
                    "attached_pic",
                ]
            )

        cmd.append(str(tmp_path))

        proc = create_process(cmd, suppress_output=True)
        proc.wait()

        # Replace original with chaptered version
        os.replace(tmp_path, output_path)
        chapters_file.unlink()

    def convert_chapters(
        self,
        chapters: list[Chapter],
        output_path: Path,
    ) -> ConversionResult:
        """
        Convert a list of chapters to audio.

        Args:
            chapters: List of Chapter objects
            output_path: Output file path

        Returns:
            ConversionResult with success status and paths
        """
        if not chapters:
            return ConversionResult(
                success=False, error_message="No chapters to convert"
            )

        if self.options.output_format not in SUPPORTED_OUTPUT_FORMATS:
            return ConversionResult(
                success=False,
                error_message=f"Unsupported format: {self.options.output_format}",
            )

        self._cancelled = False
        prevent_sleep_start()

        try:
            self._init_pipeline()

            total_chars = sum(ch.char_count for ch in chapters)
            chars_processed = 0
            start_time = time.time()
            current_time = 0.0
            chapter_times: list[dict[str, Any]] = []

            # Set up output
            out_file, ffmpeg_proc = self._setup_output(output_path)

            # Determine splitting strategy
            use_phrasplit = self._use_phrasplit_mode()
            split_pattern = None if use_phrasplit else self._get_split_pattern()

            progress = ConversionProgress(
                total_chapters=len(chapters),
                total_chars=total_chars,
            )

            for chapter_idx, chapter in enumerate(chapters):
                if self._cancelled:
                    self._finalize_output(out_file, ffmpeg_proc)
                    return ConversionResult(success=False, error_message="Cancelled")

                chapter_start = current_time
                chapter_times.append(
                    {
                        "title": chapter.title,
                        "start": chapter_start,
                        "end": 0.0,
                    }
                )

                progress.current_chapter = chapter_idx + 1
                progress.chapter_name = chapter.title

                self.log(f"Chapter {chapter_idx + 1}/{len(chapters)}: {chapter.title}")

                # Generate TTS for this chapter
                if use_phrasplit:
                    # Pre-split text using phrasplit, then process each segment
                    segments = self._split_text_with_phrasplit(chapter.content)
                    for segment in segments:
                        if self._cancelled:
                            break
                        if not segment.strip():
                            continue

                        # Process each segment without further splitting
                        for result in self._pipeline(
                            segment,
                            voice=self.options.voice,
                            speed=self.options.speed,
                            split_pattern=None,  # No further splitting
                        ):
                            if self._cancelled:
                                break

                            # Write audio
                            self._write_audio_chunk(result.audio, out_file, ffmpeg_proc)

                            # Update timing
                            chunk_duration = len(result.audio) / SAMPLE_RATE
                            current_time += chunk_duration

                            # Update progress
                            grapheme_len = len(result.graphemes)
                            chars_processed += grapheme_len
                            progress.chars_processed = chars_processed
                            progress.current_text = result.graphemes[:100]

                            elapsed = time.time() - start_time
                            if chars_processed > 0 and elapsed > 0.5:
                                avg_time = elapsed / chars_processed
                                remaining = total_chars - chars_processed
                                progress.estimated_remaining = avg_time * remaining
                            progress.elapsed_time = elapsed

                            if self.progress_callback:
                                self.progress_callback(progress)
                else:
                    # Use pattern-based splitting (auto/line modes)
                    for result in self._pipeline(
                        chapter.content,
                        voice=self.options.voice,
                        speed=self.options.speed,
                        split_pattern=split_pattern,
                    ):
                        if self._cancelled:
                            break

                        # Write audio
                        self._write_audio_chunk(result.audio, out_file, ffmpeg_proc)

                        # Update timing
                        chunk_duration = len(result.audio) / SAMPLE_RATE
                        current_time += chunk_duration

                        # Update progress
                        grapheme_len = len(result.graphemes)
                        chars_processed += grapheme_len
                        progress.chars_processed = chars_processed
                        progress.current_text = result.graphemes[:100]

                        elapsed = time.time() - start_time
                        if chars_processed > 0 and elapsed > 0.5:
                            avg_time = elapsed / chars_processed
                            remaining = total_chars - chars_processed
                            progress.estimated_remaining = avg_time * remaining
                        progress.elapsed_time = elapsed

                        if self.progress_callback:
                            self.progress_callback(progress)

                # Add silence between chapters (except after last)
                if chapter_idx < len(chapters) - 1:
                    silence = self._generate_silence(
                        self.options.silence_between_chapters
                    )
                    self._write_audio_chunk(silence, out_file, ffmpeg_proc)
                    current_time += self.options.silence_between_chapters

                chapter_times[-1]["end"] = current_time

            # Finalize
            self.log("Finalizing audio...")
            self._finalize_output(out_file, ffmpeg_proc)

            # Add chapters to m4b
            self._add_chapters_to_m4b(output_path, chapter_times)

            return ConversionResult(
                success=True,
                output_path=output_path,
            )

        except Exception as e:
            return ConversionResult(success=False, error_message=str(e))
        finally:
            prevent_sleep_end()

    def convert_text(self, text: str, output_path: Path) -> ConversionResult:
        """
        Convert plain text to audio.

        Args:
            text: Text to convert
            output_path: Output file path

        Returns:
            ConversionResult
        """
        chapters = [Chapter(title="Text", content=text, index=0)]
        return self.convert_chapters(chapters, output_path)

    def convert_epub(
        self,
        epub_path: Path,
        output_path: Path,
        selected_chapters: Optional[list[int]] = None,
    ) -> ConversionResult:
        """
        Convert an EPUB file to audio.

        Args:
            epub_path: Path to EPUB file
            output_path: Output file path
            selected_chapters: Optional list of chapter indices to convert

        Returns:
            ConversionResult
        """
        from epub2text import EPUBParser

        self.log(f"Parsing EPUB: {epub_path}")

        # Parse EPUB using epub2text
        try:
            parser = EPUBParser(str(epub_path))
            epub_chapters = parser.get_chapters()
        except Exception as e:
            return ConversionResult(
                success=False,
                error_message=f"Failed to parse EPUB: {e}",
            )

        if not epub_chapters:
            return ConversionResult(
                success=False,
                error_message="No chapters found in EPUB",
            )

        # Filter chapters if selection provided
        if selected_chapters:
            epub_chapters = [
                ch for i, ch in enumerate(epub_chapters) if i in selected_chapters
            ]

        # Convert to our Chapter format - epub2text Chapter has .text attribute
        chapters = [
            Chapter(title=ch.title, content=ch.text, index=i)
            for i, ch in enumerate(epub_chapters)
        ]

        self.log(f"Found {len(chapters)} chapters")

        # Try to get metadata from EPUB for m4b
        if self.options.output_format == "m4b":
            try:
                metadata = parser.get_metadata()
                if metadata:
                    if not self.options.title and metadata.title:
                        self.options.title = metadata.title
                    if not self.options.author and metadata.authors:
                        self.options.author = metadata.authors[0]
            except Exception:
                pass

        return self.convert_chapters(chapters, output_path)


def parse_text_chapters(text: str) -> list[Chapter]:
    """
    Parse text content into chapters based on chapter markers.

    Args:
        text: Text content

    Returns:
        List of Chapter objects
    """
    matches = list(CHAPTER_PATTERN.finditer(text))

    if not matches:
        return [Chapter(title="Text", content=text.strip(), index=0)]

    chapters = []

    # Add introduction if content before first marker
    first_start = matches[0].start()
    if first_start > 0:
        intro_text = text[:first_start].strip()
        if intro_text:
            chapters.append(Chapter(title="Introduction", content=intro_text, index=0))

    # Parse chapters
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        chapter_name = match.group().strip()
        chapter_text = text[start:end].strip()

        if chapter_text:
            chapters.append(
                Chapter(title=chapter_name, content=chapter_text, index=len(chapters))
            )

    return chapters
