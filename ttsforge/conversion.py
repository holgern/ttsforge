"""TTS conversion module for ttsforge - converts text/EPUB to audiobooks."""

import hashlib
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
import soundfile as sf

from .constants import (
    DEFAULT_VOICE_FOR_LANG,
    ISO_TO_LANG_CODE,
    SAMPLE_RATE,
    SUPPORTED_OUTPUT_FORMATS,
    VOICE_PREFIX_TO_LANG,
)
from .onnx_backend import (
    KokoroONNX,
    VoiceBlend,
    are_models_downloaded,
    download_all_models,
    get_onnx_lang_code,
)
from .utils import (
    create_process,
    ensure_ffmpeg,
    format_duration,
    format_filename_template,
    prevent_sleep_end,
    prevent_sleep_start,
    sanitize_filename,
)

if TYPE_CHECKING:
    from phrasplit import Segment


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


@dataclass
class ChapterState:
    """State of a single chapter conversion."""

    index: int
    title: str
    content_hash: str  # Hash of chapter content for integrity check
    completed: bool = False
    audio_file: Optional[str] = None  # Relative path to chapter audio
    duration: float = 0.0  # Duration in seconds
    char_count: int = 0


@dataclass
class ConversionState:
    """Persistent state for resumable conversions."""

    version: int = 1
    source_file: str = ""
    source_hash: str = ""  # Hash of source file for change detection
    output_file: str = ""
    work_dir: str = ""
    voice: str = ""
    language: str = ""
    speed: float = 1.0
    split_mode: str = "auto"
    output_format: str = "m4b"
    silence_between_chapters: float = 2.0
    segment_pause_min: float = 0.1
    segment_pause_max: float = 0.3
    paragraph_pause_min: float = 0.5
    paragraph_pause_max: float = 1.0
    lang: Optional[str] = None  # Language override for phonemization
    chapters: list[ChapterState] = field(default_factory=list)
    started_at: str = ""
    last_updated: str = ""

    @classmethod
    def load(cls, state_file: Path) -> Optional["ConversionState"]:
        """Load state from a JSON file."""
        if not state_file.exists():
            return None
        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct ChapterState objects
            chapters = [ChapterState(**ch) for ch in data.get("chapters", [])]
            data["chapters"] = chapters

            # Handle missing fields for backward compatibility
            if "silence_between_chapters" not in data:
                data["silence_between_chapters"] = 2.0
            if "segment_pause_min" not in data:
                data["segment_pause_min"] = 0.1
            if "segment_pause_max" not in data:
                data["segment_pause_max"] = 0.3
            if "paragraph_pause_min" not in data:
                data["paragraph_pause_min"] = 0.5
            if "paragraph_pause_max" not in data:
                data["paragraph_pause_max"] = 1.0
            if "lang" not in data:
                data["lang"] = None

            return cls(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def save(self, state_file: Path) -> None:
        """Save state to a JSON file."""
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "version": self.version,
            "source_file": self.source_file,
            "source_hash": self.source_hash,
            "output_file": self.output_file,
            "work_dir": self.work_dir,
            "voice": self.voice,
            "language": self.language,
            "speed": self.speed,
            "split_mode": self.split_mode,
            "output_format": self.output_format,
            "silence_between_chapters": self.silence_between_chapters,
            "segment_pause_min": self.segment_pause_min,
            "segment_pause_max": self.segment_pause_max,
            "paragraph_pause_min": self.paragraph_pause_min,
            "paragraph_pause_max": self.paragraph_pause_max,
            "lang": self.lang,
            "chapters": [
                {
                    "index": ch.index,
                    "title": ch.title,
                    "content_hash": ch.content_hash,
                    "completed": ch.completed,
                    "audio_file": ch.audio_file,
                    "duration": ch.duration,
                    "char_count": ch.char_count,
                }
                for ch in self.chapters
            ],
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_completed_count(self) -> int:
        """Get the number of completed chapters."""
        return sum(1 for ch in self.chapters if ch.completed)

    def get_next_incomplete_index(self) -> Optional[int]:
        """Get the index of the next incomplete chapter."""
        for ch in self.chapters:
            if not ch.completed:
                return ch.index
        return None

    def is_complete(self) -> bool:
        """Check if all chapters are completed."""
        return all(ch.completed for ch in self.chapters)


def _hash_content(content: str) -> str:
    """Generate a hash of content for integrity checking."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def _hash_file(file_path: Path) -> str:
    """Generate a hash of a file for change detection."""
    if not file_path.exists():
        return ""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


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
    use_gpu: bool = False  # GPU requires onnxruntime-gpu
    silence_between_chapters: float = 2.0
    # Language override for phonemization (e.g., 'de', 'en-us', 'fr')
    # If None, language is determined from voice prefix
    lang: Optional[str] = None
    # Segment pause (random silence between sentences within a paragraph)
    segment_pause_min: float = 0.1
    segment_pause_max: float = 0.3
    # Paragraph pause (random silence between paragraphs - longer than segment pause)
    paragraph_pause_min: float = 0.5
    paragraph_pause_max: float = 1.0
    # Chapter announcement settings
    announce_chapters: bool = True  # Read chapter titles aloud before content
    chapter_pause_after_title: float = 2.0  # Pause after chapter title (seconds)
    save_chapters_separately: bool = False
    merge_at_end: bool = True
    # Split mode: auto, line, paragraph, sentence, clause
    split_mode: str = "auto"
    # Resume capability
    resume: bool = True  # Enable resume by default for long conversions
    keep_chapter_files: bool = False  # Keep individual chapter files after merge
    # Metadata for m4b
    title: Optional[str] = None
    author: Optional[str] = None
    cover_image: Optional[Path] = None
    # Voice blending (e.g., "af_nicole:50,am_michael:50")
    voice_blend: Optional[str] = None
    # Voice database for custom/synthetic voices
    voice_database: Optional[Path] = None
    # Filename template for chapter files
    chapter_filename_template: str = "{chapter_num:03d}_{book_title}_{chapter_title}"
    # Custom ONNX model path (None = use default downloaded model)
    model_path: Optional[Path] = None
    # Custom voices.bin path (None = use default downloaded voices)
    voices_path: Optional[Path] = None


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
    """Converts text to speech using Kokoro ONNX TTS."""

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
        self._kokoro: Optional[KokoroONNX] = None
        self._np = np
        self._voice_style: Optional[Union[str, np.ndarray]] = None

    def log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        if self.log_callback:
            self.log_callback(message, level)

    def cancel(self) -> None:
        """Request cancellation of the conversion."""
        self._cancelled = True

    def _init_pipeline(self) -> None:
        """Initialize the TTS pipeline using ONNX backend."""
        if self._kokoro is not None:
            return

        self.log("Initializing ONNX TTS pipeline...")

        # Check if models are downloaded
        if not are_models_downloaded():
            self.log("Downloading ONNX model files...")
            download_all_models()
            self.log("Model download complete.")

        # Initialize ONNX backend
        self._kokoro = KokoroONNX(
            model_path=self.options.model_path,
            voices_path=self.options.voices_path,
            use_gpu=self.options.use_gpu,
        )

        # Load voice database if specified
        if self.options.voice_database and self.options.voice_database.exists():
            self._kokoro.load_voice_database(self.options.voice_database)
            self.log(f"Loaded voice database: {self.options.voice_database}")

        # Resolve voice (handle blending)
        if self.options.voice_blend:
            blend = VoiceBlend.parse(self.options.voice_blend)
            self._voice_style = self._kokoro.create_blended_voice(blend)
            voice_names = ", ".join(f"{v}:{int(w * 100)}%" for v, w in blend.voices)
            self.log(f"Using blended voice: {voice_names}")
        else:
            # Check if voice is in database first
            if self.options.voice_database:
                db_voice = self._kokoro.get_voice_from_database(self.options.voice)
                if db_voice is not None:
                    self._voice_style = db_voice
                    self.log(f"Using voice from database: {self.options.voice}")
                else:
                    self._voice_style = self.options.voice
                    self.log(f"Using voice: {self.options.voice}")
            else:
                self._voice_style = self.options.voice
                self.log(f"Using voice: {self.options.voice}")

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

    def _split_text_with_phrasplit(self, text: str) -> list["Segment"]:
        """
        Split text using phrasplit's split_text function.

        Uses the unified split_text function which returns Segment namedtuples
        containing text, paragraph index, and sentence index. This allows for
        different pause lengths between paragraphs vs. sentences.

        Args:
            text: Text to split

        Returns:
            List of Segment namedtuples with text, paragraph, and sentence info
        """
        mode = self.options.split_mode

        if mode == "line":
            # Simple line splitting - create pseudo-segments
            # Each line is its own paragraph
            from phrasplit import Segment

            segments = []
            for i, line in enumerate(text.split("\n")):
                if line.strip():
                    segments.append(Segment(text=line.strip(), paragraph=i, sentence=0))
            return segments

        if mode in ["paragraph", "sentence", "clause"]:
            try:
                from phrasplit import split_text

                spacy_model = self._get_spacy_model()
                return split_text(
                    text,
                    mode=mode,
                    language_model=spacy_model,
                    apply_corrections=True,
                    split_on_colon=True,
                )
            except ImportError:
                self.log(
                    "phrasplit not installed, falling back to line mode", "warning"
                )
                from phrasplit import Segment

                segments = []
                for i, line in enumerate(text.split("\n")):
                    if line.strip():
                        segments.append(
                            Segment(text=line.strip(), paragraph=i, sentence=0)
                        )
                return segments
            except OSError as e:
                self.log(
                    f"spaCy model error: {e}, falling back to line mode", "warning"
                )
                from phrasplit import Segment

                segments = []
                for i, line in enumerate(text.split("\n")):
                    if line.strip():
                        segments.append(
                            Segment(text=line.strip(), paragraph=i, sentence=0)
                        )
                return segments

        # "auto" mode - return empty to use pattern-based splitting
        return []

    def _use_phrasplit_mode(self) -> bool:
        """Check if we should use phrasplit for text splitting."""
        return self.options.split_mode in ["paragraph", "sentence", "clause"]

    def _generate_silence(self, duration: float) -> Any:
        """Generate silence audio of given duration."""
        samples = int(duration * SAMPLE_RATE)
        return self._np.zeros(samples, dtype="float32")

    def _generate_segment_pause(self) -> Any:
        """Generate random silence for inter-segment (sentence) pause."""
        min_pause = self.options.segment_pause_min
        max_pause = self.options.segment_pause_max
        if min_pause <= 0 and max_pause <= 0:
            return self._np.array([], dtype="float32")
        duration = random.uniform(min_pause, max_pause)
        return self._generate_silence(duration)

    def _generate_paragraph_pause(self) -> Any:
        """Generate random silence for inter-paragraph pause (longer than segment)."""
        min_pause = self.options.paragraph_pause_min
        max_pause = self.options.paragraph_pause_max
        if min_pause <= 0 and max_pause <= 0:
            return self._np.array([], dtype="float32")
        duration = random.uniform(min_pause, max_pause)
        return self._generate_silence(duration)

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

    def _convert_single_chapter_to_wav(
        self,
        chapter: Chapter,
        output_file: Path,
        progress: Optional[ConversionProgress] = None,
        start_time: Optional[float] = None,
        total_chars: int = 0,
        chars_before: int = 0,
    ) -> tuple[float, int]:
        """
        Convert a single chapter to a WAV file.

        Args:
            chapter: Chapter to convert
            output_file: Output WAV file path
            progress: Optional progress object to update
            start_time: Conversion start time for ETA calculation
            total_chars: Total characters in conversion
            chars_before: Characters processed before this chapter

        Returns:
            Tuple of (duration in seconds, characters processed)
        """
        use_phrasplit = self._use_phrasplit_mode()
        chars_processed = 0

        # Get language code for ONNX
        # Use lang override if provided, otherwise use language from options
        effective_lang = (
            self.options.lang if self.options.lang else self.options.language
        )
        lang_code = get_onnx_lang_code(effective_lang)

        # Open WAV file for writing
        with sf.SoundFile(
            str(output_file),
            "w",
            samplerate=SAMPLE_RATE,
            channels=1,
            format="wav",
        ) as out_file:
            duration = 0.0

            # Announce chapter title if enabled
            if self.options.announce_chapters and chapter.title:
                # Format: "Chapter N. Title"
                announcement_text = f"Chapter {chapter.index + 1}. {chapter.title}"
                assert self._kokoro is not None
                title_samples, _ = self._kokoro.create(
                    announcement_text,
                    voice=self._voice_style or self.options.voice,
                    speed=self.options.speed,
                    lang=lang_code,
                )
                out_file.write(title_samples)
                duration += len(title_samples) / SAMPLE_RATE

                # Add pause after chapter title
                pause_duration = self.options.chapter_pause_after_title
                if pause_duration > 0:
                    pause_samples = int(pause_duration * SAMPLE_RATE)
                    pause_audio = self._np.zeros(pause_samples, dtype="float32")
                    out_file.write(pause_audio)
                    duration += pause_duration

            if use_phrasplit:
                segments = self._split_text_with_phrasplit(chapter.content)
                # Filter out empty segments first to know total count
                segments = [s for s in segments if s.text.strip()]
                total_segments = len(segments)

                for seg_idx, segment in enumerate(segments):
                    if self._cancelled:
                        break

                    # Use ONNX backend
                    assert self._kokoro is not None
                    samples, sample_rate = self._kokoro.create(
                        segment.text,
                        voice=self._voice_style or self.options.voice,
                        speed=self.options.speed,
                        lang=lang_code,
                    )

                    if self._cancelled:
                        break
                    out_file.write(samples)
                    duration += len(samples) / SAMPLE_RATE

                    # Add pause between segments (not after the last segment)
                    # Use longer pause for paragraph boundaries, shorter for sentences
                    if seg_idx < total_segments - 1:
                        next_segment = segments[seg_idx + 1]
                        if next_segment.paragraph != segment.paragraph:
                            # Paragraph change - use longer pause
                            pause_audio = self._generate_paragraph_pause()
                        else:
                            # Same paragraph - use shorter sentence pause
                            pause_audio = self._generate_segment_pause()
                        if len(pause_audio) > 0:
                            out_file.write(pause_audio)
                            duration += len(pause_audio) / SAMPLE_RATE

                    # Update progress
                    grapheme_len = len(segment.text)
                    chars_processed += grapheme_len
                    if progress and self.progress_callback:
                        progress.chars_processed = chars_before + chars_processed
                        progress.current_text = segment.text[:100]
                        if start_time and total_chars > 0:
                            elapsed = time.time() - start_time
                            if chars_processed > 0 and elapsed > 0.5:
                                avg_time = elapsed / chars_processed
                                remaining = total_chars - progress.chars_processed
                                progress.estimated_remaining = avg_time * remaining
                            progress.elapsed_time = elapsed
                        self.progress_callback(progress)
            else:
                # Use chunk-based generation for progress tracking
                assert self._kokoro is not None
                # Collect chunks to know total count for proper pause insertion
                chunks_data = list(
                    self._kokoro.generate_chunks(
                        chapter.content,
                        voice=self._voice_style or self.options.voice,
                        speed=self.options.speed,
                        lang=lang_code,
                    )
                )
                total_chunks = len(chunks_data)

                for chunk_idx, (samples, _sample_rate, chunk) in enumerate(chunks_data):
                    if self._cancelled:
                        break
                    out_file.write(samples)
                    duration += len(samples) / SAMPLE_RATE

                    # Add random pause between chunks (not after the last chunk)
                    if chunk_idx < total_chunks - 1:
                        pause_audio = self._generate_segment_pause()
                        if len(pause_audio) > 0:
                            out_file.write(pause_audio)
                            duration += len(pause_audio) / SAMPLE_RATE

                    # Update progress
                    grapheme_len = len(chunk)
                    chars_processed += grapheme_len
                    if progress and self.progress_callback:
                        progress.chars_processed = chars_before + chars_processed
                        progress.current_text = chunk[:100]
                        if start_time and total_chars > 0:
                            elapsed = time.time() - start_time
                            if chars_processed > 0 and elapsed > 0.5:
                                avg_time = elapsed / chars_processed
                                remaining = total_chars - progress.chars_processed
                                progress.estimated_remaining = avg_time * remaining
                            progress.elapsed_time = elapsed
                        self.progress_callback(progress)

        return duration, chars_processed

    def _merge_chapter_files(
        self,
        chapter_files: list[Path],
        chapter_durations: list[float],
        chapter_titles: list[str],
        output_path: Path,
    ) -> None:
        """
        Merge individual chapter WAV files into the final output format.

        Args:
            chapter_files: List of chapter WAV file paths
            chapter_durations: Duration of each chapter in seconds
            chapter_titles: Title of each chapter
            output_path: Final output file path
        """
        ensure_ffmpeg()
        import static_ffmpeg

        static_ffmpeg.add_paths()

        fmt = self.options.output_format
        silence_duration = self.options.silence_between_chapters

        # Create a concat file for ffmpeg
        concat_file = output_path.with_suffix(".concat.txt")
        silence_file = output_path.parent / "_silence.wav"

        # Generate silence file if needed
        if silence_duration > 0 and len(chapter_files) > 1:
            silence_samples = int(silence_duration * SAMPLE_RATE)
            silence_audio = self._np.zeros(silence_samples, dtype="float32")
            with sf.SoundFile(
                str(silence_file),
                "w",
                samplerate=SAMPLE_RATE,
                channels=1,
                format="wav",
            ) as f:
                f.write(silence_audio)

        # Write concat file
        with open(concat_file, "w", encoding="utf-8") as f:
            for i, chapter_file in enumerate(chapter_files):
                f.write(f"file '{chapter_file.absolute()}'\n")
                if i < len(chapter_files) - 1 and silence_duration > 0:
                    f.write(f"file '{silence_file.absolute()}'\n")

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
        ]

        if fmt == "m4b":
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
            if self.options.title:
                cmd.extend(["-metadata", f"title={self.options.title}"])
            if self.options.author:
                cmd.extend(["-metadata", f"artist={self.options.author}"])
        elif fmt == "opus":
            cmd.extend(["-c:a", "libopus", "-b:a", "24000"])
        elif fmt == "mp3":
            cmd.extend(["-c:a", "libmp3lame", "-q:a", "2"])
        elif fmt == "flac":
            cmd.extend(["-c:a", "flac"])
        elif fmt == "wav":
            cmd.extend(["-c:a", "pcm_s16le"])

        cmd.append(str(output_path))

        proc = create_process(cmd, suppress_output=True)
        proc.wait()

        # Clean up concat and silence files
        concat_file.unlink(missing_ok=True)
        silence_file.unlink(missing_ok=True)

        # Add chapter markers for m4b
        if fmt == "m4b" and len(chapter_files) > 1:
            chapter_times: list[dict[str, Any]] = []
            current_time = 0.0
            for i, (duration, title) in enumerate(
                zip(chapter_durations, chapter_titles)
            ):
                chapter_times.append(
                    {
                        "title": title,
                        "start": current_time,
                        "end": current_time + duration,
                    }
                )
                current_time += duration
                if i < len(chapter_durations) - 1:
                    current_time += silence_duration

            self._add_chapters_to_m4b(output_path, chapter_times)

    def convert_chapters_resumable(
        self,
        chapters: list[Chapter],
        output_path: Path,
        source_file: Optional[Path] = None,
        resume: bool = True,
    ) -> ConversionResult:
        """
        Convert chapters to audio with resume capability.

        Each chapter is saved as a separate WAV file, allowing conversion
        to be resumed if interrupted. A state file tracks progress.

        Args:
            chapters: List of Chapter objects
            output_path: Output file path
            source_file: Original source file (for state tracking)
            resume: Whether to resume from previous state

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
            # Set up work directory for chapter files (use book title)
            safe_book_title = sanitize_filename(self.options.title or output_path.stem)[
                :50
            ]
            work_dir = output_path.parent / f".{safe_book_title}_chapters"
            work_dir.mkdir(parents=True, exist_ok=True)
            state_file = work_dir / f"{safe_book_title}_state.json"

            # Load or create state
            state: Optional[ConversionState] = None
            if resume and state_file.exists():
                state = ConversionState.load(state_file)
                if state:
                    # Verify source file hasn't changed
                    source_hash = _hash_file(source_file) if source_file else ""
                    if source_file and state.source_hash != source_hash:
                        self.log(
                            "Source file changed, starting fresh conversion",
                            "warning",
                        )
                        state = None
                    else:
                        # Check if settings differ from saved state
                        settings_changed = (
                            state.voice != self.options.voice
                            or state.language != self.options.language
                            or state.speed != self.options.speed
                            or state.split_mode != self.options.split_mode
                            or state.silence_between_chapters
                            != self.options.silence_between_chapters
                            or state.segment_pause_min != self.options.segment_pause_min
                            or state.segment_pause_max != self.options.segment_pause_max
                            or state.paragraph_pause_min
                            != self.options.paragraph_pause_min
                            or state.paragraph_pause_max
                            != self.options.paragraph_pause_max
                            or state.lang != self.options.lang
                        )

                        if settings_changed:
                            self.log(
                                f"Restoring settings from previous session: "
                                f"voice={state.voice}, language={state.language}, "
                                f"lang_override={state.lang}, "
                                f"speed={state.speed}, split_mode={state.split_mode}, "
                                f"silence={state.silence_between_chapters}s, "
                                f"segment_pause={state.segment_pause_min}-"
                                f"{state.segment_pause_max}s, "
                                f"paragraph_pause={state.paragraph_pause_min}-"
                                f"{state.paragraph_pause_max}s",
                                "info",
                            )

                        # Apply saved settings to options for consistency
                        self.options.voice = state.voice
                        self.options.language = state.language
                        self.options.speed = state.speed
                        self.options.split_mode = state.split_mode
                        self.options.output_format = state.output_format
                        self.options.silence_between_chapters = (
                            state.silence_between_chapters
                        )
                        self.options.segment_pause_min = state.segment_pause_min
                        self.options.segment_pause_max = state.segment_pause_max
                        self.options.paragraph_pause_min = state.paragraph_pause_min
                        self.options.paragraph_pause_max = state.paragraph_pause_max
                        self.options.lang = state.lang

            if state is None:
                # Create new state
                source_hash = _hash_file(source_file) if source_file else ""
                state = ConversionState(
                    source_file=str(source_file) if source_file else "",
                    source_hash=source_hash,
                    output_file=str(output_path),
                    work_dir=str(work_dir),
                    voice=self.options.voice,
                    language=self.options.language,
                    speed=self.options.speed,
                    split_mode=self.options.split_mode,
                    output_format=self.options.output_format,
                    silence_between_chapters=self.options.silence_between_chapters,
                    segment_pause_min=self.options.segment_pause_min,
                    segment_pause_max=self.options.segment_pause_max,
                    paragraph_pause_min=self.options.paragraph_pause_min,
                    paragraph_pause_max=self.options.paragraph_pause_max,
                    lang=self.options.lang,
                    chapters=[
                        ChapterState(
                            index=i,
                            title=ch.title,
                            content_hash=_hash_content(ch.content),
                            char_count=ch.char_count,
                        )
                        for i, ch in enumerate(chapters)
                    ],
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                state.save(state_file)
            else:
                completed = state.get_completed_count()
                total = len(chapters)
                self.log(f"Resuming conversion: {completed}/{total} chapters completed")

            # Initialize pipeline
            self._init_pipeline()

            total_chars = sum(ch.char_count for ch in chapters)
            # Account for already completed chapters
            chars_already_done = sum(
                state.chapters[i].char_count
                for i in range(len(state.chapters))
                if state.chapters[i].completed
            )
            chars_processed = chars_already_done
            start_time = time.time()

            progress = ConversionProgress(
                total_chapters=len(chapters),
                total_chars=total_chars,
                chars_processed=chars_processed,
            )

            # Convert each chapter
            for chapter_idx, chapter in enumerate(chapters):
                if self._cancelled:
                    state.save(state_file)
                    return ConversionResult(
                        success=False,
                        error_message="Cancelled",
                        chapters_dir=work_dir,
                    )

                chapter_state = state.chapters[chapter_idx]

                # Skip already completed chapters
                if chapter_state.completed and chapter_state.audio_file:
                    chapter_file = work_dir / chapter_state.audio_file
                    if chapter_file.exists():
                        ch_num = chapter_idx + 1
                        self.log(
                            f"Skipping completed chapter {ch_num}: {chapter.title}"
                        )
                        continue
                    else:
                        # File missing, need to reconvert
                        chapter_state.completed = False

                progress.current_chapter = chapter_idx + 1
                progress.chapter_name = chapter.title

                ch_num = chapter_idx + 1
                self.log(
                    f"Converting chapter {ch_num}/{len(chapters)}: {chapter.title}"
                )

                # Generate chapter filename using template
                chapter_filename = (
                    format_filename_template(
                        self.options.chapter_filename_template,
                        book_title=self.options.title or "Untitled",
                        chapter_title=chapter.title,
                        chapter_num=chapter_idx + 1,
                    )
                    + ".wav"
                )
                chapter_file = work_dir / chapter_filename

                # Convert chapter to WAV
                duration, _ = self._convert_single_chapter_to_wav(
                    chapter,
                    chapter_file,
                    progress=progress,
                    start_time=start_time,
                    total_chars=total_chars,
                    chars_before=chars_processed,
                )

                if self._cancelled:
                    # Remove incomplete file
                    chapter_file.unlink(missing_ok=True)
                    state.save(state_file)
                    return ConversionResult(
                        success=False,
                        error_message="Cancelled",
                        chapters_dir=work_dir,
                    )

                # Update state
                chapter_state.completed = True
                chapter_state.audio_file = chapter_filename
                chapter_state.duration = duration
                state.save(state_file)

                # Update progress
                chars_processed += chapter.char_count
                progress.chars_processed = chars_processed
                elapsed = time.time() - start_time
                if chars_processed > chars_already_done and elapsed > 0.5:
                    chars_in_session = chars_processed - chars_already_done
                    avg_time = elapsed / chars_in_session
                    remaining = total_chars - chars_processed
                    progress.estimated_remaining = avg_time * remaining
                progress.elapsed_time = elapsed

                if self.progress_callback:
                    self.progress_callback(progress)

            # All chapters completed, merge into final output
            self.log("Merging chapters into final audiobook...")

            chapter_files = [
                work_dir / ch.audio_file for ch in state.chapters if ch.audio_file
            ]
            chapter_durations = [ch.duration for ch in state.chapters]
            chapter_titles = [ch.title for ch in state.chapters]

            self._merge_chapter_files(
                chapter_files,
                chapter_durations,
                chapter_titles,
                output_path,
            )

            self.log("Conversion complete!")

            return ConversionResult(
                success=True,
                output_path=output_path,
                chapters_dir=work_dir,
            )

        except Exception as e:
            return ConversionResult(success=False, error_message=str(e))
        finally:
            prevent_sleep_end()

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

                # Get language code for ONNX
                # Use lang override if provided, otherwise use language from options
                effective_lang = (
                    self.options.lang if self.options.lang else self.options.language
                )
                lang_code = get_onnx_lang_code(effective_lang)

                # Generate TTS for this chapter
                if use_phrasplit:
                    # Pre-split text using phrasplit, then process each segment
                    segments = self._split_text_with_phrasplit(chapter.content)
                    # Filter out empty segments to know total count
                    segments = [s for s in segments if s.text.strip()]
                    total_segments = len(segments)

                    for seg_idx, segment in enumerate(segments):
                        if self._cancelled:
                            break

                        # Process each segment using ONNX backend
                        assert self._kokoro is not None
                        samples, sample_rate = self._kokoro.create(
                            segment.text,
                            voice=self._voice_style or self.options.voice,
                            speed=self.options.speed,
                            lang=lang_code,
                        )

                        if self._cancelled:
                            break

                        # Write audio
                        self._write_audio_chunk(samples, out_file, ffmpeg_proc)

                        # Update timing
                        chunk_duration = len(samples) / SAMPLE_RATE
                        current_time += chunk_duration

                        # Add pause between segments (not after the last segment)
                        # Longer pause for paragraph boundaries, shorter for sentences
                        if seg_idx < total_segments - 1:
                            next_segment = segments[seg_idx + 1]
                            if next_segment.paragraph != segment.paragraph:
                                # Paragraph change - use longer pause
                                pause_audio = self._generate_paragraph_pause()
                            else:
                                # Same paragraph - use shorter sentence pause
                                pause_audio = self._generate_segment_pause()
                            if len(pause_audio) > 0:
                                self._write_audio_chunk(
                                    pause_audio, out_file, ffmpeg_proc
                                )
                                current_time += len(pause_audio) / SAMPLE_RATE

                        # Update progress
                        grapheme_len = len(segment.text)
                        chars_processed += grapheme_len
                        progress.chars_processed = chars_processed
                        progress.current_text = segment.text[:100]

                        elapsed = time.time() - start_time
                        if chars_processed > 0 and elapsed > 0.5:
                            avg_time = elapsed / chars_processed
                            remaining = total_chars - chars_processed
                            progress.estimated_remaining = avg_time * remaining
                        progress.elapsed_time = elapsed

                        if self.progress_callback:
                            self.progress_callback(progress)
                else:
                    # Use chunk-based generation for progress tracking
                    assert self._kokoro is not None
                    # Collect chunks to know total count for proper pause insertion
                    chunks_data = list(
                        self._kokoro.generate_chunks(
                            chapter.content,
                            voice=self._voice_style or self.options.voice,
                            speed=self.options.speed,
                            lang=lang_code,
                        )
                    )
                    total_chunks = len(chunks_data)

                    for chunk_idx, (samples, _sample_rate, chunk) in enumerate(
                        chunks_data
                    ):
                        if self._cancelled:
                            break

                        # Write audio
                        self._write_audio_chunk(samples, out_file, ffmpeg_proc)

                        # Update timing
                        chunk_duration = len(samples) / SAMPLE_RATE
                        current_time += chunk_duration

                        # Add random pause between chunks (not after the last chunk)
                        if chunk_idx < total_chunks - 1:
                            pause_audio = self._generate_segment_pause()
                            if len(pause_audio) > 0:
                                self._write_audio_chunk(
                                    pause_audio, out_file, ffmpeg_proc
                                )
                                current_time += len(pause_audio) / SAMPLE_RATE

                        # Update progress
                        grapheme_len = len(chunk)
                        chars_processed += grapheme_len
                        progress.chars_processed = chars_processed
                        progress.current_text = chunk[:100]

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

        # Use resumable conversion if enabled
        if self.options.resume:
            result = self.convert_chapters_resumable(
                chapters, output_path, source_file=epub_path, resume=True
            )
            # Clean up chapter files unless keep_chapter_files is set
            if (
                result.success
                and result.chapters_dir
                and not self.options.keep_chapter_files
            ):
                import shutil

                try:
                    shutil.rmtree(result.chapters_dir)
                except Exception:
                    pass
            return result

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
