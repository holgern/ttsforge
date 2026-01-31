"""TTS conversion module for ttsforge - converts text/EPUB to audiobooks."""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import soundfile as sf

from .audio_merge import AudioMerger, MergeMeta
from .constants import (
    DEFAULT_VOICE_FOR_LANG,
    ISO_TO_LANG_CODE,
    SAMPLE_RATE,
    SUPPORTED_OUTPUT_FORMATS,
    VOICE_PREFIX_TO_LANG,
)
from .kokoro_lang import get_onnx_lang_code
from .kokoro_runner import KokoroRunner, KokoroRunOptions
from .ssmd_generator import (
    SSMDGenerationError,
    chapter_to_ssmd,
    load_ssmd_file,
    save_ssmd_file,
)
from .utils import (
    format_duration,
    format_filename_template,
    load_phoneme_dictionary,
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
    html_content: Optional[str] = None  # Optional HTML for emphasis detection
    is_ssmd: bool = False

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def text(self) -> str:
        """Alias for content to maintain compatibility with input_reader.Chapter."""
        return self.content


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
    ssmd_file: Optional[str] = None  # Relative path to SSMD file
    ssmd_hash: Optional[str] = None  # Hash of SSMD content for change detection


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
    pause_clause: float = 0.25
    pause_sentence: float = 0.2
    pause_paragraph: float = 0.75
    pause_variance: float = 0.05
    pause_mode: str = "auto"  # "tts", "manual", or "auto
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

            # Migrate old pause parameters to new system
            if "segment_pause_min" in data or "segment_pause_max" in data:
                seg_min = data.get("segment_pause_min", 0.1)
                seg_max = data.get("segment_pause_max", 0.3)
                data["pause_sentence"] = (seg_min + seg_max) / 2.0
                if "pause_variance" not in data:
                    data["pause_variance"] = max(0.01, (seg_max - seg_min) / 4.0)

            if "paragraph_pause_min" in data or "paragraph_pause_max" in data:
                para_min = data.get("paragraph_pause_min", 0.5)
                para_max = data.get("paragraph_pause_max", 1.0)
                data["pause_paragraph"] = (para_min + para_max) / 2.0

            # Set defaults for new parameters
            if "pause_clause" not in data:
                data["pause_clause"] = 0.25
            if "pause_sentence" not in data:
                data["pause_sentence"] = 0.2
            if "pause_paragraph" not in data:
                data["pause_paragraph"] = 0.75
            if "pause_variance" not in data:
                data["pause_variance"] = 0.05
            if "pause_mode" not in data:
                data["pause_mode"] = "auto"
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
            "pause_clause": self.pause_clause,
            "pause_sentence": self.pause_sentence,
            "pause_paragraph": self.pause_paragraph,
            "pause_variance": self.pause_variance,
            "pause_mode": self.pause_mode,
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
                    "ssmd_file": ch.ssmd_file,
                    "ssmd_hash": ch.ssmd_hash,
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
    # Mixed-language support (auto-detect and handle multiple languages)
    use_mixed_language: bool = False
    mixed_language_primary: Optional[str] = None
    mixed_language_allowed: Optional[list[str]] = None
    mixed_language_confidence: float = 0.7
    # Custom phoneme dictionary for pronunciation overrides
    phoneme_dictionary_path: Optional[str] = None
    phoneme_dict_case_sensitive: bool = False
    # Pause settings (pykokoro built-in pause handling)
    pause_clause: float = 0.25  # For clause boundaries (commas)
    pause_sentence: float = 0.2  # For sentence boundaries
    pause_paragraph: float = 0.75  # For paragraph boundaries
    pause_variance: float = 0.05  # Standard deviation for natural variation
    pause_mode: str = "auto"  # "tts", "manual", or "auto
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
    # SSMD generation control
    generate_ssmd_only: bool = False  # If True, only generate SSMD files, no audio
    detect_emphasis: bool = False  # If True, detect emphasis from HTML tags in EPUB


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
        self._runner: KokoroRunner | None = None
        self._merger = AudioMerger(log=self.log)

    def log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        if self.log_callback:
            self.log_callback(message, level)

    def cancel(self) -> None:
        """Request cancellation of the conversion."""
        self._cancelled = True

    def _init_runner(self) -> None:
        """Initialize the Kokoro runner."""
        if self._runner is not None:
            return

        self.log("Initializing ONNX TTS pipeline...")

        # Create TokenizerConfig from ConversionOptions (for mixed-language support)
        from pykokoro.tokenizer import TokenizerConfig

        tokenizer_config = TokenizerConfig(
            use_mixed_language=self.options.use_mixed_language,
            mixed_language_primary=self.options.mixed_language_primary,
            mixed_language_allowed=self.options.mixed_language_allowed,
            mixed_language_confidence=self.options.mixed_language_confidence,
            phoneme_dictionary_path=self.options.phoneme_dictionary_path,
            phoneme_dict_case_sensitive=self.options.phoneme_dict_case_sensitive,
        )

        opts = KokoroRunOptions(
            voice=self.options.voice,
            speed=self.options.speed,
            use_gpu=self.options.use_gpu,
            pause_clause=self.options.pause_clause,
            pause_sentence=self.options.pause_sentence,
            pause_paragraph=self.options.pause_paragraph,
            pause_variance=self.options.pause_variance,
            model_path=self.options.model_path,
            voices_path=self.options.voices_path,
            voice_blend=self.options.voice_blend,
            voice_database=self.options.voice_database,
            tokenizer_config=tokenizer_config,
        )
        self._runner = KokoroRunner(opts, log=self.log)
        self._runner.ensure_ready()

    def _build_ssmd_content(
        self,
        chapter: Chapter,
        phoneme_dict: Optional[dict[str, str]],
        mixed_language_config: Optional[dict[str, Any]],
        html_content: Optional[str],
    ) -> str:
        """Generate SSMD content for a chapter, falling back to plain text."""
        try:
            return chapter_to_ssmd(
                chapter_title=chapter.title,
                chapter_text=chapter.text,
                phoneme_dict=phoneme_dict,
                phoneme_dict_case_sensitive=self.options.phoneme_dict_case_sensitive,
                mixed_language_config=mixed_language_config,
                html_content=html_content,
                include_title=self.options.announce_chapters,
            )
        except SSMDGenerationError as e:
            self.log(f"SSMD generation failed: {e}, using plain text", "error")
            return chapter.text

    def _load_or_generate_ssmd(
        self,
        chapter: Chapter,
        ssmd_file: Path,
        phoneme_dict: Optional[dict[str, str]],
        mixed_language_config: Optional[dict[str, Any]],
        html_content: Optional[str],
    ) -> tuple[str, str]:
        """Load SSMD from disk or generate and save it."""
        ssmd_content: Optional[str] = None
        ssmd_hash = ""

        if chapter.is_ssmd:
            if ssmd_file.exists():
                try:
                    ssmd_content, ssmd_hash = load_ssmd_file(ssmd_file)
                    self.log(f"Loaded SSMD from {ssmd_file.name}")
                except SSMDGenerationError as e:
                    self.log(f"Failed to load SSMD: {e}, using input", "warning")
                    ssmd_content = None

            if ssmd_content is None:
                ssmd_content = chapter.text
                ssmd_hash = save_ssmd_file(ssmd_content, ssmd_file)
                self.log(f"Saved SSMD to {ssmd_file.name}")

            return ssmd_content, ssmd_hash

        if ssmd_file.exists():
            try:
                ssmd_content, ssmd_hash = load_ssmd_file(ssmd_file)
                self.log(f"Loaded SSMD from {ssmd_file.name}")
            except SSMDGenerationError as e:
                self.log(f"Failed to load SSMD: {e}, regenerating...", "warning")
                ssmd_content = None

        if ssmd_content is None:
            self.log(f"Generating SSMD for chapter: {chapter.title}")
            ssmd_content = self._build_ssmd_content(
                chapter,
                phoneme_dict=phoneme_dict,
                mixed_language_config=mixed_language_config,
                html_content=html_content,
            )
            ssmd_hash = save_ssmd_file(ssmd_content, ssmd_file)
            self.log(f"Saved SSMD to {ssmd_file.name}")

        return ssmd_content, ssmd_hash

    def _render_chapter_wav(
        self,
        chapter: Chapter,
        output_file: Path,
        ssmd_content: str,
    ) -> float:
        """Render SSMD content to a chapter WAV file."""
        effective_lang = (
            self.options.lang if self.options.lang else self.options.language
        )
        lang_code = get_onnx_lang_code(effective_lang)

        with sf.SoundFile(
            str(output_file),
            "w",
            samplerate=SAMPLE_RATE,
            channels=1,
            format="wav",
        ) as out_file:
            assert self._runner is not None
            samples = self._runner.synthesize(
                ssmd_content,
                lang_code=lang_code,
                pause_mode=self.options.pause_mode,
                is_phonemes=False,
            )
            out_file.write(samples)

        return len(samples) / SAMPLE_RATE

    def convert_chapters_resumable(  # noqa: C901 - Complex but necessary for resume logic
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
                    # Verify chapter count matches
                    elif len(state.chapters) != len(chapters):
                        self.log(
                            f"Chapter count changed "
                            f"({len(state.chapters)} -> {len(chapters)}), "
                            "starting fresh conversion",
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
                            or state.pause_clause != self.options.pause_clause
                            or state.pause_sentence != self.options.pause_sentence
                            or state.pause_paragraph != self.options.pause_paragraph
                            or state.pause_variance != self.options.pause_variance
                            or state.pause_mode != self.options.pause_mode
                            or state.lang != self.options.lang
                        )

                        if settings_changed:
                            self.log(
                                f"Restoring settings from previous session: "
                                f"voice={state.voice}, language={state.language}, "
                                f"lang_override={state.lang}, "
                                f"speed={state.speed}, "
                                f"split_mode={state.split_mode}, "
                                f"silence={state.silence_between_chapters}s, "
                                f"pauses: clause={state.pause_clause}s "
                                f"sent={state.pause_sentence}s "
                                f"para={state.pause_paragraph}s "
                                f"var={state.pause_variance}s "
                                f"pause_mode={state.pause_mode}",
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
                        self.options.pause_clause = state.pause_clause
                        self.options.pause_sentence = state.pause_sentence
                        self.options.pause_paragraph = state.pause_paragraph
                        self.options.pause_variance = state.pause_variance
                        self.options.pause_mode = state.pause_mode
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
                    pause_clause=self.options.pause_clause,
                    pause_sentence=self.options.pause_sentence,
                    pause_paragraph=self.options.pause_paragraph,
                    pause_variance=self.options.pause_variance,
                    pause_mode=self.options.pause_mode,
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

            # Initialize runner
            self._init_runner()

            phoneme_dict = None
            if self.options.phoneme_dictionary_path:
                phoneme_dict = load_phoneme_dictionary(
                    self.options.phoneme_dictionary_path,
                    case_sensitive=self.options.phoneme_dict_case_sensitive,
                    log_callback=lambda message: self.log(message, "warning"),
                )

            mixed_language_config = None
            if self.options.use_mixed_language:
                mixed_language_config = {
                    "use_mixed_language": True,
                    "primary": self.options.mixed_language_primary,
                    "allowed": self.options.mixed_language_allowed,
                    "confidence": self.options.mixed_language_confidence,
                }

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

                # Validate chapter index to prevent index errors
                if chapter_idx >= len(state.chapters):
                    error_msg = (
                        f"Chapter index {chapter_idx} out of range. "
                        f"State has {len(state.chapters)} chapters "
                        f"but trying to access "
                        f"chapter {chapter_idx + 1}/{len(chapters)}. "
                        "This usually means the state file is corrupted. "
                        "Try using --fresh to start a new conversion."
                    )
                    return ConversionResult(
                        success=False,
                        error_message=error_msg,
                    )

                chapter_state = state.chapters[chapter_idx]

                # Check if SSMD file was manually edited
                ssmd_edited = False
                if chapter_state.ssmd_file and chapter_state.ssmd_hash:
                    ssmd_path = work_dir / chapter_state.ssmd_file
                    if ssmd_path.exists():
                        try:
                            _, current_hash = load_ssmd_file(ssmd_path)
                            if current_hash != chapter_state.ssmd_hash:
                                self.log(
                                    f"Chapter {chapter_idx + 1} SSMD file was edited, "
                                    "will regenerate audio",
                                    "info",
                                )
                                ssmd_edited = True
                                chapter_state.completed = False
                        except SSMDGenerationError:
                            # SSMD file corrupted, will regenerate
                            ssmd_edited = True
                            chapter_state.completed = False

                # Skip already completed chapters (unless SSMD was edited)
                if (
                    chapter_state.completed
                    and chapter_state.audio_file
                    and not ssmd_edited
                ):
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

                # Generate SSMD filename (same as WAV but with .ssmd extension)
                ssmd_filename = chapter_filename.replace(".wav", ".ssmd")
                ssmd_file = work_dir / ssmd_filename
                html_content = (
                    chapter.html_content if self.options.detect_emphasis else None
                )
                ssmd_content, ssmd_hash = self._load_or_generate_ssmd(
                    chapter,
                    ssmd_file,
                    phoneme_dict=phoneme_dict,
                    mixed_language_config=mixed_language_config,
                    html_content=html_content,
                )

                # If generate_ssmd_only mode, just generate SSMD and skip audio
                if self.options.generate_ssmd_only:
                    chapter_state.completed = True
                    chapter_state.ssmd_file = ssmd_filename
                    chapter_state.ssmd_hash = ssmd_hash
                    state.save(state_file)

                    chars_processed += chapter.char_count
                    progress.chars_processed = chars_processed
                    if self.progress_callback:
                        self.progress_callback(progress)
                    continue

                duration = self._render_chapter_wav(
                    chapter,
                    chapter_file,
                    ssmd_content,
                )

                if self._cancelled:
                    # Remove incomplete files
                    chapter_file.unlink(missing_ok=True)
                    ssmd_file.unlink(missing_ok=True)
                    state.save(state_file)
                    return ConversionResult(
                        success=False,
                        error_message="Cancelled",
                        chapters_dir=work_dir,
                    )

                # Update state
                chapter_state.completed = True
                chapter_state.audio_file = chapter_filename
                chapter_state.ssmd_file = ssmd_filename
                chapter_state.ssmd_hash = ssmd_hash
                chapter_state.duration = duration
                state.save(state_file)

                # Update progress
                chars_processed += chapter.char_count
                progress.chars_processed = chars_processed
                progress.current_text = (
                    f"Completed chapter: {chapter.title or 'Untitled'}"
                )
                elapsed = time.time() - start_time
                if chars_processed > chars_already_done and elapsed > 0.5:
                    chars_in_session = chars_processed - chars_already_done
                    avg_time = elapsed / chars_in_session
                    remaining = total_chars - chars_processed
                    progress.estimated_remaining = avg_time * remaining
                progress.elapsed_time = elapsed

                if self.progress_callback:
                    self.progress_callback(progress)

            # If generate_ssmd_only mode, exit here without merging
            if self.options.generate_ssmd_only:
                self.log("SSMD generation complete!")
                self.log(f"SSMD files saved in: {work_dir}")
                return ConversionResult(
                    success=True,
                    chapters_dir=work_dir,
                    output_path=None,  # No audio output in SSMD-only mode
                )

            # All chapters completed, merge into final output
            self.log("Merging chapters into final audiobook...")

            chapter_files = [
                work_dir / ch.audio_file for ch in state.chapters if ch.audio_file
            ]
            chapter_durations = [ch.duration for ch in state.chapters]
            chapter_titles = [ch.title for ch in state.chapters]

            meta = MergeMeta(
                fmt=self.options.output_format,
                silence_between_chapters=self.options.silence_between_chapters,
                title=self.options.title,
                author=self.options.author,
                cover_image=self.options.cover_image,
            )
            self._merger.merge_chapter_wavs(
                chapter_files,
                chapter_durations,
                chapter_titles,
                output_path,
                meta,
            )

            self.log("Conversion complete!")

            return ConversionResult(
                success=True,
                output_path=output_path,
                chapters_dir=work_dir,
            )

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return ConversionResult(success=False, error_message=error_msg)
        finally:
            prevent_sleep_end()

    def convert_chapters(
        self,
        chapters: list[Chapter],
        output_path: Path,
    ) -> ConversionResult:
        """Convert a list of chapters to audio using the SSMD pipeline."""
        result = self.convert_chapters_resumable(
            chapters=chapters,
            output_path=output_path,
            resume=self.options.resume,
        )
        self._cleanup_chapter_dir(result)
        return result

    def _cleanup_chapter_dir(self, result: ConversionResult) -> None:
        if self.options.generate_ssmd_only:
            return
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
        # Remove <<CHAPTER: ...>> markers that epub2text adds at the start of content
        # since we now announce chapter titles separately
        chapters = []
        for i, ch in enumerate(epub_chapters):
            # Remove the <<CHAPTER: title>> marker from the beginning of content
            content = ch.text
            # Pattern matches: <<CHAPTER: anything>> followed by whitespace/newlines
            content = re.sub(
                r"^\s*<<CHAPTER:[^>]*>>\s*\n*", "", content, count=1, flags=re.MULTILINE
            )
            chapters.append(Chapter(title=ch.title, content=content, index=i))

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

        result = self.convert_chapters_resumable(
            chapters,
            output_path,
            source_file=epub_path,
            resume=self.options.resume,
        )
        self._cleanup_chapter_dir(result)
        return result


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
