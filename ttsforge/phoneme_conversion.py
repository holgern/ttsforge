"""Phoneme-based TTS conversion module for ttsforge.

This module converts pre-tokenized PhonemeBook files to audio,
bypassing text-to-phoneme conversion since phonemes/tokens are pre-computed.
"""

import json
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, cast

import numpy as np
import soundfile as sf

from .audio_merge import AudioMerger, MergeMeta
from .chapter_selection import parse_chapter_selection
from .constants import SAMPLE_RATE, SUPPORTED_OUTPUT_FORMATS
from .kokoro_lang import get_onnx_lang_code
from .kokoro_runner import KokoroRunner, KokoroRunOptions
from .phonemes import PhonemeBook, PhonemeChapter, PhonemeSegment
from .utils import (
    atomic_write_json,
    create_process,
    format_duration,
    format_filename_template,
    get_ffmpeg_path,
    prevent_sleep_end,
    prevent_sleep_start,
    sanitize_filename,
)


@dataclass
class PhonemeConversionProgress:
    """Progress information during phoneme conversion."""

    current_chapter: int = 0
    total_chapters: int = 0
    chapter_name: str = ""
    current_segment: int = 0
    total_segments: int = 0
    segments_processed: int = 0  # Global segment count
    total_segments_all: int = 0  # Total segments across all chapters
    current_text: str = ""
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0

    @property
    def percent(self) -> int:
        if self.total_segments_all == 0:
            return 0
        return min(int(self.segments_processed / self.total_segments_all * 100), 99)

    @property
    def etr_formatted(self) -> str:
        return format_duration(self.estimated_remaining)


@dataclass
class PhonemeConversionResult:
    """Result of a phoneme conversion operation."""

    success: bool
    output_path: Path | None = None
    error_message: str | None = None
    chapters_dir: Path | None = None
    duration: float = 0.0


@dataclass
class PhonemeChapterState:
    """State of a single chapter conversion."""

    index: int
    title: str
    segment_count: int
    completed: bool = False
    audio_file: str | None = None  # Relative path to chapter audio
    duration: float = 0.0


@dataclass
class PhonemeConversionState:
    """Persistent state for resumable phoneme conversions."""

    version: int = 1
    source_file: str = ""
    output_file: str = ""
    work_dir: str = ""
    voice: str = ""
    speed: float = 1.0
    output_format: str = "m4b"
    silence_between_chapters: float = 2.0
    pause_clause: float = 0.25
    pause_sentence: float = 0.2
    pause_paragraph: float = 0.75
    pause_variance: float = 0.05
    pause_mode: str = "auto"
    lang: str | None = None  # Language override for phonemization
    chapters: list[PhonemeChapterState] = field(default_factory=list)
    started_at: str = ""
    last_updated: str = ""
    # Track selected chapters (0-based indices)
    selected_chapters: list[int] = field(default_factory=list)

    def get_completed_count(self) -> int:
        """Get number of completed chapters."""
        return sum(1 for ch in self.chapters if ch.completed)

    @classmethod
    def load(cls, state_file: Path) -> Optional["PhonemeConversionState"]:
        """Load state from a JSON file."""
        if not state_file.exists():
            return None
        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct PhonemeChapterState objects
            chapters = [PhonemeChapterState(**ch) for ch in data.get("chapters", [])]
            data["chapters"] = chapters

            # Handle missing fields for backward compatibility
            if "silence_between_chapters" not in data:
                data["silence_between_chapters"] = 2.0
            if "selected_chapters" not in data:
                data["selected_chapters"] = []

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
            "output_file": self.output_file,
            "work_dir": self.work_dir,
            "voice": self.voice,
            "speed": self.speed,
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
                    "segment_count": ch.segment_count,
                    "completed": ch.completed,
                    "audio_file": ch.audio_file,
                    "duration": ch.duration,
                }
                for ch in self.chapters
            ],
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "selected_chapters": self.selected_chapters,
        }
        atomic_write_json(state_file, data, indent=2, ensure_ascii=True)


@dataclass
class PhonemeConversionOptions:
    """Options for phoneme-based TTS conversion."""

    voice: str = "af_heart"
    speed: float = 1.0
    output_format: str = "m4b"
    use_gpu: bool = False
    silence_between_chapters: float = 2.0
    # Language override for phonemization (e.g., 'de', 'en-us', 'fr')
    # If None, language from PhonemeSegments is used
    lang: str | None = None
    # Pause settings (pykokoro built-in pause handling)
    pause_clause: float = 0.25  # For clause boundaries (commas)
    pause_sentence: float = 0.2  # For sentence boundaries
    pause_paragraph: float = 0.75  # For paragraph boundaries
    pause_variance: float = 0.05  # Standard deviation for natural variation
    pause_mode: str = "auto"  # "tts", "manual", or "auto"
    # Chapter announcement settings
    announce_chapters: bool = True  # Read chapter titles aloud before content
    chapter_pause_after_title: float = 2.0  # Pause after chapter title (seconds)
    # Metadata for m4b
    title: str | None = None
    author: str | None = None
    cover_image: Path | None = None
    # Voice blending (e.g., "af_nicole:50,am_michael:50")
    voice_blend: str | None = None
    # Voice database for custom/synthetic voices
    voice_database: Path | None = None
    # Chapter selection (e.g., "1-5" or "3,5,7") - 1-based
    chapters: str | None = None
    # Resume capability
    resume: bool = True
    # Keep chapter files after merge
    keep_chapter_files: bool = False
    # Filename template for chapter files
    chapter_filename_template: str = "{chapter_num:03d}_{book_title}_{chapter_title}"
    # Custom ONNX model path (None = use default downloaded model)
    model_path: Path | None = None
    # Custom voices.bin path (None = use default downloaded voices)
    voices_path: Path | None = None


class PhonemeConverter:
    """Converts PhonemeBook to audio using pre-tokenized phonemes/tokens."""

    def __init__(
        self,
        book: PhonemeBook,
        options: PhonemeConversionOptions,
        progress_callback: Callable[[PhonemeConversionProgress], None] | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """
        Initialize the phoneme converter.

        Args:
            book: PhonemeBook to convert
            options: Conversion options
            progress_callback: Called with progress updates
            log_callback: Called with log messages (message, level)
        """
        self.book = book
        self.options = options
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self._cancel_event = threading.Event()
        self._runner: KokoroRunner | None = None
        self._merger = AudioMerger(log=self.log)

    @property
    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        if self.log_callback:
            self.log_callback(message, level)

    def cancel(self) -> None:
        """Request cancellation of the conversion."""
        self._cancel_event.set()

    def _phoneme_segments_to_ssmd(self, segments: list[PhonemeSegment]) -> str:
        """Build SSMD text from phoneme segments."""
        parts: list[str] = []
        for idx, segment in enumerate(segments):
            phonemes = segment.phonemes.strip()
            if not phonemes:
                continue
            parts.append(phonemes)
            if idx >= len(segments) - 1:
                continue
            next_segment = segments[idx + 1]
            strength = "p" if next_segment.paragraph != segment.paragraph else "s"
            parts.append(f"...{strength}")
            parts.append("\n" if strength == "p" else " ")
        return "".join(parts).strip()

    def _generate_silence(self, duration: float) -> np.ndarray:
        """Generate silence audio of given duration."""
        samples = int(duration * SAMPLE_RATE)
        return np.zeros(samples, dtype="float32")

    def _setup_output(
        self, output_path: Path
    ) -> tuple[sf.SoundFile | None, subprocess.Popen[bytes] | None]:
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
        ffmpeg = get_ffmpeg_path()

        cmd = [
            ffmpeg,
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

        ffmpeg_proc = cast(
            subprocess.Popen[bytes],
            create_process(
                cmd, stdin=subprocess.PIPE, text=False, suppress_output=True
            ),
        )
        return None, ffmpeg_proc

    def _finalize_output(
        self,
        out_file: sf.SoundFile | None,
        ffmpeg_proc: subprocess.Popen[bytes] | None,
    ) -> None:
        """Finalize and close output file/process."""
        if out_file is not None:
            out_file.close()
        elif ffmpeg_proc is not None:
            if ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

    def _write_audio_chunk(
        self,
        audio: np.ndarray,
        out_file: sf.SoundFile | None,
        ffmpeg_proc: subprocess.Popen[bytes] | None,
    ) -> None:
        """Write audio chunk to file or ffmpeg process."""
        if out_file is not None:
            out_file.write(audio)
        elif ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
            audio_bytes = audio.astype("float32").tobytes()
            ffmpeg_proc.stdin.write(audio_bytes)

    def _convert_chapter_to_wav(
        self,
        chapter: PhonemeChapter,
        output_file: Path,
        progress: PhonemeConversionProgress | None = None,
        start_time: float | None = None,
        segments_before: int = 0,
    ) -> tuple[float, int]:
        """
        Convert a single chapter to a WAV file.

        Args:
            chapter: PhonemeChapter to convert
            output_file: Output WAV file path
            progress: Optional progress object to update
            start_time: Conversion start time for ETA calculation
            segments_before: Segments processed before this chapter

        Returns:
            Tuple of (duration in seconds, segments processed)
        """
        segments_processed = 0
        total_segments = len(chapter.segments)
        assert self._runner is not None
        lang_code = (
            get_onnx_lang_code(self.options.lang)
            if self.options.lang
            else (chapter.segments[0].lang if chapter.segments else "en-us")
        )

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
            # Only announce if there are segments to follow
            if self.options.announce_chapters and chapter.title and chapter.segments:
                title_samples = self._runner.synthesize(
                    chapter.title,
                    lang_code=lang_code,
                    pause_mode="tts",
                    is_phonemes=False,
                )
                out_file.write(title_samples)
                duration += len(title_samples) / SAMPLE_RATE

                # Add pause after chapter title
                pause_duration = self.options.chapter_pause_after_title
                if pause_duration > 0:
                    pause_samples = int(pause_duration * SAMPLE_RATE)
                    pause_audio = np.zeros(pause_samples, dtype=np.float32)
                    out_file.write(pause_audio)
                    duration += pause_duration

            if not self._cancel_event.is_set() and chapter.segments:
                # Single pipeline call for entire chapter
                ssmd_text = self._phoneme_segments_to_ssmd(chapter.segments)
                samples = self._runner.synthesize(
                    ssmd_text,
                    lang_code=lang_code,
                    pause_mode=cast(
                        Literal["tts", "manual", "auto"], self.options.pause_mode
                    ),
                    is_phonemes=True,
                )

                out_file.write(samples)
                duration += len(samples) / SAMPLE_RATE
                segments_processed = total_segments

                # Update progress once per chapter
                if progress and self.progress_callback:
                    progress.current_segment = segments_processed
                    progress.segments_processed = segments_before + segments_processed
                    ch_title = chapter.title or "chapter"
                    progress.current_text = (
                        f"Completed {ch_title} ({segments_processed} segments)"
                    )
                    if start_time and progress.total_segments_all > 0:
                        elapsed = time.time() - start_time
                        if progress.segments_processed > 0 and elapsed > 0.5:
                            avg_time = elapsed / progress.segments_processed
                            remaining = (
                                progress.total_segments_all
                                - progress.segments_processed
                            )
                            progress.estimated_remaining = avg_time * remaining
                        progress.elapsed_time = elapsed
                    self.progress_callback(progress)

        return duration, segments_processed

    def _get_selected_chapters(self) -> list[PhonemeChapter]:
        """Get chapters based on selection option."""
        if not self.options.chapters:
            return list(self.book.chapters)

        indices = parse_chapter_selection(
            self.options.chapters, len(self.book.chapters)
        )
        return [self.book.chapters[i] for i in indices]

    def _get_selected_indices(self) -> list[int]:
        """Get 0-based chapter indices based on selection option."""
        if not self.options.chapters:
            return list(range(len(self.book.chapters)))

        return parse_chapter_selection(self.options.chapters, len(self.book.chapters))

    def convert(self, output_path: Path) -> PhonemeConversionResult:
        """
        Convert PhonemeBook to audio with resume capability.

        Each chapter is saved as a separate WAV file, allowing conversion
        to be resumed if interrupted. A state file tracks progress.

        Args:
            output_path: Output file path

        Returns:
            PhonemeConversionResult with success status and paths
        """
        selected_chapters = self._get_selected_chapters()
        selected_indices = self._get_selected_indices()

        if not selected_chapters:
            return PhonemeConversionResult(
                success=False, error_message="No chapters to convert"
            )

        if self.options.output_format not in SUPPORTED_OUTPUT_FORMATS:
            return PhonemeConversionResult(
                success=False,
                error_message=f"Unsupported format: {self.options.output_format}",
            )

        self._cancel_event.clear()
        prevent_sleep_start()

        try:
            # Set up work directory for chapter files (use book title)
            safe_book_title = sanitize_filename(
                self.options.title or self.book.title or output_path.stem
            )[:50]
            work_dir = output_path.parent / f".{safe_book_title}_chapters"
            work_dir.mkdir(parents=True, exist_ok=True)
            state_file = work_dir / f"{safe_book_title}_state.json"

            # Load or create state
            state: PhonemeConversionState | None = None
            if self.options.resume and state_file.exists():
                state = PhonemeConversionState.load(state_file)
                if state:
                    # Check if selected chapters match
                    if state.selected_chapters != selected_indices:
                        self.log(
                            "Chapter selection changed, starting fresh conversion",
                            "warning",
                        )
                        state = None
                    # Check if settings differ from saved state
                    elif (
                        state.voice != self.options.voice
                        or state.speed != self.options.speed
                        or state.silence_between_chapters
                        != self.options.silence_between_chapters
                        or state.pause_clause != self.options.pause_clause
                        or state.pause_sentence != self.options.pause_sentence
                        or state.pause_paragraph != self.options.pause_paragraph
                        or state.pause_variance != self.options.pause_variance
                        or state.pause_mode != self.options.pause_mode
                    ):
                        self.log(
                            f"Restoring settings from previous session: "
                            f"voice={state.voice}, speed={state.speed}, "
                            f"silence={state.silence_between_chapters}s, "
                            f"pause_clause={state.pause_clause}s, "
                            f"pause_sentence={state.pause_sentence}s, "
                            f"pause_paragraph={state.pause_paragraph}s, "
                            f"pause_variance={state.pause_variance}s, "
                            f"pause_mode={state.pause_mode}",
                            "info",
                        )
                        # Apply saved settings for consistency
                        self.options.voice = state.voice
                        self.options.speed = state.speed
                        self.options.output_format = state.output_format
                        self.options.silence_between_chapters = (
                            state.silence_between_chapters
                        )
                        self.options.pause_clause = state.pause_clause
                        self.options.pause_sentence = state.pause_sentence
                        self.options.pause_paragraph = state.pause_paragraph
                        self.options.pause_variance = state.pause_variance
                        self.options.pause_mode = state.pause_mode

            if state is None:
                # Create new state
                state = PhonemeConversionState(
                    source_file=str(self.book.title),
                    output_file=str(output_path),
                    work_dir=str(work_dir),
                    voice=self.options.voice,
                    speed=self.options.speed,
                    output_format=self.options.output_format,
                    silence_between_chapters=self.options.silence_between_chapters,
                    pause_clause=self.options.pause_clause,
                    pause_sentence=self.options.pause_sentence,
                    pause_paragraph=self.options.pause_paragraph,
                    pause_variance=self.options.pause_variance,
                    pause_mode=self.options.pause_mode,
                    chapters=[
                        PhonemeChapterState(
                            index=idx,
                            title=self.book.chapters[idx].title,
                            segment_count=len(self.book.chapters[idx].segments),
                        )
                        for idx in selected_indices
                    ],
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    selected_chapters=selected_indices,
                )
                state.save(state_file)
            else:
                completed = state.get_completed_count()
                total = len(selected_chapters)
                self.log(f"Resuming conversion: {completed}/{total} chapters completed")

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
            )
            self._runner = KokoroRunner(opts, log=self.log)
            self._runner.ensure_ready()

            total_segments = sum(len(ch.segments) for ch in selected_chapters)
            # Account for already completed chapters
            segments_already_done = sum(
                state.chapters[i].segment_count
                for i in range(len(state.chapters))
                if state.chapters[i].completed
            )
            segments_processed = segments_already_done
            start_time = time.time()

            progress = PhonemeConversionProgress(
                total_chapters=len(selected_chapters),
                total_segments_all=total_segments,
                segments_processed=segments_processed,
            )

            # Convert each chapter
            for state_idx, chapter_state in enumerate(state.chapters):
                if self._cancel_event.is_set():
                    state.save(state_file)
                    return PhonemeConversionResult(
                        success=False,
                        error_message="Cancelled",
                        chapters_dir=work_dir,
                    )

                chapter_idx = chapter_state.index
                chapter = self.book.chapters[chapter_idx]

                # Skip already completed chapters
                if chapter_state.completed and chapter_state.audio_file:
                    chapter_file = work_dir / chapter_state.audio_file
                    if chapter_file.exists():
                        ch_num = state_idx + 1
                        self.log(
                            f"Skipping completed chapter {ch_num}: {chapter.title}"
                        )
                        continue
                    else:
                        # File missing, need to reconvert
                        chapter_state.completed = False

                progress.current_chapter = state_idx + 1
                progress.chapter_name = chapter.title
                progress.total_segments = len(chapter.segments)
                progress.current_segment = 0

                ch_num = state_idx + 1
                total_ch = len(state.chapters)
                self.log(f"Converting chapter {ch_num}/{total_ch}: {chapter.title}")

                # Generate chapter filename using template
                chapter_filename = (
                    format_filename_template(
                        self.options.chapter_filename_template,
                        book_title=self.options.title or self.book.title or "Untitled",
                        chapter_title=chapter.title,
                        chapter_num=state_idx + 1,
                    )
                    + ".wav"
                )
                chapter_file = work_dir / chapter_filename

                # Convert chapter to WAV
                duration, segs_done = self._convert_chapter_to_wav(
                    chapter,
                    chapter_file,
                    progress=progress,
                    start_time=start_time,
                    segments_before=segments_processed,
                )

                if self._cancel_event.is_set():
                    # Remove incomplete file
                    chapter_file.unlink(missing_ok=True)
                    state.save(state_file)
                    return PhonemeConversionResult(
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
                segments_processed += segs_done
                progress.segments_processed = segments_processed
                elapsed = time.time() - start_time
                if segments_processed > segments_already_done and elapsed > 0.5:
                    segs_in_session = segments_processed - segments_already_done
                    avg_time = elapsed / segs_in_session
                    remaining = total_segments - segments_processed
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

            total_duration = sum(chapter_durations)
            self.log(
                f"Conversion complete! Duration: {format_duration(total_duration)}"
            )

            # Clean up work directory if not keeping chapter files
            if not self.options.keep_chapter_files:
                for f in work_dir.iterdir():
                    f.unlink()
                work_dir.rmdir()
                work_dir = None  # type: ignore

            return PhonemeConversionResult(
                success=True,
                output_path=output_path,
                chapters_dir=work_dir,
                duration=total_duration,
            )

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return PhonemeConversionResult(success=False, error_message=error_msg)
        finally:
            prevent_sleep_end()

    def convert_streaming(self, output_path: Path) -> PhonemeConversionResult:
        """
        Convert PhonemeBook to audio in streaming mode.

        Audio is written directly to the output file/process without
        intermediate chapter files. This is faster but doesn't support
        resume capability.

        Args:
            output_path: Output file path

        Returns:
            PhonemeConversionResult with success status and paths
        """
        selected_chapters = self._get_selected_chapters()

        if not selected_chapters:
            return PhonemeConversionResult(
                success=False, error_message="No chapters to convert"
            )

        if self.options.output_format not in SUPPORTED_OUTPUT_FORMATS:
            return PhonemeConversionResult(
                success=False,
                error_message=f"Unsupported format: {self.options.output_format}",
            )

        self._cancel_event.clear()
        prevent_sleep_start()

        try:
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
            )
            self._runner = KokoroRunner(opts, log=self.log)
            self._runner.ensure_ready()

            total_segments = sum(len(ch.segments) for ch in selected_chapters)
            segments_processed = 0
            start_time = time.time()
            current_time = 0.0
            chapter_times: list[dict[str, Any]] = []

            progress = PhonemeConversionProgress(
                total_chapters=len(selected_chapters),
                total_segments_all=total_segments,
            )

            # Set up output
            out_file, ffmpeg_proc = self._setup_output(output_path)

            for chapter_idx, chapter in enumerate(selected_chapters):
                if self._cancel_event.is_set():
                    break

                progress.current_chapter = chapter_idx + 1
                progress.chapter_name = chapter.title
                progress.total_segments = len(chapter.segments)
                progress.current_segment = 0

                ch_num = chapter_idx + 1
                total_ch = len(selected_chapters)
                self.log(f"Converting chapter {ch_num}/{total_ch}: {chapter.title}")

                chapter_start = current_time

                total_chapter_segments = len(chapter.segments)
                if not self._cancel_event.is_set() and chapter.segments:
                    assert self._runner is not None
                    lang_code = (
                        get_onnx_lang_code(self.options.lang)
                        if self.options.lang
                        else (chapter.segments[0].lang if chapter.segments else "en-us")
                    )
                    ssmd_text = self._phoneme_segments_to_ssmd(chapter.segments)
                    samples = self._runner.synthesize(
                        ssmd_text,
                        lang_code=lang_code,
                        pause_mode=cast(
                            Literal["tts", "manual", "auto"],
                            self.options.pause_mode,
                        ),
                        is_phonemes=True,
                    )

                    self._write_audio_chunk(samples, out_file, ffmpeg_proc)
                    current_time += len(samples) / SAMPLE_RATE
                    segments_processed += total_chapter_segments

                    # Update progress once per chapter
                    progress.current_segment = total_chapter_segments
                    progress.segments_processed = segments_processed
                    progress.current_text = (
                        f"Completed {chapter.title} ({total_chapter_segments} segments)"
                    )
                    if segments_processed > 0:
                        elapsed = time.time() - start_time
                        if elapsed > 0.5:
                            avg_time = elapsed / segments_processed
                            remaining = total_segments - segments_processed
                            progress.estimated_remaining = avg_time * remaining
                        progress.elapsed_time = elapsed

                    if self.progress_callback:
                        self.progress_callback(progress)

                # Record chapter timing
                chapter_times.append(
                    {
                        "title": chapter.title,
                        "start": chapter_start,
                        "end": current_time,
                    }
                )

                # Add silence between chapters
                if (
                    chapter_idx < len(selected_chapters) - 1
                    and self.options.silence_between_chapters > 0
                ):
                    silence = self._generate_silence(
                        self.options.silence_between_chapters
                    )
                    self._write_audio_chunk(silence, out_file, ffmpeg_proc)
                    current_time += self.options.silence_between_chapters

            # Finalize output
            self._finalize_output(out_file, ffmpeg_proc)

            if self._cancel_event.is_set():
                # Clean up partial file
                output_path.unlink(missing_ok=True)
                return PhonemeConversionResult(success=False, error_message="Cancelled")

            # Add chapter markers for m4b
            if self.options.output_format == "m4b" and len(chapter_times) > 1:
                self._merger.add_chapters_to_m4b(
                    output_path,
                    chapter_times,
                    self.options.cover_image,
                )

            self.log(f"Conversion complete! Duration: {format_duration(current_time)}")

            return PhonemeConversionResult(
                success=True,
                output_path=output_path,
                duration=current_time,
            )

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return PhonemeConversionResult(success=False, error_message=error_msg)
        finally:
            prevent_sleep_end()
