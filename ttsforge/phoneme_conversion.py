"""Phoneme-based TTS conversion module for ttsforge.

This module converts pre-tokenized PhonemeBook files to audio,
bypassing text-to-phoneme conversion since phonemes/tokens are pre-computed.
"""

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import soundfile as sf

from .constants import SAMPLE_RATE, SUPPORTED_OUTPUT_FORMATS
from .onnx_backend import (
    KokoroONNX,
    VoiceBlend,
    are_models_downloaded,
    download_all_models,
)
from .phonemes import PhonemeBook, PhonemeChapter
from .utils import (
    create_process,
    ensure_ffmpeg,
    format_duration,
    format_filename_template,
    prevent_sleep_end,
    prevent_sleep_start,
    sanitize_filename,
)


def parse_chapter_selection(selection: str, total_chapters: int) -> list[int]:
    """Parse chapter selection string into list of 0-based chapter indices.

    Supports formats like:
    - "3" -> [2] (single chapter, 1-based to 0-based)
    - "1-5" -> [0, 1, 2, 3, 4] (range, inclusive)
    - "3,5,7" -> [2, 4, 6] (comma-separated)
    - "1-3,7,9-10" -> [0, 1, 2, 6, 8, 9] (mixed)

    Args:
        selection: Chapter selection string (1-based indexing)
        total_chapters: Total number of chapters available

    Returns:
        List of 0-based chapter indices

    Raises:
        ValueError: If selection format is invalid or chapters out of range
    """
    indices: set[int] = set()

    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            # Range: "1-5"
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
            except ValueError as e:
                raise ValueError(f"Invalid range format: {part}") from e

            if start < 1 or end < 1:
                raise ValueError(f"Chapter numbers must be >= 1: {part}")
            if start > end:
                raise ValueError(f"Invalid range (start > end): {part}")
            if end > total_chapters:
                raise ValueError(
                    f"Chapter {end} exceeds total chapters ({total_chapters})"
                )

            # Convert to 0-based indices
            for i in range(start - 1, end):
                indices.add(i)
        else:
            # Single chapter: "3"
            try:
                chapter_num = int(part)
            except ValueError as e:
                raise ValueError(f"Invalid chapter number: {part}") from e

            if chapter_num < 1:
                raise ValueError(f"Chapter number must be >= 1: {chapter_num}")
            if chapter_num > total_chapters:
                raise ValueError(
                    f"Chapter {chapter_num} exceeds total chapters ({total_chapters})"
                )

            # Convert to 0-based index
            indices.add(chapter_num - 1)

    return sorted(indices)


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
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    chapters_dir: Optional[Path] = None
    duration: float = 0.0


@dataclass
class PhonemeChapterState:
    """State of a single chapter conversion."""

    index: int
    title: str
    segment_count: int
    completed: bool = False
    audio_file: Optional[str] = None  # Relative path to chapter audio
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
    segment_pause_min: float = 0.1
    segment_pause_max: float = 0.3
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
            if "segment_pause_min" not in data:
                data["segment_pause_min"] = 0.1
            if "segment_pause_max" not in data:
                data["segment_pause_max"] = 0.3

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
            "segment_pause_min": self.segment_pause_min,
            "segment_pause_max": self.segment_pause_max,
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
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


@dataclass
class PhonemeConversionOptions:
    """Options for phoneme-based TTS conversion."""

    voice: str = "af_heart"
    speed: float = 1.0
    output_format: str = "m4b"
    use_gpu: bool = False
    silence_between_chapters: float = 2.0
    # Segment pause (random silence between segments within a chapter)
    segment_pause_min: float = 0.1
    segment_pause_max: float = 0.3
    # Metadata for m4b
    title: Optional[str] = None
    author: Optional[str] = None
    cover_image: Optional[Path] = None
    # Voice blending (e.g., "af_nicole:50,am_michael:50")
    voice_blend: Optional[str] = None
    # Voice database for custom/synthetic voices
    voice_database: Optional[Path] = None
    # Chapter selection (e.g., "1-5" or "3,5,7") - 1-based
    chapters: Optional[str] = None
    # Resume capability
    resume: bool = True
    # Keep chapter files after merge
    keep_chapter_files: bool = False
    # Filename template for chapter files
    chapter_filename_template: str = "{chapter_num:03d}_{book_title}_{chapter_title}"


class PhonemeConverter:
    """Converts PhonemeBook to audio using pre-tokenized phonemes/tokens."""

    def __init__(
        self,
        book: PhonemeBook,
        options: PhonemeConversionOptions,
        progress_callback: Optional[Callable[[PhonemeConversionProgress], None]] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
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
        self._cancelled = False
        self._kokoro: Optional[KokoroONNX] = None
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
        self._kokoro = KokoroONNX(use_gpu=self.options.use_gpu)

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

    def _generate_silence(self, duration: float) -> np.ndarray:
        """Generate silence audio of given duration."""
        samples = int(duration * SAMPLE_RATE)
        return np.zeros(samples, dtype="float32")

    def _generate_segment_pause(self) -> np.ndarray:
        """Generate random silence for inter-segment pause."""
        min_pause = self.options.segment_pause_min
        max_pause = self.options.segment_pause_max
        if min_pause <= 0 and max_pause <= 0:
            return np.array([], dtype="float32")
        duration = random.uniform(min_pause, max_pause)
        return self._generate_silence(duration)

    def _setup_output(
        self, output_path: Path
    ) -> tuple[Optional[sf.SoundFile], Optional[subprocess.Popen[bytes]]]:
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
        out_file: Optional[sf.SoundFile],
        ffmpeg_proc: Optional[subprocess.Popen[bytes]],
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
        out_file: Optional[sf.SoundFile],
        ffmpeg_proc: Optional[subprocess.Popen[bytes]],
    ) -> None:
        """Write audio chunk to file or ffmpeg process."""
        if out_file is not None:
            out_file.write(audio)
        elif ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
            audio_bytes = audio.astype("float32").tobytes()
            ffmpeg_proc.stdin.write(audio_bytes)

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

    def _convert_chapter_to_wav(
        self,
        chapter: PhonemeChapter,
        output_file: Path,
        progress: Optional[PhonemeConversionProgress] = None,
        start_time: Optional[float] = None,
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

        # Open WAV file for writing
        with sf.SoundFile(
            str(output_file),
            "w",
            samplerate=SAMPLE_RATE,
            channels=1,
            format="wav",
        ) as out_file:
            duration = 0.0

            for seg_idx, segment in enumerate(chapter.segments):
                if self._cancelled:
                    break

                # Generate audio from segment
                assert self._kokoro is not None
                samples, sample_rate = self._kokoro.create_from_segment(
                    segment,
                    voice=self._voice_style or self.options.voice,
                    speed=self.options.speed,
                )

                if self._cancelled:
                    break

                out_file.write(samples)
                duration += len(samples) / SAMPLE_RATE
                segments_processed += 1

                # Add random pause between segments (not after the last segment)
                if seg_idx < total_segments - 1:
                    pause_audio = self._generate_segment_pause()
                    if len(pause_audio) > 0:
                        out_file.write(pause_audio)
                        duration += len(pause_audio) / SAMPLE_RATE

                # Update progress
                if progress and self.progress_callback:
                    progress.current_segment = segments_processed
                    progress.segments_processed = segments_before + segments_processed
                    progress.current_text = segment.text[:100]
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
            silence_audio = np.zeros(silence_samples, dtype="float32")
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

        self._cancelled = False
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
            state: Optional[PhonemeConversionState] = None
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
                        or state.segment_pause_min != self.options.segment_pause_min
                        or state.segment_pause_max != self.options.segment_pause_max
                    ):
                        self.log(
                            f"Restoring settings from previous session: "
                            f"voice={state.voice}, speed={state.speed}, "
                            f"silence={state.silence_between_chapters}s, "
                            f"segment_pause={state.segment_pause_min}-{state.segment_pause_max}s",
                            "info",
                        )
                        # Apply saved settings for consistency
                        self.options.voice = state.voice
                        self.options.speed = state.speed
                        self.options.output_format = state.output_format
                        self.options.silence_between_chapters = (
                            state.silence_between_chapters
                        )
                        self.options.segment_pause_min = state.segment_pause_min
                        self.options.segment_pause_max = state.segment_pause_max

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
                    segment_pause_min=self.options.segment_pause_min,
                    segment_pause_max=self.options.segment_pause_max,
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

            # Initialize pipeline
            self._init_pipeline()

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
                if self._cancelled:
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

                if self._cancelled:
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

            self._merge_chapter_files(
                chapter_files,
                chapter_durations,
                chapter_titles,
                output_path,
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
            return PhonemeConversionResult(success=False, error_message=str(e))
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

        self._cancelled = False
        prevent_sleep_start()

        try:
            self._init_pipeline()

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
                if self._cancelled:
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

                for seg_idx, segment in enumerate(chapter.segments):
                    if self._cancelled:
                        break

                    # Generate audio from segment
                    assert self._kokoro is not None
                    samples, sample_rate = self._kokoro.create_from_segment(
                        segment,
                        voice=self._voice_style or self.options.voice,
                        speed=self.options.speed,
                    )

                    if self._cancelled:
                        break

                    self._write_audio_chunk(samples, out_file, ffmpeg_proc)
                    current_time += len(samples) / SAMPLE_RATE
                    segments_processed += 1

                    # Add random pause between segments (not after the last segment)
                    if seg_idx < total_chapter_segments - 1:
                        pause_audio = self._generate_segment_pause()
                        if len(pause_audio) > 0:
                            self._write_audio_chunk(pause_audio, out_file, ffmpeg_proc)
                            current_time += len(pause_audio) / SAMPLE_RATE

                    # Update progress
                    progress.current_segment = seg_idx + 1
                    progress.segments_processed = segments_processed
                    progress.current_text = segment.text[:100]
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

            if self._cancelled:
                # Clean up partial file
                output_path.unlink(missing_ok=True)
                return PhonemeConversionResult(success=False, error_message="Cancelled")

            # Add chapter markers for m4b
            if self.options.output_format == "m4b" and len(chapter_times) > 1:
                self._add_chapters_to_m4b(output_path, chapter_times)

            self.log(f"Conversion complete! Duration: {format_duration(current_time)}")

            return PhonemeConversionResult(
                success=True,
                output_path=output_path,
                duration=current_time,
            )

        except Exception as e:
            return PhonemeConversionResult(success=False, error_message=str(e))
        finally:
            prevent_sleep_end()
