"""Audio streaming player for ttsforge using sounddevice.

This module provides a continuous audio streaming player that can accept
audio chunks and play them seamlessly without gaps.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .utils import atomic_write_json

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

# Default sample rate for Kokoro models
DEFAULT_SAMPLE_RATE = 24000


@dataclass
class PlaybackPosition:
    """Represents the current playback position for resume functionality."""

    file_path: str
    chapter_index: int
    segment_index: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "chapter_index": self.chapter_index,
            "segment_index": self.segment_index,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaybackPosition:
        """Create from dictionary."""
        return cls(
            file_path=data["file_path"],
            chapter_index=data["chapter_index"],
            segment_index=data["segment_index"],
            timestamp=data.get("timestamp", time.time()),
        )


class StreamingAudioPlayer:
    """
    A continuous audio streaming player using sounddevice.

    This player accepts audio chunks and plays them seamlessly using a
    callback-based OutputStream. It handles buffering to prevent gaps
    between chunks and supports pause/resume/stop functionality.

    Example:
        player = StreamingAudioPlayer(sample_rate=24000)
        player.start()

        for audio_chunk in audio_generator:
            player.add_audio(audio_chunk)
            if player.should_stop:
                break

        player.wait_until_done()
        player.stop()
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        buffer_size: int = 2048,
        on_chunk_played: Callable[[int], None] | None = None,
        max_buffer_seconds: float = 10.0,
    ):
        """
        Initialize the streaming audio player.

        Args:
            sample_rate: Audio sample rate (default: 24000 for Kokoro)
            channels: Number of audio channels (default: 1 for mono)
            buffer_size: Size of audio buffer frames (default: 2048)
            on_chunk_played: Optional callback when a chunk finishes playing
            max_buffer_seconds: Max queued audio in seconds before blocking
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.on_chunk_played = on_chunk_played
        self.max_buffer_seconds = max_buffer_seconds

        max_samples = int(max_buffer_seconds * sample_rate)
        self._max_buffer_samples = max(max_samples, buffer_size * 2)
        self._max_queue_chunks = max(1, int(self._max_buffer_samples / buffer_size))

        # Audio queue for buffering chunks
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=self._max_queue_chunks
        )
        self._queue_lock = threading.Lock()
        self._queue_not_full = threading.Condition(self._queue_lock)
        self._queued_samples = 0

        # Current audio buffer being played
        self._current_buffer: np.ndarray | None = None
        self._buffer_position: int = 0

        # Control flags
        self._stream = None
        self._is_playing = False
        self._is_paused = False
        self._should_stop = False
        self._finished = threading.Event()
        self._all_audio_added = threading.Event()

        # Statistics
        self._chunks_played = 0
        self._total_samples_played = 0

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
        return self._is_playing and not self._is_paused

    @property
    def is_paused(self) -> bool:
        """Whether playback is paused."""
        return self._is_paused

    @property
    def should_stop(self) -> bool:
        """Whether playback should stop (e.g., user pressed Ctrl+C)."""
        return self._should_stop

    @property
    def chunks_played(self) -> int:
        """Number of audio chunks that have been played."""
        return self._chunks_played

    @property
    def duration_played(self) -> float:
        """Total duration of audio played in seconds."""
        return self._total_samples_played / self.sample_rate

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """
        Callback function called by sounddevice to fill the output buffer.

        This runs in a separate thread and must be fast to avoid audio glitches.
        """
        if status:
            # Log any stream errors (underflow, overflow)
            pass

        if self._should_stop or self._is_paused:
            # Fill with silence when stopped or paused
            outdata.fill(0)
            return

        output_pos = 0
        while output_pos < frames:
            # If we have no current buffer, try to get one from the queue
            if self._current_buffer is None or self._buffer_position >= len(
                self._current_buffer
            ):
                try:
                    self._current_buffer = self._audio_queue.get_nowait()
                    self._buffer_position = 0

                    if self._current_buffer is not None:
                        with self._queue_not_full:
                            self._queued_samples = max(
                                0, self._queued_samples - len(self._current_buffer)
                            )
                            self._queue_not_full.notify_all()

                    if self._current_buffer is None:
                        # None signals end of audio
                        outdata[output_pos:].fill(0)
                        self._finished.set()
                        return

                    self._chunks_played += 1
                    if self.on_chunk_played:
                        self.on_chunk_played(self._chunks_played)

                except queue.Empty:
                    # No audio available, fill with silence
                    outdata[output_pos:].fill(0)

                    # Check if we're done
                    if self._all_audio_added.is_set():
                        self._finished.set()
                    return

            # Copy audio from buffer to output
            available = len(self._current_buffer) - self._buffer_position
            needed = frames - output_pos
            to_copy = min(available, needed)

            # Handle mono/stereo conversion if needed
            audio_slice = self._current_buffer[
                self._buffer_position : self._buffer_position + to_copy
            ]

            if self.channels == 1:
                outdata[output_pos : output_pos + to_copy, 0] = audio_slice
            else:
                # Duplicate mono to all channels
                for ch in range(self.channels):
                    outdata[output_pos : output_pos + to_copy, ch] = audio_slice

            self._buffer_position += to_copy
            output_pos += to_copy
            self._total_samples_played += to_copy

    def start(self) -> None:
        """Start the audio output stream."""
        import sounddevice as sd  # type: ignore[import-untyped]

        if self._stream is not None:
            return

        self._should_stop = False
        self._finished.clear()
        self._all_audio_added.clear()
        self._is_playing = True

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.buffer_size,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop playback and close the stream."""
        self._should_stop = True
        self._is_playing = False
        self._current_buffer = None
        self._buffer_position = 0

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        with self._queue_not_full:
            self._queued_samples = 0
            self._queue_not_full.notify_all()

        self._all_audio_added.set()
        self._finished.set()

    def pause(self) -> None:
        """Pause playback."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume playback."""
        self._is_paused = False

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new pause state."""
        self._is_paused = not self._is_paused
        return self._is_paused

    def add_audio(self, audio: np.ndarray) -> None:
        """
        Add an audio chunk to the playback queue.

        Args:
            audio: Audio samples as numpy array (float32)
        """
        if self._should_stop:
            return

        # Ensure float32 format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Flatten if needed (handle potential 2D arrays)
        if audio.ndim > 1:
            audio = audio.flatten()

        audio_len = len(audio)
        with self._queue_not_full:
            while (
                not self._should_stop
                and self._queued_samples + audio_len > self._max_buffer_samples
            ):
                self._queue_not_full.wait(timeout=0.1)

            if self._should_stop:
                return

            self._queued_samples += audio_len

        while True:
            try:
                self._audio_queue.put(audio, timeout=0.1)
                break
            except queue.Full:
                if self._should_stop:
                    with self._queue_not_full:
                        self._queued_samples = max(0, self._queued_samples - audio_len)
                        self._queue_not_full.notify_all()
                    return

    def finish_adding(self) -> None:
        """Signal that no more audio will be added."""
        self._all_audio_added.set()
        while True:
            try:
                self._audio_queue.put(None, timeout=0.1)
                break
            except queue.Full:
                if self._should_stop:
                    break

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """
        Wait until all audio has been played.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if finished, False if timeout occurred
        """
        return self._finished.wait(timeout=timeout)

    def request_stop(self) -> None:
        """Request playback to stop (used for Ctrl+C handling)."""
        self._should_stop = True
        self._all_audio_added.set()
        self._finished.set()
        with self._queue_not_full:
            self._queue_not_full.notify_all()


def save_playback_position(
    position: PlaybackPosition, cache_dir: Path | None = None
) -> None:
    """
    Save the current playback position for resume functionality.

    Args:
        position: PlaybackPosition to save
        cache_dir: Directory to save to (default: ~/.cache/ttsforge)
    """
    from .utils import get_user_cache_path

    if cache_dir is None:
        cache_dir = get_user_cache_path()

    position_file = cache_dir / "reading_position.json"
    position_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        atomic_write_json(
            position_file, position.to_dict(), indent=2, ensure_ascii=True
        )
    except (OSError, TypeError, ValueError) as exc:
        _LOGGER.debug("Failed to save playback position: %s", exc)


def load_playback_position(
    cache_dir: Path | None = None,
) -> PlaybackPosition | None:
    """
    Load the saved playback position.

    Args:
        cache_dir: Directory to load from (default: ~/.cache/ttsforge)

    Returns:
        PlaybackPosition if found, None otherwise
    """
    import json

    from .utils import get_user_cache_path

    if cache_dir is None:
        cache_dir = get_user_cache_path()

    position_file = cache_dir / "reading_position.json"

    if not position_file.exists():
        return None

    try:
        with open(position_file, encoding="utf-8") as f:
            data = json.load(f)
        return PlaybackPosition.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def clear_playback_position(cache_dir: Path | None = None) -> None:
    """
    Clear the saved playback position.

    Args:
        cache_dir: Directory containing the position file
    """
    from .utils import get_user_cache_path

    if cache_dir is None:
        cache_dir = get_user_cache_path()

    position_file = cache_dir / "reading_position.json"

    if position_file.exists():
        position_file.unlink()


def play_audio_blocking(
    audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    """
    Play audio and block until finished.

    Simple utility for one-shot audio playback.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate (default: 24000)
    """
    import sounddevice as sd  # type: ignore[import-untyped]

    sd.play(audio, sample_rate)
    sd.wait()
