"""ONNX backend for ttsforge - uses kokoro-onnx for TTS without torch dependency."""

import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Union
from urllib.request import urlretrieve

import numpy as np

from .tokenizer import SAMPLE_RATE, EspeakConfig, Tokenizer
from .trim import trim
from .utils import get_user_cache_path

if TYPE_CHECKING:
    from .phonemes import PhonemeSegment

# ONNX model files required for kokoro-onnx
ONNX_MODEL_FILES = [
    "kokoro-v1.0.onnx",
    "voices-v1.0.bin",
]

# URLs for model files
MODEL_BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)


@dataclass
class VoiceBlend:
    """Represents a blend of multiple voices."""

    voices: list[tuple[str, float]]  # List of (voice_name, weight) tuples

    @classmethod
    def parse(cls, blend_str: str) -> "VoiceBlend":
        """
        Parse a voice blend string.

        Format: "voice1:weight1,voice2:weight2" or "voice1:50,voice2:50"
        Weights should sum to 100 (percentages).

        Example: "af_nicole:50,am_michael:50"
        """
        voices = []
        for part in blend_str.split(","):
            part = part.strip()
            if ":" in part:
                voice_name, weight_str = part.split(":", 1)
                weight = float(weight_str) / 100.0  # Convert percentage to fraction
            else:
                voice_name = part
                weight = 1.0
            voices.append((voice_name.strip(), weight))

        # Normalize weights if they don't sum to 1
        total_weight = sum(w for _, w in voices)
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            voices = [(v, w / total_weight) for v, w in voices]

        return cls(voices=voices)


def get_model_dir() -> Path:
    """Get the directory for storing ONNX model files."""
    return get_user_cache_path("models")


def get_model_path(filename: str) -> Path:
    """Get the full path to a model file."""
    return get_model_dir() / filename


def is_model_downloaded(filename: str) -> bool:
    """Check if a model file is already downloaded."""
    model_path = get_model_path(filename)
    return model_path.exists() and model_path.stat().st_size > 0


def are_models_downloaded() -> bool:
    """Check if all required model files are downloaded."""
    return all(is_model_downloaded(f) for f in ONNX_MODEL_FILES)


def download_model(
    filename: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    force: bool = False,
) -> Path:
    """
    Download a model file if not already present.

    Args:
        filename: Name of the model file to download
        progress_callback: Optional callback for progress updates (current, total)
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    model_path = get_model_path(filename)

    if model_path.exists() and not force:
        return model_path

    url = f"{MODEL_BASE_URL}/{filename}"
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download with progress
    temp_path = model_path.with_suffix(".tmp")

    def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
        if progress_callback and total_size > 0:
            downloaded = block_num * block_size
            progress_callback(min(downloaded, total_size), total_size)

    try:
        urlretrieve(url, temp_path, reporthook=_report_progress)
        temp_path.rename(model_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return model_path


def download_all_models(
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download all required model files.

    Args:
        progress_callback: Optional callback (filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Dict mapping filename to path
    """
    paths = {}
    for filename in ONNX_MODEL_FILES:
        file_progress: Optional[Callable[[int, int], None]] = None
        if progress_callback:

            def make_file_progress(
                fname: str,
            ) -> Callable[[int, int], None]:
                def inner(current: int, total: int) -> None:
                    progress_callback(fname, current, total)

                return inner

            file_progress = make_file_progress(filename)

        paths[filename] = download_model(filename, file_progress, force)

    return paths


class KokoroONNX:
    """
    Wrapper around kokoro-onnx for TTS generation.

    This class provides a consistent interface for the ttsforge conversion system.
    Now includes embedded tokenizer for phoneme/token-based generation.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        voices_path: Optional[Path] = None,
        use_gpu: bool = False,
        vocab_version: str = "v1.0",
        espeak_config: Optional[EspeakConfig] = None,
    ) -> None:
        """
        Initialize the Kokoro ONNX backend.

        Args:
            model_path: Path to the ONNX model file (auto-downloaded if None)
            voices_path: Path to the voices.bin file (auto-downloaded if None)
            use_gpu: Whether to use GPU acceleration (requires onnxruntime-gpu)
            vocab_version: Vocabulary version for tokenizer
            espeak_config: Optional espeak-ng configuration
        """
        self._kokoro: Any = None
        self._np = np
        self._use_gpu = use_gpu

        # Resolve paths
        if model_path is None:
            model_path = get_model_path(ONNX_MODEL_FILES[0])
        if voices_path is None:
            voices_path = get_model_path(ONNX_MODEL_FILES[1])

        self._model_path = model_path
        self._voices_path = voices_path

        # Voice database connection (for kokovoicelab integration)
        self._voice_db: Optional[sqlite3.Connection] = None

        # Tokenizer for phoneme-based generation
        self._tokenizer: Optional[Tokenizer] = None
        self._vocab_version = vocab_version
        self._espeak_config = espeak_config

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance (lazily initialized)."""
        if self._tokenizer is None:
            self._tokenizer = Tokenizer(
                espeak_config=self._espeak_config,
                vocab_version=self._vocab_version,
            )
        return self._tokenizer

    def _ensure_models(self) -> None:
        """Ensure model files are downloaded."""
        if not self._model_path.exists():
            download_model(ONNX_MODEL_FILES[0])
        if not self._voices_path.exists():
            download_model(ONNX_MODEL_FILES[1])

    def _init_kokoro(self) -> None:
        """Initialize the kokoro-onnx instance."""
        if self._kokoro is not None:
            return

        self._ensure_models()

        from kokoro_onnx import Kokoro

        # Set up execution providers based on GPU preference
        if self._use_gpu:
            try:
                # Try CUDA first, then CoreML for Mac
                self._kokoro = Kokoro(
                    str(self._model_path),
                    str(self._voices_path),
                )
            except Exception:
                # Fall back to CPU
                self._kokoro = Kokoro(
                    str(self._model_path),
                    str(self._voices_path),
                )
        else:
            self._kokoro = Kokoro(
                str(self._model_path),
                str(self._voices_path),
            )

    def get_voices(self) -> list[str]:
        """Get list of available voice names."""
        self._init_kokoro()
        return self._kokoro.get_voices()

    def get_voice_style(self, voice_name: str) -> np.ndarray:
        """
        Get the style vector for a voice.

        Args:
            voice_name: Name of the voice

        Returns:
            Numpy array representing the voice style
        """
        self._init_kokoro()
        return self._kokoro.get_voice_style(voice_name)

    def create_blended_voice(self, blend: VoiceBlend) -> np.ndarray:
        """
        Create a blended voice from multiple voices.

        Args:
            blend: VoiceBlend object specifying voices and weights

        Returns:
            Numpy array representing the blended voice style
        """
        self._init_kokoro()

        if len(blend.voices) == 1:
            voice_name, _ = blend.voices[0]
            return self.get_voice_style(voice_name)

        # Get style vectors and blend them
        blended: Optional[np.ndarray] = None
        for voice_name, weight in blend.voices:
            style = self.get_voice_style(voice_name)
            weighted = style * weight
            if blended is None:
                blended = weighted
            else:
                blended = np.add(blended, weighted)

        # This should never be None if blend.voices is not empty
        assert blended is not None, "No voices in blend"
        return blended

    def create(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text.

        Args:
            text: Text to synthesize
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code (e.g., 'en-us', 'en-gb', 'es', 'fr')

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Resolve voice to style vector if needed
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = voice  # kokoro-onnx accepts string voice names
        else:
            voice_style = voice  # Assume it's already a numpy array

        samples, sample_rate = self._kokoro.create(
            text,
            voice=voice_style,
            speed=speed,
            lang=lang,
        )

        return samples, sample_rate

    def create_from_phonemes(
        self,
        phonemes: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from phonemes directly.

        This bypasses text-to-phoneme conversion, useful when working
        with pre-tokenized content from PhonemeBook.

        Args:
            phonemes: IPA phoneme string
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Resolve voice to style vector if needed
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # kokoro-onnx supports direct phoneme input via create method with ps parameter
        # But we need to convert to tokens first
        tokens = self.tokenizer.tokenize(phonemes)
        return self.create_from_tokens(tokens, voice_style, speed)

    def create_from_tokens(
        self,
        tokens: list[int],
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from token IDs directly.

        This provides the lowest-level interface, useful for pre-tokenized
        content and maximum control.

        Args:
            tokens: List of token IDs
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Resolve voice to style vector if needed
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Use kokoro-onnx's create method with tokens parameter
        # Note: kokoro-onnx may require phonemes, in which case we detokenize
        phonemes = self.tokenizer.detokenize(tokens)
        samples, sample_rate = self._kokoro.create(
            phonemes,
            voice=voice_style,
            speed=speed,
            lang="en-us",  # Language doesn't matter for phonemes
            is_phonemes=True,  # Skip re-phonemization since input is already phonemes
        )

        return samples, sample_rate

    def create_from_segment(
        self,
        segment: "PhonemeSegment",
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from a PhonemeSegment.

        Args:
            segment: PhonemeSegment with phonemes and tokens
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        # Use tokens if available, otherwise use phonemes
        if segment.tokens:
            return self.create_from_tokens(segment.tokens, voice, speed)
        elif segment.phonemes:
            return self.create_from_phonemes(segment.phonemes, voice, speed)
        else:
            # Fall back to text
            return self.create(segment.text, voice, speed, segment.lang)

    def phonemize(self, text: str, lang: str = "en-us") -> str:
        """
        Convert text to phonemes.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Phoneme string
        """
        return self.tokenizer.phonemize(text, lang=lang)

    def tokenize(self, phonemes: str) -> list[int]:
        """
        Convert phonemes to tokens.

        Args:
            phonemes: Phoneme string

        Returns:
            List of token IDs
        """
        return self.tokenizer.tokenize(phonemes)

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert tokens back to phonemes.

        Args:
            tokens: List of token IDs

        Returns:
            Phoneme string
        """
        return self.tokenizer.detokenize(tokens)

    def text_to_tokens(self, text: str, lang: str = "en-us") -> list[int]:
        """
        Convert text directly to tokens.

        Args:
            text: Input text
            lang: Language code

        Returns:
            List of token IDs
        """
        return self.tokenizer.text_to_tokens(text, lang=lang)

    def generate_chunks(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
        chunk_size: int = 500,
    ):
        """
        Generate audio in chunks for long text.

        This splits text into manageable chunks and yields audio for each.
        Useful for progress tracking during long conversions.

        Args:
            text: Text to synthesize
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed
            lang: Language code
            chunk_size: Approximate character count per chunk

        Yields:
            Tuple of (audio samples, sample rate, text chunk)
        """
        self._init_kokoro()

        # Resolve voice once
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = voice
        else:
            voice_style = voice

        # Split text into chunks at sentence boundaries
        chunks = self._split_text(text, chunk_size)

        for chunk in chunks:
            if not chunk.strip():
                continue

            samples, sample_rate = self._kokoro.create(
                chunk,
                voice=voice_style,
                speed=speed,
                lang=lang,
            )

            yield samples, sample_rate, chunk

    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: Text to split
            chunk_size: Target chunk size in characters

        Returns:
            List of text chunks
        """
        # Split on sentence-ending punctuation
        import re

        # Split on sentence boundaries while keeping the delimiter
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # Voice Database Integration (from kokovoicelab)

    def load_voice_database(self, db_path: Path) -> None:
        """
        Load a voice database for custom/synthetic voices.

        Args:
            db_path: Path to the SQLite voice database
        """
        if self._voice_db is not None:
            self._voice_db.close()

        # Register numpy array converter
        sqlite3.register_converter("array", self._convert_array)
        self._voice_db = sqlite3.connect(
            str(db_path), detect_types=sqlite3.PARSE_DECLTYPES
        )

    def _convert_array(self, blob: bytes) -> np.ndarray:
        """Convert binary blob back to numpy array."""
        out = io.BytesIO(blob)
        return np.load(out)

    def get_voice_from_database(self, voice_name: str) -> Optional[np.ndarray]:
        """
        Get a voice style vector from the database.

        Args:
            voice_name: Name of the voice in the database

        Returns:
            Voice style vector or None if not found
        """
        if self._voice_db is None:
            return None

        cursor = self._voice_db.cursor()
        cursor.execute(
            "SELECT style_vector FROM voices WHERE name = ?",
            (voice_name,),
        )
        row = cursor.fetchone()

        if row:
            return row[0]
        return None

    def list_database_voices(self) -> list[dict[str, Any]]:
        """
        List all voices in the database.

        Returns:
            List of voice metadata dictionaries
        """
        if self._voice_db is None:
            return []

        cursor = self._voice_db.cursor()
        cursor.execute(
            """
            SELECT name, gender, language, quality, is_synthetic, notes
            FROM voices
            ORDER BY quality DESC
            """
        )

        voices = []
        for row in cursor.fetchall():
            voices.append(
                {
                    "name": row[0],
                    "gender": row[1],
                    "language": row[2],
                    "quality": row[3],
                    "is_synthetic": bool(row[4]),
                    "notes": row[5],
                }
            )

        return voices

    def interpolate_voices(
        self,
        voice1: str | np.ndarray,
        voice2: str | np.ndarray,
        factor: float = 0.5,
    ) -> np.ndarray:
        """
        Interpolate between two voices.

        This uses the interpolation method from kokovoicelab to create
        voices that lie on the line between two source voices.

        Args:
            voice1: First voice (name or style vector)
            voice2: Second voice (name or style vector)
            factor: Interpolation factor (0.0 = voice1, 1.0 = voice2)

        Returns:
            Interpolated voice style vector
        """
        self._init_kokoro()

        # Resolve to style vectors
        if isinstance(voice1, str):
            style1 = self.get_voice_style(voice1)
        else:
            style1 = voice1

        if isinstance(voice2, str):
            style2 = self.get_voice_style(voice2)
        else:
            style2 = voice2

        # Use kokovoicelab's interpolation method
        diff_vector = style2 - style1
        midpoint = (style1 + style2) / 2
        return midpoint + (diff_vector * factor / 2)

    def close(self) -> None:
        """Clean up resources."""
        if self._voice_db is not None:
            self._voice_db.close()
            self._voice_db = None


# Language code mapping for kokoro-onnx
LANG_CODE_TO_ONNX = {
    "a": "en-us",  # American English
    "b": "en-gb",  # British English
    "e": "es",  # Spanish
    "f": "fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "j": "ja",  # Japanese
    "p": "pt",  # Portuguese
    "z": "zh",  # Chinese
}


def get_onnx_lang_code(ttsforge_lang: str) -> str:
    """Convert ttsforge language code to kokoro-onnx language code."""
    return LANG_CODE_TO_ONNX.get(ttsforge_lang, "en-us")
