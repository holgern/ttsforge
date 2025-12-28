"""Tokenizer for ttsforge - converts text to phonemes and tokens.

This module provides text-to-phoneme and phoneme-to-token conversion using
espeak-ng as the phonemizer backend.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import espeakng_loader
import phonemizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper

try:
    from .vocab import DEFAULT_VERSION, load_vocab
except ImportError:
    from ttsforge.vocab import DEFAULT_VERSION, load_vocab

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000

# Supported languages for phonemization
# Format: language code -> espeak language code
SUPPORTED_LANGUAGES = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "en": "en-us",  # Default English to US
    "es": "es",
    "fr": "fr-fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "ko": "ko",
    "ja": "ja",
    "zh": "cmn",  # Mandarin Chinese
}


@dataclass
class EspeakConfig:
    """Configuration for espeak-ng backend.

    Attributes:
        lib_path: Path to the espeak-ng shared library
        data_path: Path to the espeak-ng data directory
    """

    lib_path: str | None = None
    data_path: str | None = None


class Tokenizer:
    """Text-to-phoneme-to-token converter using espeak-ng.

    This class handles:
    1. Text normalization
    2. Text to phoneme conversion (via espeak-ng/phonemizer)
    3. Phoneme to token conversion (via vocabulary)
    4. Token to phoneme conversion (reverse lookup)

    Args:
        espeak_config: Optional espeak-ng configuration
        vocab_version: Vocabulary version to use (default: "v1.0")
        vocab: Optional custom vocabulary dict (overrides vocab_version)

    Example:
        >>> tokenizer = Tokenizer()
        >>> phonemes = tokenizer.phonemize("Hello world")
        >>> tokens = tokenizer.tokenize(phonemes)
        >>> text_tokens = tokenizer.text_to_tokens("Hello world")
    """

    def __init__(
        self,
        espeak_config: EspeakConfig | None = None,
        vocab_version: str = DEFAULT_VERSION,
        vocab: dict[str, int] | None = None,
    ):
        """Initialize the tokenizer.

        Args:
            espeak_config: Optional espeak-ng configuration
            vocab_version: Vocabulary version to use
            vocab: Optional custom vocabulary (overrides vocab_version)
        """
        self.vocab_version = vocab_version
        self.vocab = vocab if vocab is not None else load_vocab(vocab_version)
        self._reverse_vocab: dict[int, str] | None = None

        # Initialize espeak-ng
        self._init_espeak(espeak_config)

    def _init_espeak(self, config: EspeakConfig | None) -> None:
        """Initialize espeak-ng library.

        Args:
            config: Optional espeak configuration
        """
        if config is None:
            config = EspeakConfig()

        # Use espeakng_loader defaults if not specified
        if not config.data_path:
            config.data_path = espeakng_loader.get_data_path()
        if not config.lib_path:
            config.lib_path = espeakng_loader.get_library_path()

        # Check environment override
        env_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
        if env_lib:
            config.lib_path = env_lib

        # Try to load the library
        try:
            if config.lib_path:
                ctypes.cdll.LoadLibrary(config.lib_path)
        except Exception as e:
            logger.warning(f"Failed to load espeak shared library: {e}")
            logger.info("Falling back to system-wide espeak-ng library")

            # Fallback to system library
            config.lib_path = ctypes.util.find_library(
                "espeak-ng"
            ) or ctypes.util.find_library("espeak")

            if not config.lib_path:
                error_info = (
                    "Failed to load espeak-ng. Please install espeak-ng system-wide.\n"
                    "\tSee https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md\n"
                    "\tNote: you can specify shared library path using PHONEMIZER_ESPEAK_LIBRARY environment variable.\n"
                    f"Environment:\n\t{platform.platform()} ({platform.release()}) | {sys.version}"
                )
                raise RuntimeError(error_info) from e

            try:
                ctypes.cdll.LoadLibrary(config.lib_path)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load espeak-ng from fallback: {e2}"
                ) from e2

        # Configure phonemizer backend
        if config.data_path:
            EspeakWrapper.set_data_path(config.data_path)
        if config.lib_path:
            EspeakWrapper.set_library(config.lib_path)

    @property
    def reverse_vocab(self) -> dict[int, str]:
        """Get the reverse vocabulary (token ID -> phoneme).

        Lazily constructed on first access.
        """
        if self._reverse_vocab is None:
            self._reverse_vocab = {v: k for k, v in self.vocab.items()}
        return self._reverse_vocab

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text before phonemization.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return text.strip()

    def phonemize(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> str:
        """Convert text to phonemes.

        Args:
            text: Input text
            lang: Language code (e.g., 'en-us', 'en-gb')
            normalize: Whether to normalize text first

        Returns:
            Phoneme string (IPA characters)

        Raises:
            ValueError: If language is not supported
        """
        if normalize:
            text = self.normalize_text(text)

        if not text:
            return ""

        # Map language to espeak code
        espeak_lang = SUPPORTED_LANGUAGES.get(lang, lang)

        # Use phonemizer library
        phonemes = phonemizer.phonemize(
            text,
            language=espeak_lang,
            preserve_punctuation=True,
            with_stress=True,
        )

        # Filter to only characters in vocabulary
        phonemes = "".join(c for c in phonemes if c in self.vocab)
        return phonemes.strip()

    def tokenize(self, phonemes: str) -> list[int]:
        """Convert phonemes to token IDs.

        Args:
            phonemes: Phoneme string (IPA characters)

        Returns:
            List of token IDs

        Raises:
            ValueError: If phoneme string exceeds MAX_PHONEME_LENGTH
        """
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"Phoneme string too long ({len(phonemes)} chars). "
                f"Maximum is {MAX_PHONEME_LENGTH} phonemes."
            )

        return [
            token_id for c in phonemes if (token_id := self.vocab.get(c)) is not None
        ]

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to phonemes.

        Args:
            tokens: List of token IDs

        Returns:
            Phoneme string
        """
        return "".join(
            phoneme
            for t in tokens
            if (phoneme := self.reverse_vocab.get(t)) is not None
        )

    def text_to_tokens(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> list[int]:
        """Convert text directly to tokens.

        Convenience method combining phonemize() and tokenize().

        Args:
            text: Input text
            lang: Language code
            normalize: Whether to normalize text first

        Returns:
            List of token IDs
        """
        phonemes = self.phonemize(text, lang=lang, normalize=normalize)
        return self.tokenize(phonemes)

    def text_to_phonemes_with_words(
        self,
        text: str,
        lang: str = "en-us",
    ) -> list[tuple[str, str]]:
        """Convert text to phonemes, preserving word boundaries.

        Useful for creating readable phoneme exports.

        Args:
            text: Input text
            lang: Language code

        Returns:
            List of (word, phonemes) tuples
        """
        words = text.split()
        result = []

        for word in words:
            phonemes = self.phonemize(word, lang=lang, normalize=True)
            result.append((word, phonemes))

        return result

    def format_readable(
        self,
        text: str,
        lang: str = "en-us",
    ) -> str:
        """Format text with phonemes in a human-readable way.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Formatted string like "Hello [həˈloʊ] world [wɜːld]"
        """
        word_phonemes = self.text_to_phonemes_with_words(text, lang=lang)
        return " ".join(f"{word} [{phonemes}]" for word, phonemes in word_phonemes)

    def get_vocab_info(self) -> dict:
        """Get information about the current vocabulary.

        Returns:
            Dictionary with vocabulary metadata
        """
        return {
            "version": self.vocab_version,
            "num_tokens": len(self.vocab),
            "max_token_id": max(self.vocab.values()) if self.vocab else 0,
            "max_phoneme_length": MAX_PHONEME_LENGTH,
        }
