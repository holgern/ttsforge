"""Tokenizer for ttsforge - converts text to phonemes and tokens.

This module provides text-to-phoneme and phoneme-to-token conversion using
kokorog2p (dictionary + espeak fallback) as the phonemizer backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kokorog2p import (
    N_TOKENS,
    GToken,
    filter_for_kokoro,
    get_g2p,
    get_kokoro_vocab,
    ids_to_phonemes,
    phonemes_to_ids,
    validate_for_kokoro,
)
from kokorog2p.base import G2PBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000

# Supported languages for phonemization
# Format: language code -> kokorog2p language code
SUPPORTED_LANGUAGES = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "en": "en-us",  # Default English to US
    # Note: kokorog2p currently only supports English
    # Other languages will fall back to espeak-only mode
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
class TokenizerConfig:
    """Configuration for the tokenizer.

    Attributes:
        use_espeak_fallback: Whether to use espeak for OOV words (default: True)
        use_spacy: Whether to use spaCy for POS tagging (default: True)
        use_dictionary: Whether to use dictionary lookup (default: True)
    """

    use_espeak_fallback: bool = True
    use_spacy: bool = True
    use_dictionary: bool = True


# Backward compatibility alias
@dataclass
class EspeakConfig:
    """Configuration for espeak-ng backend (deprecated, use TokenizerConfig).

    Kept for backward compatibility. The lib_path and data_path are now
    managed by kokorog2p internally.

    Attributes:
        lib_path: Path to the espeak-ng shared library (ignored)
        data_path: Path to the espeak-ng data directory (ignored)
    """

    lib_path: str | None = None
    data_path: str | None = None


@dataclass
class PhonemeResult:
    """Result of phonemization with quality metadata.

    Attributes:
        phonemes: The phoneme string
        tokens: List of GToken objects with per-word phonemes
        low_confidence_words: Words that used espeak fallback
    """

    phonemes: str
    tokens: list[GToken] = field(default_factory=list)
    low_confidence_words: list[str] = field(default_factory=list)


class Tokenizer:
    """Text-to-phoneme-to-token converter using kokorog2p.

    This class handles:
    1. Text normalization
    2. Text to phoneme conversion (via kokorog2p dictionary + espeak fallback)
    3. Phoneme to token conversion (via Kokoro vocabulary)
    4. Token to phoneme conversion (reverse lookup)

    Args:
        espeak_config: Deprecated, kept for backward compatibility
        vocab_version: Ignored (uses kokorog2p's embedded vocabulary)
        vocab: Optional custom vocabulary dict (overrides default)
        config: Optional TokenizerConfig for phonemization settings

    Example:
        >>> tokenizer = Tokenizer()
        >>> phonemes = tokenizer.phonemize("Hello world")
        >>> tokens = tokenizer.tokenize(phonemes)
        >>> text_tokens = tokenizer.text_to_tokens("Hello world")
    """

    def __init__(
        self,
        espeak_config: EspeakConfig | None = None,
        vocab_version: str = "v1.0",
        vocab: dict[str, int] | None = None,
        config: TokenizerConfig | None = None,
    ):
        """Initialize the tokenizer.

        Args:
            espeak_config: Deprecated, kept for backward compatibility
            vocab_version: Ignored (uses kokorog2p's embedded vocabulary)
            vocab: Optional custom vocabulary (overrides default)
            config: Optional TokenizerConfig for phonemization settings
        """
        self.vocab_version = vocab_version
        self.vocab = vocab if vocab is not None else get_kokoro_vocab()
        self._reverse_vocab: dict[int, str] | None = None
        self.config = config or TokenizerConfig()

        # G2P instances cache (lazy loaded per language)
        self._g2p_cache: dict[str, G2PBase] = {}

        # Log if espeak_config was provided (deprecated)
        if espeak_config is not None and (
            espeak_config.lib_path or espeak_config.data_path
        ):
            logger.warning(
                "EspeakConfig is deprecated. kokorog2p manages espeak internally."
            )

    def _get_g2p(self, lang: str) -> G2PBase:
        """Get or create a G2P instance for the given language.

        Args:
            lang: Language code (e.g., 'en-us', 'en-gb')

        Returns:
            G2P instance for the language
        """
        if lang not in self._g2p_cache:
            # Map language to kokorog2p format
            kokorog2p_lang = SUPPORTED_LANGUAGES.get(lang, lang)

            # Only English is fully supported by kokorog2p
            if kokorog2p_lang.startswith("en"):
                self._g2p_cache[lang] = get_g2p(
                    language=kokorog2p_lang,
                    use_espeak_fallback=self.config.use_espeak_fallback,
                    use_spacy=self.config.use_spacy,
                )
            else:
                # For non-English, use espeak-only mode
                logger.info(
                    f"Language {lang} not fully supported, using espeak fallback only"
                )
                # Create a minimal G2P that just uses espeak
                self._g2p_cache[lang] = self._create_espeak_only_g2p(kokorog2p_lang)

        return self._g2p_cache[lang]

    def _create_espeak_only_g2p(self, lang: str) -> G2PBase:
        """Create an espeak-only G2P for non-English languages.

        Args:
            lang: Language code

        Returns:
            G2P instance using espeak backend
        """
        from kokorog2p.backends.espeak import EspeakBackend

        class EspeakOnlyG2P(G2PBase):
            """Minimal G2P that only uses espeak."""

            def __init__(self, language: str):
                super().__init__(language=language, use_espeak_fallback=True)
                self._backend = EspeakBackend(language=language)

            def __call__(self, text: str) -> list[GToken]:
                if not text.strip():
                    return []
                phonemes = self._backend.phonemize(text)
                return [GToken(text=text, phonemes=phonemes)]

            def lookup(self, word: str, tag: str | None = None) -> str | None:
                return self._backend.phonemize(word)

        return EspeakOnlyG2P(language=lang)

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
            Phoneme string (Kokoro format)

        Raises:
            ValueError: If language is not supported
        """
        if normalize:
            text = self.normalize_text(text)

        if not text:
            return ""

        # Get G2P instance for language
        g2p = self._get_g2p(lang)

        # Convert text to phonemes using kokorog2p
        phonemes = g2p.phonemize(text)

        # Filter to only characters in vocabulary
        phonemes = filter_for_kokoro(phonemes)

        return phonemes.strip()

    def phonemize_detailed(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> PhonemeResult:
        """Convert text to phonemes with detailed token information.

        Args:
            text: Input text
            lang: Language code (e.g., 'en-us', 'en-gb')
            normalize: Whether to normalize text first

        Returns:
            PhonemeResult with phonemes, tokens, and quality metadata
        """
        if normalize:
            text = self.normalize_text(text)

        if not text:
            return PhonemeResult(phonemes="", tokens=[], low_confidence_words=[])

        # Get G2P instance for language
        g2p = self._get_g2p(lang)

        # Get tokens with per-word phonemes
        tokens = g2p(text)

        # Build phoneme string and identify low-confidence words
        phoneme_parts = []
        low_confidence = []

        for token in tokens:
            if token.phonemes:
                phoneme_parts.append(token.phonemes)
                # Check rating (1 = espeak fallback, 3-4 = dictionary)
                rating = token.get("rating", 4)
                if rating is not None and rating < 2:
                    low_confidence.append(token.text)
            if token.whitespace:
                phoneme_parts.append(" ")

        phonemes = "".join(phoneme_parts)
        phonemes = filter_for_kokoro(phonemes)

        return PhonemeResult(
            phonemes=phonemes.strip(),
            tokens=tokens,
            low_confidence_words=low_confidence,
        )

    def tokenize(self, phonemes: str) -> list[int]:
        """Convert phonemes to token IDs.

        Args:
            phonemes: Phoneme string (Kokoro format)

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

        return phonemes_to_ids(phonemes)

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to phonemes.

        Args:
            tokens: List of token IDs

        Returns:
            Phoneme string
        """
        return ids_to_phonemes(tokens)

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
        g2p = self._get_g2p(lang)
        tokens = g2p(text)

        result = []
        for token in tokens:
            if token.phonemes and token.text.strip():
                # Filter phonemes for Kokoro vocabulary
                filtered_phonemes = filter_for_kokoro(token.phonemes)
                result.append((token.text, filtered_phonemes))

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
            "n_tokens": N_TOKENS,
            "backend": "kokorog2p",
        }

    def validate_phonemes(self, phonemes: str) -> tuple[bool, list[str]]:
        """Validate that all characters are in the Kokoro vocabulary.

        Args:
            phonemes: Phoneme string to validate

        Returns:
            Tuple of (is_valid, list_of_invalid_chars)
        """
        return validate_for_kokoro(phonemes)


# Convenience function for simple usage
def create_tokenizer(
    use_espeak_fallback: bool = True,
    use_spacy: bool = True,
) -> Tokenizer:
    """Create a tokenizer with the specified configuration.

    Args:
        use_espeak_fallback: Whether to use espeak for OOV words
        use_spacy: Whether to use spaCy for POS tagging

    Returns:
        Configured Tokenizer instance
    """
    config = TokenizerConfig(
        use_espeak_fallback=use_espeak_fallback,
        use_spacy=use_spacy,
    )
    return Tokenizer(config=config)
