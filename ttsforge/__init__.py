"""
ttsforge - Generate audiobooks from EPUB files with TTS.

A CLI tool for converting EPUB books to audiobooks using Kokoro ONNX TTS.
"""

from .constants import (
    DEFAULT_CONFIG,
    LANGUAGE_DESCRIPTIONS,
    SUPPORTED_OUTPUT_FORMATS,
    VOICES,
)
from .conversion import (
    Chapter,
    ConversionOptions,
    ConversionProgress,
    ConversionResult,
    TTSConverter,
)
from .onnx_backend import (
    KokoroONNX,
    VoiceBlend,
    are_models_downloaded,
    download_all_models,
    download_model,
    get_model_dir,
    get_onnx_lang_code,
)
from .phonemes import (
    FORMAT_VERSION,
    PhonemeBook,
    PhonemeChapter,
    PhonemeSegment,
    create_phoneme_book_from_chapters,
    phonemize_text_list,
)
from .tokenizer import (
    MAX_PHONEME_LENGTH,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    EspeakConfig,
    Tokenizer,
)
from .trim import trim
from .utils import (
    load_config,
    save_config,
)

__all__ = [
    # Constants
    "DEFAULT_CONFIG",
    "LANGUAGE_DESCRIPTIONS",
    "SUPPORTED_OUTPUT_FORMATS",
    "VOICES",
    # Conversion
    "Chapter",
    "ConversionOptions",
    "ConversionProgress",
    "ConversionResult",
    "TTSConverter",
    # ONNX Backend
    "KokoroONNX",
    "VoiceBlend",
    "are_models_downloaded",
    "download_all_models",
    "download_model",
    "get_model_dir",
    "get_onnx_lang_code",
    # Tokenizer
    "EspeakConfig",
    "MAX_PHONEME_LENGTH",
    "SAMPLE_RATE",
    "SUPPORTED_LANGUAGES",
    "Tokenizer",
    # Phonemes
    "FORMAT_VERSION",
    "PhonemeBook",
    "PhonemeChapter",
    "PhonemeSegment",
    "create_phoneme_book_from_chapters",
    "phonemize_text_list",
    # Trim
    "trim",
    # Utils
    "load_config",
    "save_config",
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
