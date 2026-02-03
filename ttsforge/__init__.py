"""
ttsforge - Generate audiobooks from EPUB files with TTS.

A CLI tool for converting EPUB books to audiobooks using Kokoro ONNX TTS.
"""

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import (
    VoiceBlend,
    are_models_downloaded,
    download_all_models,
    download_model,
    get_model_dir,
)
from pykokoro.tokenizer import (
    EspeakConfig,
    MAX_PHONEME_LENGTH,
    Tokenizer,
)
from pykokoro.constants import SUPPORTED_LANGUAGES
from pykokoro.onnx_backend import VOICE_NAMES_BY_VARIANT
from .constants import (
    DEFAULT_CONFIG,
    LANGUAGE_DESCRIPTIONS,
    SUPPORTED_OUTPUT_FORMATS,
    VOICES,
)

# Import from pykokoro
from pykokoro.constants import SAMPLE_RATE

from .conversion import (
    Chapter,
    ConversionOptions,
    ConversionProgress,
    ConversionResult,
    TTSConverter,
)
from .phonemes import (
    FORMAT_VERSION,
    PhonemeBook,
    PhonemeChapter,
    PhonemeSegment,
    create_phoneme_book_from_chapters,
    phonemize_text_list,
)
from .utils import (
    load_config,
    save_config,
)
from .cli.helpers import DEFAULT_SAMPLE_TEXT


__all__ = [
    # Constants
    "DEFAULT_CONFIG",
    "LANGUAGE_DESCRIPTIONS",
    "SUPPORTED_OUTPUT_FORMATS",
    "VOICES",
    "VOICE_NAMES_BY_VARIANT",
    # Conversion
    "Chapter",
    "ConversionOptions",
    "ConversionProgress",
    "ConversionResult",
    "TTSConverter",
    # Pipeline (from pykokoro)
    "GenerationConfig",
    "KokoroPipeline",
    "PipelineConfig",
    "VoiceBlend",
    "are_models_downloaded",
    "download_all_models",
    "download_model",
    "get_model_dir",
    # Tokenizer (from pykokoro)
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
    # Utils
    "load_config",
    "save_config",
    # herlpers
    "DEFAULT_SAMPLE_TEXT",
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
