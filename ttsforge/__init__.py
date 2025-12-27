"""
ttsforge - Generate audiobooks from EPUB files with TTS.

A CLI tool for converting EPUB books to audiobooks using Kokoro TTS.
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
from .utils import (
    get_device,
    get_gpu_info,
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
    # Utils
    "get_device",
    "get_gpu_info",
    "load_config",
    "save_config",
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
