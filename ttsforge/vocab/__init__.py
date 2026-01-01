"""Vocabulary management for ttsforge tokenizer.

This module provides a compatibility layer that wraps kokorog2p's vocabulary
functions for backward compatibility with existing ttsforge code.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import kokorog2p
from kokorog2p.vocab import (
    get_vocab as _get_kokoro_vocab,
    get_vocab_reverse as _get_vocab_reverse,
    get_config as _get_kokoro_config,
    N_TOKENS,
    PAD_IDX,
    encode,
    decode,
    validate_for_kokoro,
    filter_for_kokoro,
    phonemes_to_ids,
    ids_to_phonemes,
)

if TYPE_CHECKING:
    pass

# Default version identifier (for compatibility)
DEFAULT_VERSION = "v1.0"

# Supported version strings (for backward compatibility)
SUPPORTED_VERSIONS = {"v1.0"}


def get_config_path() -> Path:
    """Get the path to the config.json (compatibility function).

    Returns:
        Path to the embedded kokoro_config.json in kokorog2p.

    Note:
        This now returns the path to kokorog2p's embedded config,
        not the downloaded config.json from onnx_backend.
    """
    import kokorog2p.data

    return Path(kokorog2p.data.__file__).parent / "kokoro_config.json"


def is_config_downloaded() -> bool:
    """Check if config is available (always True with kokorog2p).

    Returns:
        True (kokorog2p embeds the vocabulary)
    """
    return True


def load_vocab(config_path: Path | str | None = None) -> dict[str, int]:
    """Load vocabulary from kokorog2p.

    Args:
        config_path: Ignored (kept for backward compatibility).
            The vocabulary is now loaded from kokorog2p's embedded data.

    Returns:
        Dictionary mapping phoneme strings to token IDs.

    Raises:
        ValueError: If an unknown version string is provided.
    """
    # Handle backward compatibility with version strings
    if isinstance(config_path, str):
        if config_path in SUPPORTED_VERSIONS:
            # Version string provided, use kokorog2p vocab
            pass
        elif config_path.startswith("v") and "." in config_path:
            # Looks like a version string but not supported
            raise ValueError(
                f"Unknown vocabulary version: {config_path}. "
                f"Supported versions: {', '.join(sorted(SUPPORTED_VERSIONS))}"
            )
        # Otherwise ignore and use kokorog2p vocab

    return _get_kokoro_vocab()


def get_vocab_info(config_path: Path | str | None = None) -> dict:
    """Get metadata about the vocabulary.

    Args:
        config_path: Ignored (kept for backward compatibility).

    Returns:
        Dictionary with vocabulary metadata.
    """
    vocab = _get_kokoro_vocab()
    return {
        "version": DEFAULT_VERSION,
        "path": str(get_config_path()),
        "num_tokens": len(vocab),
        "max_token_id": max(vocab.values()) if vocab else 0,
        "n_tokens": N_TOKENS,
        "downloaded": True,
        "backend": "kokorog2p",
    }


def list_versions() -> list[str]:
    """List all available vocabulary versions.

    Returns:
        List of version strings. Currently only "v1.0" is supported.
    """
    return [DEFAULT_VERSION]


# Re-export kokorog2p vocabulary functions for convenience
__all__ = [
    # Compatibility functions
    "DEFAULT_VERSION",
    "SUPPORTED_VERSIONS",
    "get_config_path",
    "is_config_downloaded",
    "load_vocab",
    "get_vocab_info",
    "list_versions",
    # kokorog2p re-exports
    "N_TOKENS",
    "PAD_IDX",
    "encode",
    "decode",
    "validate_for_kokoro",
    "filter_for_kokoro",
    "phonemes_to_ids",
    "ids_to_phonemes",
]
