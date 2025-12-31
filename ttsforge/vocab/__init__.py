"""Vocabulary management for ttsforge tokenizer.

This module handles loading phoneme-to-token vocabularies from the
downloaded config.json file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Default version identifier (for compatibility)
DEFAULT_VERSION = "v1.0"

# Supported version strings (for backward compatibility)
SUPPORTED_VERSIONS = {"v1.0"}


def get_config_path() -> Path:
    """Get the path to the cached config.json.

    Returns:
        Path to the config.json file in the cache directory.

    Note:
        This imports from onnx_backend to avoid circular imports.
    """
    from ..onnx_backend import get_config_path as _get_config_path

    return _get_config_path()


def is_config_downloaded() -> bool:
    """Check if config.json is downloaded.

    Returns:
        True if config.json exists in the cache directory.
    """
    from ..onnx_backend import is_config_downloaded as _is_config_downloaded

    return _is_config_downloaded()


def load_vocab(config_path: Path | str | None = None) -> dict[str, int]:
    """Load vocabulary from cached config.json.

    Args:
        config_path: Optional path to config.json, or a version string
            (e.g., "v1.0") for backward compatibility. If None, uses the
            cached config.json from the download directory.

    Returns:
        Dictionary mapping phoneme strings to token IDs.

    Raises:
        FileNotFoundError: If config.json is not found. Run
            `ttsforge download` to download the required files.
        ValueError: If an unknown version string is provided.
    """
    # Handle backward compatibility with version strings
    if isinstance(config_path, str):
        if config_path in SUPPORTED_VERSIONS:
            # Version string provided, use default config path
            config_path = get_config_path()
        elif config_path.startswith("v") and "." in config_path:
            # Looks like a version string but not supported
            raise ValueError(
                f"Unknown vocabulary version: {config_path}. "
                f"Supported versions: {', '.join(sorted(SUPPORTED_VERSIONS))}"
            )
        else:
            # Try to interpret as a path
            config_path = Path(config_path)

    if config_path is None:
        config_path = get_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Please run 'ttsforge download' to download the required model files."
        )

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    return data["vocab"]


def get_vocab_info(config_path: Path | str | None = None) -> dict:
    """Get metadata about the vocabulary.

    Args:
        config_path: Optional path to config.json, or a version string
            for backward compatibility.

    Returns:
        Dictionary with vocabulary metadata.
    """
    # Handle backward compatibility with version strings
    if isinstance(config_path, str):
        if config_path in SUPPORTED_VERSIONS:
            config_path = get_config_path()
        else:
            config_path = Path(config_path)

    if config_path is None:
        config_path = get_config_path()

    if not config_path.exists():
        return {
            "version": DEFAULT_VERSION,
            "path": str(config_path),
            "num_tokens": 0,
            "max_token_id": 0,
            "downloaded": False,
        }

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocab"]
    return {
        "version": DEFAULT_VERSION,
        "path": str(config_path),
        "num_tokens": len(vocab),
        "max_token_id": max(vocab.values()) if vocab else 0,
        "downloaded": True,
    }


def list_versions() -> list[str]:
    """List all available vocabulary versions.

    Returns:
        List of version strings. Currently only "v1.0" is supported.
    """
    return [DEFAULT_VERSION]
