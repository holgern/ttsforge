"""Vocabulary management for ttsforge tokenizer.

This module handles loading and versioning of phoneme-to-token vocabularies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Available vocabulary versions
VOCAB_VERSIONS = {
    "v1.0": "v1_0.json",
}

DEFAULT_VERSION = "v1.0"


def get_vocab_path(version: str = DEFAULT_VERSION) -> Path:
    """Get the path to a vocabulary file.

    Args:
        version: Vocabulary version string (e.g., "v1.0")

    Returns:
        Path to the vocabulary JSON file

    Raises:
        ValueError: If the version is not found
    """
    if version not in VOCAB_VERSIONS:
        available = ", ".join(VOCAB_VERSIONS.keys())
        raise ValueError(
            f"Unknown vocabulary version '{version}'. Available: {available}"
        )

    return Path(__file__).parent / VOCAB_VERSIONS[version]


def load_vocab(version: str = DEFAULT_VERSION) -> dict[str, int]:
    """Load a vocabulary from file.

    Args:
        version: Vocabulary version string (e.g., "v1.0")

    Returns:
        Dictionary mapping phoneme strings to token IDs
    """
    vocab_path = get_vocab_path(version)
    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["vocab"]


def get_vocab_info(version: str = DEFAULT_VERSION) -> dict:
    """Get metadata about a vocabulary.

    Args:
        version: Vocabulary version string (e.g., "v1.0")

    Returns:
        Dictionary with vocabulary metadata
    """
    vocab_path = get_vocab_path(version)
    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocab"]
    return {
        "version": version,
        "path": str(vocab_path),
        "num_tokens": len(vocab),
        "max_token_id": max(vocab.values()),
        "description": data.get("description", ""),
    }


def list_versions() -> list[str]:
    """List all available vocabulary versions.

    Returns:
        List of version strings
    """
    return list(VOCAB_VERSIONS.keys())
