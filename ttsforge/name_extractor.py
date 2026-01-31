"""Name extraction module for automatic phoneme dictionary generation.

This module extracts proper names from text and generates phoneme suggestions
using kokorog2p, making it easy to create custom phoneme dictionaries for books.
"""

import functools
import json
import logging
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = "en_core_web_sm"


@functools.lru_cache(maxsize=1)
def _get_nlp(model_name: str = DEFAULT_SPACY_MODEL):
    """Load and cache the spaCy pipeline."""
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy is required for name extraction. "
            "Install with: pip install spacy && python -m spacy download en_core_web_sm"
        ) from e

    try:
        return spacy.load(model_name)
    except OSError as e:
        raise ImportError(
            f"spaCy model '{model_name}' not found. "
            f"Install with: python -m spacy download {model_name}"
        ) from e


def _split_text_into_chunks(text: str, chunk_size: int = 100000) -> list[str]:
    """Split text into chunks at paragraph boundaries.

    Args:
        text: Input text to split
        chunk_size: Target size for each chunk in characters (default: 100000)

    Returns:
        List of text chunks
    """
    # If text is small enough, return as single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # If adding this paragraph would exceed chunk size, start new chunk
        if current_size > 0 and current_size + para_size > chunk_size:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size + 2  # +2 for \n\n

    # Add final chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def extract_names_from_text(
    text: str,
    min_count: int = 3,
    max_names: int = 100,
    include_all: bool = False,
    chunk_size: int = 100000,
    progress_callback: Callable[[int, int], None] | None = None,
    model_name: str = DEFAULT_SPACY_MODEL,
    batch_size: int = 4,
) -> dict[str, int]:
    """Extract proper names from text using spaCy NER and POS tagging.

    Args:
        text: Input text to analyze
        min_count: Minimum occurrences for a name to be included (default: 3)
        max_names: Maximum number of names to return (default: 100)
        include_all: Include all capitalized proper nouns, not just PERSON entities
            (default: False)
        chunk_size: Size of text chunks to process at once in characters
            (default: 100000)
        progress_callback: Optional callback function(current, total) for
            progress updates
        model_name: spaCy model name to load (default: en_core_web_sm)
        batch_size: Batch size for spaCy pipe processing

    Returns:
        Dictionary mapping name -> occurrence count, sorted by frequency

    Raises:
        ImportError: If spaCy is not installed
    """
    nlp = _get_nlp(model_name)

    # Split text into manageable chunks
    chunks = _split_text_into_chunks(text, chunk_size)
    total_chunks = len(chunks)

    logger.info(f"Processing {total_chunks} chunks of text (chunk_size={chunk_size})")

    # Process each chunk and accumulate candidates
    candidates = []

    for chunk_idx, doc in enumerate(nlp.pipe(chunks, batch_size=batch_size), 1):
        if progress_callback:
            progress_callback(chunk_idx, total_chunks)

        chunk_len = len(chunks[chunk_idx - 1])
        logger.debug(f"Processing chunk {chunk_idx}/{total_chunks} ({chunk_len} chars)")

        # Method 1: Named Entity Recognition (PERSON entities)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidates.append(ent.text)

        # Method 2: Capitalized proper nouns (if include_all=True)
        if include_all:
            for sent in doc.sents:
                for token in sent:
                    # Skip first word of sentence, common words, and short words
                    if (
                        token.i != sent.start
                        and token.text[0].isupper()
                        and token.pos_ == "PROPN"
                        and len(token.text) > 2
                    ):
                        candidates.append(token.text)

    # Count occurrences across all chunks
    name_counts = Counter(candidates)

    # Filter by frequency and limit
    filtered = {
        name: count
        for name, count in name_counts.most_common(max_names)
        if count >= min_count
    }

    logger.info(
        f"Extracted {len(filtered)} names from {total_chunks} chunks "
        f"(min_count={min_count}, max={max_names})"
    )

    return filtered


def generate_phoneme_suggestions(
    names: dict[str, int], language: str = "en-us"
) -> dict[str, dict[str, any]]:
    """Generate phoneme suggestions for a list of names.

    Args:
        names: Dictionary of name -> occurrence count
        language: Language code for phonemization (default: 'en-us')

    Returns:
        Dictionary with phoneme suggestions and metadata:
        {
            "name": {
                "phoneme": "/phoneme/",
                "occurrences": count,
                "suggestion_quality": "auto"
            }
        }
    """
    from kokorog2p import phonemize

    suggestions = {}

    for name, count in names.items():
        try:
            # Generate phoneme using kokorog2p
            phoneme = phonemize(name, language=language).phonemes

            # Wrap in / / format for dictionary
            phoneme_formatted = f"/{phoneme}/"

            suggestions[name] = {
                "phoneme": phoneme_formatted,
                "occurrences": count,
                "suggestion_quality": "auto",
            }
        except Exception as e:
            logger.warning(f"Failed to generate phoneme for '{name}': {e}")
            # Add placeholder
            suggestions[name] = {
                "phoneme": "/FIXME/",
                "occurrences": count,
                "suggestion_quality": "error",
                "error": str(e),
            }

    return suggestions


def save_phoneme_dictionary(
    names_with_phonemes: dict[str, dict],
    output_path: Path,
    source_file: str | None = None,
    language: str = "en-us",
) -> None:
    """Save phoneme dictionary to JSON file with metadata.

    Args:
        names_with_phonemes: Dictionary from generate_phoneme_suggestions()
        output_path: Path to save JSON file
        source_file: Optional source file name for metadata
        language: Language code for metadata
    """
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "language": language,
        "total_names": len(names_with_phonemes),
        "note": (
            "Review and edit phonemes before using. "
            "Auto-generated suggestions may need correction."
        ),
    }

    if source_file:
        metadata["generated_from"] = source_file

    output_data = {"_metadata": metadata, "entries": names_with_phonemes}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved phoneme dictionary to {output_path}")


def load_simple_dictionary(file_path: Path) -> dict[str, dict]:
    """Load a simple phoneme dictionary and convert to metadata format.

    Args:
        file_path: Path to JSON dictionary file

    Returns:
        Dictionary in metadata format (for editing/merging)
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # If already in metadata format, return as-is
    if "_metadata" in data and "entries" in data:
        return data["entries"]

    # Convert simple format to metadata format
    entries = {}
    for name, phoneme in data.items():
        if isinstance(phoneme, str):
            entries[name] = {"phoneme": phoneme, "verified": False}
        elif isinstance(phoneme, dict):
            entries[name] = phoneme
        else:
            logger.warning(f"Skipping invalid entry: {name} -> {phoneme}")

    return entries


def merge_dictionaries(
    auto_generated: dict[str, dict], manual: dict[str, dict]
) -> dict[str, dict]:
    """Merge auto-generated dictionary with manual corrections.

    Manual entries take precedence over auto-generated ones.

    Args:
        auto_generated: Auto-generated phoneme dictionary
        manual: Manually created/edited dictionary

    Returns:
        Merged dictionary
    """
    merged = auto_generated.copy()

    for name, entry in manual.items():
        if name in merged:
            # Update with manual entry, preserving occurrence count
            merged[name] = {
                **merged[name],  # Keep occurrences
                **entry,  # Override with manual data
                "verified": True,
            }
        else:
            # New manual entry
            merged[name] = {**entry, "verified": True}

    return merged
