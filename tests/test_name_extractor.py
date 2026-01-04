"""Tests for name extraction functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from ttsforge.name_extractor import (
    _split_text_into_chunks,
    extract_names_from_text,
    generate_phoneme_suggestions,
    load_simple_dictionary,
    merge_dictionaries,
    save_phoneme_dictionary,
)

# Import will fail if spaCy is not installed - that's okay, tests will be skipped
spacy_available = False
try:
    import spacy  # noqa: F401

    spacy_available = True
except ImportError:
    pass

# Skip all tests if spaCy is not available
pytestmark = pytest.mark.skipif(not spacy_available, reason="spaCy not installed")


class TestNameExtraction:
    """Test suite for name extraction functionality."""

    def test_extract_names_basic(self):
        """Test basic name extraction."""
        text = """
        Alice went to the store with Bob. Alice met Charlie there.
        Bob and Charlie greeted Alice warmly. They all had a great time.
        """

        names = extract_names_from_text(text, min_count=2, max_names=10)

        # Alice, Bob, and Charlie should be detected
        assert "Alice" in names or "alice" in {k.lower() for k in names}
        assert len(names) >= 1  # At least one name found

    def test_extract_names_min_count(self):
        """Test minimum count filtering."""
        text = """
        Alice appeared once. Bob appeared twice. Bob was here. Charlie appeared many
        times. Charlie was there. Charlie did this. Charlie did that.
        """

        # With min_count=3, only Charlie should be included
        names = extract_names_from_text(text, min_count=3, max_names=10)

        # Charlie should be present (appears 4 times)
        assert any("charlie" in name.lower() for name in names)

    def test_extract_names_max_limit(self):
        """Test maximum names limit."""
        # Generate text with many names
        text = " ".join(
            [f"{chr(65 + i)}{chr(65 + i)} appeared here." for i in range(20)]
        )

        names = extract_names_from_text(text, min_count=1, max_names=5)

        # Should not exceed max_names
        assert len(names) <= 5

    def test_generate_phoneme_suggestions(self):
        """Test phoneme generation."""
        names = {"Alice": 5, "Bob": 3}

        suggestions = generate_phoneme_suggestions(names, "en-us")

        assert "Alice" in suggestions
        assert "Bob" in suggestions

        # Check format
        assert suggestions["Alice"]["phoneme"].startswith("/")
        assert suggestions["Alice"]["phoneme"].endswith("/")
        assert suggestions["Alice"]["occurrences"] == 5
        assert suggestions["Alice"]["suggestion_quality"] == "auto"

    def test_save_and_load_dictionary(self):
        """Test saving and loading phoneme dictionary."""
        suggestions = {
            "Alice": {
                "phoneme": "/ˈælɪs/",
                "occurrences": 5,
                "suggestion_quality": "auto",
            },
            "Bob": {"phoneme": "/bɑb/", "occurrences": 3, "suggestion_quality": "auto"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            save_phoneme_dictionary(
                suggestions, temp_path, source_file="test.txt", language="en-us"
            )

            # Load back
            with open(temp_path, encoding="utf-8") as f:
                data = json.load(f)

            assert "_metadata" in data
            assert "entries" in data
            assert data["_metadata"]["language"] == "en-us"
            assert "Alice" in data["entries"]
            assert data["entries"]["Alice"]["phoneme"] == "/ˈælɪs/"

        finally:
            temp_path.unlink()

    def test_load_simple_dictionary(self):
        """Test loading simple dictionary format."""
        simple_dict = {"Alice": "/ˈælɪs/", "Bob": "/bɑb/"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(simple_dict, f)
            temp_path = Path(f.name)

        try:
            entries = load_simple_dictionary(temp_path)

            assert "Alice" in entries
            assert entries["Alice"]["phoneme"] == "/ˈælɪs/"
            assert entries["Alice"]["verified"] is False

        finally:
            temp_path.unlink()

    def test_load_metadata_dictionary(self):
        """Test loading metadata dictionary format."""
        metadata_dict = {
            "_metadata": {"language": "en-us"},
            "entries": {
                "Alice": {
                    "phoneme": "/ˈælɪs/",
                    "occurrences": 5,
                    "verified": True,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metadata_dict, f)
            temp_path = Path(f.name)

        try:
            entries = load_simple_dictionary(temp_path)

            assert "Alice" in entries
            assert entries["Alice"]["phoneme"] == "/ˈælɪs/"
            assert entries["Alice"]["verified"] is True

        finally:
            temp_path.unlink()

    def test_merge_dictionaries(self):
        """Test merging auto-generated and manual dictionaries."""
        auto_generated = {
            "Alice": {
                "phoneme": "/ˈælɪs/",
                "occurrences": 5,
                "suggestion_quality": "auto",
            },
            "Bob": {
                "phoneme": "/bɑb/",
                "occurrences": 3,
                "suggestion_quality": "auto",
            },
        }

        manual = {"Alice": {"phoneme": "/ælˈis/"}}  # Corrected pronunciation

        merged = merge_dictionaries(auto_generated, manual)

        # Alice should have manual phoneme but keep occurrence count
        assert merged["Alice"]["phoneme"] == "/ælˈis/"
        assert merged["Alice"]["occurrences"] == 5
        assert merged["Alice"]["verified"] is True

        # Bob should remain unchanged
        assert merged["Bob"]["phoneme"] == "/bɑb/"

    def test_extract_names_empty_text(self):
        """Test extraction from empty text."""
        names = extract_names_from_text("", min_count=1, max_names=10)

        assert len(names) == 0

    def test_extract_names_no_names(self):
        """Test extraction from text with no names."""
        text = "The quick brown fox jumps over the lazy dog."

        names = extract_names_from_text(text, min_count=1, max_names=10)

        # Might find some depending on spaCy's NER, but should be few or none
        assert len(names) <= 2  # Allowing for some false positives

    def test_text_splitting_small(self):
        """Test that small text is not split."""
        text = "This is a short text."
        chunks = _split_text_into_chunks(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_splitting_large(self):
        """Test that large text is split into chunks."""
        # Create text with multiple paragraphs
        paragraphs = [f"Paragraph {i} with some content." for i in range(100)]
        text = "\n\n".join(paragraphs)

        chunks = _split_text_into_chunks(text, chunk_size=500)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Chunks should be roughly the target size (allowing some variance)
        for chunk in chunks[:-1]:  # All but last chunk
            # Should be close to chunk_size, allowing for paragraph boundaries
            assert 300 <= len(chunk) <= 800

        # All text should be preserved
        reconstructed = "\n\n".join(chunks)
        assert reconstructed == text

    def test_extract_with_chunking(self):
        """Test extraction with chunking on large text."""
        # Create large text with repeated names
        paragraphs = []
        for _ in range(50):
            paragraphs.append(
                "Alice and Bob went shopping. Charlie joined them. "
                "Alice, Bob, and Charlie had fun."
            )
        text = "\n\n".join(paragraphs)

        # Extract with small chunk size to force chunking
        names = extract_names_from_text(text, min_count=10, chunk_size=1000)

        # Should still find the names across chunks
        assert any("alice" in name.lower() for name in names)
        assert any("bob" in name.lower() for name in names)

    def test_extract_with_progress_callback(self):
        """Test extraction with progress callback."""
        text = "Alice and Bob met Charlie. " * 50

        callback_calls = []

        def progress_callback(current: int, total: int) -> None:
            callback_calls.append((current, total))

        extract_names_from_text(
            text, min_count=5, chunk_size=200, progress_callback=progress_callback
        )

        # Callback should have been called
        assert len(callback_calls) > 0

        # Last call should have current == total
        last_call = callback_calls[-1]
        assert last_call[0] == last_call[1]
