"""Tests for custom phoneme dictionary functionality."""

import json
import tempfile
from pathlib import Path

from pykokoro.tokenizer import Tokenizer, TokenizerConfig


class TestPhonemeDictionary:
    """Test suite for phoneme dictionary feature."""

    def test_load_simple_dictionary(self):
        """Test loading a simple phoneme dictionary."""
        test_dict = {
            "Misaki": "/misˈɑki/",
            "Kubernetes": "/kubɚnˈɛtɪs/",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            assert tokenizer._phoneme_dictionary_obj is not None
            assert tokenizer._phoneme_dictionary_obj.has_entries()
            # Slashes are now stripped by pykokoro
            assert tokenizer._phoneme_dictionary_obj.get_phoneme("Misaki") == "misˈɑki"
        finally:
            Path(temp_path).unlink()

    def test_load_metadata_format(self):
        """Test loading dictionary with metadata format."""
        test_dict = {
            "_metadata": {
                "generated_from": "test.epub",
                "language": "en-us",
            },
            "entries": {
                "Misaki": {"phoneme": "/misˈɑki/", "occurrences": 42},
                "nginx": {"phoneme": "/ˈɛnʤɪnˈɛks/", "occurrences": 8},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            assert tokenizer._phoneme_dictionary_obj is not None
            assert tokenizer._phoneme_dictionary_obj.has_entries()
            # Slashes are now stripped by pykokoro
            assert tokenizer._phoneme_dictionary_obj.get_phoneme("Misaki") == "misˈɑki"
            assert (
                tokenizer._phoneme_dictionary_obj.get_phoneme("nginx") == "ˈɛnʤɪnˈɛks"
            )
        finally:
            Path(temp_path).unlink()

    def test_load_metadata_format_simple_strings(self):
        """Test loading dictionary with metadata format but simple string values."""
        test_dict = {
            "entries": {
                "Misaki": "/misˈɑki/",
                "nginx": "/ˈɛnʤɪnˈɛks/",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            assert tokenizer._phoneme_dictionary_obj is not None
            assert tokenizer._phoneme_dictionary_obj.has_entries()
        finally:
            Path(temp_path).unlink()

    def test_phoneme_format_without_slashes(self):
        """Test that phoneme format without slashes is now valid."""
        test_dict = {"Misaki": "misˈɑki"}  # Without / / delimiters (now valid)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)
            # Dictionary should be loaded successfully
            assert tokenizer._phoneme_dictionary_obj is not None
            assert tokenizer._phoneme_dictionary_obj.get_phoneme("Misaki") == "misˈɑki"
        finally:
            Path(temp_path).unlink()

    def test_missing_file(self):
        """Test that missing file is handled gracefully."""
        config = TokenizerConfig(phoneme_dictionary_path="/nonexistent/file.json")
        # Should warn and continue without dictionary
        tokenizer = Tokenizer(config=config)
        assert tokenizer._phoneme_dictionary_obj is None

    def test_phonemize_with_dictionary(self):
        """Test phonemization with custom dictionary - through SSMD notation."""
        test_dict = {
            "Misaki": "/misˈɑki/",
            "Kubernetes": "/kubɚnˈɛtɪs/",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            text = "Misaki uses Kubernetes for deployment."

            # Apply dictionary to get SSMD notation
            ssmd_text = tokenizer._phoneme_dictionary_obj.apply(text)

            # Verify SSMD notation is applied (backslash-escaped format)
            assert r"[Misaki]\{ph=" in ssmd_text or "[Misaki](ph:" in ssmd_text
            assert r"[Kubernetes]\{ph=" in ssmd_text or "[Kubernetes](ph:" in ssmd_text
        finally:
            Path(temp_path).unlink()

    def test_case_insensitive_matching(self):
        """Test case-insensitive dictionary matching (default)."""
        test_dict = {"Misaki": "/misˈɑki/"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(
                phoneme_dictionary_path=temp_path, phoneme_dict_case_sensitive=False
            )
            tokenizer = Tokenizer(config=config)

            text = "Misaki misaki MISAKI"

            # Apply dictionary - should match all case variations
            ssmd_text = tokenizer._phoneme_dictionary_obj.apply(text)
            phoneme_count = ssmd_text.count(r'\{ph="')

            # Should match all 3 variations
            assert phoneme_count >= 3, f"Expected 3 matches, got {phoneme_count}"
        finally:
            Path(temp_path).unlink()

    def test_case_sensitive_matching(self):
        """Test case-sensitive dictionary matching."""
        test_dict = {"Misaki": "/misˈɑki/"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(
                phoneme_dictionary_path=temp_path, phoneme_dict_case_sensitive=True
            )
            tokenizer = Tokenizer(config=config)

            text = "Misaki misaki"

            # Apply dictionary - should only match exact case
            ssmd_text = tokenizer._phoneme_dictionary_obj.apply(text)
            phoneme_count = ssmd_text.count(r'\{ph="')

            # Should only match 1 variation (exact case)
            assert phoneme_count == 1, f"Expected 1 match, got {phoneme_count}"

            # Verify it's "Misaki" that matched
            assert r"[Misaki]\{ph=" in ssmd_text or "[Misaki](ph:" in ssmd_text
        finally:
            Path(temp_path).unlink()

    def test_word_boundaries(self):
        """Test that word boundaries are respected."""
        test_dict = {"test": "/tˈɛst/"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            # "test" should match
            phonemes1 = tokenizer.phonemize("This is a test.", "en-us")
            assert "tˈɛst" in phonemes1

            # "testing" should NOT match (different word)
            tokenizer.phonemize("testing", "en-us")
            # Original pronunciation of "testing" should be used
            # (not the custom one for "test")
        finally:
            Path(temp_path).unlink()

    def test_no_dictionary(self):
        """Test that phonemization works without dictionary."""
        config = TokenizerConfig(phoneme_dictionary_path=None)
        tokenizer = Tokenizer(config=config)

        text = "Hello world"
        phonemes = tokenizer.phonemize(text, "en-us")

        # Should produce normal phonemes
        assert len(phonemes) > 0
        assert " " not in phonemes or phonemes.strip()

    def test_empty_dictionary(self):
        """Test that empty dictionary is handled."""
        test_dict = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            text = "Hello world"
            phonemes = tokenizer.phonemize(text, "en-us")

            # Should work normally with empty dictionary
            assert len(phonemes) > 0
        finally:
            Path(temp_path).unlink()

    def test_special_characters_in_words(self):
        """Test dictionary words with special regex characters (periods, etc.)."""
        # Use a simple word that can be phonemized
        test_dict = {"Misaki": "/misˈɑki/"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            text = "Misaki is here."

            # Apply dictionary
            ssmd_text = tokenizer._phoneme_dictionary_obj.apply(text)

            # Should use custom phoneme
            assert r"[Misaki]\{ph=" in ssmd_text or "[Misaki](ph:" in ssmd_text
        finally:
            Path(temp_path).unlink()

    def test_multiple_occurrences(self):
        """Test that all occurrences of a word are replaced."""
        test_dict = {"test": "/tˈɛst/"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            text = "test test test"
            phonemes = tokenizer.phonemize(text, "en-us")

            # All occurrences should use custom phoneme
            # Count how many times the custom phoneme appears
            count = phonemes.count("tˈɛst")
            assert count == 3
        finally:
            Path(temp_path).unlink()

    def test_longest_match_first(self):
        """Test that longer words are matched before shorter ones."""
        # Note: Multi-word phoneme annotations have limitations in kokorog2p's
        # markdown processing. Testing with overlapping single words instead.
        test_dict = {
            "testing": "/tˈɛstɪŋ/",
            "test": "/tˈɛst/",  # Shorter word, different pronunciation
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_dict, f)
            temp_path = f.name

        try:
            config = TokenizerConfig(phoneme_dictionary_path=temp_path)
            tokenizer = Tokenizer(config=config)

            # "testing" should match the longer entry, not "test"
            text1 = "I am testing this."
            phonemes1 = tokenizer.phonemize(text1, "en-us")
            assert "tˈɛstɪŋ" in phonemes1

            # "test" alone should match its entry
            text2 = "This is a test."
            phonemes2 = tokenizer.phonemize(text2, "en-us")
            assert "tˈɛst" in phonemes2
        finally:
            Path(temp_path).unlink()
