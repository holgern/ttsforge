"""Tests for ttsforge.tokenizer module."""

import pytest

from ttsforge.tokenizer import (
    MAX_PHONEME_LENGTH,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    EspeakConfig,
    Tokenizer,
)
from ttsforge.vocab import DEFAULT_VERSION, get_vocab_info, list_versions, load_vocab


class TestEspeakConfig:
    """Tests for EspeakConfig dataclass."""

    def test_default_values(self):
        """Test default config has None values."""
        config = EspeakConfig()
        assert config.lib_path is None
        assert config.data_path is None

    def test_custom_values(self):
        """Test config with custom values."""
        config = EspeakConfig(lib_path="/path/to/lib", data_path="/path/to/data")
        assert config.lib_path == "/path/to/lib"
        assert config.data_path == "/path/to/data"


class TestConstants:
    """Tests for module constants."""

    def test_max_phoneme_length(self):
        """Test MAX_PHONEME_LENGTH is defined."""
        assert MAX_PHONEME_LENGTH == 510

    def test_sample_rate(self):
        """Test SAMPLE_RATE is defined."""
        assert SAMPLE_RATE == 24000

    def test_supported_languages(self):
        """Test SUPPORTED_LANGUAGES contains expected entries."""
        assert "en-us" in SUPPORTED_LANGUAGES
        assert "en-gb" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["en"] == "en-us"


class TestVocab:
    """Tests for vocabulary loading."""

    def test_list_versions(self):
        """Test listing vocabulary versions."""
        versions = list_versions()
        assert "v1.0" in versions

    def test_load_vocab_default(self):
        """Test loading default vocabulary."""
        vocab = load_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 100  # Should have many phonemes

    def test_load_vocab_v1_0(self):
        """Test loading v1.0 vocabulary."""
        vocab = load_vocab("v1.0")
        assert isinstance(vocab, dict)
        # Check some expected phonemes
        assert " " in vocab  # Space
        assert "." in vocab  # Period
        assert "ə" in vocab  # Schwa

    def test_get_vocab_info(self):
        """Test getting vocabulary info."""
        info = get_vocab_info()
        assert info["version"] == DEFAULT_VERSION
        assert "num_tokens" in info
        assert "max_token_id" in info
        assert info["num_tokens"] > 100

    def test_invalid_version_raises(self):
        """Test that invalid version raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vocabulary version"):
            load_vocab("v999.0")


class TestTokenizer:
    """Tests for Tokenizer class."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()

    def test_init_default(self, tokenizer):
        """Test default initialization."""
        assert tokenizer.vocab is not None
        assert len(tokenizer.vocab) > 100
        assert tokenizer.vocab_version == DEFAULT_VERSION

    def test_init_custom_vocab(self):
        """Test initialization with custom vocab."""
        custom_vocab = {"a": 1, "b": 2}
        tokenizer = Tokenizer(vocab=custom_vocab)
        assert tokenizer.vocab == custom_vocab

    def test_normalize_text(self, tokenizer):
        """Test text normalization."""
        assert tokenizer.normalize_text("  hello  ") == "hello"
        assert tokenizer.normalize_text("\n\nhello\n\n") == "hello"
        assert tokenizer.normalize_text("hello") == "hello"

    def test_phonemize_basic(self, tokenizer):
        """Test basic phonemization."""
        phonemes = tokenizer.phonemize("hello")
        assert isinstance(phonemes, str)
        assert len(phonemes) > 0
        # Should contain valid phonemes (all chars should be in vocab)
        for char in phonemes:
            assert char in tokenizer.vocab

    def test_phonemize_empty(self, tokenizer):
        """Test phonemization of empty string."""
        phonemes = tokenizer.phonemize("")
        assert phonemes == ""

    def test_phonemize_whitespace_only(self, tokenizer):
        """Test phonemization of whitespace-only string."""
        phonemes = tokenizer.phonemize("   ")
        assert phonemes == ""

    def test_phonemize_with_language(self, tokenizer):
        """Test phonemization with different languages."""
        # US English
        us_phonemes = tokenizer.phonemize("hello", lang="en-us")
        # UK English
        gb_phonemes = tokenizer.phonemize("hello", lang="en-gb")
        # Both should produce phonemes
        assert len(us_phonemes) > 0
        assert len(gb_phonemes) > 0

    def test_tokenize_basic(self, tokenizer):
        """Test basic tokenization."""
        phonemes = tokenizer.phonemize("hello")
        tokens = tokenizer.tokenize(phonemes)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_tokenize_empty(self, tokenizer):
        """Test tokenization of empty string."""
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_tokenize_length_limit(self, tokenizer):
        """Test tokenization length limit."""
        # Create a very long phoneme string
        long_phonemes = "a" * (MAX_PHONEME_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            tokenizer.tokenize(long_phonemes)

    def test_tokenize_at_limit(self, tokenizer):
        """Test tokenization at exactly the limit."""
        phonemes = "a" * MAX_PHONEME_LENGTH
        # Should not raise
        tokens = tokenizer.tokenize(phonemes)
        assert len(tokens) <= MAX_PHONEME_LENGTH

    def test_detokenize_basic(self, tokenizer):
        """Test basic detokenization."""
        phonemes = tokenizer.phonemize("hello")
        tokens = tokenizer.tokenize(phonemes)
        recovered = tokenizer.detokenize(tokens)
        # Should recover the original phonemes
        assert recovered == phonemes

    def test_detokenize_empty(self, tokenizer):
        """Test detokenization of empty list."""
        phonemes = tokenizer.detokenize([])
        assert phonemes == ""

    def test_detokenize_unknown_tokens(self, tokenizer):
        """Test detokenization with unknown token IDs."""
        # Token ID 99999 doesn't exist
        phonemes = tokenizer.detokenize([99999])
        assert phonemes == ""

    def test_text_to_tokens(self, tokenizer):
        """Test text-to-tokens convenience method."""
        tokens = tokenizer.text_to_tokens("hello")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should be same as phonemize + tokenize
        phonemes = tokenizer.phonemize("hello")
        expected = tokenizer.tokenize(phonemes)
        assert tokens == expected

    def test_reverse_vocab_lazy_init(self, tokenizer):
        """Test reverse vocab is lazily initialized."""
        # Access should create it
        reverse = tokenizer.reverse_vocab
        assert isinstance(reverse, dict)
        assert len(reverse) == len(tokenizer.vocab)
        # Keys and values should be swapped
        for phoneme, token_id in tokenizer.vocab.items():
            assert reverse[token_id] == phoneme

    def test_text_to_phonemes_with_words(self, tokenizer):
        """Test word-by-word phonemization."""
        result = tokenizer.text_to_phonemes_with_words("hello world")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0][0] == "hello"
        assert result[1][0] == "world"
        # Each tuple should have (word, phonemes)
        for word, phonemes in result:
            assert isinstance(word, str)
            assert isinstance(phonemes, str)

    def test_format_readable(self, tokenizer):
        """Test human-readable formatting."""
        readable = tokenizer.format_readable("hello world")
        assert "hello [" in readable
        assert "world [" in readable
        assert "]" in readable

    def test_get_vocab_info_method(self, tokenizer):
        """Test get_vocab_info method."""
        info = tokenizer.get_vocab_info()
        assert info["version"] == tokenizer.vocab_version
        assert info["num_tokens"] == len(tokenizer.vocab)
        assert info["max_phoneme_length"] == MAX_PHONEME_LENGTH


class TestTokenizerRoundTrip:
    """Tests for complete roundtrip conversions."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()

    def test_roundtrip_simple(self, tokenizer):
        """Test simple text roundtrip."""
        text = "Hello world"
        phonemes = tokenizer.phonemize(text)
        tokens = tokenizer.tokenize(phonemes)
        recovered = tokenizer.detokenize(tokens)
        assert recovered == phonemes

    def test_roundtrip_punctuation(self, tokenizer):
        """Test roundtrip with punctuation."""
        text = "Hello, world! How are you?"
        phonemes = tokenizer.phonemize(text)
        tokens = tokenizer.tokenize(phonemes)
        recovered = tokenizer.detokenize(tokens)
        assert recovered == phonemes

    def test_roundtrip_numbers(self, tokenizer):
        """Test roundtrip with numbers."""
        text = "I have 5 apples"
        phonemes = tokenizer.phonemize(text)
        tokens = tokenizer.tokenize(phonemes)
        recovered = tokenizer.detokenize(tokens)
        assert recovered == phonemes
