"""Tests for ttsforge.onnx_backend module."""

from pathlib import Path

from ttsforge.onnx_backend import (
    LANG_CODE_TO_ONNX,
    MODEL_BASE_URL,
    ONNX_MODEL_FILES,
    VoiceBlend,
    get_model_dir,
    get_model_path,
    get_onnx_lang_code,
    is_model_downloaded,
)


class TestVoiceBlend:
    """Tests for VoiceBlend dataclass."""

    def test_parse_single_voice(self):
        """Should parse single voice with weight."""
        blend = VoiceBlend.parse("af_nicole:100")
        assert len(blend.voices) == 1
        assert blend.voices[0] == ("af_nicole", 1.0)

    def test_parse_single_voice_no_weight(self):
        """Should parse single voice without weight."""
        blend = VoiceBlend.parse("af_nicole")
        assert len(blend.voices) == 1
        assert blend.voices[0] == ("af_nicole", 1.0)

    def test_parse_two_voices_equal_weight(self):
        """Should parse two voices with equal weights."""
        blend = VoiceBlend.parse("af_nicole:50,am_michael:50")
        assert len(blend.voices) == 2
        assert blend.voices[0] == ("af_nicole", 0.5)
        assert blend.voices[1] == ("am_michael", 0.5)

    def test_parse_three_voices(self):
        """Should parse three voices."""
        blend = VoiceBlend.parse("af_nicole:40,am_michael:30,bf_emma:30")
        assert len(blend.voices) == 3
        assert abs(blend.voices[0][1] - 0.4) < 0.01
        assert abs(blend.voices[1][1] - 0.3) < 0.01
        assert abs(blend.voices[2][1] - 0.3) < 0.01

    def test_parse_normalizes_weights(self):
        """Should normalize weights that don't sum to 100."""
        blend = VoiceBlend.parse("af_nicole:20,am_michael:20")
        # Total is 40, should normalize to 0.5 each
        assert len(blend.voices) == 2
        assert abs(blend.voices[0][1] - 0.5) < 0.01
        assert abs(blend.voices[1][1] - 0.5) < 0.01

    def test_parse_handles_whitespace(self):
        """Should handle whitespace in blend string."""
        blend = VoiceBlend.parse("  af_nicole : 50 , am_michael : 50  ")
        assert len(blend.voices) == 2
        assert blend.voices[0][0] == "af_nicole"
        assert blend.voices[1][0] == "am_michael"

    def test_parse_percentage_conversion(self):
        """Weights should be converted from percentages to fractions."""
        blend = VoiceBlend.parse("af_nicole:75,am_michael:25")
        assert abs(blend.voices[0][1] - 0.75) < 0.01
        assert abs(blend.voices[1][1] - 0.25) < 0.01


class TestModelPaths:
    """Tests for model path functions."""

    def test_onnx_model_files_not_empty(self):
        """Should have model files defined."""
        assert len(ONNX_MODEL_FILES) > 0
        assert "kokoro-v1.0.onnx" in ONNX_MODEL_FILES
        assert "voices-v1.0.bin" in ONNX_MODEL_FILES

    def test_model_base_url_valid(self):
        """Should have valid model base URL."""
        assert MODEL_BASE_URL.startswith("https://")
        assert "kokoro-onnx" in MODEL_BASE_URL

    def test_get_model_dir_returns_path(self):
        """Should return a Path object."""
        model_dir = get_model_dir()
        assert isinstance(model_dir, Path)

    def test_get_model_path_returns_full_path(self):
        """Should return full path to model file."""
        path = get_model_path("test_model.onnx")
        assert isinstance(path, Path)
        assert path.name == "test_model.onnx"
        assert get_model_dir() in path.parents or path.parent == get_model_dir()

    def test_is_model_downloaded_false_for_nonexistent(self):
        """Should return False for nonexistent file."""
        result = is_model_downloaded("nonexistent_model_xyz.onnx")
        assert result is False


class TestLangCodeMapping:
    """Tests for language code mapping."""

    def test_lang_code_to_onnx_has_entries(self):
        """Should have language code mappings."""
        assert len(LANG_CODE_TO_ONNX) > 0

    def test_american_english_mapping(self):
        """American English should map to en-us."""
        assert LANG_CODE_TO_ONNX.get("a") == "en-us"

    def test_british_english_mapping(self):
        """British English should map to en-gb."""
        assert LANG_CODE_TO_ONNX.get("b") == "en-gb"

    def test_other_languages_mapped(self):
        """Other languages should be mapped."""
        assert LANG_CODE_TO_ONNX.get("e") == "es"  # Spanish
        assert LANG_CODE_TO_ONNX.get("f") == "fr"  # French
        assert LANG_CODE_TO_ONNX.get("j") == "ja"  # Japanese
        assert LANG_CODE_TO_ONNX.get("z") == "zh"  # Chinese


class TestGetOnnxLangCode:
    """Tests for get_onnx_lang_code function."""

    def test_valid_language_code(self):
        """Should return correct ONNX language code."""
        assert get_onnx_lang_code("a") == "en-us"
        assert get_onnx_lang_code("b") == "en-gb"
        assert get_onnx_lang_code("e") == "es"

    def test_unknown_language_returns_default(self):
        """Unknown language should return en-us default."""
        assert get_onnx_lang_code("x") == "en-us"
        assert get_onnx_lang_code("unknown") == "en-us"

    def test_empty_string_returns_default(self):
        """Empty string should return default."""
        assert get_onnx_lang_code("") == "en-us"


class TestKokoroONNXClass:
    """Tests for KokoroONNX class initialization."""

    def test_import_kokoro_onnx_class(self):
        """Should be able to import KokoroONNX class."""
        from ttsforge.onnx_backend import KokoroONNX

        assert KokoroONNX is not None

    def test_kokoro_onnx_init_parameters(self):
        """Should accept expected initialization parameters."""
        from ttsforge.onnx_backend import KokoroONNX

        # Should not raise - just test that the constructor signature is correct
        kokoro = KokoroONNX(
            model_path=Path("/fake/path.onnx"),
            voices_path=Path("/fake/voices.bin"),
            use_gpu=False,
        )
        assert kokoro._use_gpu is False

    def test_kokoro_onnx_lazy_init(self):
        """KokoroONNX should use lazy initialization."""
        from ttsforge.onnx_backend import KokoroONNX

        kokoro = KokoroONNX()
        # Internal kokoro instance should be None until first use
        assert kokoro._kokoro is None

    def test_split_text_method(self):
        """Should split text into chunks."""
        from ttsforge.onnx_backend import KokoroONNX

        kokoro = KokoroONNX()
        text = "Hello world. This is a test. Another sentence here."
        chunks = kokoro._split_text(text, chunk_size=30)

        assert len(chunks) > 0
        # All text should be included in chunks
        combined = " ".join(chunks)
        assert "Hello world" in combined
        assert "This is a test" in combined

    def test_split_text_respects_chunk_size(self):
        """Chunks should respect approximate chunk size."""
        from ttsforge.onnx_backend import KokoroONNX

        kokoro = KokoroONNX()
        text = "Short. " * 50  # Many short sentences
        chunks = kokoro._split_text(text, chunk_size=50)

        # Most chunks should be around chunk_size
        for chunk in chunks[:-1]:  # Last chunk can be smaller
            assert len(chunk) <= 100  # Allow some flexibility

    def test_split_text_preserves_sentences(self):
        """Split should preserve sentence boundaries."""
        from ttsforge.onnx_backend import KokoroONNX

        kokoro = KokoroONNX()
        text = "First sentence. Second sentence. Third sentence."
        chunks = kokoro._split_text(text, chunk_size=1000)

        # With large chunk size, all should be in one chunk
        assert len(chunks) == 1
        assert chunks[0] == text


class TestVoiceDatabaseMethods:
    """Tests for voice database integration."""

    def test_get_voice_from_database_returns_none_without_db(self):
        """Should return None when no database is loaded."""
        from ttsforge.onnx_backend import KokoroONNX

        kokoro = KokoroONNX()
        result = kokoro.get_voice_from_database("any_voice")
        assert result is None

    def test_list_database_voices_empty_without_db(self):
        """Should return empty list when no database is loaded."""
        from ttsforge.onnx_backend import KokoroONNX

        kokoro = KokoroONNX()
        result = kokoro.list_database_voices()
        assert result == []

    def test_close_method(self):
        """Close method should not raise."""
        from ttsforge.onnx_backend import KokoroONNX

        kokoro = KokoroONNX()
        # Should not raise even without database
        kokoro.close()
        assert kokoro._voice_db is None
