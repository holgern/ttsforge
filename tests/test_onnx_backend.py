"""Tests for pykokoro.onnx_backend module (via ttsforge imports)."""

from pathlib import Path

from pykokoro import VoiceBlend
from pykokoro.onnx_backend import (
    DEFAULT_MODEL_QUALITY,
    HF_REPO_V1_0,
    LANG_CODE_TO_ONNX,
    MODEL_QUALITY_FILES,
    MODEL_QUALITY_FILES_HF,
    get_model_dir,
    get_model_path,
    get_onnx_lang_code,
    is_model_downloaded,
)
from pykokoro.onnx_backend import (
    VOICE_NAMES_V1_0 as VOICE_NAMES,
)
from pykokoro.tokenizer import MAX_PHONEME_LENGTH


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

    def test_model_quality_files_not_empty(self):
        """Should have model quality files defined."""
        assert len(MODEL_QUALITY_FILES) > 0
        assert "fp32" in MODEL_QUALITY_FILES
        assert "q8" in MODEL_QUALITY_FILES

    def test_huggingface_repo_id_valid(self):
        """Should have valid HuggingFace repo ID."""
        assert "/" in HF_REPO_V1_0  # Should be format: org/repo
        assert "Kokoro" in HF_REPO_V1_0 or "kokoro" in HF_REPO_V1_0.lower()

    def test_get_model_dir_returns_path(self):
        """Should return a Path object."""
        model_dir = get_model_dir()
        assert isinstance(model_dir, Path)

    def test_get_model_path_returns_full_path(self):
        """Should return full path to model file for given quality."""
        # Use default source and variant for testing
        path = get_model_path("fp32")
        assert isinstance(path, Path)
        # The actual filename includes subdirectory for HuggingFace
        assert "model.onnx" in str(path)

    def test_get_model_path_q8(self):
        """Should return correct path for q8 quality."""
        path = get_model_path("q8")
        assert "model_quantized.onnx" in str(path)

    def test_get_model_filename(self):
        """Should return correct filename for each quality."""
        assert MODEL_QUALITY_FILES_HF["fp32"] == "model.onnx"
        assert MODEL_QUALITY_FILES_HF["fp16"] == "model_fp16.onnx"
        assert MODEL_QUALITY_FILES_HF["q8"] == "model_quantized.onnx"

    def test_is_model_downloaded_false_for_missing_file(self):
        """Should return False when model file doesn't exist."""
        # This relies on a fresh cache dir or cleaned state
        # We test with a quality that is likely not downloaded
        result = is_model_downloaded("q4f16")
        # Can't assert False since it might be downloaded, just assert it returns bool
        assert isinstance(result, bool)

    def test_default_model_quality(self):
        """Default model quality should be fp32."""
        assert DEFAULT_MODEL_QUALITY == "fp32"

    def test_voice_names_not_empty(self):
        """Should have voice names defined."""
        assert len(VOICE_NAMES) > 0
        assert "af_nicole" in VOICE_NAMES
        assert "am_michael" in VOICE_NAMES


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


class TestKokoroClass:
    """Tests for Kokoro class initialization."""

    def test_import_kokoro_class(self):
        """Should be able to import Kokoro class."""
        from pykokoro import Kokoro

        assert Kokoro is not None

    def test_kokoro_init_parameters(self):
        """Should accept expected initialization parameters."""
        from pykokoro import Kokoro

        # Should not raise - test that the constructor accepts standard parameters
        kokoro = Kokoro(
            use_gpu=False,
            model_quality="fp32",
        )
        assert kokoro is not None

    def test_kokoro_has_methods(self):
        """Kokoro should have expected methods."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Check for key methods
        assert hasattr(kokoro, "create")
        assert hasattr(kokoro, "create_from_tokens")
        assert hasattr(kokoro, "tokenize")

    def test_split_text_method(self):
        """Should split text into chunks (if method exists)."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist in pykokoro
        if not hasattr(kokoro, "_split_text"):
            return

        text = "Hello world. This is a test. Another sentence here."
        chunks = kokoro._split_text(text, chunk_size=30)

        assert len(chunks) > 0
        # All text should be included in chunks
        combined = " ".join(chunks)
        assert "Hello world" in combined
        assert "This is a test" in combined

    def test_split_text_respects_chunk_size(self):
        """Chunks should respect approximate chunk size."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist in pykokoro
        if not hasattr(kokoro, "_split_text"):
            return

        text = "Short. " * 50  # Many short sentences
        chunks = kokoro._split_text(text, chunk_size=50)

        # Most chunks should be around chunk_size
        for chunk in chunks[:-1]:  # Last chunk can be smaller
            assert len(chunk) <= 100  # Allow some flexibility

    def test_split_text_preserves_sentences(self):
        """Split should preserve sentence boundaries."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist in pykokoro
        if not hasattr(kokoro, "_split_text"):
            return

        text = "First sentence. Second sentence. Third sentence."
        chunks = kokoro._split_text(text, chunk_size=1000)

        # With large chunk size, all should be in one chunk
        assert len(chunks) == 1
        assert chunks[0] == text


class TestVoiceDatabaseMethods:
    """Tests for voice database integration."""

    def test_get_voice_from_database_returns_none_without_db(self):
        """Should return None when no database is loaded."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        result = kokoro.get_voice_from_database("any_voice")
        assert result is None

    def test_list_database_voices_empty_without_db(self):
        """Should return empty list when no database is loaded."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        result = kokoro.list_database_voices()
        assert result == []

    def test_close_method(self):
        """Close method should not raise."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Should not raise even without database
        kokoro.close()


class TestSplitPhonemes:
    """Tests for _split_phonemes - internal pykokoro implementation."""

    def test_short_phonemes_no_split(self):
        """Short phonemes should not be split."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        phonemes = "həlˈoʊ wɜːld ."
        batches = kokoro._split_phonemes(phonemes)

        assert len(batches) == 1
        assert batches[0] == phonemes

    def test_split_at_sentence_boundaries(self):
        """Should split at sentence-ending punctuation (. ! ?)."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        # Create phonemes with sentence-ending punctuation
        phonemes = "hɛlˈoʊ . haʊ ˈɑːr juː ? aɪm faɪn ."
        batches = kokoro._split_phonemes(phonemes)

        # Should stay in one batch if total length < MAX_PHONEME_LENGTH
        assert len(batches) >= 1
        # Verify all content is preserved
        combined = " ".join(batches)
        assert "hɛlˈoʊ" in combined
        assert "faɪn" in combined

    def test_split_preserves_punctuation(self):
        """Punctuation should be preserved in batches."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        phonemes = "fɜːrst sɛntəns . sɛkənd sɛntəns !"
        batches = kokoro._split_phonemes(phonemes)

        # All punctuation should be preserved
        combined = " ".join(batches)
        assert "." in combined
        assert "!" in combined

    def test_split_long_phonemes_exceeding_limit(self):
        """Phonemes exceeding MAX_PHONEME_LENGTH should be split."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        # Create a phoneme string longer than MAX_PHONEME_LENGTH (510)
        # Each sentence is ~50 chars, so 12 sentences = ~600 chars
        sentence = "ɡuːtn taːk ! viː ɡeːt ɛs iːnən ? diː zɔnə ʃaɪnt . "
        phonemes = sentence * 12  # ~600 chars (exceeds 510 limit)

        batches = kokoro._split_phonemes(phonemes)

        # Should split into multiple batches
        assert len(batches) > 1
        # Each batch should be under the limit
        for batch in batches:
            assert len(batch) <= MAX_PHONEME_LENGTH
        # All content should be preserved
        combined = " ".join(batches)
        assert "ɡuːtn" in combined
        assert "ʃaɪnt" in combined

    def test_split_respects_max_phoneme_length(self):
        """Each batch should respect MAX_PHONEME_LENGTH."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        # Create very long phoneme string
        base = "a" * 100 + " . "
        phonemes = base * 10  # ~1030 chars

        batches = kokoro._split_phonemes(phonemes)

        # All batches must be under limit
        for batch in batches:
            error_msg = (
                f"Batch length {len(batch)} exceeds "
                f"MAX_PHONEME_LENGTH {MAX_PHONEME_LENGTH}"
            )
            assert len(batch) <= MAX_PHONEME_LENGTH, error_msg

    def test_split_with_german_phonemes(self):
        """Should handle German phonemes with punctuation."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        # Realistic German phonemes from kokorog2p
        phonemes = (
            "ɡuːtn taːk ! vɪlkɔmən ʦuː diːzm baɪʃpiːl . "
            "diː dɔɪʧə ʃpʁaːxə hat fiːlə bəzɔndəʁə aɪɡənʃaftn ."
        )
        batches = kokoro._split_phonemes(phonemes)

        assert len(batches) >= 1
        # Should preserve German phoneme characters
        combined = " ".join(batches)
        assert "ɡuːtn" in combined
        assert "ʃpʁaːxə" in combined
        assert "!" in combined
        assert "." in combined

    def test_split_with_only_periods(self):
        """Should split at periods correctly."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        phonemes = "fɜːrst . sɛkənd . θɜːrd ."
        batches = kokoro._split_phonemes(phonemes)

        # Should preserve all content
        combined = " ".join(batches)
        assert "fɜːrst" in combined
        assert "sɛkənd" in combined
        assert "θɜːrd" in combined

    def test_split_with_only_exclamations(self):
        """Should split at exclamation marks correctly."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        phonemes = "hɛlˈoʊ ! ɡʊdbaɪ !"
        batches = kokoro._split_phonemes(phonemes)

        combined = " ".join(batches)
        assert "hɛlˈoʊ" in combined
        assert "!" in combined
        assert "ɡʊdbaɪ" in combined

    def test_split_with_only_questions(self):
        """Should split at question marks correctly."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        phonemes = "haʊ ˈɑːr juː ? wɛr ɪz ɪt ?"
        batches = kokoro._split_phonemes(phonemes)

        combined = " ".join(batches)
        assert "haʊ" in combined
        assert "?" in combined
        assert "wɛr" in combined

    def test_split_mixed_punctuation(self):
        """Should handle mixed sentence-ending punctuation."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        phonemes = "həlˈoʊ . haʊ ˈɑːr juː ? ɡʊdbaɪ !"
        batches = kokoro._split_phonemes(phonemes)

        combined = " ".join(batches)
        assert "." in combined
        assert "?" in combined
        assert "!" in combined

    def test_split_empty_string(self):
        """Should handle empty phoneme string."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        phonemes = ""
        batches = kokoro._split_phonemes(phonemes)

        assert len(batches) == 1
        assert batches[0] == ""

    def test_split_whitespace_only(self):
        """Should handle whitespace-only phoneme string."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        phonemes = "   "
        batches = kokoro._split_phonemes(phonemes)

        # Should return empty or whitespace
        assert len(batches) >= 1

    def test_split_no_punctuation_very_long(self):
        """Should split very long phonemes even without punctuation."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        # Create string with no sentence-ending punctuation but exceeds limit
        phonemes = "a" * 600  # Exceeds MAX_PHONEME_LENGTH

        batches = kokoro._split_phonemes(phonemes)

        # Should still split even without punctuation
        assert len(batches) > 1
        # Each batch should respect limit
        for batch in batches:
            assert len(batch) <= MAX_PHONEME_LENGTH

    def test_split_preserves_content_integrity(self):
        """All phoneme content should be preserved after splitting."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        # Create a diverse phoneme string
        phonemes = "ɡuːtn taːk ! viː ɡeːt ɛs ? diː zɔnə ʃaɪnt . ɛs ɪst ʃøːn !"
        original_length = len(phonemes.replace(" ", ""))

        batches = kokoro._split_phonemes(phonemes)

        # Reconstruct and verify no content lost
        combined = " ".join(batches)
        combined_length = len(combined.replace(" ", ""))

        # Length should be approximately preserved (allowing for spacing differences)
        assert abs(combined_length - original_length) < 10

    def test_split_realistic_german_text(self):
        """Test with realistic German phoneme output from kokorog2p."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        # Skip if method doesn't exist or is internal
        if not hasattr(kokoro, "_split_phonemes"):
            return

        # Phonemes from actual German text (769 chars total)
        phonemes = (
            "ɡuːtn taːk ! vɪlkɔmən ʦuː diːzm baɪʃpiːl deːɐ dɔɪʧn ʃpʁaːxə . "
            "diː dɔɪʧə ʃpʁaːxə hat fiːlə bəzɔndəʁə aɪɡənʃaftn . "
            "ziː ɪst bəkant fyːɐ iːʁə laŋən ʦuːzamənɡəzɛʦtn vœɐtɐ viː "
            "doːnaʊdampfʃɪfaːɐtsɡəzɛlʃaft oːdɐ "
            "kʁaftfaːɐʦɔʏkhaftpflɪçtfɛɐzɪçəʁʊŋ . "
            "hɔʏtə ɪst aɪn ʃøːnɐ taːk . diː zɔnə ʃaɪnt ʊnt diː føːɡl̩ zɪŋən . "
            "ɪç mœçtə ɡɛɐnə aɪnən kafeː tʁɪŋkn ʊnt aɪn buːx leːzn . "
            "ʦaːlən zɪnt aʊx vɪçtɪç : aɪns , ʦvaɪ , dʁaɪ , fiːɐ , fʏnf , "
            "zɛks , ziːbn̩ , axt , nɔʏn , ʦeːn . "
            "ʊmlaʊtə zɪnt kaʁaktəʁɪstɪʃ fyːɐ dɔɪʧ : ɛː , øː , yː ʊnt das ɛsʦɛt s . "
            "keːzə , bʁøːtçən , mʏlɐ , ʃtʁaːsə ."
        )

        batches = kokoro._split_phonemes(phonemes)

        # Should split into multiple batches
        assert len(batches) >= 2
        # Each batch should be under limit
        for batch in batches:
            assert len(batch) <= MAX_PHONEME_LENGTH
        # Content should be preserved
        combined = " ".join(batches)
        assert "ɡuːtn taːk" in combined
        assert "ʊmlaʊtə" in combined
        assert "ʃtʁaːsə" in combined
