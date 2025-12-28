"""Tests for ttsforge.constants module."""

from ttsforge.constants import (
    DEFAULT_CONFIG,
    DEFAULT_VOICE_FOR_LANG,
    FFMPEG_FORMATS,
    ISO_TO_LANG_CODE,
    LANGUAGE_DESCRIPTIONS,
    SAMPLE_RATE,
    SAMPLE_TEXTS,
    SOUNDFILE_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    VOICE_PREFIX_TO_LANG,
    VOICES,
)


class TestLanguageDescriptions:
    """Tests for language descriptions."""

    def test_all_language_codes_have_descriptions(self):
        """All language codes should have descriptions."""
        expected_codes = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}
        assert set(LANGUAGE_DESCRIPTIONS.keys()) == expected_codes

    def test_english_variants(self):
        """English variants should be correctly named."""
        assert LANGUAGE_DESCRIPTIONS["a"] == "American English"
        assert LANGUAGE_DESCRIPTIONS["b"] == "British English"

    def test_all_descriptions_are_non_empty_strings(self):
        """All descriptions should be non-empty strings."""
        for code, desc in LANGUAGE_DESCRIPTIONS.items():
            assert isinstance(desc, str), f"Description for {code} should be string"
            assert len(desc) > 0, f"Description for {code} should not be empty"


class TestIsoToLangCode:
    """Tests for ISO language code mapping."""

    def test_english_iso_codes(self):
        """English ISO codes should map correctly."""
        assert ISO_TO_LANG_CODE["en"] == "a"
        assert ISO_TO_LANG_CODE["en-us"] == "a"
        assert ISO_TO_LANG_CODE["en-gb"] == "b"
        assert ISO_TO_LANG_CODE["en-au"] == "b"

    def test_other_language_iso_codes(self):
        """Other language ISO codes should map correctly."""
        assert ISO_TO_LANG_CODE["es"] == "e"
        assert ISO_TO_LANG_CODE["fr"] == "f"
        assert ISO_TO_LANG_CODE["hi"] == "h"
        assert ISO_TO_LANG_CODE["it"] == "i"
        assert ISO_TO_LANG_CODE["ja"] == "j"
        assert ISO_TO_LANG_CODE["pt"] == "p"
        assert ISO_TO_LANG_CODE["zh"] == "z"

    def test_regional_variants(self):
        """Regional variants should map to correct language."""
        assert ISO_TO_LANG_CODE["es-es"] == "e"
        assert ISO_TO_LANG_CODE["es-mx"] == "e"
        assert ISO_TO_LANG_CODE["fr-fr"] == "f"
        assert ISO_TO_LANG_CODE["fr-ca"] == "f"
        assert ISO_TO_LANG_CODE["pt-br"] == "p"
        assert ISO_TO_LANG_CODE["pt-pt"] == "p"
        assert ISO_TO_LANG_CODE["zh-cn"] == "z"
        assert ISO_TO_LANG_CODE["zh-tw"] == "z"


class TestVoices:
    """Tests for voice definitions."""

    def test_voices_is_non_empty_list(self):
        """VOICES should be a non-empty list."""
        assert isinstance(VOICES, list)
        assert len(VOICES) > 0

    def test_voice_naming_convention(self):
        """All voices should follow naming convention: XX_name."""
        for voice in VOICES:
            assert "_" in voice, f"Voice {voice} should contain underscore"
            prefix = voice[:2]
            assert prefix.isalpha(), f"Voice {voice} prefix should be alphabetic"

    def test_voice_prefixes_are_valid(self):
        """All voice prefixes should be in VOICE_PREFIX_TO_LANG."""
        for voice in VOICES:
            prefix = voice[:2]
            assert (
                prefix in VOICE_PREFIX_TO_LANG
            ), f"Prefix {prefix} from {voice} not in mapping"

    def test_american_english_voices_exist(self):
        """American English voices should exist."""
        am_voices = [v for v in VOICES if v.startswith("af_") or v.startswith("am_")]
        assert len(am_voices) > 0, "Should have American English voices"

    def test_british_english_voices_exist(self):
        """British English voices should exist."""
        br_voices = [v for v in VOICES if v.startswith("bf_") or v.startswith("bm_")]
        assert len(br_voices) > 0, "Should have British English voices"


class TestVoicePrefixToLang:
    """Tests for voice prefix to language mapping."""

    def test_all_prefixes_map_to_valid_language(self):
        """All prefixes should map to valid language codes."""
        valid_langs = set(LANGUAGE_DESCRIPTIONS.keys())
        for prefix, lang in VOICE_PREFIX_TO_LANG.items():
            assert lang in valid_langs, f"Prefix {prefix} maps to invalid lang {lang}"

    def test_gender_suffix_convention(self):
        """Prefixes should follow gender suffix convention (f=female, m=male)."""
        for prefix in VOICE_PREFIX_TO_LANG:
            assert len(prefix) == 2, f"Prefix {prefix} should be 2 characters"
            assert prefix[1] in ("f", "m"), f"Prefix {prefix} should end with f or m"


class TestDefaultVoiceForLang:
    """Tests for default voice per language."""

    def test_all_languages_have_default_voice(self):
        """All languages should have a default voice."""
        for lang in LANGUAGE_DESCRIPTIONS:
            assert (
                lang in DEFAULT_VOICE_FOR_LANG
            ), f"Language {lang} needs default voice"

    def test_default_voices_exist_in_voices_list(self):
        """All default voices should exist in VOICES list."""
        for lang, voice in DEFAULT_VOICE_FOR_LANG.items():
            assert voice in VOICES, f"Default voice {voice} for {lang} not in VOICES"

    def test_default_voices_match_language(self):
        """Default voices should match their language."""
        for lang, voice in DEFAULT_VOICE_FOR_LANG.items():
            voice_prefix = voice[:2]
            voice_lang = VOICE_PREFIX_TO_LANG.get(voice_prefix)
            assert voice_lang == lang, f"Voice {voice} doesn't match language {lang}"


class TestOutputFormats:
    """Tests for output format definitions."""

    def test_supported_formats(self):
        """Should have expected output formats."""
        expected = {"wav", "mp3", "flac", "opus", "m4b"}
        assert set(SUPPORTED_OUTPUT_FORMATS) == expected

    def test_ffmpeg_formats_subset(self):
        """FFMPEG formats should be subset of supported formats."""
        assert set(FFMPEG_FORMATS).issubset(set(SUPPORTED_OUTPUT_FORMATS))

    def test_soundfile_formats_subset(self):
        """Soundfile formats should be subset of supported formats."""
        assert set(SOUNDFILE_FORMATS).issubset(set(SUPPORTED_OUTPUT_FORMATS))

    def test_all_formats_covered(self):
        """All formats should be in either ffmpeg or soundfile."""
        covered = set(FFMPEG_FORMATS) | set(SOUNDFILE_FORMATS)
        assert set(SUPPORTED_OUTPUT_FORMATS) == covered


class TestDefaultConfig:
    """Tests for default configuration."""

    def test_required_keys_exist(self):
        """Default config should have required keys."""
        required_keys = {
            "default_voice",
            "default_language",
            "default_speed",
            "default_format",
            "use_gpu",
            "silence_between_chapters",
        }
        assert required_keys.issubset(set(DEFAULT_CONFIG.keys()))

    def test_default_voice_is_valid(self):
        """Default voice should be in VOICES list."""
        assert DEFAULT_CONFIG["default_voice"] in VOICES

    def test_default_language_is_valid(self):
        """Default language should be valid."""
        assert DEFAULT_CONFIG["default_language"] in LANGUAGE_DESCRIPTIONS

    def test_default_format_is_valid(self):
        """Default format should be supported."""
        assert DEFAULT_CONFIG["default_format"] in SUPPORTED_OUTPUT_FORMATS

    def test_default_speed_is_reasonable(self):
        """Default speed should be reasonable (0.5 to 2.0)."""
        speed = DEFAULT_CONFIG["default_speed"]
        assert 0.5 <= speed <= 2.0

    def test_silence_is_non_negative(self):
        """Silence between chapters should be non-negative."""
        assert DEFAULT_CONFIG["silence_between_chapters"] >= 0


class TestAudioSettings:
    """Tests for audio settings."""

    def test_sample_rate_is_standard(self):
        """Sample rate should be a standard value."""
        assert SAMPLE_RATE == 24000  # Kokoro's native sample rate


class TestSampleTexts:
    """Tests for sample texts."""

    def test_all_languages_have_sample_text(self):
        """All languages should have sample text."""
        for lang in LANGUAGE_DESCRIPTIONS:
            assert lang in SAMPLE_TEXTS, f"Language {lang} needs sample text"

    def test_sample_texts_are_non_empty(self):
        """All sample texts should be non-empty."""
        for lang, text in SAMPLE_TEXTS.items():
            assert isinstance(text, str), f"Sample text for {lang} should be string"
            assert len(text) > 0, f"Sample text for {lang} should not be empty"
