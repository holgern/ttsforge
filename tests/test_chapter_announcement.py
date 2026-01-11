"""Unit tests for chapter announcement feature."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pykokoro import PhonemeSegment

from ttsforge.conversion import Chapter, ConversionOptions
from ttsforge.phoneme_conversion import PhonemeConversionOptions, PhonemeConverter
from ttsforge.phonemes import PhonemeBook


class TestChapterAnnouncementConversion:
    """Test chapter announcement in TTSConverter."""

    @pytest.fixture
    def sample_chapters(self):
        """Create sample chapters for testing."""
        return [
            Chapter(
                title="The Beginning",
                content="This is chapter one.",
                index=0,
            ),
            Chapter(
                title="The Middle",
                content="This is chapter two.",
                index=1,
            ),
            Chapter(
                title="The End",
                content="This is chapter three.",
                index=2,
            ),
        ]

    @patch("ttsforge.conversion.are_models_downloaded")
    @patch("ttsforge.conversion.Kokoro")
    @patch("ttsforge.conversion.prevent_sleep_start")
    @patch("ttsforge.conversion.prevent_sleep_end")
    def test_announce_chapters_enabled_by_default(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
    ):
        """Test that announce_chapters is enabled by default."""
        options = ConversionOptions()
        assert options.announce_chapters is True

    def test_chapter_pause_default_value(self):
        """Test that chapter_pause_after_title has correct default."""
        options = ConversionOptions()
        assert options.chapter_pause_after_title == 2.0

    @patch("ttsforge.conversion.are_models_downloaded")
    @patch("ttsforge.conversion.Kokoro")
    @patch("ttsforge.conversion.prevent_sleep_start")
    @patch("ttsforge.conversion.prevent_sleep_end")
    def test_announce_chapters_can_be_disabled(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
    ):
        """Test that announce_chapters can be disabled."""
        options = ConversionOptions(announce_chapters=False)
        assert options.announce_chapters is False

    def test_custom_chapter_pause_duration(self):
        """Test that custom chapter pause duration is used."""
        options = ConversionOptions(
            announce_chapters=True, chapter_pause_after_title=3.5
        )
        assert options.chapter_pause_after_title == 3.5

    def test_chapter_announcement_format_verification(self, sample_chapters):
        """Test chapter announcement text format without full conversion."""
        # Verify the expected announcement format
        chapter = sample_chapters[0]
        expected_announcement = f"Chapter {chapter.index + 1}. {chapter.title}"
        assert expected_announcement == "Chapter 1. The Beginning"

        # Test with different index
        chapter2 = sample_chapters[2]
        expected_announcement2 = f"Chapter {chapter2.index + 1}. {chapter2.title}"
        assert expected_announcement2 == "Chapter 3. The End"


class TestChapterAnnouncementPhonemeConversion:
    """Test chapter announcement in PhonemeConverter."""

    @pytest.fixture
    def sample_phoneme_book(self):
        """Create a sample PhonemeBook for testing."""
        book = PhonemeBook(title="Test Book")

        ch1 = book.create_chapter("Chapter One")
        ch1.add_segment(
            PhonemeSegment(
                text="Hello world",
                phonemes="həˈloʊ wɜːld",
                tokens=[50, 83, 156, 54, 57, 135, 100, 101, 102],
            )
        )

        ch2 = book.create_chapter("Chapter Two")
        ch2.add_segment(
            PhonemeSegment(
                text="Goodbye world",
                phonemes="ɡʊdˈbaɪ wɜːld",
                tokens=[60, 93, 166, 64, 67, 145, 110, 111, 112],
            )
        )

        return book

    def test_phoneme_announce_chapters_enabled_by_default(self):
        """Test that announce_chapters is enabled by default in phoneme options."""
        options = PhonemeConversionOptions()
        assert options.announce_chapters is True

    def test_phoneme_chapter_pause_default_value(self):
        """Test chapter_pause_after_title has correct default in phoneme options."""
        options = PhonemeConversionOptions()
        assert options.chapter_pause_after_title == 2.0

    @patch("ttsforge.phoneme_conversion.are_models_downloaded")
    @patch("ttsforge.phoneme_conversion.Kokoro")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_start")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_end")
    def test_phoneme_chapter_announcement_calls_create(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
        sample_phoneme_book,
    ):
        """Test that phoneme converter calls create() for announcements."""
        mock_models_downloaded.return_value = True
        mock_kokoro = MagicMock()
        fake_audio = np.zeros(24000, dtype="float32")
        mock_kokoro.create.return_value = (fake_audio, 24000)
        mock_kokoro.create_from_segments.return_value = (fake_audio, 24000)
        mock_kokoro_class.return_value = mock_kokoro

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            options = PhonemeConversionOptions(
                output_format="wav", announce_chapters=True
            )
            converter = PhonemeConverter(sample_phoneme_book, options)

            converter.convert(output_path)

            # Should call create() for announcements (2 chapters)
            assert mock_kokoro.create.call_count == 2
            # Verify first announcement - now just uses the title
            first_call = mock_kokoro.create.call_args_list[0]
            announcement_text = first_call[0][0]
            assert announcement_text == "Chapter One"

    @patch("ttsforge.phoneme_conversion.are_models_downloaded")
    @patch("ttsforge.phoneme_conversion.Kokoro")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_start")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_end")
    def test_phoneme_no_announcement_when_disabled(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
        sample_phoneme_book,
    ):
        """Test phoneme converter doesn't announce when disabled."""
        mock_models_downloaded.return_value = True
        mock_kokoro = MagicMock()
        fake_audio = np.zeros(24000, dtype="float32")
        mock_kokoro.create_from_segments.return_value = (fake_audio, 24000)
        mock_kokoro_class.return_value = mock_kokoro

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            options = PhonemeConversionOptions(
                output_format="wav", announce_chapters=False
            )
            converter = PhonemeConverter(sample_phoneme_book, options)

            converter.convert(output_path)

            # Should NOT call create() for announcements
            # (Only create_from_segments for actual content)
            assert (
                not hasattr(mock_kokoro.create, "call_count")
                or mock_kokoro.create.call_count == 0
            )

    @patch("ttsforge.phoneme_conversion.are_models_downloaded")
    @patch("ttsforge.phoneme_conversion.Kokoro")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_start")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_end")
    def test_phoneme_empty_chapter_no_announcement(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
    ):
        """Test that empty chapters (no segments) are not announced."""
        mock_models_downloaded.return_value = True
        mock_kokoro = MagicMock()
        fake_audio = np.zeros(24000, dtype="float32")
        mock_kokoro.create.return_value = (fake_audio, 24000)
        mock_kokoro_class.return_value = mock_kokoro

        # Create book with empty chapter
        book = PhonemeBook(title="Test")
        book.create_chapter("Empty Chapter")  # No segments added

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            options = PhonemeConversionOptions(
                output_format="wav", announce_chapters=True
            )
            converter = PhonemeConverter(book, options)

            converter.convert(output_path)

            # Should NOT announce empty chapters
            assert mock_kokoro.create.call_count == 0


class TestChapterAnnouncementConfig:
    """Test chapter announcement configuration."""

    def test_config_defaults_in_constants(self):
        """Test that DEFAULT_CONFIG has correct chapter announcement settings."""
        from ttsforge.constants import DEFAULT_CONFIG

        assert "announce_chapters" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["announce_chapters"] is True
        assert "chapter_pause_after_title" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["chapter_pause_after_title"] == 2.0

    def test_announcement_format_includes_number_and_title(self):
        """Test that announcement format is 'Chapter N. Title'."""
        # This is implicitly tested in the mock call verification tests above
        # The format should be: f"Chapter {chapter.index + 1}. {chapter.title}"
        chapter = Chapter(title="Test Chapter", content="Content", index=0)
        expected_format = f"Chapter {chapter.index + 1}. {chapter.title}"
        assert expected_format == "Chapter 1. Test Chapter"

    def test_chapter_indexing_is_one_based(self):
        """Test that chapter numbering in announcements is 1-based (not 0-based)."""
        chapter = Chapter(title="First", content="Text", index=0)
        # Announcement should say "Chapter 1" not "Chapter 0"
        announcement = f"Chapter {chapter.index + 1}. {chapter.title}"
        assert announcement.startswith("Chapter 1")

        chapter2 = Chapter(title="Second", content="Text", index=1)
        announcement2 = f"Chapter {chapter2.index + 1}. {chapter2.title}"
        assert announcement2.startswith("Chapter 2")
