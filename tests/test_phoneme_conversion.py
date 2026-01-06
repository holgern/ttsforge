"""Tests for ttsforge.phoneme_conversion module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pykokoro import PhonemeSegment
from ttsforge.phoneme_conversion import (
    PhonemeChapterState,
    PhonemeConversionOptions,
    PhonemeConversionProgress,
    PhonemeConversionResult,
    PhonemeConversionState,
    PhonemeConverter,
    parse_chapter_selection,
)
from ttsforge.phonemes import PhonemeBook


class TestParseChapterSelection:
    """Tests for parse_chapter_selection function."""

    def test_single_chapter(self):
        """Test parsing single chapter."""
        result = parse_chapter_selection("3", 10)
        assert result == [2]  # 0-based index

    def test_range(self):
        """Test parsing chapter range."""
        result = parse_chapter_selection("1-5", 10)
        assert result == [0, 1, 2, 3, 4]

    def test_comma_separated(self):
        """Test parsing comma-separated chapters."""
        result = parse_chapter_selection("3,5,7", 10)
        assert result == [2, 4, 6]

    def test_mixed_format(self):
        """Test parsing mixed range and single chapters."""
        result = parse_chapter_selection("1-3,7,9-10", 10)
        assert result == [0, 1, 2, 6, 8, 9]

    def test_with_spaces(self):
        """Test parsing with spaces."""
        result = parse_chapter_selection("1 - 3, 5, 7 - 8", 10)
        assert result == [0, 1, 2, 4, 6, 7]

    def test_sorted_output(self):
        """Test that output is sorted."""
        result = parse_chapter_selection("5,2,8,1", 10)
        assert result == [0, 1, 4, 7]

    def test_deduplicated(self):
        """Test that duplicates are removed."""
        result = parse_chapter_selection("1,1,2,1-3", 10)
        assert result == [0, 1, 2]

    def test_invalid_format_non_numeric(self):
        """Test error on non-numeric input."""
        with pytest.raises(ValueError, match="Invalid chapter number"):
            parse_chapter_selection("abc", 10)

    def test_invalid_format_bad_range(self):
        """Test error on invalid range format."""
        with pytest.raises(ValueError, match="Invalid range format"):
            parse_chapter_selection("1-2-3", 10)

    def test_chapter_zero(self):
        """Test error on chapter 0."""
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_chapter_selection("0", 10)

    def test_negative_chapter(self):
        """Test error on negative chapter format."""
        # "-1" is parsed as a range with empty start, so it's an invalid format
        with pytest.raises(ValueError, match="Invalid range format"):
            parse_chapter_selection("-1", 10)

    def test_chapter_exceeds_total(self):
        """Test error when chapter exceeds total."""
        with pytest.raises(ValueError, match="exceeds total chapters"):
            parse_chapter_selection("15", 10)

    def test_range_exceeds_total(self):
        """Test error when range exceeds total."""
        with pytest.raises(ValueError, match="exceeds total chapters"):
            parse_chapter_selection("8-12", 10)

    def test_reversed_range(self):
        """Test error on reversed range."""
        with pytest.raises(ValueError, match="start > end"):
            parse_chapter_selection("5-3", 10)


class TestPhonemeConversionProgress:
    """Tests for PhonemeConversionProgress dataclass."""

    def test_percent_zero_total(self):
        """Test percent with zero total segments."""
        progress = PhonemeConversionProgress(total_segments_all=0)
        assert progress.percent == 0

    def test_percent_partial(self):
        """Test percent calculation."""
        progress = PhonemeConversionProgress(
            segments_processed=50,
            total_segments_all=100,
        )
        assert progress.percent == 50

    def test_percent_capped(self):
        """Test percent is capped at 99."""
        progress = PhonemeConversionProgress(
            segments_processed=100,
            total_segments_all=100,
        )
        assert progress.percent == 99

    def test_etr_formatted(self):
        """Test ETR formatting."""
        progress = PhonemeConversionProgress(estimated_remaining=3661)
        # Should format as "1:01:01" or similar
        assert progress.etr_formatted


class TestPhonemeConversionState:
    """Tests for PhonemeConversionState dataclass."""

    def test_get_completed_count(self):
        """Test counting completed chapters."""
        state = PhonemeConversionState(
            chapters=[
                PhonemeChapterState(
                    index=0, title="Ch 1", segment_count=10, completed=True
                ),
                PhonemeChapterState(
                    index=1, title="Ch 2", segment_count=10, completed=False
                ),
                PhonemeChapterState(
                    index=2, title="Ch 3", segment_count=10, completed=True
                ),
            ]
        )
        assert state.get_completed_count() == 2

    def test_save_and_load(self):
        """Test saving and loading state."""
        state = PhonemeConversionState(
            source_file="test.json",
            output_file="test.m4b",
            work_dir="/tmp/work",
            voice="af_heart",
            speed=1.0,
            output_format="m4b",
            silence_between_chapters=2.0,
            chapters=[
                PhonemeChapterState(
                    index=0,
                    title="Chapter 1",
                    segment_count=10,
                    completed=True,
                    audio_file="001_chapter_1.wav",
                    duration=60.0,
                ),
            ],
            started_at="2024-01-01 12:00:00",
            selected_chapters=[0, 1, 2],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state.save(state_file)

            # Verify file exists
            assert state_file.exists()

            # Load and verify
            loaded = PhonemeConversionState.load(state_file)
            assert loaded is not None
            assert loaded.source_file == "test.json"
            assert loaded.voice == "af_heart"
            assert len(loaded.chapters) == 1
            assert loaded.chapters[0].completed is True
            assert loaded.selected_chapters == [0, 1, 2]

    def test_load_nonexistent(self):
        """Test loading nonexistent file."""
        result = PhonemeConversionState.load(Path("/nonexistent/state.json"))
        assert result is None

    def test_load_invalid_json(self):
        """Test loading invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text("not valid json")
            result = PhonemeConversionState.load(state_file)
            assert result is None


class TestPhonemeConversionOptions:
    """Tests for PhonemeConversionOptions dataclass."""

    def test_defaults(self):
        """Test default values."""
        options = PhonemeConversionOptions()
        assert options.voice == "af_heart"
        assert options.speed == 1.0
        assert options.output_format == "m4b"
        assert options.use_gpu is False
        assert options.silence_between_chapters == 2.0
        assert options.resume is True
        assert options.keep_chapter_files is False

    def test_custom_values(self):
        """Test custom values."""
        options = PhonemeConversionOptions(
            voice="am_adam",
            speed=1.5,
            output_format="mp3",
            chapters="1-5",
        )
        assert options.voice == "am_adam"
        assert options.speed == 1.5
        assert options.output_format == "mp3"
        assert options.chapters == "1-5"


class TestPhonemeConverter:
    """Tests for PhonemeConverter class."""

    @pytest.fixture
    def sample_book(self):
        """Create a sample PhonemeBook for testing."""
        book = PhonemeBook(title="Test Book")

        # Chapter 1 with 2 segments
        ch1 = book.create_chapter("Chapter 1")
        ch1.add_segment(
            PhonemeSegment(
                text="Hello world",
                phonemes="həˈloʊ wɜːld",
                tokens=[50, 83, 156, 54, 57, 135, 100, 101, 102],
            )
        )
        ch1.add_segment(
            PhonemeSegment(
                text="How are you",
                phonemes="haʊ ɑːr juː",
                tokens=[60, 61, 62, 63, 64, 65],
            )
        )

        # Chapter 2 with 1 segment
        ch2 = book.create_chapter("Chapter 2")
        ch2.add_segment(
            PhonemeSegment(
                text="Goodbye",
                phonemes="ɡʊdˈbaɪ",
                tokens=[70, 71, 72, 73],
            )
        )

        return book

    @pytest.fixture
    def mock_kokoro(self):
        """Create a mock KokoroONNX."""
        mock = MagicMock()
        # Return fake audio samples (1 second of silence)
        fake_audio = np.zeros(24000, dtype="float32")
        mock.create_from_segment.return_value = (fake_audio, 24000)
        return mock

    def test_init(self, sample_book):
        """Test converter initialization."""
        options = PhonemeConversionOptions()
        converter = PhonemeConverter(sample_book, options)
        assert converter.book is sample_book
        assert converter.options is options
        assert converter._cancelled is False

    def test_log_callback(self, sample_book):
        """Test log callback is called."""
        logs = []

        def log_callback(msg, level):
            logs.append((msg, level))

        options = PhonemeConversionOptions()
        converter = PhonemeConverter(sample_book, options, log_callback=log_callback)
        converter.log("Test message", "info")

        assert len(logs) == 1
        assert logs[0] == ("Test message", "info")

    def test_cancel(self, sample_book):
        """Test cancel method."""
        options = PhonemeConversionOptions()
        converter = PhonemeConverter(sample_book, options)
        assert converter._cancelled is False
        converter.cancel()
        assert converter._cancelled is True

    def test_generate_silence(self, sample_book):
        """Test silence generation."""
        options = PhonemeConversionOptions()
        converter = PhonemeConverter(sample_book, options)
        silence = converter._generate_silence(1.0)
        assert isinstance(silence, np.ndarray)
        assert len(silence) == 24000  # 1 second at 24kHz
        assert silence.dtype == np.float32
        assert np.all(silence == 0)

    def test_get_selected_chapters_all(self, sample_book):
        """Test getting all chapters when no selection."""
        options = PhonemeConversionOptions()
        converter = PhonemeConverter(sample_book, options)
        chapters = converter._get_selected_chapters()
        assert len(chapters) == 2

    def test_get_selected_chapters_subset(self, sample_book):
        """Test getting selected chapters."""
        options = PhonemeConversionOptions(chapters="1")
        converter = PhonemeConverter(sample_book, options)
        chapters = converter._get_selected_chapters()
        assert len(chapters) == 1
        assert chapters[0].title == "Chapter 1"

    def test_get_selected_indices_all(self, sample_book):
        """Test getting all indices when no selection."""
        options = PhonemeConversionOptions()
        converter = PhonemeConverter(sample_book, options)
        indices = converter._get_selected_indices()
        assert indices == [0, 1]

    def test_get_selected_indices_subset(self, sample_book):
        """Test getting selected indices."""
        options = PhonemeConversionOptions(chapters="2")
        converter = PhonemeConverter(sample_book, options)
        indices = converter._get_selected_indices()
        assert indices == [1]


class TestPhonemeConverterConversion:
    """Tests for PhonemeConverter conversion methods (with mocked TTS)."""

    @pytest.fixture
    def sample_book(self):
        """Create a sample PhonemeBook for testing."""
        book = PhonemeBook(title="Test Book")

        ch1 = book.create_chapter("Chapter 1")
        ch1.add_segment(
            PhonemeSegment(
                text="Hello",
                phonemes="həˈloʊ",
                tokens=[50, 83, 156],
            )
        )

        ch2 = book.create_chapter("Chapter 2")
        ch2.add_segment(
            PhonemeSegment(
                text="World",
                phonemes="wɜːld",
                tokens=[100, 101],
            )
        )

        return book

    @patch("ttsforge.phoneme_conversion.are_models_downloaded")
    @patch("ttsforge.phoneme_conversion.Kokoro")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_start")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_end")
    def test_convert_creates_output(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
        sample_book,
    ):
        """Test that convert creates output file."""
        # Setup mocks
        mock_models_downloaded.return_value = True
        mock_kokoro = MagicMock()
        fake_audio = np.zeros(24000, dtype="float32")
        mock_kokoro.create_from_segment.return_value = (fake_audio, 24000)
        mock_kokoro.create.return_value = (
            fake_audio,
            24000,
        )  # For chapter announcements
        mock_kokoro_class.return_value = mock_kokoro

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            options = PhonemeConversionOptions(output_format="wav")
            converter = PhonemeConverter(sample_book, options)

            result = converter.convert(output_path)

            assert result.success is True
            assert result.output_path == output_path
            assert output_path.exists()

    @patch("ttsforge.phoneme_conversion.are_models_downloaded")
    @patch("ttsforge.phoneme_conversion.Kokoro")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_start")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_end")
    def test_convert_streaming_creates_output(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
        sample_book,
    ):
        """Test that convert_streaming creates output file."""
        # Setup mocks
        mock_models_downloaded.return_value = True
        mock_kokoro = MagicMock()
        fake_audio = np.zeros(24000, dtype="float32")
        mock_kokoro.create_from_segment.return_value = (fake_audio, 24000)
        mock_kokoro_class.return_value = mock_kokoro

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            options = PhonemeConversionOptions(output_format="wav")
            converter = PhonemeConverter(sample_book, options)

            result = converter.convert_streaming(output_path)

            assert result.success is True
            assert result.output_path == output_path
            assert output_path.exists()

    def test_convert_no_chapters(self):
        """Test convert with empty book."""
        book = PhonemeBook(title="Empty Book")
        options = PhonemeConversionOptions()
        converter = PhonemeConverter(book, options)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.m4b"
            result = converter.convert(output_path)

            assert result.success is False
            assert "No chapters" in result.error_message

    def test_convert_invalid_format(self, sample_book):
        """Test convert with invalid format."""
        options = PhonemeConversionOptions(output_format="invalid")
        converter = PhonemeConverter(sample_book, options)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.invalid"
            result = converter.convert(output_path)

            assert result.success is False
            assert "Unsupported format" in result.error_message

    @patch("ttsforge.phoneme_conversion.are_models_downloaded")
    @patch("ttsforge.phoneme_conversion.Kokoro")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_start")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_end")
    def test_convert_with_chapter_selection(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
        sample_book,
    ):
        """Test conversion with chapter selection."""
        mock_models_downloaded.return_value = True
        mock_kokoro = MagicMock()
        fake_audio = np.zeros(24000, dtype="float32")
        mock_kokoro.create_from_segment.return_value = (fake_audio, 24000)
        mock_kokoro.create.return_value = (
            fake_audio,
            24000,
        )  # For chapter announcements
        mock_kokoro_class.return_value = mock_kokoro

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            options = PhonemeConversionOptions(
                output_format="wav",
                chapters="1",  # Only first chapter
            )
            converter = PhonemeConverter(sample_book, options)

            result = converter.convert(output_path)

            assert result.success is True
            # Should only process chapter 1 (1 segment)
            # Note: with announce_chapters enabled, create() called for title
            assert mock_kokoro.create_from_segment.call_count == 1

    @patch("ttsforge.phoneme_conversion.are_models_downloaded")
    @patch("ttsforge.phoneme_conversion.Kokoro")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_start")
    @patch("ttsforge.phoneme_conversion.prevent_sleep_end")
    def test_progress_callback_called(
        self,
        mock_prevent_end,
        mock_prevent_start,
        mock_kokoro_class,
        mock_models_downloaded,
        sample_book,
    ):
        """Test that progress callback is called."""
        mock_models_downloaded.return_value = True
        mock_kokoro = MagicMock()
        fake_audio = np.zeros(24000, dtype="float32")
        mock_kokoro.create_from_segment.return_value = (fake_audio, 24000)
        mock_kokoro.create.return_value = (
            fake_audio,
            24000,
        )  # For chapter announcements
        mock_kokoro_class.return_value = mock_kokoro

        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            options = PhonemeConversionOptions(output_format="wav")
            converter = PhonemeConverter(
                sample_book,
                options,
                progress_callback=progress_callback,
            )

            converter.convert(output_path)

            # Should have progress updates for each segment
            assert len(progress_updates) >= 2  # At least 2 segments


class TestPhonemeConversionResult:
    """Tests for PhonemeConversionResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = PhonemeConversionResult(
            success=True,
            output_path=Path("/test/output.m4b"),
            duration=3600.0,
        )
        assert result.success is True
        assert result.output_path == Path("/test/output.m4b")
        assert result.duration == 3600.0

    def test_failure_result(self):
        """Test failure result."""
        result = PhonemeConversionResult(
            success=False,
            error_message="Test error",
        )
        assert result.success is False
        assert result.error_message == "Test error"
