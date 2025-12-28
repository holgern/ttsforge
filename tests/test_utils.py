"""Tests for ttsforge.utils module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from ttsforge.utils import (
    DEFAULT_ENCODING,
    detect_encoding,
    format_duration,
    format_size,
    get_device,
    get_gpu_info,
    get_user_cache_path,
    get_user_config_path,
    load_config,
    reset_config,
    sanitize_filename,
    save_config,
)


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_zero_seconds(self):
        """Zero seconds should format correctly."""
        assert format_duration(0) == "00:00:00"

    def test_seconds_only(self):
        """Seconds only should format correctly."""
        assert format_duration(45) == "00:00:45"

    def test_minutes_and_seconds(self):
        """Minutes and seconds should format correctly."""
        assert format_duration(125) == "00:02:05"

    def test_hours_minutes_seconds(self):
        """Hours, minutes, and seconds should format correctly."""
        assert format_duration(3661) == "01:01:01"

    def test_large_duration(self):
        """Large durations should format correctly."""
        assert format_duration(36000) == "10:00:00"

    def test_float_seconds(self):
        """Float seconds should be truncated."""
        assert format_duration(65.9) == "00:01:05"


class TestFormatSize:
    """Tests for format_size function."""

    def test_bytes(self):
        """Small sizes should show in bytes."""
        assert format_size(500) == "500.0 B"

    def test_kilobytes(self):
        """KB range should format correctly."""
        assert format_size(1500) == "1.5 KB"

    def test_megabytes(self):
        """MB range should format correctly."""
        assert format_size(1500000) == "1.4 MB"

    def test_gigabytes(self):
        """GB range should format correctly."""
        assert format_size(1500000000) == "1.4 GB"

    def test_terabytes(self):
        """TB range should format correctly."""
        assert format_size(1500000000000) == "1.4 TB"

    def test_zero(self):
        """Zero should format as bytes."""
        assert format_size(0) == "0.0 B"


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_normal_string(self):
        """Normal strings should pass through."""
        assert sanitize_filename("hello_world") == "hello_world"

    def test_removes_invalid_chars(self):
        """Invalid characters should be removed."""
        assert sanitize_filename('file<>:"/\\|?*name') == "filename"

    def test_replaces_spaces_with_underscore(self):
        """Spaces should be replaced with underscores."""
        assert sanitize_filename("hello world") == "hello_world"

    def test_collapses_multiple_spaces(self):
        """Multiple spaces should collapse to single underscore."""
        assert sanitize_filename("hello   world") == "hello_world"

    def test_strips_leading_trailing_underscores(self):
        """Leading and trailing underscores should be stripped."""
        assert sanitize_filename("_hello_world_") == "hello_world"

    def test_max_length_truncation(self):
        """Long names should be truncated."""
        long_name = "a" * 150
        result = sanitize_filename(long_name, max_length=100)
        assert len(result) <= 100

    def test_empty_returns_output(self):
        """Empty or all-invalid strings should return 'output'."""
        assert sanitize_filename("") == "output"
        assert sanitize_filename("<>:") == "output"

    def test_custom_max_length(self):
        """Custom max length should be respected."""
        result = sanitize_filename("hello_world_test", max_length=10)
        assert len(result) <= 10


class TestGetDevice:
    """Tests for get_device function."""

    def test_cpu_when_disabled(self):
        """Should return 'cpu' when GPU disabled."""
        assert get_device(use_gpu=False) == "cpu"

    def test_returns_valid_device(self):
        """Should return a valid device string."""
        device = get_device(use_gpu=True)
        assert device in ("cpu", "cuda", "mps")


class TestGetGpuInfo:
    """Tests for get_gpu_info function."""

    def test_returns_tuple(self):
        """Should return a tuple of (message, available)."""
        result = get_gpu_info(enabled=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], bool)

    def test_disabled_message(self):
        """Should indicate when GPU is disabled."""
        message, available = get_gpu_info(enabled=False)
        # Either GPU is available but disabled, or just not available
        assert isinstance(message, str)
        assert len(message) > 0


class TestConfigFunctions:
    """Tests for configuration functions."""

    def test_get_user_config_path_returns_path(self):
        """Should return a Path object."""
        path = get_user_config_path()
        assert isinstance(path, Path)
        assert path.name == "config.json"

    def test_get_user_cache_path_returns_path(self):
        """Should return a Path object."""
        path = get_user_cache_path()
        assert isinstance(path, Path)

    def test_get_user_cache_path_with_folder(self):
        """Should create subfolder in cache path."""
        path = get_user_cache_path("test_folder")
        assert isinstance(path, Path)
        assert "test_folder" in str(path)

    def test_load_config_returns_dict(self):
        """Should return a dictionary."""
        config = load_config()
        assert isinstance(config, dict)

    def test_load_config_has_defaults(self):
        """Should have default keys."""
        config = load_config()
        assert "default_voice" in config
        assert "default_language" in config
        assert "default_speed" in config

    def test_save_and_load_config(self):
        """Should be able to save and load config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("ttsforge.utils.get_user_config_path", return_value=config_path):
                test_config = {"test_key": "test_value", "default_voice": "am_adam"}
                result = save_config(test_config)
                assert result is True

                loaded = load_config()
                assert loaded["test_key"] == "test_value"
                assert loaded["default_voice"] == "am_adam"

    def test_reset_config_returns_defaults(self):
        """Reset should return default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("ttsforge.utils.get_user_config_path", return_value=config_path):
                # First save custom config
                save_config({"custom_key": "value"})

                # Reset should return defaults
                config = reset_config()
                assert "custom_key" not in config
                assert "default_voice" in config


class TestDetectEncoding:
    """Tests for detect_encoding function."""

    def test_detect_utf8(self):
        """Should detect UTF-8 encoding."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello, World! \u00e4\u00f6\u00fc")
            f.flush()
            try:
                encoding = detect_encoding(f.name)
                assert encoding in ("utf-8", "ascii")
            finally:
                os.unlink(f.name)

    def test_detect_ascii(self):
        """Should detect ASCII encoding."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="ascii"
        ) as f:
            f.write("Hello, World!")
            f.flush()
            try:
                encoding = detect_encoding(f.name)
                assert encoding in ("utf-8", "ascii")  # ASCII is subset of UTF-8
            finally:
                os.unlink(f.name)

    def test_returns_lowercase(self):
        """Encoding should be returned in lowercase."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            f.flush()
            try:
                encoding = detect_encoding(f.name)
                assert encoding == encoding.lower()
            finally:
                os.unlink(f.name)


class TestDefaultEncoding:
    """Tests for DEFAULT_ENCODING constant."""

    def test_default_encoding_is_string(self):
        """DEFAULT_ENCODING should be a string."""
        assert isinstance(DEFAULT_ENCODING, str)

    def test_default_encoding_is_valid(self):
        """DEFAULT_ENCODING should be a valid encoding."""
        # Should not raise an exception
        "test".encode(DEFAULT_ENCODING)
