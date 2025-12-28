"""Tests for ttsforge.cli module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ttsforge.cli import DEFAULT_SAMPLE_TEXT, main


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


class TestMainCommand:
    """Tests for main CLI group."""

    def test_main_help(self, runner):
        """Should show help text."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ttsforge" in result.output.lower() or "epub" in result.output.lower()

    def test_main_version(self, runner):
        """Should show version."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0


class TestVoicesCommand:
    """Tests for voices command."""

    def test_voices_list(self, runner):
        """Should list available voices."""
        result = runner.invoke(main, ["voices"])
        assert result.exit_code == 0
        assert "af_bella" in result.output or "Voice" in result.output

    def test_voices_filter_by_language(self, runner):
        """Should filter voices by language."""
        result = runner.invoke(main, ["voices", "--language", "a"])
        assert result.exit_code == 0
        # American English voices should be shown
        assert "af_" in result.output or "am_" in result.output


class TestConfigCommand:
    """Tests for config command."""

    def test_config_show(self, runner):
        """Should show current configuration."""
        result = runner.invoke(main, ["config", "--show"])
        assert result.exit_code == 0

    def test_config_reset(self, runner):
        """Should reset configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("ttsforge.utils.get_user_config_path", return_value=config_path):
                result = runner.invoke(main, ["config", "--reset"])
                assert result.exit_code == 0
                assert "reset" in result.output.lower()


class TestSampleCommand:
    """Tests for sample command."""

    def test_sample_help(self, runner):
        """Should show sample command help."""
        result = runner.invoke(main, ["sample", "--help"])
        assert result.exit_code == 0
        assert "sample" in result.output.lower()
        assert "--voice" in result.output
        assert "--speed" in result.output

    def test_sample_default_text_defined(self):
        """DEFAULT_SAMPLE_TEXT should be defined."""
        assert isinstance(DEFAULT_SAMPLE_TEXT, str)
        assert len(DEFAULT_SAMPLE_TEXT) > 0

    def test_sample_displays_settings(self, runner):
        """Sample should display current settings (before TTS init fails)."""
        # This test doesn't mock TTSConverter, so it will fail during conversion
        # but we can verify that settings are displayed first
        with tempfile.TemporaryDirectory() as tmpdir:
            with runner.isolated_filesystem(temp_dir=tmpdir):
                result = runner.invoke(main, ["sample"])
                # Settings should be displayed even if conversion fails
                assert "Voice:" in result.output
                assert "Language:" in result.output
                assert "Speed:" in result.output


class TestConvertCommand:
    """Tests for convert command."""

    def test_convert_help(self, runner):
        """Should show convert command help."""
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0
        assert "convert" in result.output.lower()
        assert "--voice" in result.output
        assert "--resume" in result.output or "resume" in result.output.lower()

    def test_convert_requires_input(self, runner):
        """Should require input file."""
        result = runner.invoke(main, ["convert"])
        assert result.exit_code != 0

    def test_convert_invalid_file(self, runner):
        """Should handle invalid file gracefully."""
        result = runner.invoke(main, ["convert", "nonexistent.epub"])
        assert result.exit_code != 0


class TestListCommand:
    """Tests for list command."""

    def test_list_help(self, runner):
        """Should show list command help."""
        result = runner.invoke(main, ["list", "--help"])
        assert result.exit_code == 0

    def test_list_requires_input(self, runner):
        """Should require input file."""
        result = runner.invoke(main, ["list"])
        assert result.exit_code != 0


class TestInfoCommand:
    """Tests for info command."""

    def test_info_help(self, runner):
        """Should show info command help."""
        result = runner.invoke(main, ["info", "--help"])
        assert result.exit_code == 0

    def test_info_requires_input(self, runner):
        """Should require input file."""
        result = runner.invoke(main, ["info"])
        assert result.exit_code != 0


class TestCliOptions:
    """Tests for CLI option validation."""

    def test_invalid_voice_rejected(self, runner):
        """Invalid voice should be rejected."""
        result = runner.invoke(main, ["sample", "--voice", "invalid_voice"])
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()

    def test_invalid_language_rejected(self, runner):
        """Invalid language should be rejected."""
        result = runner.invoke(main, ["sample", "--language", "x"])
        assert result.exit_code != 0

    def test_invalid_format_rejected(self, runner):
        """Invalid format should be rejected."""
        result = runner.invoke(main, ["sample", "--format", "invalid"])
        assert result.exit_code != 0

    def test_invalid_split_mode_rejected(self, runner):
        """Invalid split mode should be rejected."""
        result = runner.invoke(main, ["sample", "--split-mode", "invalid"])
        assert result.exit_code != 0
