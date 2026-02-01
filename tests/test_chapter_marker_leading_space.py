"""Test chapter marker removal with leading whitespace."""

import re


class TestChapterMarkerLeadingWhitespace:
    """Test that chapter markers are removed even with leading whitespace."""

    def test_marker_with_no_leading_space(self):
        """Test normal case - marker at start of line."""
        text = "<<CHAPTER: Test Chapter>>\n\nThis is the content."
        pattern = r"^<<CHAPTER:[^>]*>>\s*\n*"
        result = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)
        assert result == "This is the content."

    def test_marker_with_leading_space(self):
        """Test marker with a leading space - should be removed with new pattern."""
        text = " <<CHAPTER: Test Chapter>>\n\nThis is the content."
        # New pattern handles leading whitespace
        pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        result = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)
        assert result == "This is the content."

    def test_marker_with_leading_tabs(self):
        """Test marker with leading tabs - should be removed with new pattern."""
        text = "\t<<CHAPTER: Test Chapter>>\n\nThis is the content."
        pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        result = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)
        assert result == "This is the content."

    def test_marker_with_multiple_spaces(self):
        """Test marker with multiple leading spaces -
        should be removed with new pattern."""
        text = "   <<CHAPTER: Test Chapter>>\n\nThis is the content."
        pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        result = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)
        assert result == "This is the content."

    def test_improved_pattern_handles_leading_whitespace(self):
        """Test that improved pattern handles all leading whitespace cases."""
        # Improved pattern that handles leading whitespace
        improved_pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"

        test_cases = [
            ("<<CHAPTER: Test>>\n\nContent", "Content"),
            (" <<CHAPTER: Test>>\n\nContent", "Content"),
            ("\t<<CHAPTER: Test>>\n\nContent", "Content"),
            ("   <<CHAPTER: Test>>\n\nContent", "Content"),
            (" \t <<CHAPTER: Test>>\n\nContent", "Content"),
        ]

        for text, expected in test_cases:
            result = re.sub(improved_pattern, "", text, count=1, flags=re.MULTILINE)
            assert result == expected, f"Failed for input: {repr(text)}"

    def test_marker_not_at_line_start_still_removed_with_multiline(self):
        """Test that marker after newline is removed (MULTILINE mode)."""
        text = "Some text\n<<CHAPTER: Test>>\n\nContent"
        improved_pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        result = re.sub(improved_pattern, "", text, count=1, flags=re.MULTILINE)
        assert result == "Some text\nContent"

    def test_only_first_marker_removed(self):
        """Test that only the first marker is removed (count=1)."""
        text = "<<CHAPTER: One>>\n\nSome text <<CHAPTER: Two>> inside it."
        improved_pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        result = re.sub(improved_pattern, "", text, count=1, flags=re.MULTILINE)
        assert result == "Some text <<CHAPTER: Two>> inside it."

    def test_real_world_epub_scenario(self):
        """Test realistic epub2text output with potential whitespace issues."""
        # Simulate what epub2text might return with whitespace quirks
        epub_content = " <<CHAPTER: THE STORY SO FAR>>\n\nIn the shadow of the Apt..."

        # Old pattern (fails)
        old_pattern = r"^<<CHAPTER:[^>]*>>\s*\n*"
        old_result = re.sub(old_pattern, "", epub_content, count=1, flags=re.MULTILINE)

        # New pattern (works)
        new_pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        new_result = re.sub(new_pattern, "", epub_content, count=1, flags=re.MULTILINE)

        # Verify old pattern fails to remove marker
        assert "<<CHAPTER:" in old_result, "Old pattern should fail with leading space"

        # Verify new pattern successfully removes marker
        assert "<<CHAPTER:" not in new_result, "New pattern should remove marker"
        assert new_result == "In the shadow of the Apt..."
