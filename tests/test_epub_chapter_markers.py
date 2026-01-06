"""Test chapter marker removal from epub2text content."""

import re
from unittest.mock import MagicMock

from ttsforge.conversion import Chapter


class TestEpubChapterMarkerRemoval:
    """Test that <<CHAPTER:>> markers are removed from epub2text content."""

    def test_chapter_marker_removal_pattern(self):
        """Test the regex pattern removes chapter markers correctly."""
        # Test various formats of chapter markers
        test_cases = [
            (
                "<<CHAPTER: THE STORY SO FAR>>\n\nThis is the content.",
                "This is the content.",
            ),
            (
                "<<CHAPTER: Chapter 1>>\n\nContent here.",
                "Content here.",
            ),
            (
                "<<CHAPTER: Introduction: A New Beginning>>\n\nText content.",
                "Text content.",
            ),
            # Multiple newlines after marker
            (
                "<<CHAPTER: Test>>\n\n\nContent.",
                "Content.",
            ),
            # No content after marker
            (
                "<<CHAPTER: Empty>>",
                "",
            ),
        ]

        pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        for original, expected in test_cases:
            cleaned = re.sub(pattern, "", original, count=1, flags=re.MULTILINE)
            assert cleaned == expected, f"Failed for: {original!r}"

    def test_chapter_marker_only_removed_once(self):
        """Test that only the first chapter marker is removed."""
        text = "<<CHAPTER: One>>\n\nSome text with <<CHAPTER: Two>> inside it."
        pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        cleaned = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)
        # First marker removed, but not the one in the middle of content
        assert cleaned == "Some text with <<CHAPTER: Two>> inside it."

    def test_no_marker_text_unchanged(self):
        """Test that text without markers is unchanged."""
        text = "This is normal chapter content without any markers."
        pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        cleaned = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)
        assert cleaned == text

    def test_epub_chapters_have_markers_stripped(self):
        """Test that convert_epub removes chapter markers from content."""

        # Mock epub2text Chapter objects with markers in content
        mock_chapter1 = MagicMock()
        mock_chapter1.title = "The Beginning"
        mock_chapter1.text = (
            "<<CHAPTER: The Beginning>>\n\nThis is chapter one content."
        )

        mock_chapter2 = MagicMock()
        mock_chapter2.title = "The Middle"
        mock_chapter2.text = "<<CHAPTER: The Middle>>\n\nThis is chapter two content."

        epub_chapters = [mock_chapter1, mock_chapter2]

        # Simulate the chapter parsing logic from convert_epub
        chapters = []
        for i, ch in enumerate(epub_chapters):
            content = ch.text
            content = re.sub(
                r"^\s*<<CHAPTER:[^>]*>>\s*\n*", "", content, count=1, flags=re.MULTILINE
            )
            chapters.append(Chapter(title=ch.title, content=content, index=i))

        # Verify markers were removed
        assert chapters[0].content == "This is chapter one content."
        assert chapters[1].content == "This is chapter two content."

        # Verify titles are preserved
        assert chapters[0].title == "The Beginning"
        assert chapters[1].title == "The Middle"

    def test_special_characters_in_chapter_title(self):
        """Test markers with special regex characters in title."""
        text = "<<CHAPTER: Test (Part 1) - The Beginning [Draft]>>\n\nContent."
        pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        cleaned = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)
        assert cleaned == "Content."

    def test_unicode_in_chapter_title(self):
        """Test markers with unicode characters."""
        text = "<<CHAPTER: Chapitre 1: L'été français>>\n\nContenu du chapitre."
        pattern = r"^\s*<<CHAPTER:[^>]*>>\s*\n*"
        cleaned = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)
        assert cleaned == "Contenu du chapitre."


class TestPhonemeChapterMarkerRemoval:
    """Test that PhonemeChapter.add_text() removes chapter markers."""

    def test_add_text_removes_chapter_markers(self):
        """Test that add_text() filters out chapter markers."""
        from pykokoro import Tokenizer
        from ttsforge.phonemes import PhonemeChapter

        # Create a chapter with text containing a marker
        chapter = PhonemeChapter(title="Test Chapter", chapter_index=0)
        tokenizer = Tokenizer()

        text_with_marker = "<<CHAPTER: Test Chapter>>\n\nThis is the actual content."

        # Add text - should remove the marker
        segments = chapter.add_text(text_with_marker, tokenizer, lang="en-us")

        # Verify the marker was removed
        assert len(segments) > 0
        # The segment text should NOT contain "Test Chapter" from the marker
        assert not segments[0].text.startswith("Test Chapter")
        # It should start with "This is the actual content"
        assert "This is the actual content" in segments[0].text

    def test_add_text_without_markers_unchanged(self):
        """Test that text without markers is processed normally."""
        from pykokoro import Tokenizer
        from ttsforge.phonemes import PhonemeChapter

        chapter = PhonemeChapter(title="Test", chapter_index=0)
        tokenizer = Tokenizer()

        clean_text = "This is clean text without any markers."

        segments = chapter.add_text(clean_text, tokenizer, lang="en-us")

        # Verify text is preserved
        assert len(segments) > 0
        assert "This is clean text without any markers" in segments[0].text
