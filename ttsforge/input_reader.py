"""Unified input file reader for EPUB, TXT, and PDF files.

This module provides a common interface for reading different input formats,
extracting metadata, chapters, and content for TTS conversion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Metadata:
    """Book metadata."""

    title: Optional[str] = None
    authors: list[str] = None
    language: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[int] = None

    def __post_init__(self):
        """Initialize authors list if None."""
        if self.authors is None:
            self.authors = []


@dataclass
class Chapter:
    """Represents a chapter with title and content."""

    title: str
    text: str
    index: int = 0

    @property
    def char_count(self) -> int:
        """Return the character count of the chapter."""
        return len(self.text)


class InputReader:
    """Unified reader for EPUB, TXT (Gutenberg), and PDF files."""

    def __init__(self, file_path: Path | str):
        """Initialize the reader with a file path.

        Args:
            file_path: Path to the input file (EPUB, TXT, or PDF)
        """
        self.file_path = Path(file_path)
        self._metadata: Optional[Metadata] = None
        self._chapters: Optional[list[Chapter]] = None

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Determine file type
        self.file_type = self._detect_file_type()

    def _detect_file_type(self) -> str:
        """Detect the file type based on extension.

        Returns:
            File type: 'epub', 'txt', or 'pdf'
        """
        suffix = self.file_path.suffix.lower()
        if suffix == ".epub":
            return "epub"
        elif suffix in [".txt", ".text"]:
            return "txt"
        elif suffix == ".pdf":
            return "pdf"
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported types: .epub, .txt, .pdf"
            )

    def get_metadata(self) -> Metadata:
        """Extract metadata from the file.

        Returns:
            Metadata object with title, author, language, etc.
        """
        if self._metadata is not None:
            return self._metadata

        if self.file_type == "epub":
            self._metadata = self._get_epub_metadata()
        elif self.file_type == "txt":
            self._metadata = self._get_gutenberg_metadata()
        elif self.file_type == "pdf":
            self._metadata = self._get_pdf_metadata()

        return self._metadata

    def get_chapters(self) -> list[Chapter]:
        """Extract chapters from the file.

        Returns:
            List of Chapter objects
        """
        if self._chapters is not None:
            return self._chapters

        if self.file_type == "epub":
            self._chapters = self._get_epub_chapters()
        elif self.file_type == "txt":
            self._chapters = self._get_gutenberg_chapters()
        elif self.file_type == "pdf":
            self._chapters = self._get_pdf_chapters()

        return self._chapters

    def get_chapters_with_html(self) -> list[tuple[Chapter, Optional[str]]]:
        """Extract chapters with their original HTML content for markup detection.

        Returns:
            List of tuples containing (Chapter, html_content or None)
        """
        if self.file_type == "epub":
            return self._get_epub_chapters_with_html()
        else:
            # For non-EPUB files, HTML content is not available
            chapters = self.get_chapters()
            return [(ch, None) for ch in chapters]

    # EPUB methods
    def _get_epub_metadata(self) -> Metadata:
        """Extract metadata from EPUB file."""
        try:
            from epub2text import EPUBParser
        except ImportError:
            raise ImportError(
                "epub2text is required for EPUB support. "
                "Install with: pip install epub2text"
            )

        parser = EPUBParser(str(self.file_path))
        epub_metadata = parser.get_metadata()

        return Metadata(
            title=epub_metadata.title,
            authors=list(epub_metadata.authors) if epub_metadata.authors else [],
            language=epub_metadata.language,
            publisher=epub_metadata.publisher,
            publication_year=epub_metadata.publication_year,
        )

    def _get_epub_chapters(self) -> list[Chapter]:
        """Extract chapters from EPUB file."""
        try:
            from epub2text import EPUBParser
        except ImportError:
            raise ImportError(
                "epub2text is required for EPUB support. "
                "Install with: pip install epub2text"
            )

        parser = EPUBParser(str(self.file_path))
        epub_chapters = parser.get_chapters()

        # Convert to our Chapter format
        chapters = []
        for i, ch in enumerate(epub_chapters):
            # Remove <<CHAPTER: ...>> markers that epub2text adds
            content = re.sub(
                r"^\s*<<CHAPTER:[^>]*>>\s*\n*", "", ch.text, count=1, flags=re.MULTILINE
            )
            chapters.append(Chapter(title=ch.title, text=content, index=i))

        return chapters

    def _get_epub_chapters_with_html(self) -> list[tuple[Chapter, Optional[str]]]:
        """Extract chapters from EPUB with HTML content preserved."""
        try:
            from epub2text import EPUBParser
        except ImportError:
            raise ImportError(
                "epub2text is required for EPUB support. "
                "Install with: pip install epub2text"
            )

        parser = EPUBParser(str(self.file_path))
        epub_chapters = parser.get_chapters()

        # Convert to our Chapter format with HTML
        chapters_with_html = []
        for i, ch in enumerate(epub_chapters):
            # Remove <<CHAPTER: ...>> markers from plain text
            content = re.sub(
                r"^\s*<<CHAPTER:[^>]*>>\s*\n*", "", ch.text, count=1, flags=re.MULTILINE
            )
            chapter = Chapter(title=ch.title, text=content, index=i)

            # Try to get HTML content
            # epub2text may have an html attribute or we need to extract it
            html_content = getattr(ch, "html", None)
            if html_content is None and hasattr(ch, "content"):
                html_content = ch.content

            chapters_with_html.append((chapter, html_content))

        return chapters_with_html

    # Gutenberg TXT methods
    def _get_gutenberg_metadata(self) -> Metadata:
        """Extract metadata from Project Gutenberg TXT file.

        Parses the header of a Gutenberg text file to extract metadata.
        """
        with open(self.file_path, encoding="utf-8", errors="ignore") as f:
            # Read first 1000 lines for metadata (Gutenberg header is typically short)
            header_lines = []
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                header_lines.append(line)
                # Stop at start of content
                if "*** START OF" in line.upper():
                    break

        header_text = "".join(header_lines)

        # Extract metadata using regex
        title = None
        authors = []
        language = None

        # Title pattern: "Title: <title>"
        title_match = re.search(
            r"^Title:\s*(.+)$", header_text, re.MULTILINE | re.IGNORECASE
        )
        if title_match:
            title = title_match.group(1).strip()

        # Author pattern: "Author: <author>"
        author_match = re.search(
            r"^Author:\s*(.+)$", header_text, re.MULTILINE | re.IGNORECASE
        )
        if author_match:
            authors = [author_match.group(1).strip()]

        # Language pattern: "Language: <language>"
        lang_match = re.search(
            r"^Language:\s*(.+)$", header_text, re.MULTILINE | re.IGNORECASE
        )
        if lang_match:
            language = lang_match.group(1).strip()

        return Metadata(title=title, authors=authors, language=language)

    def _get_gutenberg_chapters(self) -> list[Chapter]:
        """Extract chapters from Project Gutenberg TXT file.

        Splits the text into chapters based on common patterns like:
        - "CHAPTER I", "CHAPTER 1", "Chapter One"
        - "ONE", "TWO", etc. (capitalized chapter titles)
        - "PART I", etc.
        """
        with open(self.file_path, encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

        # Find the start and end markers
        start_match = re.search(
            r"\*\*\* START OF (?:THE|THIS) (?:PROJECT )?GUTENBERG (?:EBOOK|E-BOOK)",
            full_text,
            re.IGNORECASE,
        )
        end_match = re.search(
            r"\*\*\* END OF (?:THE|THIS) (?:PROJECT )?GUTENBERG (?:EBOOK|E-BOOK)",
            full_text,
            re.IGNORECASE,
        )

        # Extract content between markers
        if start_match:
            start_pos = start_match.end()
        else:
            start_pos = 0

        if end_match:
            end_pos = end_match.start()
        else:
            end_pos = len(full_text)

        content = full_text[start_pos:end_pos].strip()

        # Try to split by chapters
        # Pattern 1: "CHAPTER X" or "Chapter X" at start of line
        chapter_pattern = re.compile(
            r"^(?:CHAPTER|Chapter|PART|Part)\s+(?:[IVXLCDM]+|\d+|[A-Z][A-Z\s-]+)$",
            re.MULTILINE,
        )

        # Find all chapter markers
        chapter_matches = list(chapter_pattern.finditer(content))

        if len(chapter_matches) > 1:
            # We found chapters, split by them
            chapters = []
            for i, match in enumerate(chapter_matches):
                title = match.group(0).strip()
                start = match.end()
                end = (
                    chapter_matches[i + 1].start()
                    if i + 1 < len(chapter_matches)
                    else len(content)
                )
                text = content[start:end].strip()

                if text:  # Only add non-empty chapters
                    chapters.append(Chapter(title=title, text=text, index=i))

            return chapters
        else:
            # No clear chapter structure, check for numbered sections
            # Pattern 2: Single words in all caps on their own line (like "ONE", "TWO", etc.)
            section_pattern = re.compile(r"^([A-Z][A-Z\s-]{2,})$", re.MULTILINE)
            section_matches = list(section_pattern.finditer(content))

            # Filter to likely chapter titles (not too long, appear multiple times)
            if len(section_matches) >= 3:
                chapters = []
                for i, match in enumerate(section_matches):
                    title = match.group(0).strip()
                    start = match.end()
                    end = (
                        section_matches[i + 1].start()
                        if i + 1 < len(section_matches)
                        else len(content)
                    )
                    text = content[start:end].strip()

                    if text and len(text) > 100:  # Only add substantial sections
                        chapters.append(Chapter(title=title, text=text, index=i))

                if chapters:
                    return chapters

            # No chapter structure found, return entire content as one chapter
            metadata = self.get_metadata()
            title = metadata.title or self.file_path.stem
            return [Chapter(title=title, text=content, index=0)]

    # PDF methods (placeholder for future implementation)
    def _get_pdf_metadata(self) -> Metadata:
        """Extract metadata from PDF file.

        TODO: Implement PDF metadata extraction.
        """
        raise NotImplementedError("PDF support is not yet implemented")

    def _get_pdf_chapters(self) -> list[Chapter]:
        """Extract chapters from PDF file.

        TODO: Implement PDF chapter extraction.
        """
        raise NotImplementedError("PDF support is not yet implemented")
