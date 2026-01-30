"""SSMD (Speech Synthesis Markdown) generator for ttsforge.

This module converts chapter text to SSMD format with markup for:
- Emphasis (*text* for moderate, **text** for strong)
- Language switches ([text](lang_code))
- Phoneme substitutions ([word](ph: /phoneme/))

Note: Structural breaks (paragraphs, sentences, clauses) are NOT automatically
added. The SSMD parser in pykokoro handles sentence detection automatically.
Users can manually add breaks in the SSMD file if desired:
- Paragraph breaks (...p)
- Sentence breaks (...s)
- Clause breaks (...c)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


class SSMDGenerationError(Exception):
    """Exception raised when SSMD generation fails."""

    pass


def _hash_content(content: str) -> str:
    """Generate a hash of content for change detection.

    Args:
        content: Text content to hash

    Returns:
        12-character hex hash
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def _detect_emphasis_from_html(html_content: str) -> dict[str, str]:
    """Detect emphasis from HTML tags and map positions to emphasis levels.

    Args:
        html_content: HTML content with formatting tags

    Returns:
        Dictionary mapping text positions to emphasis markers
    """
    emphasis_map = {}

    # Pattern for <em>, <i> tags (moderate emphasis)
    em_pattern = r"<(?:em|i)(?:\s[^>]*)?>([^<]+)</(?:em|i)>"
    for match in re.finditer(em_pattern, html_content, re.IGNORECASE):
        text = match.group(1)
        emphasis_map[text] = "*"

    # Pattern for <strong>, <b> tags (strong emphasis)
    strong_pattern = r"<(?:strong|b)(?:\s[^>]*)?>([^<]+)</(?:strong|b)>"
    for match in re.finditer(strong_pattern, html_content, re.IGNORECASE):
        text = match.group(1)
        emphasis_map[text] = "**"

    return emphasis_map


def _apply_emphasis_markers(text: str, emphasis_map: dict[str, str]) -> str:
    """Apply emphasis markers to text based on emphasis map.

    Args:
        text: Plain text
        emphasis_map: Dictionary mapping text segments to emphasis markers

    Returns:
        Text with emphasis markers applied
    """
    result = text
    # Sort by length (longest first) to avoid partial replacements
    for emphasized_text in sorted(emphasis_map.keys(), key=len, reverse=True):
        marker = emphasis_map[emphasized_text]
        # Use word boundaries to avoid partial matches
        pattern = rf"\b{re.escape(emphasized_text)}\b"
        replacement = f"{marker}{emphasized_text}{marker}"
        result = re.sub(pattern, replacement, result, count=1)

    return result


def _inject_phoneme_substitutions(
    text: str, phoneme_dict: dict[str, str], case_sensitive: bool = False
) -> str:
    """Inject phoneme substitutions into text using SSMD [word](ph: /phoneme/) syntax.

    Args:
        text: Text to process
        phoneme_dict: Dictionary mapping words to IPA phonemes
        case_sensitive: Whether to match case-sensitively

    Returns:
        Text with phoneme substitutions injected
    """
    if not phoneme_dict:
        return text

    link_pattern = re.compile(r"\[[^\]]+\]\([^\)]+\)")

    words = [word for word in phoneme_dict.keys() if word]
    if not words:
        return text

    words = sorted(words, key=len, reverse=True)
    alternation = "|".join(re.escape(word) for word in words)
    boundary_pattern = rf"(?<!\w)({alternation})(?!\w)"
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile(boundary_pattern, flags=flags)

    if case_sensitive:
        lookup = phoneme_dict
    else:
        lookup = {}
        for word, phoneme in phoneme_dict.items():
            key = word.lower()
            if key not in lookup:
                lookup[key] = phoneme

    def replace(match: re.Match[str]) -> str:
        matched_word = match.group(1)
        key = matched_word if case_sensitive else matched_word.lower()
        phoneme = lookup.get(key)
        if not phoneme:
            return matched_word
        clean_phoneme = phoneme.strip("/")
        return f"[{matched_word}](ph: /{clean_phoneme}/)"

    segments: list[str] = []
    last_index = 0
    for match in link_pattern.finditer(text):
        if match.start() > last_index:
            segment = text[last_index : match.start()]
            segments.append(compiled.sub(replace, segment))
        segments.append(match.group(0))
        last_index = match.end()

    if last_index < len(text):
        segments.append(compiled.sub(replace, text[last_index:]))

    return "".join(segments)


def _add_language_markers(text: str, mixed_language_config: dict | None = None) -> str:
    """Add language markers for mixed-language segments.

    Note: This is a placeholder for now. Full implementation would require
    language detection library (lingua-language-detector).

    Args:
        text: Text to process
        mixed_language_config: Configuration for mixed-language mode

    Returns:
        Text with language markers (currently returns text unchanged)
    """
    # TODO: Implement language detection and wrapping
    # For now, return text unchanged
    # Future: Use lingua-language-detector to identify foreign segments
    # and wrap them with [segment](lang_code)
    return text


def _add_structural_breaks(text: str) -> str:
    """Preserve paragraph structure without adding automatic SSMD breaks.

    The SSMD parser in pykokoro will handle sentence detection automatically.
    This function only preserves existing paragraph breaks as double newlines.

    Args:
        text: Plain text to process

    Returns:
        Text with normalized paragraph spacing (no SSMD break markers)
    """
    # Split into paragraphs and normalize spacing
    paragraphs = re.split(r"\n\s*\n+", text)
    result_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if para:
            result_paragraphs.append(para)

    # Join paragraphs with double newlines (standard paragraph separation)
    # No SSMD markers - let pykokoro's parser handle sentence detection
    result = "\n\n".join(result_paragraphs)

    return result


def _strip_redundant_title(chapter_title: str, chapter_text: str) -> str:
    """Remove a duplicated chapter title from the start of the text."""
    title = chapter_title.strip()
    if not title:
        return chapter_text

    lines = chapter_text.splitlines()
    first_idx = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_idx = idx
            break

    if first_idx is None:
        return chapter_text

    first_line = lines[first_idx]
    title_pattern = re.compile(
        rf"^\s*{re.escape(title)}(?:\b|[\s:;\-\u2013\u2014])",
        re.IGNORECASE,
    )
    if not title_pattern.search(first_line):
        return chapter_text

    trimmed_line = title_pattern.sub("", first_line, count=1).lstrip(
        " \t:;\-\u2013\u2014"
    )
    if trimmed_line:
        lines[first_idx] = trimmed_line
        return "\n".join(lines[first_idx:]).lstrip()

    remaining = lines[first_idx + 1 :]
    while remaining and not remaining[0].strip():
        remaining = remaining[1:]
    return "\n".join(remaining).lstrip()


def chapter_to_ssmd(
    chapter_title: str,
    chapter_text: str,
    phoneme_dict: dict[str, str] | None = None,
    phoneme_dict_case_sensitive: bool = False,
    mixed_language_config: dict | None = None,
    html_content: str | None = None,
    include_title: bool = True,
) -> str:
    """Convert a chapter to SSMD format.

    Args:
        chapter_title: Title of the chapter
        chapter_text: Plain text content of the chapter
        phoneme_dict: Optional dictionary mapping words to IPA phonemes
        phoneme_dict_case_sensitive: Whether phoneme matching is case-sensitive
        mixed_language_config: Optional config for mixed-language mode
        html_content: Optional HTML content for emphasis detection
        include_title: Whether to include chapter title in SSMD

    Returns:
        SSMD formatted text

    Raises:
        SSMDGenerationError: If generation fails
    """
    try:
        result = chapter_text
        if include_title and chapter_title:
            result = _strip_redundant_title(chapter_title, result)

        # Step 1: Detect emphasis from HTML if available
        emphasis_map = {}
        if html_content:
            emphasis_map = _detect_emphasis_from_html(html_content)

        # Step 2: Apply emphasis markers
        if emphasis_map:
            result = _apply_emphasis_markers(result, emphasis_map)

        # Step 3: Inject phoneme substitutions
        if phoneme_dict:
            result = _inject_phoneme_substitutions(
                result, phoneme_dict, phoneme_dict_case_sensitive
            )

        # Step 4: Add language markers (if mixed-language mode)
        if mixed_language_config and mixed_language_config.get("use_mixed_language"):
            result = _add_language_markers(result, mixed_language_config)

        # Step 5: Add structural breaks (paragraphs, sentences, clauses)
        result = _add_structural_breaks(result)

        # Step 6: Add chapter title if requested
        if include_title and chapter_title:
            # Clean title and add as heading with double newline separation
            clean_title = chapter_title.strip()
            result = f"# {clean_title}\n\n{result}"

        return result

    except Exception as e:
        raise SSMDGenerationError(
            f"Failed to generate SSMD for chapter '{chapter_title}': {str(e)}"
        ) from e


def save_ssmd_file(ssmd_content: str, output_path: Path) -> str:
    """Save SSMD content to a file and return its hash.

    Args:
        ssmd_content: SSMD formatted text
        output_path: Path to save the SSMD file

    Returns:
        Hash of the saved content

    Raises:
        SSMDGenerationError: If file save fails
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ssmd_content)
        return _hash_content(ssmd_content)
    except Exception as e:
        raise SSMDGenerationError(
            f"Failed to save SSMD file to {output_path}: {str(e)}"
        ) from e


def load_ssmd_file(ssmd_path: Path) -> tuple[str, str]:
    """Load SSMD file and return content with hash.

    Args:
        ssmd_path: Path to the SSMD file

    Returns:
        Tuple of (content, hash)

    Raises:
        SSMDGenerationError: If file load fails or doesn't exist
    """
    try:
        if not ssmd_path.exists():
            raise SSMDGenerationError(f"SSMD file not found: {ssmd_path}")

        with open(ssmd_path, encoding="utf-8") as f:
            content = f.read()

        return content, _hash_content(content)
    except SSMDGenerationError:
        raise
    except Exception as e:
        raise SSMDGenerationError(
            f"Failed to load SSMD file from {ssmd_path}: {str(e)}"
        ) from e


