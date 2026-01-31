from __future__ import annotations


def parse_chapter_selection(selection: str, total_chapters: int) -> list[int]:
    """Parse chapter selection string into list of 0-based chapter indices.

    Supports formats like:
    - "3" -> [2] (single chapter, 1-based to 0-based)
    - "1-5" -> [0, 1, 2, 3, 4] (range, inclusive)
    - "3,5,7" -> [2, 4, 6] (comma-separated)
    - "1-3,7,9-10" -> [0, 1, 2, 6, 8, 9] (mixed)

    Args:
        selection: Chapter selection string (1-based indexing)
        total_chapters: Total number of chapters available

    Returns:
        List of 0-based chapter indices

    Raises:
        ValueError: If selection format is invalid or chapters out of range
    """
    indices: set[int] = set()

    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            # Range: "1-5"
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
            except ValueError as e:
                raise ValueError(f"Invalid range format: {part}") from e

            if start < 1 or end < 1:
                raise ValueError(f"Chapter numbers must be >= 1: {part}")
            if start > end:
                raise ValueError(f"Invalid range (start > end): {part}")
            if end > total_chapters:
                raise ValueError(
                    f"Chapter {end} exceeds total chapters ({total_chapters})"
                )

            # Convert to 0-based indices
            for i in range(start - 1, end):
                indices.add(i)
        else:
            # Single chapter: "3"
            try:
                chapter_num = int(part)
            except ValueError as e:
                raise ValueError(f"Invalid chapter number: {part}") from e

            if chapter_num < 1:
                raise ValueError(f"Chapter number must be >= 1: {chapter_num}")
            if chapter_num > total_chapters:
                raise ValueError(
                    f"Chapter {chapter_num} exceeds total chapters ({total_chapters})"
                )

            # Convert to 0-based index
            indices.add(chapter_num - 1)

    return sorted(indices)
