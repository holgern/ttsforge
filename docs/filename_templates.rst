Filename Templates
==================

ttsforge uses configurable filename templates to name output files based on book 
metadata. This allows you to organize your audiobook library with consistent naming.


Template Syntax
---------------

Templates use Python's format string syntax with curly braces for variable 
substitution:

.. code-block:: text

   {variable_name}
   {variable_name:format_spec}

For example:

- ``{book_title}`` → ``My Great Novel``
- ``{chapter_num:03d}`` → ``001`` (zero-padded to 3 digits)


Available Variables
-------------------

The following variables are available in filename templates:

``{book_title}``
   The title of the book from EPUB metadata, or the configured ``default_title``
   if no title is found.

   Example: ``Empire in Black and Gold``

``{author}``
   The author name from EPUB metadata, or "Unknown" if not found.

   Example: ``Adrian Tchaikovsky``

``{chapter_title}``
   The title of the current chapter (only available in chapter filename templates).

   Example: ``Chapter 1 - The Beginning``

``{chapter_num}``
   The chapter number (1-based). Supports format specifiers for padding.

   - ``{chapter_num}`` → ``1``
   - ``{chapter_num:02d}`` → ``01``
   - ``{chapter_num:03d}`` → ``001``

``{input_stem}``
   The input filename without extension (useful for maintaining original naming).

   Example: If input is ``my_book.epub``, this gives ``my_book``

``{chapters_range}``
   A string representing the chapter selection, or empty if all chapters are selected.

   - Single chapter: ``chapter_1``
   - Range: ``chapters_1-5``
   - Multiple: ``chapters_1-3_5_7-10``


Template Types
--------------

ttsforge uses three different filename templates for different purposes:

Output Filename Template
^^^^^^^^^^^^^^^^^^^^^^^^

Controls the name of the final audiobook file.

**Config key:** ``output_filename_template``

**Default:** ``{book_title}``

**Used by:**
- ``ttsforge convert`` command
- ``ttsforge phonemes convert`` command

**Example:**

.. code-block:: bash

   # Set template
   ttsforge config --set output_filename_template "{author} - {book_title}"

   # Convert
   ttsforge convert book.epub

   # Output: "Adrian Tchaikovsky - Empire in Black and Gold.m4b"

Chapter Filename Template
^^^^^^^^^^^^^^^^^^^^^^^^^

Controls the names of intermediate chapter WAV files created during conversion.

**Config key:** ``chapter_filename_template``

**Default:** ``{chapter_num:03d}_{book_title}_{chapter_title}``

**Used by:**
- ``ttsforge convert`` command (chapter files in work directory)
- ``ttsforge phonemes convert`` command

**Example:**

.. code-block:: bash

   # Set template
   ttsforge config --set chapter_filename_template "{chapter_num:03d}_{chapter_title}"

   # Chapters will be named like:
   # 001_Prologue.wav
   # 002_The Beginning.wav
   # 003_Dark Times.wav

Phoneme Export Template
^^^^^^^^^^^^^^^^^^^^^^^

Controls the name of phoneme JSON files created during export.

**Config key:** ``phoneme_export_template``

**Default:** ``{book_title}``

**Used by:**
- ``ttsforge phonemes export`` command

**Example:**

.. code-block:: bash

   # Set template
   ttsforge config --set phoneme_export_template "{book_title}_phonemes"

   # Export
   ttsforge phonemes export book.epub

   # Output: "Empire in Black and Gold_phonemes.phonemes.json"


Format Specifiers
-----------------

Variables can include format specifiers after a colon:

Number Formatting
^^^^^^^^^^^^^^^^^

.. code-block:: text

   {chapter_num:d}     → "1"          (integer)
   {chapter_num:02d}   → "01"         (zero-padded, 2 digits)
   {chapter_num:03d}   → "001"        (zero-padded, 3 digits)
   {chapter_num:04d}   → "0001"       (zero-padded, 4 digits)

String Formatting
^^^^^^^^^^^^^^^^^

.. code-block:: text

   {book_title}        → "My Book"    (full string)
   {book_title:.20}    → "My Book"    (max 20 chars, truncated if longer)


Filename Sanitization
---------------------

All template values are automatically sanitized to be safe for filenames:

- Invalid characters (``/\:*?"<>|``) are replaced with underscores
- Leading/trailing whitespace and dots are removed
- Multiple consecutive underscores are collapsed to single underscore

For example:

- ``"Book: The Story"`` → ``"Book_ The Story"``
- ``"What?"`` → ``"What_"``


Partial Chapter Selections
--------------------------

When converting a subset of chapters, the ``{chapters_range}`` variable contains
the selection, and it's automatically appended to output filenames:

.. code-block:: bash

   # Convert chapters 1-5
   ttsforge convert book.epub --chapters 1-5

   # Output: "Empire in Black and Gold_chapters_1-5.m4b"

This ensures partial conversions don't overwrite complete audiobooks.


Examples
--------

Default Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Default templates
   output_filename_template = "{book_title}"
   chapter_filename_template = "{chapter_num:03d}_{book_title}_{chapter_title}"

   # Input: "Shadows of the Apt - Book 1.epub"
   # Output: "Empire in Black and Gold.m4b"
   # Chapters: "001_Empire in Black and Gold_Prologue.wav", etc.

Author-First Naming
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Set templates
   ttsforge config --set output_filename_template "{author} - {book_title}"

   # Input: "shadows.epub" (metadata: Author="Adrian Tchaikovsky", Title="Empire in Black and Gold")
   # Output: "Adrian Tchaikovsky - Empire in Black and Gold.m4b"

Preserve Original Filename
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Use input filename
   ttsforge config --set output_filename_template "{input_stem}"

   # Input: "my_audiobook.epub"
   # Output: "my_audiobook.m4b"

Series Naming
^^^^^^^^^^^^^

For books in a series, you might use the input filename if it contains series info:

.. code-block:: bash

   ttsforge config --set output_filename_template "{author} - {input_stem}"

   # Input: "01 - Empire in Black and Gold.epub"
   # Output: "Adrian Tchaikovsky - 01 - Empire in Black and Gold.m4b"

Simple Chapter Names
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ttsforge config --set chapter_filename_template "{chapter_num:03d}"

   # Chapters: "001.wav", "002.wav", "003.wav", etc.


Work Directory
--------------

During conversion, ttsforge creates a hidden work directory to store chapter files 
and state information:

.. code-block:: text

   .{book_title}_chapters/
   ├── 001_Empire in Black and Gold_Prologue.wav
   ├── 002_Empire in Black and Gold_Chapter 1.wav
   ├── ...
   └── Empire in Black and Gold_state.json

The work directory is named after the book title and is cleaned up after successful
conversion (unless ``--keep-chapters`` is used).


Troubleshooting
---------------

Filename too long
^^^^^^^^^^^^^^^^^

If output filenames are too long for your filesystem:

1. Use shorter templates: ``{input_stem}`` instead of ``{author} - {book_title}``
2. Truncate long values: ``{book_title:.50}`` limits title to 50 characters
3. Use simpler chapter templates: ``{chapter_num:03d}``

Special characters in titles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Characters that are invalid in filenames are automatically replaced with underscores.
If you see unexpected underscores in filenames, check the original EPUB metadata
for special characters.

Duplicate filenames
^^^^^^^^^^^^^^^^^^^

If converting multiple books with the same title, use templates that include
unique identifiers:

.. code-block:: bash

   ttsforge config --set output_filename_template "{author} - {book_title}"
   # Or use input stem:
   ttsforge config --set output_filename_template "{input_stem}"
