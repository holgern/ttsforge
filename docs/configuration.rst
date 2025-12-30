Configuration
=============

ttsforge stores its configuration in a JSON file and provides a CLI interface for
managing settings.


Configuration File Location
---------------------------

The configuration file is stored at:

- **Linux**: ``~/.config/ttsforge/config.json``
- **macOS**: ``~/Library/Application Support/ttsforge/config.json``
- **Windows**: ``%APPDATA%\ttsforge\config.json``


Managing Configuration
----------------------

View current configuration:

.. code-block:: bash

   ttsforge config --show

Set a configuration option:

.. code-block:: bash

   ttsforge config --set KEY VALUE

Set multiple options:

.. code-block:: bash

   ttsforge config --set default_voice am_adam --set default_speed 1.1

Reset to defaults:

.. code-block:: bash

   ttsforge config --reset


Configuration Options
---------------------

Voice and Language Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``default_voice``
   Default TTS voice to use.

   - Type: string
   - Default: ``af_heart``
   - Example: ``ttsforge config --set default_voice am_adam``

``default_language``
   Default language code.

   - Type: string
   - Default: ``a`` (American English)
   - Choices: ``a``, ``b``, ``e``, ``f``, ``h``, ``i``, ``j``, ``p``, ``z``
   - Example: ``ttsforge config --set default_language b``

``default_speed``
   Default speech speed multiplier.

   - Type: float
   - Default: ``1.0``
   - Range: ``0.5`` to ``2.0``
   - Example: ``ttsforge config --set default_speed 1.1``

Output Settings
^^^^^^^^^^^^^^^

``default_format``
   Default output audio format.

   - Type: string
   - Default: ``m4b``
   - Choices: ``wav``, ``mp3``, ``flac``, ``opus``, ``m4b``
   - Example: ``ttsforge config --set default_format mp3``

Processing Settings
^^^^^^^^^^^^^^^^^^^

``use_gpu``
   Enable GPU acceleration for TTS inference.

   - Type: boolean
   - Default: ``false``
   - Requires: ``onnxruntime-gpu`` package
   - Example: ``ttsforge config --set use_gpu true``

``auto_detect_language``
   Automatically detect language from EPUB metadata.

   - Type: boolean
   - Default: ``true``
   - Example: ``ttsforge config --set auto_detect_language false``

``default_split_mode``
   Default text splitting mode for processing.

   - Type: string
   - Default: ``auto``
   - Choices: ``auto``, ``line``, ``paragraph``, ``sentence``, ``clause``
   - Example: ``ttsforge config --set default_split_mode sentence``

Audio Timing Settings
^^^^^^^^^^^^^^^^^^^^^

``silence_between_chapters``
   Silence duration between chapters in seconds.

   - Type: float
   - Default: ``2.0``
   - Example: ``ttsforge config --set silence_between_chapters 3.0``

``segment_pause_min``
   Minimum pause between segments (sentences) in seconds.

   - Type: float
   - Default: ``0.1``
   - Example: ``ttsforge config --set segment_pause_min 0.15``

``segment_pause_max``
   Maximum pause between segments (sentences) in seconds.

   - Type: float
   - Default: ``0.3``
   - Example: ``ttsforge config --set segment_pause_max 0.5``

``paragraph_pause_min``
   Minimum pause between paragraphs in seconds. Paragraph pauses are typically
   longer than sentence pauses to create natural breaks in the audio.

   - Type: float
   - Default: ``0.5``
   - Example: ``ttsforge config --set paragraph_pause_min 0.6``

``paragraph_pause_max``
   Maximum pause between paragraphs in seconds.

   - Type: float
   - Default: ``1.0``
   - Example: ``ttsforge config --set paragraph_pause_max 1.5``

File Output Settings
^^^^^^^^^^^^^^^^^^^^

``save_chapters_separately``
   Save individual chapter audio files.

   - Type: boolean
   - Default: ``false``
   - Example: ``ttsforge config --set save_chapters_separately true``

``merge_at_end``
   Merge chapter files into final audiobook.

   - Type: boolean
   - Default: ``true``
   - Example: ``ttsforge config --set merge_at_end false``

Filename Template Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^

These settings control how output files are named. See :doc:`filename_templates` for details.

``output_filename_template``
   Template for final audiobook filenames.

   - Type: string
   - Default: ``{book_title}``
   - Example: ``ttsforge config --set output_filename_template "{author}_{book_title}"``

``chapter_filename_template``
   Template for chapter WAV file names during conversion.

   - Type: string
   - Default: ``{chapter_num:03d}_{book_title}_{chapter_title}``
   - Example: ``ttsforge config --set chapter_filename_template "{chapter_num:03d}_{chapter_title}"``

``phoneme_export_template``
   Template for phoneme export filenames.

   - Type: string
   - Default: ``{book_title}``
   - Example: ``ttsforge config --set phoneme_export_template "{book_title}_phonemes"``

``default_title``
   Fallback title when book has no metadata.

   - Type: string
   - Default: ``Untitled``
   - Example: ``ttsforge config --set default_title "Unknown Book"``


Complete Configuration Reference
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - Option
     - Type
     - Default
     - Description
   * - ``default_voice``
     - string
     - ``af_heart``
     - Default TTS voice
   * - ``default_language``
     - string
     - ``a``
     - Default language code
   * - ``default_speed``
     - float
     - ``1.0``
     - Speech speed multiplier
   * - ``default_format``
     - string
     - ``m4b``
     - Output audio format
   * - ``use_gpu``
     - boolean
     - ``false``
     - Enable GPU acceleration
   * - ``silence_between_chapters``
     - float
     - ``2.0``
     - Silence between chapters (seconds)
   * - ``segment_pause_min``
     - float
     - ``0.1``
     - Minimum segment pause (seconds)
   * - ``segment_pause_max``
     - float
     - ``0.3``
     - Maximum segment pause (seconds)
   * - ``paragraph_pause_min``
     - float
     - ``0.5``
     - Minimum paragraph pause (seconds)
   * - ``paragraph_pause_max``
     - float
     - ``1.0``
     - Maximum paragraph pause (seconds)
   * - ``save_chapters_separately``
     - boolean
     - ``false``
     - Keep chapter audio files
   * - ``merge_at_end``
     - boolean
     - ``true``
     - Merge chapters into final file
   * - ``auto_detect_language``
     - boolean
     - ``true``
     - Auto-detect language from EPUB
   * - ``default_split_mode``
     - string
     - ``auto``
     - Text splitting mode
   * - ``output_filename_template``
     - string
     - ``{book_title}``
     - Output filename template
   * - ``chapter_filename_template``
     - string
     - ``{chapter_num:03d}_...``
     - Chapter filename template
   * - ``phoneme_export_template``
     - string
     - ``{book_title}``
     - Phoneme export template
   * - ``default_title``
     - string
     - ``Untitled``
     - Fallback title


Example Configuration File
--------------------------

Here's an example ``config.json`` with custom settings:

.. code-block:: json

   {
     "default_voice": "am_adam",
     "default_language": "a",
     "default_speed": 1.1,
     "default_format": "m4b",
     "use_gpu": true,
     "silence_between_chapters": 2.5,
     "segment_pause_min": 0.1,
     "segment_pause_max": 0.3,
     "paragraph_pause_min": 0.5,
     "paragraph_pause_max": 1.0,
     "save_chapters_separately": false,
     "merge_at_end": true,
     "auto_detect_language": true,
     "default_split_mode": "sentence",
     "output_filename_template": "{author} - {book_title}",
     "chapter_filename_template": "{chapter_num:03d}_{chapter_title}",
     "phoneme_export_template": "{book_title}",
     "default_title": "Untitled"
   }


Command-Line Override
---------------------

Configuration values can be overridden on the command line. Command-line options
take precedence over configuration file settings:

.. code-block:: bash

   # Use configured voice, but override speed
   ttsforge convert book.epub -s 1.2

   # Override voice and format
   ttsforge convert book.epub -v bf_emma -f mp3


Environment Variables
---------------------

ttsforge does not currently support environment variables for configuration.
Use the config file or command-line options instead.
