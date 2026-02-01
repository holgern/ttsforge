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

``phonemization_lang``
   Override language for phonemization (e.g., ``de``, ``fr``, ``en-us``).

   - Type: string or null
   - Default: ``None``
   - Example: ``ttsforge config --set phonemization_lang de``

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

``model_quality``
   ONNX model quality/quantization.

   - Type: string
   - Default: ``fp32``
   - Choices: ``fp32``, ``fp16``, ``q8``, ``q8f16``, ``q4``, ``q4f16``, ``uint8``, ``uint8f16``
   - Example: ``ttsforge config --set model_quality fp16``

``model_variant``
   Model variant to download.

   - Type: string
   - Default: ``v1.0``
   - Choices: ``v1.0``, ``v1.1-zh``, ``v1.1-de``
   - Example: ``ttsforge config --set model_variant v1.1-de``

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

Read Settings
^^^^^^^^^^^^^

``default_content_mode``
   Default content mode for ``read`` (``chapters`` or ``pages``).

   - Type: string
   - Default: ``chapters``
   - Example: ``ttsforge config --set default_content_mode pages``

``default_page_size``
   Synthetic page size in characters for ``read`` pages mode.

   - Type: integer
   - Default: ``2000``
   - Example: ``ttsforge config --set default_page_size 2500``

Mixed-Language Settings
^^^^^^^^^^^^^^^^^^^^^^^

``use_mixed_language``
   Enable automatic detection and handling of multiple languages in text.

   - Type: boolean
   - Default: ``false``
   - Requires: ``lingua-language-detector`` package (``pip install lingua-language-detector``)
   - Example: ``ttsforge config --set use_mixed_language true``

``mixed_language_primary``
   Primary/fallback language for mixed-language mode.

   - Type: string or null
   - Default: ``None``
   - Supported: ``en-us``, ``en-gb``, ``de``, ``fr-fr``, ``es``, ``it``, ``pt``, ``pl``, ``tr``, ``ru``, ``ko``, ``ja``, ``zh``/``cmn``
   - Example: ``ttsforge config --set mixed_language_primary de``

``mixed_language_allowed``
   List of languages allowed for auto-detection in mixed-language mode.

   - Type: list of strings or null
   - Default: ``None``
   - Required when ``use_mixed_language`` is enabled
   - Example: ``ttsforge config --set mixed_language_allowed "['de', 'en-us']"``

``mixed_language_confidence``
   Confidence threshold for language detection (0.0-1.0).

   - Type: float
   - Default: ``0.7``
   - Range: ``0.0`` to ``1.0``
   - Higher values require more confidence before switching languages
   - Example: ``ttsforge config --set mixed_language_confidence 0.8``

Audio Timing Settings
^^^^^^^^^^^^^^^^^^^^^

``silence_between_chapters``
   Silence duration between chapters in seconds.

   - Type: float
   - Default: ``2.0``
   - Example: ``ttsforge config --set silence_between_chapters 3.0``

``pause_clause``
   Pause after clauses in seconds.

   - Type: float
   - Default: ``0.5``
   - Example: ``ttsforge config --set pause_clause 0.4``

``pause_sentence``
   Pause after sentences in seconds.

   - Type: float
   - Default: ``0.7``
   - Example: ``ttsforge config --set pause_sentence 0.6``

``pause_paragraph``
   Pause after paragraphs in seconds.

   - Type: float
   - Default: ``0.9``
   - Example: ``ttsforge config --set pause_paragraph 1.1``

``pause_variance``
   Random variance added to pause durations in seconds.

   - Type: float
   - Default: ``0.05``
   - Example: ``ttsforge config --set pause_variance 0.08``

``pause_mode``
   Pause mode: ``tts``, ``manual``, or ``auto``.

   - Type: string
   - Default: ``auto``
   - Example: ``ttsforge config --set pause_mode manual``

Chapter Announcement Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``announce_chapters``
   Read chapter titles aloud before chapter content.

   - Type: boolean
   - Default: ``true``
   - Example: ``ttsforge config --set announce_chapters false``

``chapter_pause_after_title``
   Pause duration after the chapter title announcement in seconds.

   - Type: float
   - Default: ``2.0``
   - Example: ``ttsforge config --set chapter_pause_after_title 1.5``

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
   * - ``model_quality``
     - string
     - ``fp32``
     - Model quality/quantization
   * - ``model_variant``
     - string
     - ``v1.0``
     - Model variant
   * - ``silence_between_chapters``
     - float
     - ``2.0``
     - Silence between chapters (seconds)
   * - ``pause_clause``
     - float
     - ``0.5``
     - Clause pause (seconds)
   * - ``pause_sentence``
     - float
     - ``0.7``
     - Sentence pause (seconds)
   * - ``pause_paragraph``
     - float
     - ``0.9``
     - Paragraph pause (seconds)
   * - ``pause_variance``
     - float
     - ``0.05``
     - Pause variance (seconds)
   * - ``pause_mode``
     - string
     - ``auto``
     - Pause mode (tts/manual/auto)
   * - ``announce_chapters``
     - boolean
     - ``true``
     - Speak chapter titles
   * - ``chapter_pause_after_title``
     - float
     - ``2.0``
     - Pause after chapter titles (seconds)
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
   * - ``phonemization_lang``
     - string/null
     - ``None``
     - Override phonemization language
   * - ``default_split_mode``
     - string
     - ``auto``
     - Text splitting mode
   * - ``default_content_mode``
     - string
     - ``chapters``
     - Default read mode (chapters/pages)
   * - ``default_page_size``
     - integer
     - ``2000``
     - Page size for read pages mode
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
   * - ``use_mixed_language``
     - boolean
     - ``false``
     - Enable mixed-language mode
   * - ``mixed_language_primary``
     - string/null
     - ``None``
     - Primary language for mixed mode
   * - ``mixed_language_allowed``
     - list/null
     - ``None``
     - Allowed languages list
   * - ``mixed_language_confidence``
     - float
     - ``0.7``
     - Language detection threshold


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
     "model_quality": "fp32",
     "model_variant": "v1.0",
      "silence_between_chapters": 2.5,
     "pause_clause": 0.5,
     "pause_sentence": 0.7,
     "pause_paragraph": 0.9,
     "pause_variance": 0.05,
     "pause_mode": "auto",
     "announce_chapters": true,
     "chapter_pause_after_title": 2.0,
      "save_chapters_separately": false,
      "merge_at_end": true,
      "auto_detect_language": true,
     "phonemization_lang": null,
      "default_split_mode": "sentence",
     "default_content_mode": "chapters",
     "default_page_size": 2000,
      "output_filename_template": "{author} - {book_title}",
      "chapter_filename_template": "{chapter_num:03d}_{chapter_title}",
      "phoneme_export_template": "{book_title}",
     "default_title": "Untitled",
     "use_mixed_language": false,
     "mixed_language_primary": null,
     "mixed_language_allowed": null,
     "mixed_language_confidence": 0.7
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
