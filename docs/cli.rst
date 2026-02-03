CLI Reference
=============

Complete command-line interface reference for ttsforge.


Global Options
--------------

.. code-block:: bash

   ttsforge --version    # Show version and exit
   ttsforge --help       # Show help message


convert
-------

Convert an EPUB file to an audiobook.

.. code-block:: bash

   ttsforge convert EPUB_FILE [OPTIONS]

Arguments
^^^^^^^^^

``EPUB_FILE``
   Path to the EPUB file to convert (required).

Options
^^^^^^^

``-o, --output PATH``
   Output file path. Defaults to input filename with new extension in the same directory.

``-f, --format FORMAT``
   Output audio format. Choices: ``wav``, ``mp3``, ``flac``, ``opus``, ``m4b``.
   Default: ``m4b``.

``-v, --voice VOICE``
   Voice to use for TTS. Can be a single voice name or a voice blend.

   - Single voice: ``af_heart``, ``am_adam``, etc.
   - Voice blend: ``af_nicole:50,am_michael:50`` (auto-detects blend format)

   See :doc:`voices` for available voices.
   Default: ``af_heart``.

``-l, --language LANG``
   Language code for TTS. Choices: ``a`` (American English), ``b`` (British English),
   ``e`` (Spanish), ``f`` (French), ``h`` (Hindi), ``i`` (Italian), ``j`` (Japanese),
   ``p`` (Brazilian Portuguese), ``z`` (Mandarin Chinese).
   Default: auto-detected from EPUB metadata.

``--lang LANG``
   Override language for phonemization (e.g., ``de``, ``fr``, ``en-us``).
   By default, language is determined from the voice.

``-s, --speed FLOAT``
   Speech speed multiplier (0.5 to 2.0). Default: ``1.0``.

``--gpu / --no-gpu``
   Enable or disable GPU acceleration.

``--chapters SELECTION``
   Chapters to convert. Examples: ``1-5``, ``1,3,5``, ``1-3,5,7-10``, ``all``.
   Default: all chapters (interactive selection if not specified).

``--silence FLOAT``
   Silence duration between chapters in seconds. Default: ``2.0``.

``--pause-clause FLOAT``
   Pause after clauses in seconds. Default: ``0.5``.

``--pause-sentence FLOAT``
   Pause after sentences in seconds. Default: ``0.7``.

``--pause-paragraph FLOAT``
   Pause after paragraphs in seconds. Default: ``0.9``.

``--pause-variance FLOAT``
   Random variance added to pauses in seconds. Default: ``0.05``.

``--pause-mode MODE``
   Pause mode: ``tts``, ``manual``, or ``auto``. Default: ``auto``.

``--enable-short-sentence``
   Enable special handling for short sentences (less than 5 words).

``--announce-chapters / --no-announce-chapters``
   Read chapter titles aloud before chapter content. Default: enabled.

``--chapter-pause FLOAT``
   Pause after chapter title announcement in seconds. Default: ``2.0``.

``--title TEXT``
   Title metadata for the audiobook. Defaults to EPUB title.

``--author TEXT``
   Author metadata for the audiobook. Defaults to EPUB author.

``--cover PATH``
   Cover image for M4B format.

``-y, --yes``
   Skip confirmation prompts.

``--verbose``
   Show detailed output during conversion.

``--split-mode MODE``
   Text splitting mode. Choices: ``auto``, ``line``, ``paragraph``, ``sentence``, ``clause``.
   Default: ``auto``.

``--resume / --no-resume``
   Enable or disable resume capability. Default: enabled.

``--fresh``
   Discard any previous progress and start conversion from scratch.

``--generate-ssmd``
   Generate only SSMD files without creating audio (for manual editing).

``--detect-emphasis / --no-detect-emphasis``
   Detect emphasis (italic/bold) from EPUB HTML. Default: disabled.

``--keep-chapters``
   Keep individual chapter audio files after conversion.

``--voice-blend SPEC``
   Blend multiple voices (traditional method). Format: ``voice1:weight1,voice2:weight2``.
   Example: ``af_nicole:50,am_michael:50``.

   **Note:** You can also specify blends directly in the ``--voice`` parameter,
   which will auto-detect the blend format. Both methods work identically.

``--voice-db PATH``
   Path to custom voice database (SQLite).

``--phoneme-dict PATH``
   Path to custom phoneme dictionary JSON file for pronunciation overrides.

``--phoneme-dict-case-sensitive``
   Make phoneme dictionary matching case-sensitive (default: case-insensitive).

``--use-mixed-language``
   Enable mixed-language support (auto-detect multiple languages in text).

``--mixed-language-primary LANG``
   Primary language for mixed-language mode (e.g., ``de``, ``en-us``).
   This language is used as the fallback when detection is uncertain.

``--mixed-language-allowed LANGS``
   Comma-separated list of allowed languages for detection (e.g., ``de,en-us``).
   Required when ``--use-mixed-language`` is enabled.

``--mixed-language-confidence FLOAT``
   Detection confidence threshold for mixed-language mode (0.0-1.0).
   Default: ``0.7``. Higher values require more confidence for language switches.

Examples
^^^^^^^^

.. code-block:: bash

   # Basic conversion
   ttsforge convert book.epub

   # Convert with specific voice and speed
   ttsforge convert book.epub -v am_adam -s 1.1

   # Convert chapters 1-5 to MP3
   ttsforge convert book.epub --chapters 1-5 -f mp3

   # Full options
   ttsforge convert book.epub \
       --voice af_sarah \
       --speed 1.1 \
       --format m4b \
       --title "My Audiobook" \
       --author "Author Name" \
       --cover cover.jpg \
       --output ./audiobooks/mybook.m4b \
       --yes

   # Resume interrupted conversion
   ttsforge convert book.epub

   # Start fresh (discard progress)
   ttsforge convert book.epub --fresh

   # Mixed-language conversion (German with English terms)
   ttsforge convert book.epub \
       --use-mixed-language \
       --mixed-language-primary de \
       --mixed-language-allowed de,en-us


list
----

List chapters in an EPUB file.

.. code-block:: bash

   ttsforge list EPUB_FILE

Arguments
^^^^^^^^^

``EPUB_FILE``
   Path to the EPUB file (required).

Example
^^^^^^^

.. code-block:: bash

   ttsforge list book.epub

Output shows chapter numbers, titles, and character counts.


info
----

Show metadata and information about an EPUB file.

.. code-block:: bash

   ttsforge info EPUB_FILE

Arguments
^^^^^^^^^

``EPUB_FILE``
   Path to the EPUB file (required).

Example
^^^^^^^

.. code-block:: bash

   ttsforge info book.epub

Shows title, author, language, publisher, year, chapter count, and file size.


sample
------

Generate a sample audio file to test TTS settings.

.. code-block:: bash

   ttsforge sample [TEXT] [OPTIONS]

Arguments
^^^^^^^^^

``TEXT``
   Text to convert. If not provided, uses default sample text.

Options
^^^^^^^

``-o, --output PATH``
   Output file path. Default: ``./sample.wav``.

``-f, --format FORMAT``
   Output audio format. Default: ``wav``.

``-v, --voice VOICE``
   TTS voice to use. Can be a single voice or voice blend.

   - Single voice: ``af_heart``
   - Voice blend: ``af_nicole:50,am_michael:50`` (auto-detects blend format)

``-l, --language LANG``
   Language for TTS.

``--lang LANG``
   Override language for phonemization (e.g., ``de``, ``fr``, ``en-us``).

``-s, --speed FLOAT``
   Speech speed. Default: ``1.0``.

``--gpu / --no-gpu``
   Enable or disable GPU acceleration.

``--split-mode MODE``
   Text splitting mode.

``--verbose``
   Show detailed output.

``-p, --play``
   Play audio directly (also saves to file if ``-o`` specified).

   **Note:** Playback requires the optional ``ttsforge[audio]`` extra.

``--use-mixed-language``
   Enable mixed-language support (auto-detect multiple languages in text).

``--mixed-language-primary LANG``
   Primary language for mixed-language mode (e.g., ``de``, ``en-us``).

``--mixed-language-allowed LANGS``
   Comma-separated list of allowed languages (e.g., ``de,en-us``).

``--mixed-language-confidence FLOAT``
   Detection confidence threshold (0.0-1.0). Default: ``0.7``.

``--phoneme-dict PATH``
   Path to custom phoneme dictionary JSON file for pronunciation overrides.

``--phoneme-dict-case-sensitive``
   Make phoneme dictionary matching case-sensitive (default: case-insensitive).

Examples
^^^^^^^^

.. code-block:: bash

   # Default sample
   ttsforge sample

   # Custom text
   ttsforge sample "Hello, this is a test."

   # With voice and output options
   ttsforge sample "Testing voice" --voice am_adam -o test.wav

   # Mixed-language sample
   ttsforge sample \
      "Das ist ein Test. This is a test." \
      --use-mixed-language \
      --mixed-language-primary de \
      --mixed-language-allowed de,en-us


read
----

Stream playback from an EPUB or text file (no output files).

.. code-block:: bash

   ttsforge read [INPUT_FILE] [OPTIONS]

Arguments
^^^^^^^^^

``INPUT_FILE``
   Path to EPUB/TXT file, or ``-`` to read from stdin. If omitted, reads stdin.

Options
^^^^^^^

``-v, --voice VOICE``
   TTS voice to use.

``-l, --language LANG``
   Language for TTS.

``-s, --speed FLOAT``
   Speech speed. Default: ``1.0``.

``--gpu / --no-gpu``
   Enable or disable GPU acceleration.

``--mode MODE``
   Content mode: ``chapters`` or ``pages``.

``-c, --chapters SELECTION``
   Chapter selection for ``chapters`` mode.

``-p, --pages SELECTION``
   Page selection for ``pages`` mode.

``--start-chapter INT``
   Start from specific chapter number (1-indexed).

``--start-page INT``
   Start from specific page number (1-indexed).

``--page-size INT``
   Synthetic page size in characters (default: 2000).

``--resume``
   Resume from last saved position.

``--list``
   List chapters/pages and exit without reading.

``--split MODE``
   Text splitting mode: ``sentence`` or ``paragraph``.

``--pause-clause FLOAT``
   Pause after clauses in seconds.

``--pause-sentence FLOAT``
   Pause after sentences in seconds.

``--pause-paragraph FLOAT``
   Pause after paragraphs in seconds.

``--pause-variance FLOAT``
   Random variance added to pauses in seconds.

``--pause-mode MODE``
   Pause mode: ``tts``, ``manual``, or ``auto``.

**Note:** Playback requires the optional ``ttsforge[audio]`` extra.

Examples
^^^^^^^^

.. code-block:: bash

   # Read an EPUB aloud
   ttsforge read book.epub

   # Read pages 1-10
   ttsforge read book.epub --mode pages --pages 1-10

   # Resume from last position
   ttsforge read book.epub --resume


voices
------

List available TTS voices.

.. code-block:: bash

   ttsforge voices [OPTIONS]

Options
^^^^^^^

``-l, --language LANG``
   Filter voices by language code.

Examples
^^^^^^^^

.. code-block:: bash

   # List all voices
   ttsforge voices

   # List American English voices
   ttsforge voices -l a

   # List British English voices
   ttsforge voices -l b


demo
----

Generate a demo audio file with voice samples.

.. code-block:: bash

   ttsforge demo [OPTIONS]

Options
^^^^^^^

``-o, --output PATH``
   Output file path. Default: ``./voices_demo.wav`` (or directory with ``--separate``).

``-l, --language LANG``
   Filter voices by language.

``-v, --voice VOICES``
   Specific voices to include (comma-separated).
   Example: ``af_heart,am_adam``.

``-s, --speed FLOAT``
   Speech speed. Default: ``1.0``.

``--gpu / --no-gpu``
   Enable or disable GPU acceleration.

``--silence FLOAT``
   Silence between voice samples in seconds. Default: ``0.5``.

``--text TEXT``
   Custom text to use. Use ``{voice}`` placeholder for voice name.

``--separate``
   Save each voice as a separate file instead of concatenating.

``--blend SPEC``
   Voice blend to demo (e.g., ``af_nicole:50,am_michael:50``).

``--blend-presets``
   Demo a curated set of voice blend combinations.

``-p, --play``
   Play audio directly instead of only saving files.

   **Note:** Playback requires the optional ``ttsforge[audio]`` extra.

Examples
^^^^^^^^

.. code-block:: bash

   # Demo all voices
   ttsforge demo

   # Demo American English voices only
   ttsforge demo -l a

   # Demo specific voices
   ttsforge demo -v af_heart,am_adam,bf_emma

   # Save separate files
   ttsforge demo --separate -o ./voice_samples/

   # Custom demo text
   ttsforge demo --text "Hi, I'm {voice}. Nice to meet you!"


download
--------

Download ONNX model files required for TTS.

.. code-block:: bash

   ttsforge download [OPTIONS]

Options
^^^^^^^

``--force``
   Force re-download even if files exist.

Examples
^^^^^^^^

.. code-block:: bash

   # Download models
   ttsforge download

   # Force re-download
   ttsforge download --force


config
------

Manage ttsforge configuration.

.. code-block:: bash

   ttsforge config [OPTIONS]

Configuration is stored in ``~/.config/ttsforge/config.json``.

Options
^^^^^^^

``--show``
   Show current configuration.

``--reset``
   Reset configuration to defaults.

``--set KEY VALUE``
   Set a configuration option. Can be used multiple times.

Examples
^^^^^^^^

.. code-block:: bash

   # Show configuration
   ttsforge config --show

   # Set default voice
   ttsforge config --set default_voice am_adam

   # Set multiple options
   ttsforge config --set default_voice af_sarah --set default_speed 1.1

   # Enable GPU
   ttsforge config --set use_gpu true

   # Reset to defaults
   ttsforge config --reset

See :doc:`configuration` for all available options.


phonemes
--------

Commands for working with phonemes and pre-tokenized content.

phonemes export
^^^^^^^^^^^^^^^

Export an EPUB as pre-tokenized phoneme data.

.. code-block:: bash

   ttsforge phonemes export EPUB_FILE [OPTIONS]

Arguments
"""""""""

``EPUB_FILE``
   Path to the EPUB file (required).

Options
"""""""

``-o, --output PATH``
   Output file path. Default: input filename with ``.phonemes.json``.

``--readable``
   Export as human-readable text format instead of JSON.

``-l, --language LANG``
   Language code for phonemization. Default: ``a``.

``--chapters SELECTION``
   Chapters to export.

``--vocab-version VERSION``
   Vocabulary version. Default: ``v1.0``.

``--split-mode MODE``
   Split mode: ``paragraph``, ``sentence``, or ``clause``. Default: ``sentence``.

``--max-chars INT``
   Maximum characters per segment. Default: ``300``.

Examples
""""""""

.. code-block:: bash

   # Export to phonemes
   ttsforge phonemes export book.epub

   # Export as readable format
   ttsforge phonemes export book.epub --readable -o book.readable.txt

   # Export specific chapters
   ttsforge phonemes export book.epub --chapters 1-5

   # Use clause splitting for shorter segments
   ttsforge phonemes export book.epub --split-mode clause

phonemes convert
^^^^^^^^^^^^^^^^

Convert a pre-tokenized phoneme file to audio.

.. code-block:: bash

   ttsforge phonemes convert PHONEME_FILE [OPTIONS]

Arguments
"""""""""

``PHONEME_FILE``
   Path to the phoneme JSON file (required).

Options
"""""""

``-o, --output PATH``
   Output file path.

``-f, --format FORMAT``
   Output audio format.

``-v, --voice VOICE``
   Voice to use for TTS.

``-s, --speed FLOAT``
   Speech speed. Default: ``1.0``.

``--gpu / --no-gpu``
   Enable or disable GPU acceleration.

``--silence FLOAT``
   Silence between chapters. Default: ``2.0``.

``--pause-clause FLOAT``
   Pause after clauses in seconds. Default: ``0.5``.

``--pause-sentence FLOAT``
   Pause after sentences in seconds. Default: ``0.7``.

``--pause-paragraph FLOAT``
   Pause after paragraphs in seconds. Default: ``0.9``.

``--pause-variance FLOAT``
   Random variance added to pauses in seconds. Default: ``0.05``.

``--pause-mode MODE``
   Pause mode: ``tts``, ``manual``, or ``auto``. Default: ``auto``.

``--announce-chapters / --no-announce-chapters``
   Read chapter titles aloud before chapter content. Default: enabled.

``--chapter-pause FLOAT``
   Pause after chapter title announcement in seconds. Default: ``2.0``.

``--chapters SELECTION``
   Select chapters to convert.

``--title TEXT``
   Audiobook title.

``--author TEXT``
   Audiobook author.

``--cover PATH``
   Cover image path.

``--voice-blend SPEC``
   Blend multiple voices.

``--voice-database PATH``
   Path to custom voice database.

``--streaming / --no-streaming``
   Use streaming mode (faster, no resume). Default: resumable.

``--keep-chapters``
   Keep intermediate chapter files.

``-y, --yes``
   Skip confirmation prompts.

Examples
""""""""

.. code-block:: bash

   # Convert phoneme file
   ttsforge phonemes convert book.phonemes.json

   # With voice and output
   ttsforge phonemes convert book.phonemes.json -v am_adam -o book.m4b

   # Streaming mode (faster but no resume)
   ttsforge phonemes convert book.phonemes.json --streaming

phonemes info
^^^^^^^^^^^^^

Show information about a phoneme file.

.. code-block:: bash

   ttsforge phonemes info PHONEME_FILE [OPTIONS]

Options
"""""""

``--stats``
   Show detailed token statistics.

Examples
""""""""

.. code-block:: bash

   # Basic info
   ttsforge phonemes info book.phonemes.json

   # With statistics
   ttsforge phonemes info book.phonemes.json --stats

phonemes preview
^^^^^^^^^^^^^^^^

Preview phonemes for given text.

.. code-block:: bash

   ttsforge phonemes preview TEXT [OPTIONS]

Options
"""""""

``-l, --language LANG``
   Language code for phonemization. Default: ``a``.

``-v, --voice VOICE``
   Voice to use for audio preview (when using ``--play``).
   Can be a single voice or voice blend (e.g., ``af_nicole:50,am_michael:50``).

``--play``
   Generate and play audio preview of the phonemes.

   **Note:** Playback requires the optional ``ttsforge[audio]`` extra.

``--tokens``
   Show token IDs in addition to phonemes.

``--vocab-version VERSION``
   Vocabulary version. Default: ``v1.0``.

Examples
""""""""

.. code-block:: bash

   # Preview phonemes
   ttsforge phonemes preview "Hello, world!"

   # With tokens
   ttsforge phonemes preview "Hello, world!" --tokens

   # Different language
   ttsforge phonemes preview "Bonjour!" -l f

   # With audio playback
   ttsforge phonemes preview "Test audio" --play

   # With voice blend
   ttsforge phonemes preview "Test blend" --voice "af_nicole:60,am_michael:40" --play
