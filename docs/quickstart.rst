Quick Start Guide
=================

This guide will help you get started with ttsforge quickly.


Basic Conversion
----------------

Convert an EPUB file to an audiobook with default settings:

.. code-block:: bash

   ttsforge convert mybook.epub

This creates ``mybook.m4b`` in the same directory with:

- Default voice: ``af_heart`` (American English female)
- Default format: M4B (with chapter markers)
- Auto-detected language from EPUB metadata


Choosing a Voice
----------------

List available voices:

.. code-block:: bash

   ttsforge voices

List voices for a specific language:

.. code-block:: bash

   ttsforge voices -l a  # American English
   ttsforge voices -l b  # British English

Convert with a specific voice:

.. code-block:: bash

   ttsforge convert mybook.epub -v am_adam  # Male voice


Voice Blending
--------------

Mix multiple voices for unique narration by specifying voice blends in the ``--voice`` parameter:

.. code-block:: bash

   # 50/50 blend of two voices
   ttsforge sample "Hello world" --voice "af_nicole:50,am_michael:50" -p

   # Weighted blend (70% Nicole, 30% Michael)
   ttsforge convert mybook.epub --voice "af_nicole:70,am_michael:30"

   # Three-way blend
   ttsforge sample "Testing" --voice "af_sky:40,af_bella:30,am_adam:30" -p

The format is: ``voice1:weight1,voice2:weight2,...`` where weights are percentages (0-100).

You can also use the traditional ``--voice-blend`` parameter:

.. code-block:: bash

   ttsforge convert mybook.epub --voice-blend "af_nicole:50,am_michael:50"


Output Formats
--------------

ttsforge supports multiple audio formats:

.. code-block:: bash

   # M4B audiobook (default) - includes chapter markers
   ttsforge convert mybook.epub -f m4b

   # MP3
   ttsforge convert mybook.epub -f mp3

   # WAV (uncompressed)
   ttsforge convert mybook.epub -f wav

   # FLAC (lossless compression)
   ttsforge convert mybook.epub -f flac

   # OPUS (efficient compression)
   ttsforge convert mybook.epub -f opus


Converting Specific Chapters
----------------------------

Preview chapter list:

.. code-block:: bash

   ttsforge list mybook.epub

Convert specific chapters:

.. code-block:: bash

   # Convert chapters 1 through 5
   ttsforge convert mybook.epub --chapters 1-5

   # Convert specific chapters
   ttsforge convert mybook.epub --chapters 1,3,5,7

   # Mixed selection
   ttsforge convert mybook.epub --chapters 1-3,5,7-10


Speed Control
-------------

Adjust speech speed (0.5 to 2.0):

.. code-block:: bash

   # Faster
   ttsforge convert mybook.epub -s 1.2

   # Slower
   ttsforge convert mybook.epub -s 0.9


Resumable Conversions
---------------------

ttsforge automatically saves progress during conversion. If interrupted:

.. code-block:: bash

   # Simply re-run the same command
   ttsforge convert mybook.epub

   # Progress is resumed from the last completed chapter

To start fresh, discarding previous progress:

.. code-block:: bash

   ttsforge convert mybook.epub --fresh


Phoneme Pre-tokenization
------------------------

For large books or batch processing, pre-tokenize text to phonemes:

.. code-block:: bash

   # Step 1: Export to phonemes (fast, no TTS)
   ttsforge phonemes export mybook.epub -o mybook.phonemes.json

   # Step 2: Convert phonemes to audio (can be run on different machine)
   ttsforge phonemes convert mybook.phonemes.json -v af_heart

Benefits:

- Review phonemes before generating audio
- Faster repeated conversions (skip phonemization)
- Separate phonemization from audio generation


Testing TTS Settings
--------------------

Generate a sample to test your settings:

.. code-block:: bash

   # Default sample
   ttsforge sample

   # Custom text
   ttsforge sample "Hello, this is a test of the voice."

   # With specific voice and speed
   ttsforge sample --voice am_adam --speed 1.1

   # Play directly (requires audio extra)
   ttsforge sample --play


Streaming Read (Optional)
-------------------------

Listen to an EPUB or text file in real-time with the ``read`` command.
This requires the optional audio playback extra:

.. code-block:: bash

   pip install "ttsforge[audio]"

.. code-block:: bash

   # Read an EPUB aloud
   ttsforge read mybook.epub

   # Read a text file
   ttsforge read story.txt


Voice Demo
----------

Listen to all voices with a demo:

.. code-block:: bash

   # Demo all voices
   ttsforge demo

   # Demo voices for a specific language
   ttsforge demo -l a  # American English only

   # Save individual voice files
   ttsforge demo --separate -o ./voice_samples/


Mixed-Language Support
----------------------

For books containing multiple languages (e.g., German text with English technical terms),
ttsforge can automatically detect and handle different languages:

.. code-block:: bash

   # Convert a book with German and English text
   ttsforge convert mybook.epub \
       --use-mixed-language \
       --mixed-language-primary de \
       --mixed-language-allowed de,en-us

   # Test with a sample
   ttsforge sample \
       "Das ist ein deutscher Satz. This is an English sentence." \
       --use-mixed-language \
       --mixed-language-primary de \
       --mixed-language-allowed de,en-us

**Requirements**: Install the language detector:

.. code-block:: bash

   pip install lingua-language-detector

**Options**:

- ``--use-mixed-language`` - Enable automatic language detection
- ``--mixed-language-primary LANG`` - Primary/fallback language (e.g., ``de``, ``en-us``)
- ``--mixed-language-allowed LANGS`` - Comma-separated list of languages to detect
- ``--mixed-language-confidence FLOAT`` - Detection confidence threshold (0.0-1.0, default: 0.7)

**Supported languages**: ``en-us``, ``en-gb``, ``de``, ``fr-fr``, ``es``, ``it``, ``pt``, ``pl``, ``tr``, ``ru``, ``ko``, ``ja``, ``zh``/``cmn``

**Configuration**: Set defaults in config:

.. code-block:: bash

   ttsforge config --set use_mixed_language true
   ttsforge config --set mixed_language_primary de
   ttsforge config --set mixed_language_allowed "['de', 'en-us']"
   ttsforge config --set mixed_language_confidence 0.7


SSMD Editing
------------

ttsforge uses SSMD (Speech Synthesis Markdown) as an intermediate format between
EPUB and audio. This allows you to fine-tune pronunciation and pacing.

During conversion, ``.ssmd`` files are automatically generated for each chapter:

.. code-block:: text

   .{book_title}_chapters/
   ├── chapter_001_intro.ssmd
   ├── chapter_001_intro.wav
   └── ...

**Basic workflow**:

.. code-block:: bash

   # 1. Start conversion
   ttsforge convert book.epub

   # 2. Pause (Ctrl+C) and edit SSMD files
   vim .book_chapters/chapter_001_intro.ssmd

   # 3. Resume - auto-detects edits and regenerates audio
   ttsforge convert book.epub

**Common SSMD syntax**:

.. code-block:: ssmd

   ...p                               # Paragraph break
   ...s                               # Sentence break
   *text*                             # Moderate emphasis
   **text**                           # Strong emphasis
   [Hermione](ph: /hɝmˈIni/)          # Custom pronunciation

**Example SSMD file**:

.. code-block:: ssmd

   Chapter One ...p

   [Harry](ph: /hæɹi/) Potter was a *highly unusual* boy. ...s
   He **hated** the summer holidays. ...p

For complete SSMD documentation, see :doc:`ssmd`.


Configuration
-------------

Set default options:

.. code-block:: bash

   # Set default voice
   ttsforge config --set default_voice am_adam

   # Set default format
   ttsforge config --set default_format mp3

   # Enable GPU acceleration
   ttsforge config --set use_gpu true

   # View all settings
   ttsforge config --show


Complete Example
----------------

Full conversion with all options:

.. code-block:: bash

   ttsforge convert mybook.epub \
       --voice af_sarah \
       --speed 1.1 \
       --format m4b \
       --chapters 1-10 \
       --title "My Audiobook" \
       --author "Author Name" \
       --cover cover.jpg \
       --output ./audiobooks/mybook.m4b


Next Steps
----------

- :doc:`ssmd` - SSMD editing and syntax reference
- :doc:`cli` - Complete command reference
- :doc:`voices` - Detailed voice information
- :doc:`configuration` - All configuration options
- :doc:`filename_templates` - Customize output filenames
