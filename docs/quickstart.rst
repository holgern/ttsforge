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

- :doc:`cli` - Complete command reference
- :doc:`voices` - Detailed voice information
- :doc:`configuration` - All configuration options
- :doc:`filename_templates` - Customize output filenames
