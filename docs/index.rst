ttsforge Documentation
======================

**ttsforge** is a command-line tool for converting EPUB files to audiobooks using
Kokoro ONNX TTS (Text-to-Speech).

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   cli
   ssmd
   configuration
   filename_templates
   voices

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index


Features
--------

- **EPUB to Audiobook Conversion**: Convert EPUB files to M4B, MP3, WAV, FLAC, or OPUS formats
- **50+ High-Quality Voices**: Support for 9 languages with multiple voice options
- **SSMD Editing**: Edit intermediate SSMD files to fine-tune pronunciation and pacing
- **Resumable Conversions**: Long audiobook conversions can be interrupted and resumed
- **Phoneme Pre-tokenization**: Pre-process text to phonemes for faster batch conversions
- **Configurable Filename Templates**: Customize output filenames with book metadata
- **Voice Blending**: Mix multiple voices for custom narration styles
- **GPU Acceleration**: Optional GPU support for faster processing
- **Chapter Selection**: Convert specific chapters or chapter ranges
- **Metadata Support**: Automatic language detection and metadata embedding


Quick Example
-------------

.. code-block:: bash

   # Install ttsforge
   pip install ttsforge

   # Convert an EPUB to audiobook (M4B format with chapters)
   ttsforge convert book.epub

   # Convert with a specific voice
   ttsforge convert book.epub -v am_adam

   # Convert specific chapters
   ttsforge convert book.epub --chapters 1-5

   # List available voices
   ttsforge voices


Supported Languages
-------------------

ttsforge supports 9 languages with native TTS voices:

- **American English** (a) - 20 voices
- **British English** (b) - 8 voices
- **Spanish** (e) - 3 voices
- **French** (f) - 1 voice
- **Hindi** (h) - 4 voices
- **Italian** (i) - 2 voices
- **Japanese** (j) - 5 voices
- **Brazilian Portuguese** (p) - 3 voices
- **Mandarin Chinese** (z) - 8 voices


Requirements
------------

- Python 3.10 or later
- ffmpeg (required for M4B and OPUS formats)
- espeak-ng (for phonemization)
- ~330MB disk space for ONNX models (downloaded automatically)


License
-------

ttsforge is released under the MIT License.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
