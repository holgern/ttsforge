API Reference
=============

This section documents the Python API for ttsforge, allowing programmatic use
of the library.

Module Overview
---------------

ttsforge is organized into the following modules:

Core Modules
^^^^^^^^^^^^

**ttsforge.cli**
   Command-line interface implementation using Click.

**ttsforge.conversion**
   Main conversion logic for EPUB to audiobook conversion.

**ttsforge.phoneme_conversion**
   Conversion logic for pre-tokenized phoneme files.

TTS Backend
^^^^^^^^^^^

**ttsforge.kokoro_runner**
   Shared Kokoro ONNX runner used by conversion paths.

**ttsforge.kokoro_lang**
   Language code helpers for Kokoro.

**ttsforge.phonemes**
   Data structures for phoneme book representation.

Utilities
^^^^^^^^^

**ttsforge.constants**
   Configuration defaults, voice definitions, and language mappings.

**ttsforge.utils**
   Utility functions for file handling, configuration, and formatting.

**ttsforge.audio_merge**
   Audio concatenation and chapter marker handling.

**ttsforge.chapter_selection**
   Parsing helpers for chapter selection strings.

**ttsforge.ssmd_generator**
   SSMD generation and validation helpers.

**ttsforge.input_reader**
   EPUB/text/PDF input parsing helpers.

**ttsforge.name_extractor**
   Name extraction utilities for dictionary building.

**ttsforge.vocab**
   Vocabulary utilities and metadata.

**ttsforge.trim**
   Audio trimming utilities for silence removal.


Quick API Examples
------------------

Basic Text-to-Speech
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ttsforge.kokoro_lang import get_onnx_lang_code
   from ttsforge.kokoro_runner import KokoroRunOptions, KokoroRunner

   # Initialize runner
   opts = KokoroRunOptions(
       voice="af_heart",
       speed=1.0,
       use_gpu=False,
       pause_clause=0.25,
       pause_sentence=0.2,
       pause_paragraph=0.75,
       pause_variance=0.05,
   )
   runner = KokoroRunner(opts, log=print)
   runner.ensure_ready()

   # Generate audio
   audio = runner.synthesize(
       "Hello, world!",
       lang_code=get_onnx_lang_code("en-us"),
       pause_mode="tts",
       is_phonemes=False,
   )

   # Save to file
   import soundfile as sf
   sf.write("output.wav", audio, 24000)

Converting an EPUB
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   from ttsforge.conversion import ConversionOptions, TTSConverter

   # Configure conversion
   options = ConversionOptions(
       voice="am_adam",
       language="a",
       speed=1.0,
       output_format="m4b",
       use_gpu=False,
   )

   # Create converter
   converter = TTSConverter(options=options)

   # Convert EPUB
   result = converter.convert_epub(
       epub_path=Path("book.epub"),
       output_path=Path("book.m4b"),
   )

   if result.success:
       print(f"Created: {result.output_path}")
   else:
       print(f"Error: {result.error_message}")

Working with Phonemes
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pykokoro.tokenizer import Tokenizer

   # Initialize tokenizer
   tokenizer = Tokenizer()

   # Convert text to phonemes
   text = "Hello, world!"
   phonemes = tokenizer.phonemize(text, lang="en-us")
   print(f"Phonemes: {phonemes}")

   # Get token IDs
   tokens = tokenizer.tokenize(phonemes)
   print(f"Tokens: {tokens}")

   # Human-readable format
   readable = tokenizer.format_readable(text, lang="en-us")
   print(f"Readable: {readable}")

Loading Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ttsforge.utils import load_config, save_config

   # Load current config
   config = load_config()
   print(f"Default voice: {config['default_voice']}")

   # Modify and save
   config['default_voice'] = 'am_adam'
   save_config(config)


Auto-generated API Documentation
--------------------------------

.. automodule:: ttsforge
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.constants
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.conversion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.phoneme_conversion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.kokoro_runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.kokoro_lang
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.phonemes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.audio_merge
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.chapter_selection
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.ssmd_generator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.input_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.name_extractor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.vocab
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.trim
   :members:
   :undoc-members:
   :show-inheritance:
