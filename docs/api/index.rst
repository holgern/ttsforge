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

**ttsforge.onnx_backend**
   ONNX Runtime backend for Kokoro TTS model inference.

**ttsforge.tokenizer**
   Text-to-phoneme tokenization using espeak-ng.

**ttsforge.phonemes**
   Data structures for phoneme book representation.

Utilities
^^^^^^^^^

**ttsforge.constants**
   Configuration defaults, voice definitions, and language mappings.

**ttsforge.utils**
   Utility functions for file handling, configuration, and formatting.

**ttsforge.trim**
   Audio trimming utilities for silence removal.


Quick API Examples
------------------

Basic Text-to-Speech
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ttsforge.onnx_backend import KokoroONNX

   # Initialize TTS engine
   kokoro = KokoroONNX(use_gpu=False)

   # Generate audio
   audio, sample_rate = kokoro.create(
       "Hello, world!",
       voice="af_heart",
       speed=1.0
   )

   # Save to file
   import soundfile as sf
   sf.write("output.wav", audio, sample_rate)

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

   from ttsforge.tokenizer import Tokenizer

   # Initialize tokenizer
   tokenizer = Tokenizer(vocab_version="v1.0")

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

.. automodule:: ttsforge.onnx_backend
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.phonemes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttsforge.trim
   :members:
   :undoc-members:
   :show-inheritance:
