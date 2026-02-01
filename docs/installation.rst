Installation
============

This guide covers the installation of ttsforge and its dependencies.


System Requirements
-------------------

- **Python**: 3.10 or later
- **Operating System**: Linux, macOS, or Windows
- **Disk Space**: ~330MB for ONNX models (downloaded automatically on first use)


Dependencies
------------

ttsforge requires the following external tools:

ffmpeg (Required for MP3/FLAC/OPUS/M4B)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ffmpeg is required for MP3/FLAC/OPUS/M4B output and chapter merging.

**Termux (Android):**

.. code-block:: bash

   pkg install ffmpeg


**Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install ffmpeg

**macOS (Homebrew):**

.. code-block:: bash

   brew install ffmpeg

**Windows:**

Download from https://ffmpeg.org/download.html and add to PATH.

Optional: bundled ffmpeg via Python (not available on all platforms)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you cannot install a system ffmpeg, you can try the optional prebuilt binaries:

.. code-block:: bash

   pip install "ttsforge[static_ffmpeg]"

espeak-ng (Required for Phonemization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

espeak-ng is used for text-to-phoneme conversion.

**Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install espeak-ng

**macOS (Homebrew):**

.. code-block:: bash

   brew install espeak-ng

**Windows:**

Download from https://github.com/espeak-ng/espeak-ng/releases

Audio Playback (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^

Audio playback features (``--play`` flags and the ``read`` command) require
``sounddevice``:

.. code-block:: bash

   pip install "ttsforge[audio]"

Or install directly:

.. code-block:: bash

   pip install sounddevice


Installing ttsforge
-------------------

From PyPI (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install ttsforge

Optional extras:

.. code-block:: bash

   # Audio playback (required for --play and read)
   pip install "ttsforge[audio]"

   # Bundled ffmpeg binaries
   pip install "ttsforge[static_ffmpeg]"

   # GPU acceleration
   pip install "ttsforge[gpu]"

From Source
^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/holgern/ttsforge.git
   cd ttsforge
   pip install -e .

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

For development with testing and linting tools:

.. code-block:: bash

   git clone https://github.com/holgern/ttsforge.git
   cd ttsforge
   pip install -e ".[dev]"


GPU Acceleration (Optional)
---------------------------

For GPU-accelerated inference, install onnxruntime-gpu:

.. code-block:: bash

   pip install onnxruntime-gpu

Then enable GPU in your configuration:

.. code-block:: bash

   ttsforge config --set use_gpu true

Or use the ``--gpu`` flag with commands:

.. code-block:: bash

   ttsforge convert book.epub --gpu


Mixed-Language Support (Optional)
----------------------------------

For automatic detection and handling of multiple languages in text (e.g., German text with English technical terms):

.. code-block:: bash

   pip install lingua-language-detector

Then enable mixed-language mode:

.. code-block:: bash

   ttsforge config --set use_mixed_language true
   ttsforge config --set mixed_language_primary de
   ttsforge config --set mixed_language_allowed "['de', 'en-us']"

Or use the ``--use-mixed-language`` flag with commands:

.. code-block:: bash

   ttsforge convert book.epub \
       --use-mixed-language \
       --mixed-language-primary de \
       --mixed-language-allowed de,en-us


Downloading Models
------------------

ttsforge uses Kokoro ONNX models (~330MB total) which are downloaded automatically
on first use. You can also download them proactively:

.. code-block:: bash

   # Download models
   ttsforge download

   # Force re-download
   ttsforge download --force

Models are stored in:

- Linux: ``~/.cache/ttsforge/``
- macOS: ``~/Library/Caches/ttsforge/``
- Windows: ``%LOCALAPPDATA%\ttsforge\Cache\``


Verifying Installation
----------------------

Verify that ttsforge is installed correctly:

.. code-block:: bash

   # Check version
   ttsforge --version

   # Show current configuration
   ttsforge config --show

   # Generate a sample audio file
   ttsforge sample "Hello, world!"

If the sample command succeeds and creates ``sample.wav``, ttsforge is ready to use.


Troubleshooting
---------------

ffmpeg not found
^^^^^^^^^^^^^^^^

If you see "ffmpeg not found" errors when creating M4B files:

1. Ensure ffmpeg is installed (see above)
2. Verify it's in your PATH: ``ffmpeg -version``
3. On Windows, you may need to restart your terminal after installation

espeak-ng not found
^^^^^^^^^^^^^^^^^^^

If phonemization fails:

1. Ensure espeak-ng is installed (see above)
2. On Linux, the library should be ``libespeak-ng.so.1``
3. On macOS with Homebrew, it's typically at ``/opt/homebrew/lib/libespeak-ng.dylib``

Model download fails
^^^^^^^^^^^^^^^^^^^^

If model download fails:

1. Check your internet connection
2. Try downloading manually with ``ttsforge download``
3. Check disk space (~330MB required)
4. The model directory can be found with ``ttsforge config --show``

GPU not detected
^^^^^^^^^^^^^^^^

If GPU acceleration isn't working:

1. Ensure ``onnxruntime-gpu`` is installed (not just ``onnxruntime``)
2. Verify CUDA is properly installed
3. Check GPU compatibility with ONNX Runtime
