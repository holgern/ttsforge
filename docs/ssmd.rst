SSMD Editing
============

SSMD (Speech Synthesis Markdown) is an intermediate text format used by ttsforge
between EPUB extraction and TTS audio generation. It allows fine-grained control
over pronunciation, pacing, and emphasis in your audiobooks.


How SSMD Works
--------------

During conversion, ttsforge automatically generates ``.ssmd`` files for each chapter:

.. code-block:: text

   .{book_title}_chapters/
   ├── {book_title}_state.json
   ├── chapter_001_intro.ssmd      # Editable text with speech markup
   ├── chapter_001_intro.wav
   ├── chapter_002_chapter1.ssmd
   ├── chapter_002_chapter1.wav
   └── ...

When you resume a conversion, ttsforge detects if you've edited any SSMD files
(using MD5 hash comparison) and automatically regenerates the corresponding audio.


Basic Workflow
--------------

1. **Start conversion**:

   .. code-block:: bash

      ttsforge convert book.epub

2. **Pause conversion** (Ctrl+C when needed)

3. **Edit SSMD files** to fix pronunciation or pacing:

   .. code-block:: bash

      vim .book_chapters/chapter_001_intro.ssmd

4. **Resume conversion** - automatically detects edits:

   .. code-block:: bash

      ttsforge convert book.epub


SSMD Syntax
-----------

SSMD uses a simple markdown-like syntax for speech control.


Structural Breaks
~~~~~~~~~~~~~~~~~

Control pauses between text segments:

.. code-block:: ssmd

   ...p    # Paragraph break (0.5-1.0s pause)
   ...s    # Sentence break (0.1-0.3s pause)
   ...c    # Clause break (shorter pause)

Example:

.. code-block:: ssmd

   This is the first paragraph. ...s
   It has multiple sentences. ...p

   This is a second paragraph. ...s


Emphasis
~~~~~~~~

Add vocal emphasis to words or phrases:

.. code-block:: ssmd

   *text*      # Moderate emphasis
   **text**    # Strong emphasis

Example:

.. code-block:: ssmd

   Harry was a *highly unusual* boy. ...s
   He **hated** the summer holidays. ...s


Custom Phonemes
~~~~~~~~~~~~~~~

Override pronunciation using IPA phonemes:

.. code-block:: ssmd

   [word](ph: /phoneme/)

Examples:

.. code-block:: ssmd

   [Hermione](ph: /hɝmˈIni/) Granger was Harry's best friend. ...s
   The [API](ph: /ˌeɪpiˈaɪ/) supports [JSON](ph: /dʒˈeɪsɑn/). ...s
   [Kubernetes](ph: /kubɚnˈɛtɪs/) is a container orchestrator. ...s


Language Switching (Planned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mark text as a different language (placeholder for future):

.. code-block:: ssmd

   [Bonjour](fr)      # French text
   [Hola](es)         # Spanish text


Complete Example
----------------

Here's a complete SSMD file example:

.. code-block:: ssmd

   Chapter One ...p

   [Harry](ph: /hæɹi/) Potter was a *highly unusual* boy in many ways. ...s
   For one thing, he **hated** the summer holidays more than any other
   time of year. ...s For another, he really wanted to do his homework,
   but was forced to do it in secret, in the dead of the night. ...p

   And he also happened to be a wizard. ...p

   The [Dursleys](ph: /dɝzliz/) had everything they wanted, but they
   also had a secret. ...s And their greatest fear was that somebody
   would discover it. ...p


Automatic Features
------------------

SSMD files are automatically enhanced with:

Phoneme Dictionary Injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use ``--phoneme-dict``, all phoneme substitutions are automatically
injected into the SSMD:

.. code-block:: bash

   ttsforge convert book.epub --phoneme-dict custom_phonemes.json

The generated SSMD will include:

.. code-block:: ssmd

   [Hermione](ph: /hɝmˈIni/) loved reading books. ...s


HTML Emphasis Detection
~~~~~~~~~~~~~~~~~~~~~~~

Emphasis from the original EPUB HTML is automatically converted:

- ``<em>text</em>`` → ``*text*``
- ``<strong>text</strong>`` → ``**text**``


Structural Break Preservation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ttsforge preserves paragraph structure but does not insert explicit
``...p`` or ``...s`` markers. Sentence detection is handled internally by
pykokoro at synthesis time. Use manual break markers only when you need
precise control over pauses.


Use Cases
---------

When to Edit SSMD
~~~~~~~~~~~~~~~~~

1. **Pronunciation issues**: Character names, technical terms, foreign words
2. **Pacing problems**: Adjust paragraph and sentence breaks for better flow
3. **Emphasis corrections**: Add or remove emphasis on specific words
4. **Consistency**: Ensure consistent pronunciation across chapters


Combining with Phoneme Dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For best results, use both features together:

1. Create a phoneme dictionary for common names/terms
2. Let ttsforge auto-inject into SSMD
3. Edit SSMD files for chapter-specific tweaks

.. code-block:: bash

   # 1. Extract and review names
   ttsforge extract-names book.epub
   vim custom_phonemes.json

   # 2. Start conversion (phonemes auto-injected into SSMD)
   ttsforge convert book.epub --phoneme-dict custom_phonemes.json

   # 3. Edit specific SSMD files as needed
   vim .book_chapters/chapter_005.ssmd

   # 4. Resume (regenerates edited chapters)
   ttsforge convert book.epub


Tips and Best Practices
------------------------

1. **Start with phoneme dictionary**: Create a global dictionary first,
   then use SSMD for chapter-specific overrides

2. **Test edits incrementally**: Edit one chapter, let it regenerate,
   listen to verify before editing more

3. **Use emphasis sparingly**: Too much emphasis can sound unnatural

4. **Keep backups**: SSMD files are regenerated if missing, but manual
   edits are preserved

5. **Consistent phonemes**: Use the same IPA notation throughout for
   consistency


Technical Details
-----------------

Hash-Based Change Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ttsforge tracks SSMD file changes using MD5 hashes (12 characters) stored
in the state file. When you resume:

1. Current SSMD file is hashed
2. Compared with saved hash in state
3. If different, audio is regenerated
4. New hash is saved


File Format
~~~~~~~~~~~

SSMD files are plain text UTF-8 files with the ``.ssmd`` extension.
They can be edited with any text editor.


Error Handling
~~~~~~~~~~~~~~

If SSMD generation fails, ttsforge falls back to plain text conversion
and logs a warning. The conversion continues without SSMD features.


Validation
~~~~~~~~~~

SSMD is not automatically validated during conversion. For manual checks,
use the ``validate_ssmd`` helper from ``ttsforge.ssmd_generator`` to get
warnings about unbalanced markers before you synthesize.


Limitations
-----------

- Language switching is not yet implemented (planned feature)
- Phoneme syntax must use valid IPA characters
- Very long lines may be truncated in some editors
- Hash detection only works with resumable conversions


See Also
--------

- :doc:`quickstart` - Getting started with ttsforge
- :doc:`cli` - Complete command reference
- :doc:`configuration` - Configuration options

For more SSMD examples and a quick reference, see ``SSMD_QUICKSTART.md``
in the repository root.
