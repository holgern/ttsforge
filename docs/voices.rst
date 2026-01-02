Voices
======

ttsforge uses Kokoro TTS which provides 54 high-quality neural voices across
9 languages.


Voice Naming Convention
-----------------------

Voices follow a consistent naming pattern:

.. code-block:: text

   {language}{gender}_{name}

Where:

- **Language**: Two-letter code (``af``, ``am``, ``bf``, etc.)
- **Gender**: ``f`` = female, ``m`` = male
- **Name**: Voice identifier

For example:

- ``af_heart`` = American English, Female, "Heart" voice
- ``am_adam`` = American English, Male, "Adam" voice
- ``bf_emma`` = British English, Female, "Emma" voice


Listing Voices
--------------

List all available voices:

.. code-block:: bash

   ttsforge voices

List voices for a specific language:

.. code-block:: bash

   ttsforge voices -l a  # American English
   ttsforge voices -l b  # British English


Voice Demo
----------

Listen to all voices:

.. code-block:: bash

   # All voices in one file
   ttsforge demo

   # Specific language
   ttsforge demo -l a

   # Save individual files
   ttsforge demo --separate -o ./voice_samples/


Voices by Language
------------------

American English (a)
^^^^^^^^^^^^^^^^^^^^

**Female Voices (11):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``af_alloy``
     - Versatile, balanced voice
     -
   * - ``af_aoede``
     - Clear, pleasant tone
     -
   * - ``af_bella``
     - Warm, friendly voice
     -
   * - ``af_heart``
     - Expressive, emotional voice
     - Yes
   * - ``af_jessica``
     - Professional, articulate
     -
   * - ``af_kore``
     - Youthful, energetic voice
     -
   * - ``af_nicole``
     - Soft, gentle voice
     -
   * - ``af_nova``
     - Modern, dynamic voice
     -
   * - ``af_river``
     - Calm, flowing voice
     -
   * - ``af_sarah``
     - Confident, clear voice
     -
   * - ``af_sky``
     - Light, airy voice
     -

**Male Voices (9):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Notes
   * - ``am_adam``
     - Deep, authoritative voice
     - Popular for audiobooks
   * - ``am_echo``
     - Resonant, clear voice
     -
   * - ``am_eric``
     - Friendly, approachable voice
     -
   * - ``am_fenrir``
     - Strong, dramatic voice
     -
   * - ``am_liam``
     - Young, energetic voice
     -
   * - ``am_michael``
     - Professional narrator voice
     -
   * - ``am_onyx``
     - Deep, smooth voice
     -
   * - ``am_puck``
     - Playful, expressive voice
     -
   * - ``am_santa``
     - Warm, jolly voice
     - Seasonal character voice

British English (b)
^^^^^^^^^^^^^^^^^^^

**Female Voices (4):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``bf_alice``
     - Elegant, refined voice
     -
   * - ``bf_emma``
     - Classic British voice
     - Yes
   * - ``bf_isabella``
     - Sophisticated voice
     -
   * - ``bf_lily``
     - Gentle, soft voice
     -

**Male Voices (4):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Notes
   * - ``bm_daniel``
     - Traditional British voice
     -
   * - ``bm_fable``
     - Storytelling voice
     - Great for narratives
   * - ``bm_george``
     - Authoritative, mature voice
     -
   * - ``bm_lewis``
     - Modern British voice
     -

Spanish (e)
^^^^^^^^^^^

**Female Voices (1):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``ef_dora``
     - Clear Spanish voice
     - Yes

**Male Voices (2):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Notes
   * - ``em_alex``
     - Natural Spanish voice
     -
   * - ``em_santa``
     - Warm Spanish voice
     - Seasonal

French (f)
^^^^^^^^^^

**Female Voices (1):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``ff_siwis``
     - Natural French voice
     - Yes

Hindi (h)
^^^^^^^^^

**Female Voices (2):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``hf_alpha``
     - Clear Hindi voice
     - Yes
   * - ``hf_beta``
     - Alternative Hindi voice
     -

**Male Voices (2):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Notes
   * - ``hm_omega``
     - Deep Hindi voice
     -
   * - ``hm_psi``
     - Natural Hindi voice
     -

Italian (i)
^^^^^^^^^^^

**Female Voices (1):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``if_sara``
     - Natural Italian voice
     - Yes

**Male Voices (1):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Notes
   * - ``im_nicola``
     - Clear Italian voice
     -

Japanese (j)
^^^^^^^^^^^^

**Female Voices (4):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``jf_alpha``
     - Standard Japanese voice
     - Yes
   * - ``jf_gongitsune``
     - Storytelling voice
     -
   * - ``jf_nezumi``
     - Soft Japanese voice
     -
   * - ``jf_tebukuro``
     - Gentle Japanese voice
     -

**Male Voices (1):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Notes
   * - ``jm_kumo``
     - Natural Japanese male voice
     -

Brazilian Portuguese (p)
^^^^^^^^^^^^^^^^^^^^^^^^

**Female Voices (1):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``pf_dora``
     - Natural Portuguese voice
     - Yes

**Male Voices (2):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Notes
   * - ``pm_alex``
     - Clear Portuguese voice
     -
   * - ``pm_santa``
     - Warm Portuguese voice
     - Seasonal

Mandarin Chinese (z)
^^^^^^^^^^^^^^^^^^^^

**Female Voices (4):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Default
   * - ``zf_xiaobei``
     - Northern accent
     -
   * - ``zf_xiaoni``
     - Soft Chinese voice
     -
   * - ``zf_xiaoxiao``
     - Popular Chinese voice
     - Yes
   * - ``zf_xiaoyi``
     - Clear Chinese voice
     -

**Male Voices (4):**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Voice
     - Description
     - Notes
   * - ``zm_yunjian``
     - Strong male voice
     -
   * - ``zm_yunxi``
     - Natural male voice
     -
   * - ``zm_yunxia``
     - Youthful male voice
     -
   * - ``zm_yunyang``
     - Mature male voice
     -


Voice Blending
--------------

Combine multiple voices for unique narration. You can specify voice blends in two ways:

**Using --voice parameter (recommended):**

The ``--voice`` parameter now auto-detects blend format when you include colons and commas:

.. code-block:: bash

   # 50/50 blend of two voices
   ttsforge convert book.epub --voice "af_nicole:50,am_michael:50"

   # Weighted blend (70% Nicole, 30% Michael)
   ttsforge convert book.epub --voice "af_nicole:70,am_michael:30"

   # Works with sample command
   ttsforge sample "Hello world" --voice "af_sky:60,bf_emma:40" -p

   # Works with phonemes preview
   ttsforge phonemes preview "Test" --voice "am_adam:50,am_michael:50" --play

**Using --voice-blend parameter (traditional):**

.. code-block:: bash

   # Explicit voice-blend parameter
   ttsforge convert book.epub --voice-blend "af_nicole:50,am_michael:50"

   # Can combine with regular voice (blend takes precedence)
   ttsforge convert book.epub --voice af_sky --voice-blend "af_nicole:60,am_michael:40"

Voice blending creates a mixed voice by interpolating the voice embeddings.
This can create interesting narrator voices, but results may vary. Blending works best
with voices of the same language and similar characteristics.


Recommendations
---------------

For Audiobooks (Fiction)
^^^^^^^^^^^^^^^^^^^^^^^^

- **American English**: ``af_heart`` (female), ``am_adam`` (male)
- **British English**: ``bf_emma`` (female), ``bm_fable`` (male)

For Audiobooks (Non-Fiction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **American English**: ``af_sarah`` (female), ``am_michael`` (male)
- **British English**: ``bf_alice`` (female), ``bm_george`` (male)

For Technical Content
^^^^^^^^^^^^^^^^^^^^^

- Clear articulation: ``af_jessica``, ``am_eric``
- Moderate speed: Use ``-s 0.95`` for complex content

For Children's Books
^^^^^^^^^^^^^^^^^^^^

- Expressive voices: ``af_kore``, ``am_puck``
- Storytelling voices: ``bf_emma``, ``bm_fable``


Language Code Reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 20 20

   * - Code
     - Language
     - Voices
     - Default Voice
   * - ``a``
     - American English
     - 20
     - ``af_heart``
   * - ``b``
     - British English
     - 8
     - ``bf_emma``
   * - ``e``
     - Spanish
     - 3
     - ``ef_dora``
   * - ``f``
     - French
     - 1
     - ``ff_siwis``
   * - ``h``
     - Hindi
     - 4
     - ``hf_alpha``
   * - ``i``
     - Italian
     - 2
     - ``if_sara``
   * - ``j``
     - Japanese
     - 5
     - ``jf_alpha``
   * - ``p``
     - Brazilian Portuguese
     - 3
     - ``pf_dora``
   * - ``z``
     - Mandarin Chinese
     - 8
     - ``zf_xiaoxiao``
