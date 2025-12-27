"""Constants for ttsforge - voices, languages, and formats."""

# Program Information
PROGRAM_NAME = "ttsforge"
PROGRAM_DESCRIPTION = "Generate audiobooks from EPUB files with TTS."

# Language code to description mapping
LANGUAGE_DESCRIPTIONS = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

# ISO language code to ttsforge language code mapping
ISO_TO_LANG_CODE = {
    "en": "a",  # Default to American English
    "en-us": "a",
    "en-gb": "b",
    "en-au": "b",
    "es": "e",
    "es-es": "e",
    "es-mx": "e",
    "fr": "f",
    "fr-fr": "f",
    "fr-ca": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "pt-br": "p",
    "pt-pt": "p",
    "zh": "z",
    "zh-cn": "z",
    "zh-tw": "z",
}

# All available Kokoro voices
VOICES = [
    # American English Female
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    # American English Male
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    # British English Female
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    # British English Male
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    # Spanish
    "ef_dora",
    "em_alex",
    "em_santa",
    # French
    "ff_siwis",
    # Hindi
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    # Italian
    "if_sara",
    "im_nicola",
    # Japanese
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    # Brazilian Portuguese
    "pf_dora",
    "pm_alex",
    "pm_santa",
    # Mandarin Chinese
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]

# Voice prefix to language code mapping
VOICE_PREFIX_TO_LANG = {
    "af": "a",  # American Female
    "am": "a",  # American Male
    "bf": "b",  # British Female
    "bm": "b",  # British Male
    "ef": "e",  # Spanish Female
    "em": "e",  # Spanish Male
    "ff": "f",  # French Female
    "fm": "f",  # French Male
    "hf": "h",  # Hindi Female
    "hm": "h",  # Hindi Male
    "if": "i",  # Italian Female
    "im": "i",  # Italian Male
    "jf": "j",  # Japanese Female
    "jm": "j",  # Japanese Male
    "pf": "p",  # Portuguese Female
    "pm": "p",  # Portuguese Male
    "zf": "z",  # Chinese Female
    "zm": "z",  # Chinese Male
}

# Language code to default voice mapping
DEFAULT_VOICE_FOR_LANG = {
    "a": "af_heart",
    "b": "bf_emma",
    "e": "ef_dora",
    "f": "ff_siwis",
    "h": "hf_alpha",
    "i": "if_sara",
    "j": "jf_alpha",
    "p": "pf_dora",
    "z": "zf_xiaoxiao",
}

# Supported output audio formats
SUPPORTED_OUTPUT_FORMATS = [
    "wav",
    "mp3",
    "flac",
    "opus",
    "m4b",
]

# Formats that require ffmpeg
FFMPEG_FORMATS = ["m4b", "opus"]

# Formats supported by soundfile directly
SOUNDFILE_FORMATS = ["wav", "mp3", "flac"]

# Default configuration values
DEFAULT_CONFIG = {
    "default_voice": "af_heart",
    "default_language": "a",
    "default_speed": 1.0,
    "default_format": "m4b",
    "use_gpu": True,
    "silence_between_chapters": 2.0,
    "save_chapters_separately": False,
    "merge_at_end": True,
    "auto_detect_language": True,
}

# Audio settings
SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1

# Sample texts for voice preview (per language)
SAMPLE_TEXTS = {
    "a": "This is a sample of the selected voice.",
    "b": "This is a sample of the selected voice.",
    "e": "Este es una muestra de la voz seleccionada.",
    "f": "Ceci est un exemple de la voix s\u00e9lectionn\u00e9e.",
    "h": "\u092f\u0939 \u091a\u092f\u0928\u093f\u0924 \u0906\u0935\u093e\u091c\u093c \u0915\u093e \u090f\u0915 \u0928\u092e\u0942\u0928\u093e \u0939\u0948\u0964",
    "i": "Questo \u00e8 un esempio della voce selezionata.",
    "j": "\u3053\u308c\u306f\u9078\u629e\u3057\u305f\u58f0\u306e\u30b5\u30f3\u30d7\u30eb\u3067\u3059\u3002",
    "p": "Este \u00e9 um exemplo da voz selecionada.",
    "z": "\u8fd9\u662f\u6240\u9009\u8bed\u97f3\u7684\u793a\u4f8b\u3002",
}
