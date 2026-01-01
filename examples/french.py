#!/usr/bin/env python3
"""
French TTS example using ttsforge.

This example demonstrates text-to-speech synthesis in French
using the Kokoro model with French voices.

Usage:
    python examples/french.py

Output:
    french_demo.wav - Generated French speech audio

Available French voices:
    - ff_siwis (female)
"""

import soundfile as sf

from ttsforge.onnx_backend import SAMPLE_RATE, KokoroONNX

# French quote about life and dreams (Victor Hugo inspired)
TEXT = (
    "La vie est un voyage, pas une destination. "
    "Chaque jour est une nouvelle page dans le livre de notre existence. "
    "Il faut rever sa vie et vivre son reve."
)

VOICE = "ff_siwis"  # French Female voice
LANG = "fr-fr"  # French


def main():
    """Generate French speech audio."""
    print("Initializing TTS engine...")
    kokoro = KokoroONNX()

    print(f"Text: {TEXT}")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    samples, _ = kokoro.create(
        TEXT,
        voice=VOICE,
        speed=1.0,
        lang=LANG,
    )

    output_file = "french_demo.wav"
    sf.write(output_file, samples, SAMPLE_RATE)

    duration = len(samples) / SAMPLE_RATE
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
