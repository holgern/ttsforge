#!/usr/bin/env python3
"""
Italian TTS example using ttsforge.

This example demonstrates text-to-speech synthesis in Italian
using the Kokoro model with Italian voices.

Usage:
    python examples/italian.py

Output:
    italian_demo.wav - Generated Italian speech audio

Available Italian voices:
    - if_sara (female)
    - im_nicola (male)
"""

import soundfile as sf

from ttsforge.onnx_backend import SAMPLE_RATE, KokoroONNX

# Italian quote about life, art, and beauty
TEXT = (
    "La vita e come una melodia: ogni nota ha il suo significato. "
    "L'arte non riproduce cio che e visibile, "
    "ma rende visibile cio che non sempre lo e. "
    "La bellezza salvera il mondo."
)

VOICE = "if_sara"  # Italian Female voice
LANG = "it"  # Italian


def main():
    """Generate Italian speech audio."""
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

    output_file = "italian_demo.wav"
    sf.write(output_file, samples, SAMPLE_RATE)

    duration = len(samples) / SAMPLE_RATE
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
