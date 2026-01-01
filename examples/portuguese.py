#!/usr/bin/env python3
"""
Brazilian Portuguese TTS example using ttsforge.

This example demonstrates text-to-speech synthesis in Brazilian Portuguese
using the Kokoro model with Portuguese voices.

Usage:
    python examples/portuguese.py

Output:
    portuguese_demo.wav - Generated Portuguese speech audio

Available Portuguese voices:
    - pf_dora (female)
    - pm_alex, pm_santa (male)
"""

import soundfile as sf

from ttsforge.onnx_backend import SAMPLE_RATE, KokoroONNX

# Brazilian Portuguese quote about dreams and life
TEXT = (
    "A vida e feita de escolhas. Cada passo que damos nos leva a um novo caminho. "
    "Sonhe grande, trabalhe duro, e nunca desista dos seus objetivos. "
    "O sucesso e a soma de pequenos esforcos repetidos dia apos dia."
)

VOICE = "pf_dora"  # Portuguese Female voice
LANG = "pt-br"  # Brazilian Portuguese


def main():
    """Generate Portuguese speech audio."""
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

    output_file = "portuguese_demo.wav"
    sf.write(output_file, samples, SAMPLE_RATE)

    duration = len(samples) / SAMPLE_RATE
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
