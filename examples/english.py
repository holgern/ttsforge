#!/usr/bin/env python3
"""
English TTS example using ttsforge.

This example demonstrates text-to-speech synthesis in English
using the Kokoro model with American and British English voices.

Usage:
    python examples/english.py

Output:
    english_demo.wav - Generated English speech audio

Available English voices:
    American Female: af_alloy, af_aoede, af_bella, af_heart, af_jessica,
                     af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
    American Male: am_adam, am_echo, am_eric, am_fenrir, am_liam,
                   am_michael, am_onyx, am_puck, am_santa
    British Female: bf_alice, bf_emma, bf_isabella, bf_lily
    British Male: bm_daniel, bm_fable, bm_george, bm_lewis
"""

import soundfile as sf

from ttsforge.onnx_backend import SAMPLE_RATE, KokoroONNX

# Quote about technology and the future
TEXT = (
    "The best way to predict the future is to create it. "
    "Technology is nothing without the imagination to use it wisely. "
    "Every great innovation begins with a simple question: what if?"
)

VOICE = "af_heart"  # American Female voice
LANG = "en-us"  # American English


def main():
    """Generate English speech audio."""
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

    output_file = "english_demo.wav"
    sf.write(output_file, samples, SAMPLE_RATE)

    duration = len(samples) / SAMPLE_RATE
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
