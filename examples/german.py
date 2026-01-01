#!/usr/bin/env python3
"""
German text example using ttsforge.

This example demonstrates how ttsforge handles German text using the af_bella voice.
Note: The Kokoro model was not explicitly trained on German, so pronunciation may
not be perfect. The model will attempt to phonemize German text using espeak-ng.

Usage:
    python examples/german.py

Output:
    german_demo.wav - Generated German speech
"""

import soundfile as sf

from ttsforge.onnx_backend import SAMPLE_RATE, KokoroONNX, VoiceBlend

# German text samples
TEXT = """
Guten Tag! Willkommen zu diesem Beispiel der deutschen Sprache.

Die deutsche Sprache hat viele besondere Eigenschaften. 
Sie ist bekannt für ihre langen zusammengesetzten Wörter wie 
Donaudampfschifffahrtsgesellschaft oder Kraftfahrzeughaftpflichtversicherung.

Heute ist ein schöner Tag. Die Sonne scheint, und die Vögel singen.
Ich möchte gerne einen Kaffee trinken und ein Buch lesen.

Zahlen sind auch wichtig: eins, zwei, drei, vier, fünf, sechs, sieben, acht, neun, zehn.

Umlaute sind charakteristisch für Deutsch: ä, ö, ü und das Eszett ß.
Käse, Brötchen, Müller, Straße.

Fragen Sie mich, wie es Ihnen geht?
Es geht mir sehr gut, danke schön!

Die Wissenschaft macht große Fortschritte.
Technologie verändert unsere Welt jeden Tag.

Auf Wiedersehen und vielen Dank fürs Zuhören!
"""

VOICE = "af_bella"  # American Female voice
VOICE = "ef_dora"  # American Female voice
VOICE = "if_sara"  # American Female voice
VOICE = "jf_alpha"  # American Female voice
BLEND = "jf_alpha:50,ef_dora:50"  # Blend of three voices
LANG = "de"  # German


def main():
    """Generate German speech using English-trained voice."""
    print("Initializing TTS engine...")
    kokoro = KokoroONNX()
    

    print("=" * 60)
    print("NOTE: Kokoro was NOT explicitly trained on German.")
    print("The model will attempt German phonemization via espeak-ng.")
    print("Pronunciation may not be perfect or native-sounding.")
    print("=" * 60)
    print(f"\nVoice: {BLEND}")
    print(f"Language: {LANG}")

    blend = VoiceBlend.parse(BLEND)
    print("\nGenerating audio...")
    samples, _ = kokoro.create(
        TEXT,
        voice=blend,
        speed=1.0,
        lang=LANG,
    )

    output_file = "german_demo.wav"
    sf.write(output_file, samples, SAMPLE_RATE)

    duration = len(samples) / SAMPLE_RATE
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
