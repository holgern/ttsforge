#!/usr/bin/env python3
"""
Advanced English contractions and edge cases stress test using ttsforge.

This example focuses on challenging contractions that may be difficult for
phoneme libraries to handle correctly, including:
- Negative contractions (don't, couldn't, shouldn't)
- Multiple contractions (I'd've, shouldn't've)
- Rare and archaic contractions ('twas, 'tis, shan't)
- Informal contractions (gonna, wanna, gotta)
- Possessive forms that look like contractions
- Contractions at sentence boundaries and with punctuation

Usage:
    python examples/contractions_advanced.py

Output:
    contractions_advanced_demo.wav - Generated speech testing edge cases
"""

import soundfile as sf

from ttsforge.onnx_backend import SAMPLE_RATE, KokoroONNX

# Text with challenging contraction patterns
TEXT = """
Advanced Contractions Challenge Text

Negative Contractions:
Don't worry about it. I won't tell anyone. They can't believe what happened.
She doesn't care anymore. He didn't show up. We couldn't find the keys.
You shouldn't have done that. They wouldn't listen to reason.
Isn't it obvious? Aren't they coming? Wasn't he supposed to be here?
Haven't you heard? Hasn't she told you? Hadn't we agreed on this?
That's not what I meant. You'll see what I mean. That's the way it is.
You'll understand eventually. That's wonderful news! You'll love this.

Common Will Contractions:
I'll be there soon. You'll see me tomorrow. He'll arrive on time.
She'll call you later. It'll be fine. We'll make it work.
They'll understand eventually. That'll be enough. This'll help you.

Multiple Contractions:
I'd've helped if I could've. You'd've done the same thing.
She'd've known better. We'd've been there earlier.
They shouldn't've said that. You shouldn't've worried.
I couldn't've imagined this. He wouldn't've believed it.
We mightn't've succeeded without help.
You'll've finished by then. They'll've arrived already.

Rare and Archaic Contractions:
'Twas the night before Christmas. 'Tis the season to be jolly.
Shan't we dance? Mayn't I join you? 'Twere better left unsaid.
Whoe'er shall find this. Whate'er you need. Where'er you go.
'Twould be nice to see you. 'Twill be a pleasure.

Informal Speech Contractions:
I'm gonna go now. You wanna come with me? I gotta finish this first.
She's gonna love it. He's gotta try harder. They wanna help out.
We're gonna make it work. You gotta believe me. I wanna thank you.
She's sorta tired. He's kinda busy. It's outta control.
I dunno what to say. Lemme think about it. Gimme a minute.

Contractions with Punctuation:
"Don't!" she shouted. "I can't," he whispered. "Won't you stay?" they asked.
Don't you see? Can't you tell? Won't they understand?
I'd've—if I could've—helped you. She'd said, "I won't," but then she did.
"Shouldn't we?" "Wouldn't you?" "Couldn't they?"

Possessive vs Contractions:
It's a dog. The dog wagged its tail. It's been raining. Its color is brown.
Who's there? Whose book is this? Who's been eating? Whose turn is it?
You're welcome. Your kindness matters. You're going. Your choice is clear.
They're here. Their house is nice. They're leaving. Their decision stands.

Challenging Sequences:
I'd've thought you'd've known. Shouldn't've, couldn't've, wouldn't've tried.
Don't, won't, can't, shan't agree. He'd've, she'd've, they'd've succeeded.
Y'all'd've loved it. Y'all're invited. Y'all've been wonderful.
That's what you'll discover. You'll know that's true. That's how you'll learn.

Contractions at Line Breaks:
Don't
stop believing. Won't
you come? I'd've
finished sooner.

Numbers and Contractions:
The '90s were great. It's '23 already. That's 'bout right.
It's 10 o'clock. Don't be late. I'll be there by 5 o'clock.

Dialect and Regional:
Y'all come back now. Ain't that something? 'Cause I said so.
How'd'y'do? What're y'all doing? Where'd'ya put it?
C'mon over here. Gonna hafta go. Oughta've known better.

Nested Apostrophes:
The class of '99's reunion was fun. The '80s' music is timeless.
Rock 'n' roll's influence continues. Fish 'n' chips' popularity grew.

Complex Negative Contractions:
Mustn't've been easy. Needn't've worried. Oughtn't've happened.
Mightn't've worked out. Daren't say more. Usedn't to care.

Literary and Poetic Contractions:
O'er the fields we go. E'en in darkness. Ne'er shall I forget.
Oft' times I wonder. 'Midst the chaos. 'Mongst the stars.
Howe'er it ends. Whene'er you're ready. Where'er the wind blows.

End Test Sequence:
This test includes don't, won't, can't, shouldn't, couldn't, wouldn't,
that's, you'll, I'll, we'll, they'll, he'll, she'll, it'll,
I'd've, you'd've, 'twas, gonna, wanna, gotta, y'all, ain't, and many more
challenging contractions to thoroughly test phoneme handling capabilities.

Final Mixed Test:
That's right, you'll see. Don't worry, you'll be fine. I won't forget that's important.
You'll find that's the truth. That's what I'll tell them. You'll know when that's done.
Don't say you'll do it if you won't. That's not what you'll need.
I'll make sure that's clear. You'll appreciate that's necessary.
"""

VOICE = "af_bella"  # American Female voice
LANG = "en-us"  # American English


def main():
    """Generate English speech testing advanced contractions."""
    print("Initializing TTS engine...")
    kokoro = KokoroONNX()

    print("=" * 70)
    print("ADVANCED CONTRACTIONS TEST")
    print("=" * 70)
    print("\nTesting challenging contractions:")
    print("  - Common: that's, you'll, I'll, we'll, don't, won't, can't")
    print("  - Negative: couldn't, shouldn't, wouldn't, hasn't, haven't, hadn't")
    print("  - Multiple: I'd've, you'll've, shouldn't've, couldn't've")
    print("  - Archaic: 'twas, 'tis, shan't, mayn't")
    print("  - Informal: gonna, wanna, gotta, dunno, lemme, gimme")
    print("  - Regional: y'all, ain't, c'mon")
    print("  - Complex: mustn't've, needn't've, oughtn't've")
    print("  - Possessive vs contraction: it's/its, who's/whose, you're/your")
    print("  - With punctuation and line breaks")
    print("=" * 70)
    print(f"\nVoice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    samples, _ = kokoro.create(
        TEXT,
        voice=VOICE,
        speed=1.0,
        lang=LANG,
    )

    output_file = "contractions_advanced_demo.wav"
    sf.write(output_file, samples, SAMPLE_RATE)

    duration = len(samples) / SAMPLE_RATE
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")
    print("\nNote: Listen carefully to verify correct pronunciation of:")
    print("  - Common contractions (that's, you'll, I'll, don't, won't, can't)")
    print("  - Negative contractions (couldn't, shouldn't, wouldn't)")
    print("  - Multiple contractions (I'd've, you'll've, shouldn't've)")
    print("  - Informal contractions (gonna, wanna, gotta)")
    print("  - Distinction between it's/its, who's/whose, you're/your")
    print("  - Mixed sequences (that's what you'll see, don't worry you'll be fine)")

    kokoro.close()


if __name__ == "__main__":
    main()
