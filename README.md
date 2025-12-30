# ttsforge

Convert EPUB files to audiobooks using Kokoro ONNX TTS.

ttsforge is a command-line tool that transforms EPUB ebooks into high-quality audiobooks with support for 54 neural voices across 9 languages.

## Features

- **EPUB to Audiobook**: Convert EPUB files to M4B, MP3, WAV, FLAC, or OPUS
- **54 Neural Voices**: High-quality TTS in 9 languages
- **Resumable Conversions**: Interrupt and resume long audiobook conversions
- **Phoneme Pre-tokenization**: Pre-process text for faster batch conversions
- **Configurable Filenames**: Template-based output naming with book metadata
- **Voice Blending**: Mix multiple voices for custom narration
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Chapter Support**: M4B files include chapter markers from EPUB

## Installation

```bash
pip install ttsforge
```

### Dependencies

- **ffmpeg**: Required for M4B and OPUS formats
- **espeak-ng**: Required for phonemization

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg espeak-ng
```

**macOS:**
```bash
brew install ffmpeg espeak-ng
```

## Quick Start

```bash
# Convert an EPUB to audiobook (M4B with chapters)
ttsforge convert book.epub

# Use a specific voice
ttsforge convert book.epub -v am_adam

# Convert specific chapters
ttsforge convert book.epub --chapters 1-5

# List available voices
ttsforge voices

# Generate a voice demo
ttsforge demo
```

## Usage

### Basic Conversion

```bash
ttsforge convert book.epub
```

Creates `book.m4b` with default settings (voice: `af_heart`, format: M4B).

### Voice Selection

```bash
# List all voices
ttsforge voices

# List voices for a language
ttsforge voices -l b  # British English

# Convert with specific voice
ttsforge convert book.epub -v bf_emma
```

### Output Formats

```bash
ttsforge convert book.epub -f mp3    # MP3
ttsforge convert book.epub -f wav    # WAV (uncompressed)
ttsforge convert book.epub -f flac   # FLAC (lossless)
ttsforge convert book.epub -f opus   # OPUS
ttsforge convert book.epub -f m4b    # M4B audiobook (default)
```

### Chapter Selection

```bash
# Preview chapters
ttsforge list book.epub

# Convert range
ttsforge convert book.epub --chapters 1-5

# Convert specific chapters
ttsforge convert book.epub --chapters 1,3,5,7

# Mixed selection
ttsforge convert book.epub --chapters 1-3,5,10-15
```

### Speed Control

```bash
ttsforge convert book.epub -s 1.2   # 20% faster
ttsforge convert book.epub -s 0.9   # 10% slower
```

### Resumable Conversions

Conversions are resumable by default. If interrupted, re-run the same command:

```bash
ttsforge convert book.epub  # Resumes from last chapter
ttsforge convert book.epub --fresh  # Start over
```

### Phoneme Workflow

For large books or batch processing, pre-tokenize to phonemes:

```bash
# Export to phonemes (fast, CPU-only)
ttsforge phonemes export book.epub

# Convert phonemes to audio (can run on different machine)
ttsforge phonemes convert book.phonemes.json -v am_adam
```

### Configuration

```bash
# View settings
ttsforge config --show

# Set defaults
ttsforge config --set default_voice am_adam
ttsforge config --set default_format mp3
ttsforge config --set use_gpu true

# Reset to defaults
ttsforge config --reset
```

### Filename Templates

Customize output filenames with metadata:

```bash
ttsforge config --set output_filename_template "{author} - {book_title}"
```

Available variables: `{book_title}`, `{author}`, `{chapter_title}`, `{chapter_num}`, `{input_stem}`, `{chapters_range}`

## Voices

ttsforge includes 54 voices across 9 languages:

| Language | Code | Voices | Default |
|----------|------|--------|---------|
| American English | `a` | 20 | `af_heart` |
| British English | `b` | 8 | `bf_emma` |
| Spanish | `e` | 3 | `ef_dora` |
| French | `f` | 1 | `ff_siwis` |
| Hindi | `h` | 4 | `hf_alpha` |
| Italian | `i` | 2 | `if_sara` |
| Japanese | `j` | 5 | `jf_alpha` |
| Brazilian Portuguese | `p` | 3 | `pf_dora` |
| Mandarin Chinese | `z` | 8 | `zf_xiaoxiao` |

Voice naming: `{lang}{gender}_{name}` (e.g., `am_adam` = American Male "Adam")

### Voice Demo

```bash
# Demo all voices
ttsforge demo

# Demo specific language
ttsforge demo -l a

# Save individual voice files
ttsforge demo --separate -o ./voices/
```

### Voice Blending

```bash
ttsforge convert book.epub --voice-blend "af_nicole:50,am_michael:50"
```

## Commands

| Command | Description |
|---------|-------------|
| `convert` | Convert EPUB to audiobook |
| `list` | List chapters in EPUB |
| `info` | Show EPUB metadata |
| `sample` | Generate sample audio |
| `voices` | List available voices |
| `demo` | Generate voice demo |
| `download` | Download ONNX models |
| `config` | Manage configuration |
| `phonemes export` | Export EPUB to phonemes |
| `phonemes convert` | Convert phonemes to audio |
| `phonemes info` | Show phoneme file info |
| `phonemes preview` | Preview text as phonemes |

## GPU Acceleration

For faster processing with CUDA:

```bash
pip install onnxruntime-gpu
ttsforge config --set use_gpu true
```

Or use per-command:

```bash
ttsforge convert book.epub --gpu
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `default_voice` | `af_heart` | Default TTS voice |
| `default_language` | `a` | Default language code |
| `default_speed` | `1.0` | Speech speed (0.5-2.0) |
| `default_format` | `m4b` | Output format |
| `use_gpu` | `false` | Enable GPU acceleration |
| `silence_between_chapters` | `2.0` | Chapter gap (seconds) |
| `segment_pause_min` | `0.1` | Min sentence pause |
| `segment_pause_max` | `0.3` | Max sentence pause |
| `paragraph_pause_min` | `0.5` | Min paragraph pause |
| `paragraph_pause_max` | `1.0` | Max paragraph pause |
| `output_filename_template` | `{book_title}` | Output filename template |

## Documentation

Full documentation: https://ttsforge.readthedocs.io/

Build locally:
```bash
cd docs
pip install sphinx sphinx-rtd-theme
make html
```

## Requirements

- Python 3.10+
- ffmpeg (for M4B/OPUS)
- espeak-ng (for phonemization)
- ~330MB disk space (ONNX models)

## License

MIT License

## Credits

- [Kokoro](https://github.com/hexgrad/kokoro) - TTS model
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) - Phonemization
- [ONNX Runtime](https://onnxruntime.ai/) - Model inference
