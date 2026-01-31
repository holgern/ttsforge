"""CLI interface for ttsforge - convert EPUB to audiobooks.

This module serves as the main entry point for the ttsforge CLI, organizing
commands into logical groups:

- Conversion commands: convert, read, sample, list, info
- Phoneme commands: phonemes export/convert/preview/info
- Utility commands: voices, demo, download, config, extract-names, list-names
"""

from pathlib import Path
from typing import Optional

import click

from ..constants import PROGRAM_NAME
from .helpers import console, get_version

# Import all command modules
from .commands_conversion import convert, info, list_chapters, read, sample
from .commands_phonemes import phonemes
from .commands_utility import config, demo, download, extract_names, list_names, voices


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit.")
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to custom kokoro.onnx model file.",
)
@click.option(
    "--voices",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to custom voices.bin file.",
)
@click.pass_context
def main(
    ctx: click.Context, version: bool, model: Path | None, voices: Path | None
) -> None:
    """ttsforge - Generate audiobooks from EPUB files with TTS."""
    ctx.ensure_object(dict)
    ctx.obj["model_path"] = model
    ctx.obj["voices_path"] = voices
    if version:
        console.print(f"[bold]{PROGRAM_NAME}[/bold] version {get_version()}")
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register all commands with the main group
main.add_command(convert)
main.add_command(list_chapters, name="list")
main.add_command(info)
main.add_command(sample)
main.add_command(read)
main.add_command(voices)
main.add_command(demo)
main.add_command(download)
main.add_command(config)
main.add_command(phonemes)
main.add_command(extract_names)
main.add_command(list_names)

# Export main for backward compatibility
__all__ = ["main"]


if __name__ == "__main__":
    main()
