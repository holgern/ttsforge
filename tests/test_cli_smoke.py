from pathlib import Path

from click.testing import CliRunner

from ttsforge.cli import main


def test_info_and_list_smoke(tmp_path: Path) -> None:
    text = """Title: Sample Book
Author: Jane Doe
Language: English

CHAPTER I
This is the first chapter.

CHAPTER II
This is the second chapter.
"""
    input_file = tmp_path / "sample.txt"
    input_file.write_text(text, encoding="utf-8")

    runner = CliRunner()
    info_result = runner.invoke(main, ["info", str(input_file)])
    assert info_result.exit_code == 0

    list_result = runner.invoke(main, ["list", str(input_file)])
    assert list_result.exit_code == 0
