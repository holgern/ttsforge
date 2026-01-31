import importlib
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

from ttsforge.utils import (
    atomic_write_json,
    create_process,
    ensure_ffmpeg,
    format_filename_template,
    run_process,
    sanitize_filename,
)


def test_sanitize_filename() -> None:
    assert sanitize_filename("Hello: World/Testing?") == "Hello_WorldTesting"


def test_format_filename_template() -> None:
    result = format_filename_template(
        "{author}_{book_title}", author="Jane Doe", book_title="My Book"
    )
    assert result == "Jane_Doe_My_Book"


def test_run_process_large_output() -> None:
    script = "import sys; sys.stdout.write('x' * (1024 * 1024))"
    result = run_process([sys.executable, "-c", script], text=True)
    assert result.returncode == 0
    assert result.stdout is not None
    assert len(result.stdout) >= 1024 * 1024


def test_create_process_capture_output() -> None:
    result = create_process(
        [sys.executable, "-c", "print('hello')"], capture_output=True
    )
    assert result.returncode == 0
    assert isinstance(result.stdout, str)
    assert "hello" in result.stdout


def test_create_process_suppressed_output() -> None:
    proc = create_process([sys.executable, "-c", "print('hi')"], suppress_output=True)
    assert proc.wait(timeout=5) == 0


def test_atomic_write_json_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"ok": true}', encoding="utf-8")

    def raise_replace(src: str, dst: str) -> None:
        raise OSError("boom")

    original = path.read_text(encoding="utf-8")
    monkeypatch.setattr(os, "replace", raise_replace)
    with pytest.raises(OSError):
        atomic_write_json(path, {"ok": False}, indent=2, ensure_ascii=True)

    assert path.read_text(encoding="utf-8") == original


def test_ensure_ffmpeg_system_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/ffmpeg")

    called = {"imported": False}

    def fake_import(name: str):
        called["imported"] = True
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert ensure_ffmpeg() is True
    assert called["imported"] is False


def test_ensure_ffmpeg_static_available(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"added": False}

    def fake_add_paths() -> None:
        state["added"] = True

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: types.SimpleNamespace(add_paths=fake_add_paths),
    )

    def fake_which(cmd: str):
        if cmd != "ffmpeg":
            return None
        return "/fake/ffmpeg" if state["added"] else None

    monkeypatch.setattr(shutil, "which", fake_which)
    assert ensure_ffmpeg() is True
    assert state["added"] is True


def test_ensure_ffmpeg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    def raise_import(name: str):
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", raise_import)
    with pytest.raises(RuntimeError):
        ensure_ffmpeg()
