# ttsforge/audio_merge.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from .constants import SAMPLE_RATE
from .utils import create_process, get_ffmpeg_path


@dataclass(slots=True)
class MergeMeta:
    fmt: str
    silence_between_chapters: float
    title: str | None = None
    author: str | None = None
    cover_image: Path | None = None


class AudioMerger:
    class LogCallback(Protocol):
        def __call__(self, message: str, level: str = "info") -> None: ...

    def __init__(self, log: LogCallback):
        self.log = log

    def add_chapters_to_m4b(
        self, output_path: Path, chapters: list[dict[str, Any]], cover: Path | None
    ) -> None:
        if len(chapters) <= 1:
            return
        ffmpeg = get_ffmpeg_path()

        chapters_file = output_path.with_suffix(".chapters.txt")
        chapters_file.write_text(self._ffmetadata(chapters), encoding="utf-8")

        tmp_path = output_path.with_suffix(".tmp.m4b")
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(output_path),
            "-i",
            str(chapters_file),
            "-map",
            "0:a",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:a",
            "copy",
        ]

        if cover and cover.exists():
            cmd += [
                "-i",
                str(cover),
                "-map",
                "2",
                "-c:v",
                "copy",
                "-disposition:v",
                "attached_pic",
            ]

        cmd.append(str(tmp_path))
        proc = create_process(cmd, suppress_output=True)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("ffmpeg failed while adding m4b chapters")

        tmp_path.replace(output_path)
        chapters_file.unlink(missing_ok=True)

    def merge_chapter_wavs(
        self,
        chapter_files: list[Path],
        chapter_durations: list[float],
        chapter_titles: list[str],
        output_path: Path,
        meta: MergeMeta,
    ) -> None:
        ffmpeg = get_ffmpeg_path()

        concat_file = output_path.with_suffix(".concat.txt")
        silence_file = output_path.parent / "_silence.wav"

        if meta.silence_between_chapters > 0 and len(chapter_files) > 1:
            self._write_silence_wav(silence_file, meta.silence_between_chapters)

        with concat_file.open("w", encoding="utf-8") as f:
            for i, ch in enumerate(chapter_files):
                f.write(f"file '{ch.absolute()}'\n")
                if i < len(chapter_files) - 1 and meta.silence_between_chapters > 0:
                    f.write(f"file '{silence_file.absolute()}'\n")

        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file)]

        if meta.fmt == "m4b":
            if meta.cover_image and meta.cover_image.exists():
                cmd += [
                    "-i",
                    str(meta.cover_image),
                    "-map",
                    "0:a",
                    "-map",
                    "1",
                    "-c:v",
                    "copy",
                    "-disposition:v",
                    "attached_pic",
                ]
            cmd += [
                "-c:a",
                "aac",
                "-q:a",
                "2",
                "-movflags",
                "+faststart+use_metadata_tags",
            ]
            if meta.title:
                cmd += ["-metadata", f"title={meta.title}"]
            if meta.author:
                cmd += ["-metadata", f"artist={meta.author}"]
        elif meta.fmt == "opus":
            cmd += ["-c:a", "libopus", "-b:a", "24000"]
        elif meta.fmt == "mp3":
            cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
        elif meta.fmt == "flac":
            cmd += ["-c:a", "flac"]
        elif meta.fmt == "wav":
            cmd += ["-c:a", "pcm_s16le"]

        cmd.append(str(output_path))
        proc = create_process(cmd, suppress_output=True)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("ffmpeg failed while merging chapters")

        concat_file.unlink(missing_ok=True)
        silence_file.unlink(missing_ok=True)

        if meta.fmt == "m4b" and len(chapter_files) > 1:
            times = []
            t = 0.0
            for i, (dur, title) in enumerate(
                zip(chapter_durations, chapter_titles, strict=False)
            ):
                times.append({"title": title, "start": t, "end": t + dur})
                t += dur
                if i < len(chapter_durations) - 1:
                    t += meta.silence_between_chapters
            self.add_chapters_to_m4b(output_path, times, meta.cover_image)

    def _write_silence_wav(self, path: Path, duration: float) -> None:
        samples = int(duration * SAMPLE_RATE)
        audio = np.zeros(samples, dtype="float32")
        with sf.SoundFile(
            str(path), "w", samplerate=SAMPLE_RATE, channels=1, format="wav"
        ) as f:
            f.write(audio)

    def _ffmetadata(self, chapters: list[dict[str, Any]]) -> str:
        lines = [";FFMETADATA1"]
        for ch in chapters:
            title = str(ch["title"]).replace("=", "\\=")
            lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={int(ch['start'] * 1000)}",
                f"END={int(ch['end'] * 1000)}",
                f"title={title}",
                "",
            ]
        return "\n".join(lines)
