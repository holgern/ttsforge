"""Utility functions for ttsforge - config, GPU detection, encoding, etc."""

import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional

from platformdirs import user_cache_dir, user_config_dir

from .constants import DEFAULT_CONFIG, PROGRAM_NAME

warnings.filterwarnings("ignore")

# Default encoding for subprocess
DEFAULT_ENCODING = sys.getfilesystemencoding()


def get_user_config_path() -> Path:
    """Get path to user configuration file."""
    if platform.system() != "Windows":
        # On non-Windows, prefer ~/.config/ttsforge if it already exists
        custom_dir = Path.home() / ".config" / "ttsforge"
        if custom_dir.exists():
            config_dir = custom_dir
        else:
            config_dir = Path(
                user_config_dir(
                    "ttsforge", appauthor=False, roaming=True, ensure_exists=True
                )
            )
    else:
        config_dir = Path(
            user_config_dir(
                "ttsforge", appauthor=False, roaming=True, ensure_exists=True
            )
        )
    return config_dir / "config.json"


def get_user_cache_path(folder: Optional[str] = None) -> Path:
    """Get path to user cache directory, optionally with a subfolder."""
    cache_dir = Path(
        user_cache_dir("ttsforge", appauthor=False, opinion=True, ensure_exists=True)
    )
    if folder:
        cache_dir = cache_dir / folder
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_config() -> dict[str, Any]:
    """Load configuration from file, returning defaults if not found."""
    try:
        config_path = get_user_config_path()
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                user_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**DEFAULT_CONFIG, **user_config}
    except Exception:
        pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> bool:
    """Save configuration to file. Returns True on success."""
    try:
        config_path = get_user_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


def reset_config() -> dict[str, Any]:
    """Reset configuration to defaults and save."""
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()


def detect_encoding(file_path: str | Path) -> str:
    """Detect the encoding of a file using chardet/charset_normalizer."""
    import chardet
    import charset_normalizer

    with open(file_path, "rb") as f:
        raw_data = f.read()

    detected_encoding = None
    for detector in (charset_normalizer, chardet):
        try:
            result = detector.detect(raw_data)
            if result and result.get("encoding"):
                detected_encoding = result["encoding"]
                break
        except Exception:
            continue

    encoding = detected_encoding if detected_encoding else "utf-8"
    return encoding.lower()


def get_gpu_info(enabled: bool = True) -> tuple[str, bool]:
    """
    Check GPU acceleration availability for ONNX runtime.

    Args:
        enabled: Whether GPU acceleration is requested

    Returns:
        Tuple of (status message, is_gpu_available)
    """
    if not enabled:
        return "GPU disabled in config. Using CPU.", False

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()

        # Check for CUDA
        if "CUDAExecutionProvider" in providers:
            return "CUDA GPU available via ONNX Runtime.", True

        # Check for CoreML (Apple)
        if "CoreMLExecutionProvider" in providers:
            return "CoreML GPU available via ONNX Runtime.", True

        # Check for DirectML (Windows)
        if "DmlExecutionProvider" in providers:
            return "DirectML GPU available via ONNX Runtime.", True

        return f"No GPU providers available. Using CPU. (Available: {providers})", False
    except ImportError:
        return "ONNX Runtime not installed. Using CPU.", False
    except Exception as e:
        return f"Error checking GPU: {e}", False


def get_device(use_gpu: bool = True) -> str:
    """
    Get the appropriate execution provider for ONNX Runtime.

    Args:
        use_gpu: Whether to attempt GPU usage

    Returns:
        Execution provider name: 'CUDAExecutionProvider', 'CoreMLExecutionProvider', or 'CPUExecutionProvider'
    """
    if not use_gpu:
        return "CPUExecutionProvider"

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()

        # Prefer CUDA
        if "CUDAExecutionProvider" in providers:
            return "CUDAExecutionProvider"

        # CoreML for Apple
        if "CoreMLExecutionProvider" in providers:
            return "CoreMLExecutionProvider"

        # DirectML for Windows
        if "DmlExecutionProvider" in providers:
            return "DmlExecutionProvider"

    except ImportError:
        pass

    return "CPUExecutionProvider"


def create_process(
    cmd: list[str] | str,
    stdin: Optional[int] = None,
    text: bool = True,
    capture_output: bool = False,
    suppress_output: bool = False,
) -> subprocess.Popen:
    """
    Create a subprocess with proper platform handling.

    Args:
        cmd: Command to execute (list or string)
        stdin: stdin pipe option (e.g., subprocess.PIPE)
        text: Whether to use text mode
        capture_output: Whether to capture output
        suppress_output: Suppress all output (for rich progress bars)

    Returns:
        Popen object
    """
    use_shell = isinstance(cmd, str)
    kwargs: dict[str, Any] = {
        "shell": use_shell,
        "bufsize": 1,
    }

    # Suppress output if requested (avoids rich progress interference)
    if suppress_output:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    else:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT

    if text and not suppress_output:
        kwargs["text"] = True
        kwargs["encoding"] = DEFAULT_ENCODING
        kwargs["errors"] = "replace"
    elif not suppress_output:
        kwargs["text"] = False
        kwargs["bufsize"] = 0

    if stdin is not None:
        kwargs["stdin"] = stdin

    if platform.system() == "Windows":
        startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
        startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
        kwargs.update(
            {"startupinfo": startupinfo, "creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
        )

    proc = subprocess.Popen(cmd, **kwargs)

    # Stream output to console in real-time if not capturing or suppressing
    if proc.stdout and not capture_output and not suppress_output:

        def _stream_output(stream: Any) -> None:
            if text:
                for line in stream:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            else:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    try:
                        sys.stdout.write(
                            chunk.decode(DEFAULT_ENCODING, errors="replace")
                        )
                        sys.stdout.flush()
                    except Exception:
                        pass
            stream.close()

        Thread(target=_stream_output, args=(proc.stdout,), daemon=True).start()

    return proc


def ensure_ffmpeg() -> bool:
    """
    Ensure ffmpeg is available, installing static-ffmpeg if needed.

    Returns:
        True if ffmpeg is available
    """
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
        return True
    except ImportError:
        return False


def load_tts_pipeline() -> tuple[Any, Any]:
    """
    Load numpy and Kokoro ONNX TTS backend.

    Returns:
        Tuple of (numpy module, KokoroONNX class)
    """
    import numpy as np

    from .onnx_backend import KokoroONNX

    return np, KokoroONNX


class LoadPipelineThread(Thread):
    """Thread for loading TTS pipeline in background."""

    def __init__(self, callback: Callable[[Any, Any, Optional[str]], None]) -> None:
        super().__init__()
        self.callback = callback

    def run(self) -> None:
        try:
            np_module, kokoro_class = load_tts_pipeline()
            self.callback(np_module, kokoro_class, None)
        except Exception as e:
            self.callback(None, None, str(e))


# Sleep prevention for long conversions
_sleep_procs: dict[str, Optional[subprocess.Popen[str]]] = {
    "Darwin": None,
    "Linux": None,
}


def prevent_sleep_start() -> None:
    """Prevent system from sleeping during conversion."""
    system = platform.system()
    if system == "Windows":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
            0x80000000 | 0x00000001 | 0x00000040
        )
    elif system == "Darwin":
        _sleep_procs["Darwin"] = create_process(["caffeinate"], capture_output=True)
    elif system == "Linux":
        import shutil

        if shutil.which("systemd-inhibit"):
            _sleep_procs["Linux"] = create_process(
                [
                    "systemd-inhibit",
                    f"--who={PROGRAM_NAME}",
                    "--why=Prevent sleep during TTS conversion",
                    "--what=sleep",
                    "--mode=block",
                    "sleep",
                    "infinity",
                ],
                capture_output=True,
            )


def prevent_sleep_end() -> None:
    """Allow system to sleep again."""
    system = platform.system()
    if system == "Windows":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # type: ignore[attr-defined]
    elif system in ("Darwin", "Linux"):
        proc = _sleep_procs.get(system)
        if proc is not None:
            try:
                proc.terminate()
                _sleep_procs[system] = None
            except Exception:
                pass


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    Sanitize a string for use as a filename.

    Args:
        name: The string to sanitize
        max_length: Maximum length of the result

    Returns:
        Sanitized filename
    """
    import re

    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", name)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r"[\s_]+", "_", sanitized).strip("_")
    # Truncate if needed
    if len(sanitized) > max_length:
        # Try to break at underscore
        pos = sanitized[:max_length].rfind("_")
        sanitized = sanitized[: pos if pos > 0 else max_length].rstrip("_")
    return sanitized or "output"


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
