# ttsforge/kokoro_lang.py
from __future__ import annotations


def get_onnx_lang_code(ttsforge_lang: str) -> str:
    """Convert ttsforge language code to kokoro ONNX language code."""
    from pykokoro.onnx_backend import LANG_CODE_TO_ONNX

    lang = LANG_CODE_TO_ONNX.get(ttsforge_lang)
    if isinstance(lang, str):
        return lang
    return ttsforge_lang or "en-us"
