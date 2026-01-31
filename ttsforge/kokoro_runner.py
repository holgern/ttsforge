# ttsforge/kokoro_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import (
    Kokoro,
    VoiceBlend,
    are_models_downloaded,
    download_all_models,
)
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter


@dataclass(slots=True)
class KokoroRunOptions:
    voice: str
    speed: float
    use_gpu: bool
    pause_clause: float
    pause_sentence: float
    pause_paragraph: float
    pause_variance: float
    model_path: Any | None = None
    voices_path: Any | None = None
    voice_blend: str | None = None
    voice_database: Any | None = None
    tokenizer_config: Any | None = None  # pykokoro.tokenizer.TokenizerConfig


class KokoroRunner:
    class LogCallback(Protocol):
        def __call__(self, message: str, level: str = "info") -> None: ...

    def __init__(self, opts: KokoroRunOptions, log: LogCallback):
        self.opts = opts
        self.log = log
        self._kokoro: Kokoro | None = None
        self._pipeline: KokoroPipeline | None = None
        self._voice_style: str | VoiceBlend | None = None

    def ensure_ready(self) -> None:
        if self._pipeline is not None:
            return

        if not are_models_downloaded():
            self.log("Downloading ONNX model files...")
            download_all_models()

        self._kokoro = Kokoro(
            model_path=self.opts.model_path,
            voices_path=self.opts.voices_path,
            use_gpu=self.opts.use_gpu,
            tokenizer_config=self.opts.tokenizer_config,
        )

        assert self._kokoro is not None

        if self.opts.voice_database:
            try:
                self._kokoro.load_voice_database(self.opts.voice_database)
                self.log(f"Loaded voice database: {self.opts.voice_database}")
            except Exception as e:
                self.log(f"Failed to load voice database: {e}", "warning")

        if self.opts.voice_blend:
            self._voice_style = VoiceBlend.parse(self.opts.voice_blend)
        else:
            # if voice_database provides overrides, let Kokoro resolve it
            if self.opts.voice_database:
                db_voice = cast(
                    str | VoiceBlend | None,
                    self._kokoro.get_voice_from_database(self.opts.voice),
                )
                self._voice_style = (
                    db_voice if db_voice is not None else self.opts.voice
                )
            else:
                self._voice_style = self.opts.voice

        # GenerationConfig will be supplied per call
        # because lang / is_phonemes can vary
        pipeline_cfg = PipelineConfig(
            voice=self._voice_style,
            generation=GenerationConfig(speed=self.opts.speed, lang="en-us"),
            model_path=self.opts.model_path,
            voices_path=self.opts.voices_path,
            tokenizer_config=self.opts.tokenizer_config,
        )

        # Use the same adapters everywhere (text + phonemes)
        self._pipeline = KokoroPipeline(
            pipeline_cfg,
            phoneme_processing=OnnxPhonemeProcessorAdapter(self._kokoro),
            audio_generation=OnnxAudioGenerationAdapter(self._kokoro),
            audio_postprocessing=OnnxAudioPostprocessingAdapter(self._kokoro),
        )

    def synthesize(
        self,
        text_or_ssmd: str,
        *,
        lang_code: str,
        pause_mode: Literal["tts", "manual", "auto"],
        is_phonemes: bool = False,
    ) -> np.ndarray:
        self.ensure_ready()
        assert self._pipeline is not None
        gen = GenerationConfig(
            speed=self.opts.speed,
            lang=lang_code,
            is_phonemes=is_phonemes,
            pause_mode=pause_mode,
            pause_clause=self.opts.pause_clause,
            pause_sentence=self.opts.pause_sentence,
            pause_paragraph=self.opts.pause_paragraph,
            pause_variance=self.opts.pause_variance,
        )
        audio = self._pipeline.run(text_or_ssmd, generation=gen).audio
        return cast(np.ndarray, audio)
