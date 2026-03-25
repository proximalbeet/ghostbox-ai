# Whisper; converts audio chunks to rough text

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator

import numpy as np
import whisper

import config
from Audio.processor import AudioSegment, AudioProcessor


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Transcription:
    """
    Raw Whisper output paired with its source channel metadata.

    Attributes
    ----------
    text        : raw transcript string (may be noisy / imperfect)
    freq_hz     : centre frequency of the source channel
    label       : human-readable channel label
    start_time  : unix timestamp of the first sample in the source segment
    end_time    : unix timestamp of the last sample in the source segment
    duration    : length of the audio segment in seconds
    language    : language code detected by Whisper (e.g. "en")
    model_size  : Whisper model used (mirrors config.WHISPER_MODEL)
    transcribed_at: unix timestamp of when transcription completed
    """
    text          : str
    freq_hz       : int
    label         : str
    start_time    : float
    end_time      : float
    duration      : float
    language      : str
    model_size    : str
    transcribed_at: float

    def __repr__(self) -> str:
        return (
            f"Transcription(label={self.label!r}, duration={self.duration:.2f}s,"
            f" text={self.text!r})"
        )


# ── model loader ──────────────────────────────────────────────────────────────

class WhisperModel:
    """
    Thin wrapper around the whisper model that loads once and is reused.

    Whisper expects 16 kHz mono float32 audio.  AudioSegment arrives at
    config.SAMPLE_RATE (22 050 Hz by default), so we resample before passing
    to the model.
    """

    WHISPER_SR       = 16_000   # Whisper's required input sample rate
    WHISPER_MIN_SAMPLES = 8_000 # 0.5s at 16 kHz — minimum Whisper can reshape

    def __init__(self, model_size: str = config.WHISPER_MODEL):
        self.model_size = model_size
        self._model = whisper.load_model(model_size)

    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Linear resample from orig_sr to 16 000 Hz using numpy interpolation.
        Avoids a scipy dependency here since the quality difference is
        negligible for speech at these rates.
        """
        if orig_sr == self.WHISPER_SR:
            return audio
        target_len = int(len(audio) * self.WHISPER_SR / orig_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    def transcribe(self, segment: AudioSegment) -> Transcription | None:
        """
        Run Whisper on a single AudioSegment and return a Transcription.

        Called directly in the main thread so Whisper gets full CPU priority.
        The model is called with fp16=False to stay compatible with CPU-only
        environments.
        """
        audio_16k = self._resample(segment.audio, segment.sample_rate)

        # Guard: Whisper crashes with reshape errors on arrays that are too
        # short, empty, or silent. Reject before touching the model.
        if len(audio_16k) < self.WHISPER_MIN_SAMPLES:
            return None
        if np.max(np.abs(audio_16k)) == 0.0:
            return None

        try:
            result = self._model.transcribe(
                audio_16k,
                fp16=False,
                verbose=False,
                language=config.WHISPER_LANGUAGE,
                condition_on_previous_text=False,
                beam_size=1,
                no_speech_threshold=0.8,
            )
        except Exception:
            return None

        return Transcription(
            text           = result["text"].strip(),
            freq_hz        = segment.freq_hz,
            label          = segment.label,
            start_time     = segment.start_time,
            end_time       = segment.end_time,
            duration       = segment.duration,
            language       = result.get("language", "unknown"),
            model_size     = self.model_size,
            transcribed_at = time.time(),
        )


# ── transcriber ───────────────────────────────────────────────────────────────

class Transcriber:
    """
    Consumes AudioSegments from an AudioProcessor and yields Transcriptions.

    A single WhisperModel instance is shared across all channels so the model
    weights are loaded into memory only once.

    Usage
    -----
        scanners = build_scanners()
        manager  = CaptureManager(scanners)
        manager.start()

        proc        = AudioProcessor(manager)
        transcriber = Transcriber()

        for result in transcriber.stream(proc):
            print(result)

        manager.stop()
    """

    def __init__(self, model_size: str = config.WHISPER_MODEL):
        self._model = WhisperModel(model_size)

    def transcribe_segment(self, segment: AudioSegment) -> Transcription | None:
        """
        Transcribe a single segment.  Returns None on timeout or empty output.
        """
        result = self._model.transcribe(segment)
        if result is None or not result.text:
            return None
        return result

    def stream(
        self, processor: AudioProcessor
    ) -> Generator[Transcription, None, None]:
        """
        Pull segments from *processor* and yield Transcriptions as they arrive.

        Empty transcriptions (Whisper returned blank text) are silently dropped.
        """
        for segment in processor.stream():
            result = self.transcribe_segment(segment)
            if result is not None:
                yield result
