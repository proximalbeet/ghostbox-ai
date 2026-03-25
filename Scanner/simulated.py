# Simulated fake frequency scanner. Contains noise & injected audio

import time
import threading
from typing import Generator

import numpy as np
from scipy.signal import butter, sosfilt

import config


def _bandpass(data: np.ndarray, low: float, high: float, fs: int) -> np.ndarray:
    """Apply a butterworth bandpass filter to colour the noise realistically."""
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def _make_chunk(
    freq_hz: int,
    sample_rate: int,
    chunk_size: int,
    inject_tone: bool,
    tone_sample_offset: int,
) -> np.ndarray:
    """
    Generate one audio chunk of white noise with an optional injected tone.

    Parameters
    ----------
    freq_hz          : centre frequency of the simulated channel (for colouring)
    sample_rate      : samples per second
    chunk_size       : number of samples in the returned array
    inject_tone      : whether to mix a voice-like tone into this chunk
    tone_sample_offset: sample index into the ongoing tone (for phase continuity)

    Returns
    -------
    float32 numpy array in the range [-1.0, 1.0]
    """
    # White noise baseline
    noise = np.random.normal(0.0, config.NOISE_AMPLITUDE, chunk_size).astype(np.float32)

    # Tint the noise spectrum based on the band — low bands sound fuller
    if freq_hz < 50_000_000:          # HF / CB
        noise = _bandpass(noise, 200, 3400, sample_rate).astype(np.float32)
    elif freq_hz < 200_000_000:       # VHF
        noise = _bandpass(noise, 300, 3000, sample_rate).astype(np.float32)
    else:                             # UHF
        noise = _bandpass(noise, 400, 2800, sample_rate).astype(np.float32)

    if inject_tone:
        # AM-modulated sine wave — mimics narrow-band voice (NFM carrier)
        t = (np.arange(chunk_size) + tone_sample_offset) / sample_rate
        carrier   = np.sin(2 * np.pi * config.TONE_FREQ_HZ * t)
        # Low-frequency amplitude envelope simulates speech rhythm
        mod_freq  = 4.0  # Hz
        modulator = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * mod_freq * t))
        tone      = (carrier * modulator * config.TONE_AMPLITUDE).astype(np.float32)
        noise    += tone

    # Hard-clip to [-1, 1]
    return np.clip(noise, -1.0, 1.0)


class SimulatedFrequency:
    """
    Simulates a single RTL-SDR channel tuned to *freq_hz*.

    Usage
    -----
        scanner = SimulatedFrequency(462_562_500, "FRS-1")
        for chunk, meta in scanner.stream():
            # chunk  → float32 numpy array of length CHUNK_SIZE
            # meta   → dict with frequency, label, has_tone, timestamp
            process(chunk, meta)
    """

    def __init__(self, freq_hz: int, label: str = ""):
        self.freq_hz     = freq_hz
        self.label       = label or str(freq_hz)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the stream generator to stop."""
        self._stop_event.set()

    def stream(self) -> Generator[tuple[np.ndarray, dict], None, None]:
        """
        Yield (chunk, meta) indefinitely until stop() is called.

        Each chunk is a float32 numpy array of length config.CHUNK_SIZE.
        meta keys: freq_hz, label, has_tone, timestamp
        """
        sample_rate  = config.SAMPLE_RATE
        chunk_size   = config.CHUNK_SIZE
        tone_samples = int(config.TONE_DURATION_SEC * sample_rate)

        tone_remaining    = 0   # samples left in the current tone event
        tone_sample_index = 0   # absolute sample counter for phase continuity

        while not self._stop_event.is_set():
            # Decide whether to start a new tone event this chunk
            if tone_remaining <= 0 and np.random.random() < config.TONE_PROBABILITY:
                tone_remaining    = tone_samples
                tone_sample_index = 0

            inject     = tone_remaining > 0
            chunk      = _make_chunk(
                self.freq_hz, sample_rate, chunk_size,
                inject, tone_sample_index
            )

            yield chunk, {
                "freq_hz":   self.freq_hz,
                "label":     self.label,
                "has_tone":  inject,
                "timestamp": time.time(),
            }

            if inject:
                tone_remaining    -= chunk_size
                tone_sample_index += chunk_size


# ── FILE SOURCE (temporary) ───────────────────────────────────────────────────
# Remove this entire block once sdr.py is implemented and the RTL-SDR dongle arrives.

def _load_audio(path: str) -> np.ndarray:
    """
    Load a WAV or MP3, mix to mono, resample to config.SAMPLE_RATE.
    Called once by build_scanners(); the result is shared across all FileSource
    instances so the file is only decoded once regardless of channel count.
    """
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if file_sr != config.SAMPLE_RATE:
        g     = gcd(config.SAMPLE_RATE, file_sr)
        audio = resample_poly(audio, config.SAMPLE_RATE // g, file_sr // g)

    return audio.astype(np.float32)


class FileSource:
    """
    Feeds a shared audio buffer through the pipeline as a fake scanner channel.

    Each instance starts at a different sample offset so the interpreter
    receives overlapping-but-distinct fragments across channels.

    The interface is identical to SimulatedFrequency so CaptureManager accepts
    it without modification.

    Temporary — will be removed when Scanner/sdr.py is implemented.
    """

    def __init__(
        self,
        audio:      np.ndarray,
        freq_hz:    int,
        label:      str   = "FileSource",
        offset_sec: float = 0.0,
    ):
        self.freq_hz     = freq_hz
        self.label       = label
        self._audio      = audio
        self._start_pos  = int(offset_sec * config.SAMPLE_RATE)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def stream(self) -> Generator[tuple[np.ndarray, dict], None, None]:
        chunk_size = config.CHUNK_SIZE
        pos        = self._start_pos % len(self._audio)

        while not self._stop_event.is_set():
            end   = pos + chunk_size
            chunk = self._audio[pos:end]

            if len(chunk) < chunk_size:
                # Wrap around end of file
                remainder = chunk_size - len(chunk)
                chunk     = np.concatenate([chunk, self._audio[:remainder]])
                pos       = remainder
                if not config.FILE_SOURCE_LOOP:
                    yield chunk.astype(np.float32), {
                        "freq_hz":   self.freq_hz,
                        "label":     self.label,
                        "has_tone":  True,
                        "timestamp": time.time(),
                    }
                    break
            else:
                pos = end

            yield chunk.astype(np.float32), {
                "freq_hz":   self.freq_hz,
                "label":     self.label,
                "has_tone":  True,
                "timestamp": time.time(),
            }

# ── END FILE SOURCE ───────────────────────────────────────────────────────────


def build_scanners() -> list[SimulatedFrequency] | list[FileSource]:
    """
    Return scanner instances based on config.SCANNER_MODE.

    "simulated" → one SimulatedFrequency per SIMULATED_FREQUENCIES entry
    "file"      → one FileSource per FILE_SOURCE_CHANNELS entry, all sharing
                  the same decoded audio buffer with staggered start offsets
    """
    if config.SCANNER_MODE == "file":
        audio = _load_audio(config.FILE_SOURCE_PATH)
        return [
            FileSource(
                audio      = audio,
                freq_hz    = freq_hz,
                label      = label,
                offset_sec = offset_sec,
            )
            for freq_hz, label, offset_sec in config.FILE_SOURCE_CHANNELS
        ]

    return [
        SimulatedFrequency(freq_hz, label)
        for freq_hz, label in config.SIMULATED_FREQUENCIES
    ]
