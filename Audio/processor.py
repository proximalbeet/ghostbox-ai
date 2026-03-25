# Cleans up audio, normalizes, and splits into segments

from dataclasses import dataclass, field
from typing import Generator

import numpy as np
from scipy.signal import butter, sosfilt

import config
from Audio.capture import CaptureManager


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class AudioSegment:
    """
    A clean, normalised speech segment ready for transcription.

    Attributes
    ----------
    audio      : float32 numpy array, values in [-1.0, 1.0]
    freq_hz    : centre frequency of the source channel
    label      : human-readable channel label
    start_time : unix timestamp of the first sample in the segment
    end_time   : unix timestamp of the last chunk added
    sample_rate: samples per second (always == config.SAMPLE_RATE)
    duration   : length in seconds
    """
    audio      : np.ndarray
    freq_hz    : int
    label      : str
    start_time : float
    end_time   : float
    sample_rate: int = field(default_factory=lambda: config.SAMPLE_RATE)

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def __repr__(self) -> str:
        return (
            f"AudioSegment(label={self.label!r}, freq={self.freq_hz},"
            f" duration={self.duration:.2f}s)"
        )


# ── DSP helpers ───────────────────────────────────────────────────────────────

def _normalise(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalise to [-1.0, 1.0].
    Returns the original array unchanged if it is silent (peak == 0).
    """
    peak = np.max(np.abs(audio))
    if peak == 0.0:
        return audio
    return (audio / peak).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of an array."""
    return float(np.sqrt(np.mean(audio ** 2)))


def _highpass(audio: np.ndarray, cutoff: float = 80.0) -> np.ndarray:
    """
    Remove sub-80 Hz rumble — keeps voice band clean.
    Uses a 2nd-order Butterworth so the cost per chunk is negligible.
    """
    sos = butter(2, cutoff, btype="high", fs=config.SAMPLE_RATE, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _gate(audio: np.ndarray) -> np.ndarray:
    """
    Hard noise gate: zero out any sample whose absolute value is below the RMS
    threshold.  Operates sample-by-sample so partial chunks are handled
    gracefully.
    """
    mask = np.abs(audio) >= config.NOISE_GATE_RMS
    return (audio * mask).astype(np.float32)


def process_chunk(raw: np.ndarray) -> np.ndarray:
    """
    Full per-chunk processing pipeline:
      1. High-pass filter  — remove sub-80 Hz rumble
      2. Normalise         — peak-normalise to [-1, 1]
      3. Noise gate        — silence samples below RMS threshold

    Returns a cleaned float32 array of the same length.
    """
    audio = _highpass(raw)
    audio = _normalise(audio)
    audio = _gate(audio)
    return audio


# ── segmentation ──────────────────────────────────────────────────────────────

class ChannelProcessor:
    """
    Stateful processor for a single frequency channel.

    Consumes a stream of raw chunks from CaptureManager, runs them through the
    DSP pipeline, and yields complete AudioSegments.

    A segment is emitted when:
      - the audio goes silent for >= SILENCE_HANGOVER seconds after speech, OR
      - the accumulated segment reaches SEGMENT_MAX_SEC (force-flush), OR
      - the stream ends (flush whatever remains if >= SEGMENT_MIN_SEC)

    Segments shorter than SEGMENT_MIN_SEC are discarded as noise bursts.

    Usage
    -----
        proc = ChannelProcessor("Fire")
        for segment in proc.process(manager):
            transcribe(segment)
    """

    def __init__(self, channel_key: int | str):
        self.channel_key = channel_key
        self._reset()

    def _reset(self) -> None:
        self._pending:     list[np.ndarray] = []
        self._start_time:  float | None     = None
        self._end_time:    float            = 0.0
        self._silence_sec: float            = 0.0  # accumulated silence time

    def _pending_duration(self) -> float:
        total_samples = sum(len(c) for c in self._pending)
        return total_samples / config.SAMPLE_RATE

    def _flush(self, label: str, freq_hz: int) -> AudioSegment | None:
        """Concatenate pending chunks into a segment; return None if too short or silent."""
        if not self._pending:
            return None
        audio = np.concatenate(self._pending)
        if len(audio) == 0:
            self._reset()
            return None
        normalised = _normalise(audio)
        # Reject segments that are entirely silent after normalisation
        if np.max(np.abs(normalised)) == 0.0:
            self._reset()
            return None
        seg = AudioSegment(
            audio       = normalised,
            freq_hz     = freq_hz,
            label       = label,
            start_time  = self._start_time,
            end_time    = self._end_time,
        )
        self._reset()
        if seg.duration < config.SEGMENT_MIN_SEC:
            return None                          # too short — noise burst
        return seg

    def process(
        self, manager: CaptureManager
    ) -> Generator[AudioSegment, None, None]:
        """
        Infinite generator.  Pulls chunks from *manager*, processes them, and
        yields AudioSegments.  Stops when the manager stops running.
        """
        chunk_sec      = config.CHUNK_SIZE / config.SAMPLE_RATE
        hangover_sec   = config.SILENCE_HANGOVER
        max_seg_sec    = config.SEGMENT_MAX_SEC

        for raw_chunk, meta in manager.stream(self.channel_key):
            clean = process_chunk(raw_chunk)
            chunk_rms = _rms(clean)
            is_speech = chunk_rms >= config.NOISE_GATE_RMS

            if is_speech:
                # Start a new segment on first active chunk
                if self._start_time is None:
                    self._start_time = meta["timestamp"]
                self._silence_sec = 0.0
                self._pending.append(clean)
                self._end_time = meta["timestamp"]

                # Force-flush if segment exceeds max length
                if self._pending_duration() >= max_seg_sec:
                    seg = self._flush(meta["label"], meta["freq_hz"])
                    if seg is not None:
                        yield seg

            else:
                if self._pending:
                    # Accumulate hangover silence into the segment
                    self._silence_sec += chunk_sec
                    self._pending.append(clean)
                    self._end_time = meta["timestamp"]

                    if self._silence_sec >= hangover_sec:
                        seg = self._flush(meta["label"], meta["freq_hz"])
                        if seg is not None:
                            yield seg

        # Stream ended — flush anything remaining
        if self._pending:
            # Grab label/freq from the last known meta via the buffer label
            buf = manager.get_buffer(self.channel_key)
            seg = self._flush(buf.label, buf.freq_hz)
            if seg is not None:
                yield seg


# ── multi-channel convenience ─────────────────────────────────────────────────

class AudioProcessor:
    """
    Wraps a CaptureManager and exposes a single merged stream of AudioSegments
    across all channels, each running in its own thread.

    Usage
    -----
        scanners = build_scanners()
        manager  = CaptureManager(scanners)
        manager.start()

        proc = AudioProcessor(manager)
        for segment in proc.stream():
            transcribe(segment)
    """

    def __init__(self, manager: CaptureManager):
        self._manager    = manager
        self._processors = [
            ChannelProcessor(buf.freq_hz)
            for buf in manager.channels
        ]

    def stream(self) -> Generator[AudioSegment, None, None]:
        """
        Round-robin across all channel processors and yield segments as they
        arrive.  Each channel processor is driven inline; for true parallelism
        hand each ChannelProcessor.process() call to its own thread and feed
        results into a shared queue.
        """
        import queue
        import threading

        out: queue.Queue[AudioSegment | None] = queue.Queue()
        n_channels = len(self._processors)

        def _worker(proc: ChannelProcessor) -> None:
            for seg in proc.process(self._manager):
                out.put(seg)
            out.put(None)   # sentinel — this channel is done

        threads = [
            threading.Thread(target=_worker, args=(p,), daemon=True)
            for p in self._processors
        ]
        for t in threads:
            t.start()

        done = 0
        while done < n_channels:
            item = out.get()
            if item is None:
                done += 1
            else:
                yield item
