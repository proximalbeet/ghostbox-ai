# Records/buffers audio chunks per frequency

import threading
from collections import deque
from typing import Generator

import numpy as np

import config
from Scanner.simulated import SimulatedFrequency


class ChannelBuffer:
    """
    Thread-safe ring buffer for one frequency channel.

    Incoming chunks are appended; when the buffer reaches BUFFER_MAXLEN the
    oldest chunk is silently dropped so fast scanners never stall on a slow
    consumer.
    """

    def __init__(self, freq_hz: int, label: str):
        self.freq_hz = freq_hz
        self.label   = label
        self._buf: deque[tuple[np.ndarray, dict]] = deque(
            maxlen=config.BUFFER_MAXLEN
        )
        self._lock      = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    # ── write side ────────────────────────────────────────────────────────────

    def push(self, chunk: np.ndarray, meta: dict) -> None:
        """Append a chunk from the scanner thread."""
        with self._not_empty:
            self._buf.append((chunk, meta))
            self._not_empty.notify()

    # ── read side ─────────────────────────────────────────────────────────────

    def pop(self, timeout: float = 1.0) -> tuple[np.ndarray, dict] | None:
        """
        Remove and return the oldest (chunk, meta) pair.
        Blocks up to *timeout* seconds if the buffer is empty; returns None on
        timeout.
        """
        with self._not_empty:
            if not self._buf:
                self._not_empty.wait(timeout)
            if self._buf:
                return self._buf.popleft()
            return None

    def drain(self) -> list[tuple[np.ndarray, dict]]:
        """Return all buffered chunks at once and clear the buffer."""
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __repr__(self) -> str:
        return (
            f"ChannelBuffer(freq={self.freq_hz}, label={self.label!r},"
            f" buffered={len(self)})"
        )


class CaptureManager:
    """
    Manages one background thread per scanner instance.

    Each thread pulls chunks from a SimulatedFrequency stream and routes them
    into the matching ChannelBuffer.  The processor layer reads from those
    buffers without touching the scanner threads directly.

    Usage
    -----
        scanners = build_scanners()
        manager  = CaptureManager(scanners)
        manager.start()

        # consume from any channel by label or freq_hz:
        buf = manager.get_buffer("FRS-1")
        chunk, meta = buf.pop()

        # or iterate forever:
        for chunk, meta in manager.stream("Fire"):
            process(chunk, meta)

        manager.stop()
    """

    def __init__(self, scanners: list[SimulatedFrequency]):
        self._scanners: list[SimulatedFrequency] = scanners
        self._buffers:  dict[int, ChannelBuffer]  = {}   # keyed by freq_hz
        self._by_label: dict[str, ChannelBuffer]  = {}   # keyed by label
        self._threads:  list[threading.Thread]    = []
        self._running   = False

        for scanner in scanners:
            buf = ChannelBuffer(scanner.freq_hz, scanner.label)
            self._buffers[scanner.freq_hz]  = buf
            self._by_label[scanner.label]   = buf

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn one capture thread per scanner and begin buffering."""
        if self._running:
            return
        self._running = True
        for scanner in self._scanners:
            t = threading.Thread(
                target=self._capture_loop,
                args=(scanner,),
                name=f"capture-{scanner.label}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        """Signal all scanners to stop and wait for threads to exit."""
        self._running = False
        for scanner in self._scanners:
            scanner.stop()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    # ── internal ──────────────────────────────────────────────────────────────

    def _capture_loop(self, scanner: SimulatedFrequency) -> None:
        buf = self._buffers[scanner.freq_hz]
        for chunk, meta in scanner.stream():
            if not self._running:
                break
            buf.push(chunk, meta)

    # ── access ────────────────────────────────────────────────────────────────

    def get_buffer(self, key: int | str) -> ChannelBuffer:
        """
        Return the ChannelBuffer for a channel.

        *key* may be the integer freq_hz or the string label.
        Raises KeyError if not found.
        """
        if isinstance(key, int):
            return self._buffers[key]
        return self._by_label[key]

    @property
    def channels(self) -> list[ChannelBuffer]:
        """All active ChannelBuffers, in the order scanners were provided."""
        return [self._buffers[s.freq_hz] for s in self._scanners]

    def stream(
        self, key: int | str, timeout: float = 1.0
    ) -> Generator[tuple[np.ndarray, dict], None, None]:
        """
        Infinite generator that yields (chunk, meta) from one channel.

        Yields nothing during empty periods (scanner still warming up or
        between tone events) and resumes automatically.

        Parameters
        ----------
        key     : freq_hz (int) or label (str)
        timeout : seconds to wait per empty-buffer poll
        """
        buf = self.get_buffer(key)
        while self._running:
            result = buf.pop(timeout=timeout)
            if result is not None:
                yield result

    def __repr__(self) -> str:
        return (
            f"CaptureManager(channels={len(self._buffers)},"
            f" running={self._running})"
        )
