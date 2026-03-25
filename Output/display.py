# Prints/logs results to terminal in real time

from __future__ import annotations

import datetime
import sys
from typing import Generator

from colorama import Fore, Back, Style, init as colorama_init

from Ai.interpreter import Interpretation, SignalAnomaly
from Ai.transcriber import Transcription

colorama_init(autoreset=True)

# ── channel colour palette ────────────────────────────────────────────────────
# Cycles through distinct foreground colours so each channel is always the
# same colour within a session.

_CHANNEL_COLOURS = [
    Fore.CYAN,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.MAGENTA,
    Fore.BLUE,
    Fore.WHITE,
]

_CONFIDENCE_COLOURS = {
    "HIGH":   Fore.RED   + Style.BRIGHT,
    "MEDIUM": Fore.YELLOW + Style.BRIGHT,
    "LOW":    Fore.WHITE  + Style.DIM,
}

_TERM_WIDTH = 72


# ── helpers ───────────────────────────────────────────────────────────────────

def _rule(char: str = "─", colour: str = Fore.WHITE + Style.DIM) -> str:
    return colour + (char * _TERM_WIDTH) + Style.RESET_ALL


def _ts(unix: float) -> str:
    return datetime.datetime.fromtimestamp(unix).strftime("%H:%M:%S")


class _ChannelPalette:
    """Maps channel labels to a consistent foreground colour for the session."""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}
        self._idx = 0

    def colour(self, label: str) -> str:
        if label not in self._map:
            self._map[label] = _CHANNEL_COLOURS[self._idx % len(_CHANNEL_COLOURS)]
            self._idx += 1
        return self._map[label]


_palette = _ChannelPalette()


# ── section renderers ─────────────────────────────────────────────────────────

def _render_header(interp: Interpretation) -> None:
    """Top bar — timestamp range, channel list, batch stats."""
    t_range = f"{_ts(interp.earliest_ts)} → {_ts(interp.latest_ts)}"
    channels = "  ".join(
        _palette.colour(lbl) + lbl + Style.RESET_ALL
        for lbl in interp.freq_labels
    )
    print(_rule("═", Fore.WHITE + Style.BRIGHT))
    print(
        Style.BRIGHT + f"  GHOSTBOX  " + Style.RESET_ALL
        + Fore.WHITE + Style.DIM + f"{t_range}   "
        + Style.RESET_ALL + f"batch:{interp.batch_size}   "
        + f"model:{interp.model}"
    )
    print(f"  channels  {channels}")
    print(_rule("─", Fore.WHITE + Style.DIM))


def _render_transcripts(batch_transcriptions: list[Transcription]) -> None:
    """One line per transcript, coloured by channel."""
    if not batch_transcriptions:
        return

    print(Style.BRIGHT + "  TRANSCRIPTS" + Style.RESET_ALL)
    for t in batch_transcriptions:
        col   = _palette.colour(t.label)
        label = col + f"  [{t.label:<10}]" + Style.RESET_ALL
        ts    = Fore.WHITE + Style.DIM + f" {_ts(t.start_time)}" + Style.RESET_ALL
        dur   = Fore.WHITE + Style.DIM + f" {t.duration:.1f}s" + Style.RESET_ALL
        text  = t.text if t.text.strip() else Fore.BLACK + Style.BRIGHT + "(silence)" + Style.RESET_ALL
        print(f"{label}{ts}{dur}  {text}")

    print(_rule("─", Fore.WHITE + Style.DIM))


def _render_no_anomaly() -> None:
    print(
        "  " + Fore.WHITE + Style.DIM
        + "── no anomaly detected ──"
        + Style.RESET_ALL
    )


def _confidence_badge(confidence: str) -> str:
    col = _CONFIDENCE_COLOURS.get(confidence.upper(), Fore.WHITE)
    return col + f"[{confidence.upper()}]" + Style.RESET_ALL


def _render_anomaly(anomaly: SignalAnomaly, index: int) -> None:
    """Render a single flagged anomaly block."""
    badge    = _confidence_badge(anomaly.confidence)
    channels = "  ".join(
        _palette.colour(ch) + ch + Style.RESET_ALL
        for ch in anomaly.channels
    )

    print(
        Style.BRIGHT + f"  ▸ SIGNAL {index}" + Style.RESET_ALL
        + f"  {badge}  {channels}"
    )
    print(
        "    " + Fore.RED + Style.BRIGHT
        + f'"{anomaly.signal}"' + Style.RESET_ALL
    )
    if anomaly.interpretation:
        print(
            "    " + Fore.WHITE + Style.DIM
            + anomaly.interpretation + Style.RESET_ALL
        )


def _render_analysis(interp: Interpretation) -> None:
    """Claude's pattern analysis section."""
    print(Style.BRIGHT + "  ANALYSIS" + Style.RESET_ALL)

    if interp.no_anomaly or not interp.anomalies:
        _render_no_anomaly()
    else:
        for i, anomaly in enumerate(interp.anomalies, 1):
            _render_anomaly(anomaly, i)
            if i < len(interp.anomalies):
                print()

    print(_rule("═", Fore.WHITE + Style.BRIGHT))


# ── public API ────────────────────────────────────────────────────────────────

def render(
    interp: Interpretation,
    transcriptions: list[Transcription] | None = None,
) -> None:
    """
    Print one complete Interpretation block to stdout.

    Parameters
    ----------
    interp          : the Interpretation returned by Interpreter.interpret()
    transcriptions  : the original Transcription batch (for per-line display).
                      If omitted, the transcript section is skipped.
    """
    _render_header(interp)
    if transcriptions:
        _render_transcripts(transcriptions)
    _render_analysis(interp)
    sys.stdout.flush()


def render_transcription(t: Transcription) -> None:
    """
    Print a single Transcription line as it arrives — before Claude has
    processed the batch.  Useful for live monitoring mode.
    """
    col   = _palette.colour(t.label)
    label = col + f"[{t.label}]" + Style.RESET_ALL
    ts    = Fore.WHITE + Style.DIM + f" {_ts(t.start_time)}" + Style.RESET_ALL
    text  = t.text if t.text.strip() else Fore.BLACK + Style.BRIGHT + "(silence)" + Style.RESET_ALL
    print(f"  {label}{ts}  {text}", flush=True)


def stream_display(
    interpretations: Generator[Interpretation, None, None],
    transcription_log: dict[int, list[Transcription]] | None = None,
) -> None:
    """
    Consume an Interpreter.stream() generator and render each result as it
    arrives.

    Parameters
    ----------
    interpretations   : generator from Interpreter.stream()
    transcription_log : optional dict mapping batch index → transcription list,
                        so each rendered block can show the source transcripts.
                        Pass None to skip the transcript section.
    """
    for i, interp in enumerate(interpretations):
        batch_ts = transcription_log.get(i) if transcription_log else None
        render(interp, batch_ts)
