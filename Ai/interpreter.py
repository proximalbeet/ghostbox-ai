# Holds the claude API and finds patterns across all fragments.

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Generator

import anthropic
from dotenv import load_dotenv

import config
from Ai.transcriber import Transcription

load_dotenv()


# ── system prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a paranormal signal analyst reviewing raw radio transcriptions captured \
across multiple frequency channels simultaneously.

Your job is to examine a batch of rough, noisy Whisper transcripts and identify \
anything that is statistically unlikely to be random noise or scanner artefacts. \
Treat each transcript as a signal fragment that may be incomplete or garbled.

Look for:
- Coherent words or phrases that appear on more than one channel
- Repeated patterns, numbers, or phonemes across frequencies
- Any sequence that carries semantic meaning despite noise
- Temporal clustering — fragments that arrived close together in time
- Anything that would be improbable by chance across independent channels

For each anomaly you find, report:
1. SIGNAL  — the exact text or pattern that caught your attention
2. CHANNELS — which frequency labels it appeared on
3. CONFIDENCE — LOW / MEDIUM / HIGH (how unlikely this is to be random noise)
4. INTERPRETATION — what this fragment may mean or represent

If nothing meaningful is found, respond only with: NO ANOMALY DETECTED

Be terse and precise. Do not speculate beyond what the signals support. \
Do not fabricate content that is not present in the transcripts.\
"""


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class SignalAnomaly:
    """A single pattern flagged by Claude across one or more channels."""
    signal        : str
    channels      : list[str]
    confidence    : str          # LOW | MEDIUM | HIGH
    interpretation: str

    def __repr__(self) -> str:
        return (
            f"SignalAnomaly(confidence={self.confidence},"
            f" channels={self.channels}, signal={self.signal!r})"
        )


@dataclass
class Interpretation:
    """
    Structured response from a single Claude call over a batch of Transcriptions.

    Attributes
    ----------
    anomalies       : list of flagged SignalAnomalys (empty if nothing found)
    raw_response    : Claude's full text reply, unmodified
    batch_size      : number of Transcriptions submitted
    freq_labels     : all channel labels present in this batch
    earliest_ts     : unix timestamp of the earliest segment in the batch
    latest_ts       : unix timestamp of the latest segment in the batch
    model           : Claude model used
    interpreted_at  : unix timestamp when the response was received
    no_anomaly      : True when Claude found nothing meaningful
    """
    anomalies     : list[SignalAnomaly]
    raw_response  : str
    batch_size    : int
    freq_labels   : list[str]
    earliest_ts   : float
    latest_ts     : float
    model         : str
    interpreted_at: float
    no_anomaly    : bool

    def __repr__(self) -> str:
        if self.no_anomaly:
            return f"Interpretation(no_anomaly=True, batch_size={self.batch_size})"
        return (
            f"Interpretation(anomalies={len(self.anomalies)},"
            f" batch_size={self.batch_size},"
            f" channels={self.freq_labels})"
        )


# ── prompt builder ────────────────────────────────────────────────────────────

def _build_user_message(batch: list[Transcription]) -> str:
    """
    Format a batch of Transcriptions into a numbered block for Claude.
    Each entry shows the channel label, frequency, timestamp offset, and text.
    """
    if not batch:
        return ""

    t0 = min(t.start_time for t in batch)
    lines = ["SIGNAL BATCH — analyze the following fragments:\n"]

    for i, t in enumerate(batch, 1):
        offset = t.start_time - t0
        lines.append(
            f"[{i}] CH:{t.label} | {t.freq_hz} Hz | "
            f"+{offset:.1f}s | dur:{t.duration:.1f}s\n"
            f"    TEXT: {t.text or '(empty)'}"
        )

    return "\n".join(lines)


# ── response parser ───────────────────────────────────────────────────────────

def _parse_response(text: str) -> tuple[bool, list[SignalAnomaly]]:
    """
    Lightly parse Claude's free-text reply into SignalAnomaly objects.

    Claude is instructed to use the SIGNAL / CHANNELS / CONFIDENCE /
    INTERPRETATION keys.  We parse line by line; anything that doesn't match
    is ignored so the raw_response is always preserved as the source of truth.
    """
    stripped = text.strip().upper()
    if "NO ANOMALY DETECTED" in stripped:
        return True, []

    anomalies: list[SignalAnomaly] = []
    current: dict = {}

    def _flush() -> None:
        if "signal" in current:
            anomalies.append(SignalAnomaly(
                signal         = current.get("signal", ""),
                channels       = [c.strip() for c in current.get("channels", "").split(",")],
                confidence     = current.get("confidence", "LOW").strip(),
                interpretation = current.get("interpretation", ""),
            ))
        current.clear()

    for line in text.splitlines():
        line = line.strip()
        for key in ("SIGNAL", "CHANNELS", "CONFIDENCE", "INTERPRETATION"):
            if line.upper().startswith(key):
                if key == "SIGNAL" and "signal" in current:
                    _flush()
                value = line[len(key):].lstrip(" :-—")
                current[key.lower()] = value
                break

    _flush()
    return False, anomalies


# ── interpreter ───────────────────────────────────────────────────────────────

class Interpreter:
    """
    Sends batches of Transcriptions to Claude and returns Interpretations.

    A single anthropic.Anthropic client is created on init and reused.  The API
    key is read from the environment (via python-dotenv) first, then falls back
    to config.ANTHROPIC_API_KEY.

    Usage
    -----
        interpreter = Interpreter()

        # one-shot batch
        result = interpreter.interpret(my_transcription_list)

        # continuous stream — collects BATCH_SIZE fragments then fires
        for result in interpreter.stream(transcriber.stream(processor)):
            display(result)
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY") or config.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env or config.ANTHROPIC_API_KEY."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def interpret(self, batch: list[Transcription]) -> Interpretation:
        """
        Send *batch* to Claude in a single API call and return a structured
        Interpretation.  Batches with no text content return a no_anomaly result
        without hitting the API.
        """
        non_empty = [t for t in batch if t.text.strip()]
        if not non_empty:
            return Interpretation(
                anomalies      = [],
                raw_response   = "",
                batch_size     = len(batch),
                freq_labels    = list({t.label for t in batch}),
                earliest_ts    = min(t.start_time for t in batch) if batch else 0.0,
                latest_ts      = max(t.end_time   for t in batch) if batch else 0.0,
                model          = config.CLAUDE_MODEL,
                interpreted_at = time.time(),
                no_anomaly     = True,
            )

        user_message = _build_user_message(non_empty)

        message = self._client.messages.create(
            model      = config.CLAUDE_MODEL,
            max_tokens = config.INTERPRETER_MAX_TOKENS,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        )

        raw = message.content[0].text
        no_anomaly, anomalies = _parse_response(raw)

        return Interpretation(
            anomalies      = anomalies,
            raw_response   = raw,
            batch_size     = len(non_empty),
            freq_labels    = sorted({t.label for t in non_empty}),
            earliest_ts    = min(t.start_time for t in non_empty),
            latest_ts      = max(t.end_time   for t in non_empty),
            model          = config.CLAUDE_MODEL,
            interpreted_at = time.time(),
            no_anomaly     = no_anomaly,
        )

    def stream(
        self,
        transcriptions: Generator[Transcription, None, None],
        batch_size: int = config.BATCH_SIZE,
    ) -> Generator[Interpretation, None, None]:
        """
        Consume a stream of Transcriptions, collect them into batches of
        *batch_size*, and yield one Interpretation per batch.
        """
        batch: list[Transcription] = []
        total_received = 0

        print(f"[DEBUG interpreter] stream started, batch_size={batch_size}", flush=True)

        for transcription in transcriptions:
            total_received += 1
            batch.append(transcription)
            print(f"[DEBUG interpreter] received transcription #{total_received} label={transcription.label!r} text={transcription.text!r} | batch={len(batch)}/{batch_size}", flush=True)

            if len(batch) >= batch_size:
                print(f"[DEBUG interpreter] batch full — firing Claude API call", flush=True)
                yield self.interpret(batch)
                batch = []

        # flush any remaining fragments at end of stream
        if batch:
            print(f"[DEBUG interpreter] stream ended — flushing {len(batch)} remaining", flush=True)
            yield self.interpret(batch)
