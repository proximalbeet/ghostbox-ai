# Entry point that runs the entire pipeline

from __future__ import annotations

import sys
import time

from colorama import Fore, Style, init as colorama_init

# ── Home (root) ───────────────────────────────────────────────────────────────
import config

# ── Scanner ───────────────────────────────────────────────────────────────────
from Scanner.simulated import build_scanners

# ── Audio ─────────────────────────────────────────────────────────────────────
from Audio.capture    import CaptureManager
from Audio.processor  import AudioProcessor

# ── Ai ────────────────────────────────────────────────────────────────────────
from Ai.transcriber  import Transcriber
from Ai.interpreter  import Interpreter, Interpretation

# ── Output ────────────────────────────────────────────────────────────────────
from Output.display  import render, render_transcription

colorama_init(autoreset=True)

_WIDTH = 72


# ── startup banner ────────────────────────────────────────────────────────────

def _print_banner() -> None:
    rule  = Fore.WHITE + Style.DIM + ("═" * _WIDTH) + Style.RESET_ALL
    title = Style.BRIGHT + "  G H O S T B O X" + Style.RESET_ALL

    print()
    print(rule)
    print(title)
    print(rule)

    print(
        Fore.WHITE + Style.DIM
        + f"  whisper model   : " + Style.RESET_ALL
        + config.WHISPER_MODEL
    )
    print(
        Fore.WHITE + Style.DIM
        + f"  claude model    : " + Style.RESET_ALL
        + config.CLAUDE_MODEL
    )
    print(
        Fore.WHITE + Style.DIM
        + f"  batch size      : " + Style.RESET_ALL
        + str(config.BATCH_SIZE) + " transcriptions"
    )
    print(
        Fore.WHITE + Style.DIM
        + f"  sample rate     : " + Style.RESET_ALL
        + f"{config.SAMPLE_RATE} Hz"
    )

    print()
    print(Style.BRIGHT + "  ACTIVE CHANNELS" + Style.RESET_ALL)

    _channel_colours = [
        Fore.CYAN, Fore.GREEN, Fore.YELLOW,
        Fore.MAGENTA, Fore.BLUE, Fore.WHITE,
    ]
    for i, (freq_hz, label) in enumerate(config.SIMULATED_FREQUENCIES):
        col  = _channel_colours[i % len(_channel_colours)]
        mhz  = freq_hz / 1_000_000
        print(
            f"  {col}{label:<12}{Style.RESET_ALL}"
            + Fore.WHITE + Style.DIM + f"  {mhz:.4f} MHz" + Style.RESET_ALL
        )

    print(rule)
    print(
        Fore.GREEN + Style.BRIGHT
        + "  scanning …  (Ctrl+C to stop)"
        + Style.RESET_ALL
    )
    print(rule)
    print()


def _print_shutdown(start_time: float, batch_count: int) -> None:
    elapsed = time.time() - start_time
    rule    = Fore.WHITE + Style.DIM + ("═" * _WIDTH) + Style.RESET_ALL
    print()
    print(rule)
    print(Style.BRIGHT + "  GHOSTBOX  shutting down" + Style.RESET_ALL)
    print(
        Fore.WHITE + Style.DIM
        + f"  runtime  : {elapsed:.1f}s   batches processed : {batch_count}"
        + Style.RESET_ALL
    )
    print(rule)
    print()


# ── pipeline ──────────────────────────────────────────────────────────────────

def run() -> None:
    _print_banner()

    # 1 — Scanner
    scanners = build_scanners()

    # 2 — Capture
    manager = CaptureManager(scanners)
    manager.start()

    # 3 — Processor
    processor = AudioProcessor(manager)

    # 4 — Transcriber  (loads Whisper weights once here)
    print(
        Fore.WHITE + Style.DIM
        + f"  loading whisper '{config.WHISPER_MODEL}' …"
        + Style.RESET_ALL,
        flush=True,
    )
    transcriber = Transcriber()
    print(
        Fore.GREEN + "  whisper ready" + Style.RESET_ALL,
        flush=True,
    )

    # 5 — Interpreter  (validates API key here — fails fast before scanning)
    interpreter = Interpreter()

    start_time  = time.time()
    batch_count = 0
    batch_log: dict[int, list] = {}   # batch index → transcription list

    try:
        def _live_transcription_stream():
            for t in transcriber.stream(processor):
                render_transcription(t)
                yield t

        for interp in interpreter.stream(_live_transcription_stream()):
            render(interp)
            batch_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
        _print_shutdown(start_time, batch_count)


if __name__ == "__main__":
    run()
