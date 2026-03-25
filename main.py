# Entry point that runs the entire pipeline

# ── Home (root) ──────────────────────────────────────────────────────────────
import config

# ── Data ─────────────────────────────────────────────────────────────────────
# Data.Samples — raw sample storage loaded at startup

# ── Scanner ──────────────────────────────────────────────────────────────────
from Scanner import simulated   # simulated frequency scanner (dev/test)
from Scanner import sdr          # real RTL-SDR scanner (production)

# ── Audio ─────────────────────────────────────────────────────────────────────
from Audio import capture        # records / buffers audio chunks per frequency
from Audio import processor      # cleans, normalizes, and splits into segments

# ── Ai ────────────────────────────────────────────────────────────────────────
from Ai import transcriber       # Whisper — converts audio chunks to rough text
from Ai import interpreter       # Claude API — finds patterns across fragments

# ── Output ────────────────────────────────────────────────────────────────────
from Output import display       # prints / logs results to terminal in real time


if __name__ == '__main__':
    pass
