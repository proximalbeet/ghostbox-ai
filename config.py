# Frequency ranges, API keys, and settings

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050    # Hz — samples per second
CHUNK_SIZE     = 4096     # samples per yielded chunk
BUFFER_MAXLEN  = 64       # max chunks held per channel before oldest is dropped

# ── Processor ─────────────────────────────────────────────────────────────────
NOISE_GATE_RMS     = 0.02   # RMS below this is treated as silence and gated out
SEGMENT_MIN_SEC    = 0.5    # shortest valid speech segment to emit (seconds)
SEGMENT_MAX_SEC    = 3.0    # flush a segment at this length even if still active
SILENCE_HANGOVER   = 0.3    # seconds of post-speech silence before segment closes

# ── Scanner mode ──────────────────────────────────────────────────────────────
# "simulated" → white-noise generator (default)
# "file"      → read a WAV/MP3 from Data/Samples/ (temporary — remove when SDR arrives)
SCANNER_MODE = "file"

# ── File source (temporary — used only when SCANNER_MODE = "file") ─────────────
FILE_SOURCE_PATH = "Data/Samples/Police Radio Chatter 3.mp3"
FILE_SOURCE_LOOP = True   # loop the file instead of stopping at EOF

# Each tuple is (center_frequency_hz, label, offset_seconds).
# The same file is loaded once and each instance starts offset_seconds into it
# so the interpreter sees overlapping-but-distinct transcription fragments.
FILE_SOURCE_CHANNELS = [
    (155_340_000, "FileSource-1", 0.0),
    (154_280_000, "FileSource-2", 0.5),
    (462_562_500, "FileSource-3", 1.0),
    (121_500_000, "FileSource-4", 1.5),
]

# ── Simulated scanner ─────────────────────────────────────────────────────────
# Each tuple is (center_frequency_hz, label)
SIMULATED_FREQUENCIES = [
    (462_562_500, "FRS-1"),    # FRS/GMRS ch 1
    (154_280_000, "Fire"),     # fire dispatch band
    (155_340_000, "Police"),   # law-enforcement simplex
    (121_500_000, "AirEmerg"), # aviation emergency
    (27_185_000,  "CB-19"),    # CB radio ch 19
]

NOISE_AMPLITUDE   = 0.08   # 0.0–1.0 — baseline white-noise level
TONE_PROBABILITY  = 0.03   # chance per chunk that a voice-tone event starts
TONE_DURATION_SEC = 0.8    # seconds a single injected tone lasts
TONE_FREQ_HZ      = 1200   # carrier frequency of the simulated voice tone (Hz)
TONE_AMPLITUDE    = 0.35   # 0.0–1.0 — amplitude of injected tone

# ── Whisper ───────────────────────────────────────────────────────────────────
# Available sizes: tiny, base, small, medium, large  (speed vs accuracy tradeoff)
WHISPER_MODEL      = "tiny"
WHISPER_LANGUAGE   = "en"  # lock language — skips auto-detect, avoids misdetection on noise
WHISPER_TIMEOUT_SEC = 10   # skip a segment if Whisper takes longer than this

# ── Interpreter ───────────────────────────────────────────────────────────────
CLAUDE_MODEL        = "claude-opus-4-6"   # model used for signal interpretation
INTERPRETER_MAX_TOKENS = 1024            # max tokens in Claude's response
BATCH_SIZE          = 3                  # max Transcriptions per Claude request

# ── AI / API keys ─────────────────────────────────────────────────────────────
# Loaded from .env at runtime — do not hard-code real keys here
ANTHROPIC_API_KEY = ""     # fallback if .env is absent
