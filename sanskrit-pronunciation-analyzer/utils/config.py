"""
config.py
---------
Centralized, typed configuration for the whole application.
Keeping all tunables in one place makes the app easy to extend
(e.g. swapping the recognition backend, changing audio parameters)
without hunting through the codebase.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
ARTICULATION_JSON_PATH = DATA_DIR / "articulation.json"


@dataclass(frozen=True)
class AudioConfig:
    """Parameters controlling microphone capture and recording behaviour."""
    sample_rate: int = 16000          # Hz - good default for speech recognition
    channels: int = 1                 # Mono microphone input
    record_seconds: float = 2.5       # Default recording duration
    block_size: int = 1024            # Frames per audio callback (affects waveform smoothness)
    dtype: str = "float32"            # sounddevice sample format
    waveform_history_seconds: float = 3.0  # How much audio history the live waveform buffer keeps


@dataclass(frozen=True)
class RecognitionConfig:
    """
    Parameters controlling the speech-recognition backend.

    `backend` selects which recognizer implementation the
    RecognizerFactory should build. Backends: "ml", "keyword",
    "whisper", "vosk", "wav2vec2", "tensorflow".
    """
    backend: str = "ml"               # default: offline acoustic ML phoneme recognizer
    language: str = "en-IN"           # language hint passed to online engines
    min_confidence_threshold: float = 0.20  # threshold for ML probability / keyword matching
    fuzzy_match_cutoff: float = 0.75  # difflib similarity cutoff for keyword matching


@dataclass(frozen=True)
class UIConfig:
    """Parameters controlling GUI appearance and behaviour."""
    window_title: str = "Sanskrit Alphabet Pronunciation Analyzer"
    window_size: str = "1000x700"
    appearance_mode: str = "dark"     # customtkinter appearance
    color_theme: str = "green"        # customtkinter color theme
    waveform_update_ms: int = 40      # GUI polling interval for live waveform redraw


# Singleton-style instances used throughout the app
AUDIO_CONFIG = AudioConfig()
RECOGNITION_CONFIG = RecognitionConfig()
UI_CONFIG = UIConfig()

# The full list of supported Sanskrit alphabets (also mirrors data/articulation.json keys)
SUPPORTED_ALPHABETS = [
    "Ka", "Kha", "Ga", "Gha", "Nga",
    "Cha", "Chha", "Ja", "Jha", "Nya",
    "Ta", "Tha", "Da", "Dha", "Na",
    "Pa", "Pha", "Ba", "Bha", "Ma",
    "Ya", "Ra", "La", "Va",
    "Sha", "Sa", "Ha",
]
