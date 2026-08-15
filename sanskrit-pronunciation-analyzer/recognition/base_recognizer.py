"""
base_recognizer.py
-------------------
Defines the abstract contract that every speech-recognition backend
must follow. The GUI only ever talks to this interface, so swapping
"keyword matching" for Whisper / Vosk / Wav2Vec2 / a TensorFlow model
later requires ZERO changes to gui/*.

To add a new backend:
    1. Create a new file in recognition/ (e.g. whisper_recognizer.py).
    2. Subclass BaseRecognizer and implement `recognize()`.
    3. Register it in recognizer_factory.py's BACKEND_REGISTRY.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RecognitionResult:
    """Standardized result returned by any recognizer implementation."""
    alphabet: str          # Detected Sanskrit alphabet, e.g. "Ka". Empty string if none detected.
    confidence: float       # 0.0 - 1.0
    raw_transcript: str = ""  # Optional: whatever raw text/label the backend produced

    @property
    def confidence_percent(self) -> int:
        return round(self.confidence * 100)

    @property
    def is_recognized(self) -> bool:
        return bool(self.alphabet)


class BaseRecognizer(ABC):
    """Abstract interface for all Sanskrit-alphabet recognition backends."""

    name: str = "base"

    @abstractmethod
    def recognize(self, samples: np.ndarray, sample_rate: int) -> RecognitionResult:
        """
        Analyze recorded audio and return the detected Sanskrit alphabet.

        Args:
            samples: 1-D float32 numpy array of mono audio samples.
            sample_rate: Sample rate of `samples` in Hz.

        Returns:
            RecognitionResult with the best guess, or an empty-alphabet
            result if nothing could be confidently detected.
        """
        raise NotImplementedError
