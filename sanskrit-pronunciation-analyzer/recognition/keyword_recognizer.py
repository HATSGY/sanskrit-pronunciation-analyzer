"""
keyword_recognizer.py
----------------------
Lightweight, dependency-friendly recognizer used as the initial/default
backend. It transcribes the recorded audio using the `SpeechRecognition`
library (Google Web Speech API by default - works without any local model
download) and then fuzzy-matches the transcript against known phonetic
spellings of each Sanskrit alphabet.

This is intentionally simple so the project runs "out of the box" on a
student laptop. It is NOT meant to be highly accurate for a language the
underlying ASR engine doesn't truly understand - it is a placeholder that
satisfies the BaseRecognizer contract so it can be swapped later for
Whisper / Vosk / Wav2Vec2 / a custom TensorFlow classifier (see
recognizer_factory.py) without touching the GUI.
"""

import difflib
from typing import Dict, List

import numpy as np

from recognition.base_recognizer import BaseRecognizer, RecognitionResult
from utils.config import RECOGNITION_CONFIG, SUPPORTED_ALPHABETS
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover
    sr = None  # type: ignore


# Phonetic variants that an English-tuned ASR engine commonly produces
# for each Sanskrit alphabet. Used to improve fuzzy-matching accuracy.
PHONETIC_VARIANTS: Dict[str, List[str]] = {
    "Ka": ["ka", "kuh", "car", "kaa"],
    "Kha": ["kha", "kaha", "khaa"],
    "Ga": ["ga", "guh", "gah"],
    "Gha": ["gha", "gaha"],
    "Nga": ["nga", "ngaa", "ang"],
    "Cha": ["cha", "chuh", "char"],
    "Chha": ["chha", "chaha"],
    "Ja": ["ja", "juh", "jaa"],
    "Jha": ["jha", "jaha"],
    "Nya": ["nya", "nyaa"],
    "Ta": ["ta", "tuh", "taa"],
    "Tha": ["tha", "thaa"],
    "Da": ["da", "duh", "daa"],
    "Dha": ["dha", "dhaa"],
    "Na": ["na", "nuh", "naa"],
    "Pa": ["pa", "puh", "paa", "pah"],
    "Pha": ["pha", "faa", "fa"],
    "Ba": ["ba", "buh", "baa"],
    "Bha": ["bha", "baha"],
    "Ma": ["ma", "muh", "maa"],
    "Ya": ["ya", "yuh", "yaa"],
    "Ra": ["ra", "ruh", "raa"],
    "La": ["la", "luh", "laa"],
    "Va": ["va", "vuh", "vaa", "wa"],
    "Sha": ["sha", "shaa", "shuh"],
    "Sa": ["sa", "suh", "saa"],
    "Ha": ["ha", "huh", "haa"],
}


class KeywordRecognizer(BaseRecognizer):
    """Default recognizer: online transcription + fuzzy keyword matching."""

    name = "keyword"

    def __init__(self) -> None:
        self._sr_available = sr is not None
        if not self._sr_available:
            logger.warning(
                "The 'SpeechRecognition' package is not installed; "
                "KeywordRecognizer will be unable to transcribe audio."
            )
        else:
            self._recognizer = sr.Recognizer()

    def recognize(self, samples: np.ndarray, sample_rate: int) -> RecognitionResult:
        transcript = self._transcribe(samples, sample_rate)
        if not transcript:
            return RecognitionResult(alphabet="", confidence=0.0, raw_transcript="")

        alphabet, confidence = self._match_alphabet(transcript)
        if confidence < RECOGNITION_CONFIG.min_confidence_threshold:
            logger.info("Transcript '%s' did not confidently match any alphabet.", transcript)
            return RecognitionResult(alphabet="", confidence=confidence, raw_transcript=transcript)

        logger.info("Matched transcript '%s' -> '%s' (%.0f%%)", transcript, alphabet, confidence * 100)
        return RecognitionResult(alphabet=alphabet, confidence=confidence, raw_transcript=transcript)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Convert raw float32 samples into text using an online ASR engine."""
        if not self._sr_available or samples.size == 0:
            return ""

        try:
            # SpeechRecognition expects 16-bit PCM audio data
            int16_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            audio_data = sr.AudioData(int16_samples.tobytes(), sample_rate, 2)

            text = self._recognizer.recognize_google(audio_data, language=RECOGNITION_CONFIG.language)
            return text.strip().lower()

        except sr.UnknownValueError:
            logger.info("ASR engine could not understand the audio.")
            return ""
        except sr.RequestError as exc:
            logger.warning("ASR request failed (likely no internet connection): %s", exc)
            return ""
        except Exception:
            logger.exception("Unexpected error during transcription")
            return ""

    def _match_alphabet(self, transcript: str) -> "tuple[str, float]":
        """Fuzzy-match a transcript against known phonetic variants of each alphabet."""
        best_alphabet = ""
        best_score = 0.0
        words = transcript.split() or [transcript]

        for alphabet in SUPPORTED_ALPHABETS:
            candidates = PHONETIC_VARIANTS.get(alphabet, [alphabet.lower()])
            for word in words:
                for candidate in candidates:
                    # Skip comparisons between words of very different length -
                    # this is what prevents unrelated filler words ("the", "is",
                    # "today") from loosely matching a short 2-3 letter candidate.
                    if abs(len(word) - len(candidate)) > 2:
                        continue
                    score = difflib.SequenceMatcher(a=word, b=candidate).ratio()
                    if score > best_score:
                        best_score = score
                        best_alphabet = alphabet

        if best_score < RECOGNITION_CONFIG.fuzzy_match_cutoff:
            return "", best_score

        return best_alphabet, best_score
