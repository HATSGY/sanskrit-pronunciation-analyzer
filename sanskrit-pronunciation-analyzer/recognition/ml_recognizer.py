"""
ml_recognizer.py
----------------
Acoustic Machine Learning recognizer for Sanskrit alphabets.

Uses extracted acoustic features (MFCCs, Spectral Centroid, Rolloff, ZCR, Deltas)
and a trained scikit-learn classifier pipeline. Operates 100% offline with
sub-50ms inference time.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from audio.features import extract_feature_vector
from recognition.base_recognizer import BaseRecognizer, RecognitionResult
from utils.config import PROJECT_ROOT, RECOGNITION_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sanskrit_phoneme_model.joblib"


class MLPhonemeRecognizer(BaseRecognizer):
    """Offline Sanskrit phoneme classifier powered by acoustic feature modeling."""

    name = "ml"

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._pipeline = None
        self._classes = []
        self._load_model()

    def _load_model(self) -> bool:
        """Load the trained model pipeline from disk."""
        if not self.model_path.exists():
            logger.warning(
                "Trained phoneme model not found at '%s'. "
                "Run 'python scripts/train_classifier.py --bootstrap' to train it.",
                self.model_path,
            )
            return False

        try:
            payload = joblib.load(self.model_path)
            self._pipeline = payload["pipeline"]
            self._classes = payload.get("classes", list(self._pipeline.classes_))
            logger.info("Loaded Sanskrit phoneme model with %d classes.", len(self._classes))
            return True
        except Exception as exc:
            logger.exception("Failed to load Sanskrit phoneme model from %s: %s", self.model_path, exc)
            self._pipeline = None
            return False

    def recognize(self, samples: np.ndarray, sample_rate: int) -> RecognitionResult:
        if self._pipeline is None:
            # Attempt lazy reload in case model was trained after startup
            if not self._load_model():
                logger.error("No trained model available to perform phoneme recognition.")
                return RecognitionResult(
                    alphabet="",
                    confidence=0.0,
                    raw_transcript="(No trained model found. Run python scripts/train_classifier.py)",
                )

        if samples.size == 0:
            return RecognitionResult(alphabet="", confidence=0.0, raw_transcript="(No audio recorded)")

        try:
            # 1. Extract 117-dimensional acoustic feature vector
            feats = extract_feature_vector(samples, sample_rate=sample_rate)

            # 2. Predict class probabilities
            probs = self._pipeline.predict_proba([feats])[0]
            best_idx = int(np.argmax(probs))
            predicted_alphabet = str(self._classes[best_idx])
            confidence = float(probs[best_idx])

            logger.info(
                "ML Phoneme Prediction: '%s' with confidence %.1f%% (raw prob: %.3f)",
                predicted_alphabet,
                confidence * 100,
                confidence,
            )

            # Check threshold
            min_thresh = getattr(RECOGNITION_CONFIG, "min_confidence_threshold", 0.20)
            # In multi-class (27 classes), uniform random chance is ~3.7%.
            # Any confidence significantly above random (e.g. >15-20%) is a strong signal.
            if confidence < min_thresh:
                logger.info(
                    "Prediction '%s' confidence (%.1f%%) below threshold (%.1f%%)",
                    predicted_alphabet,
                    confidence * 100,
                    min_thresh * 100,
                )
                return RecognitionResult(
                    alphabet="",
                    confidence=confidence,
                    raw_transcript=f"Uncertain: {predicted_alphabet} ({confidence*100:.0f}%)",
                )

            return RecognitionResult(
                alphabet=predicted_alphabet,
                confidence=confidence,
                raw_transcript=f"{predicted_alphabet} (prob: {confidence*100:.0f}%)",
            )

        except Exception:
            logger.exception("Error during ML phoneme recognition")
            return RecognitionResult(alphabet="", confidence=0.0, raw_transcript="(Error analyzing audio)")
